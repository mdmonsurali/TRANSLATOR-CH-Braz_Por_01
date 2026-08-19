"""Table rendering: HTML / Markdown table parsing into a cell grid that
honours colspan/rowspan, then OOXML emission with:
- column widths inferred from cell content + any embedded pictures
- row heights normalised so the table matches its source bbox dimensions
- per-cell font fit to avoid clipping in narrow columns
- inline picture placement inside the matching cell
- fallback to a plain text box when the HTML parse yields no rows
"""
from __future__ import annotations

import logging
import os
import re
from html.parser import HTMLParser
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from .geometry import (
    EMU_PER_PT, bbox_px_to_emu, is_quarter_turn, normalize_rotation,
    rotated_shape_geometry,
)
from .ooxml import (
    NS_W,
    build_anchored_textbox_xml, build_paragraph_xml, build_run_xml,
    build_inline_picture_xml, add_image_relationship,
)
from .picture import apply_pixel_rotation
from .text_fit import (
    fit_multiline, get_font, is_cjk_char, wrap_to_width, has_cjk,
    measure_width_px,
)
# Top-level module, NOT a package sibling: shared verbatim with
# translation_reconstruction so this rule cannot drift between the two copies.
# Both Dockerfiles COPY it to /workspace/, which is already on PYTHONPATH.
from table_geometry import (
    bbox_is_container, clamp_to_natural_width, downward_room_pt,
    dropped_column_index, natural_table_width_emu, rightward_room_pt,
)


# Deepest level of table nesting kept as STRUCTURE. Beyond this a nested table
# is flattened into its parent cell's text — a guard against hallucinated
# runaway nesting, far above anything real (observed max: 1 level).
_MAX_NEST_DEPTH = 3


class Cell(tuple):
    """One parsed table cell.

    A STRICT SUPERSET of the ``(text, colspan, rowspan, img_count)`` 4-tuple this
    parser used to emit, so every existing consumer keeps working unchanged:
    ``c[0]``..``c[3]`` index the same fields and ``len(cell) >= 4`` still holds
    (see `_cell_parts`, and the five ocr_service modules that consume
    `parse_html_table_rows`).

    ``blocks`` is the new 5th field: the cell's content in SOURCE ORDER as a
    tuple of ``("text", str)`` / ``("table", rows)`` items, where ``rows`` has
    the same shape as a top-level parse result. That makes it recursive, so
    arbitrary nesting depth costs nothing extra. It is ``()`` for the
    overwhelmingly common flat cell.

    INVARIANT: ``text`` always holds the cell's FULL visible text INCLUDING the
    text of any nested tables (tab/newline flattened). Renderers that understand
    ``blocks`` draw real nested tables; everything else — and every OCR-quality
    consumer — degrades to that flattened text rather than losing content.

    Subclasses ``tuple`` rather than ``NamedTuple`` so that equality against a
    plain 4-tuple still compares only the first four fields, which several
    call sites rely on.
    """
    # NOTE: no ``__slots__`` — CPython rejects a nonempty ``__slots__`` on a
    # tuple subclass, and an empty one would make the class-level default
    # read-only. The instance ``__dict__`` is what carries ``blocks``, OUTSIDE
    # the tuple payload, so tuple equality, hashing and ``len()`` all behave
    # exactly as the old 4-tuple did.

    def __new__(cls, text: str, colspan: int = 1, rowspan: int = 1,
                imgs: int = 0, blocks: Tuple = ()):
        self = super().__new__(cls, (text, colspan, rowspan, imgs))
        self._blocks = blocks
        return self

    @property
    def text(self) -> str:
        return self[0]

    @property
    def colspan(self) -> int:
        return self[1]

    @property
    def rowspan(self) -> int:
        return self[2]

    @property
    def imgs(self) -> int:
        return self[3]

    @property
    def blocks(self) -> Tuple:
        return self._blocks

    def with_span(self, text: str, colspan: int, rowspan: int,
                  imgs: int) -> "Cell":
        """Copy with new span values, PRESERVING ``blocks`` (used by
        `_normalize_column_granularity`'s re-span, which must not drop nested
        tables)."""
        return Cell(text, colspan, rowspan, imgs, self._blocks)


def flatten_block_text(blocks: Tuple) -> str:
    """Plain-text rendering of a cell's ``blocks``, used for `Cell.text`.

    Nested-table rows become tab-separated cells, one line per row, so the
    content is never lost for consumers that don't understand `Cell.blocks`.
    """
    out: List[str] = []
    for kind, payload in blocks:
        if kind == "text":
            if payload:
                out.append(payload)
        else:                                  # ("table", rows)
            for row, _is_header in payload:
                out.append("\t".join((c[0] or "") for c in row))
    return "\n".join(p for p in out if p)


class _TableFrame:
    """State of ONE open ``<table>``."""
    __slots__ = ("table", "row", "in_row", "in_header")

    def __init__(self):
        self.table: List[Tuple[List[Cell], bool]] = []
        self.row: List[Cell] = []
        self.in_row = False
        self.in_header = False


class _CellFrame:
    """State of ONE open ``<td>``/``<th>``.

    ``depth`` records how many table frames were open when this cell started.
    Implicit closes (a ``<tr>``/``<td>`` that arrives while a previous cell is
    still open) must only close cells belonging to the CURRENT table — without
    this, the ``<tr>`` of a NESTED table would pop the enclosing table's open
    cell and the nested rows would escape to top level.
    """
    __slots__ = ("blocks", "buf", "colspan", "rowspan", "imgs", "is_header",
                 "depth")

    def __init__(self, colspan: int, rowspan: int, is_header: bool,
                 depth: int):
        self.blocks: List[Tuple[str, object]] = []
        self.buf: List[str] = []
        self.colspan = colspan
        self.rowspan = rowspan
        self.imgs = 0
        self.is_header = is_header
        self.depth = depth


class TableHTMLParser(HTMLParser):
    """Parses ``<table>``/``<tr>``/``<th>``/``<td>`` + colspan/rowspan.

    Each cell is captured as a `Cell` — a 4-tuple superset carrying
    ``(text, colspan, rowspan, img_count)`` plus structured ``blocks``. Caller
    can then build a proper OOXML gridSpan / vMerge layout to honour merged
    header rows seen in the source PDFs (very common in CJK technical reports).

    FULLY RE-ENTRANT over nesting. Parse state lives on two explicit stacks
    (``_tables`` / ``_cells``) rather than flat instance attributes. A nested
    ``<table>`` inside a ``<td>`` used to CLOBBER the enclosing table's
    in-progress row/cell — the same attributes it was using — which split the
    document into several bogus top-level tables, promoted the nested rows to
    top level, destroyed the outer row's other cells and duplicated rows. Now a
    nested table is captured in its own frame and attached to the enclosing cell
    as structure (see `Cell.blocks`).
    """

    def __init__(self):
        super().__init__()
        self.tables: List[List[Tuple[List[Cell], bool]]] = []
        self._tables: List[_TableFrame] = []
        self._cells: List[_CellFrame] = []

    # ── Derived flags (kept as read-only properties so the old attribute names
    # still read correctly from outside; they are now a function of depth). ──

    @property
    def in_table(self) -> bool:
        return bool(self._tables)

    @property
    def in_cell(self) -> bool:
        return bool(self._cells)

    @property
    def in_row(self) -> bool:
        return bool(self._tables) and self._tables[-1].in_row

    @property
    def in_header(self) -> bool:
        return bool(self._tables) and self._tables[-1].in_header

    @staticmethod
    def _attr_int(attrs, name: str, default: int = 1) -> int:
        for k, v in attrs:
            if k == name and v:
                try:
                    n = int(v)
                    return n if n > 0 else default
                except (TypeError, ValueError):
                    return default
        return default

    # ── Cell-content helpers ─────────────────────────────────────────────────

    def _append_text(self, s: str) -> None:
        if self._cells:
            self._cells[-1].buf.append(s)

    def _flush_text(self) -> None:
        """Move the pending character run into the open cell's ordered blocks."""
        if not self._cells:
            return
        cf = self._cells[-1]
        if cf.buf:
            chunk = "".join(cf.buf)
            cf.buf = []
            if chunk.strip():
                cf.blocks.append(("text", chunk.strip()))

    def _close_own_cell(self) -> None:
        """Implicitly close an open cell ONLY if it belongs to the current
        table frame.

        A ``<tr>`` or ``<td>`` inside a NESTED table must not close the
        enclosing table's still-open cell — that cell is what the nested table
        is being collected into.
        """
        if self._cells and self._cells[-1].depth == len(self._tables):
            self._close_cell()

    def _close_cell(self) -> None:
        """Finish the innermost open cell and append it to its table's row."""
        if not self._cells:
            return
        self._flush_text()
        cf = self._cells.pop()
        blocks = tuple(cf.blocks)
        cell = Cell(
            flatten_block_text(blocks).strip(),
            cf.colspan, cf.rowspan, cf.imgs, blocks,
        )
        if self._tables:
            self._tables[-1].row.append(cell)

    def _close_row(self) -> None:
        """Finish the innermost open row."""
        if not self._tables:
            return
        tf = self._tables[-1]
        if tf.row:
            tf.table.append((tf.row, tf.in_header))
            tf.row = []
        tf.in_row = False
        tf.in_header = False

    def _close_table(self) -> None:
        """Finish the innermost open table, attaching it to the enclosing cell
        when there is one (nested), else publishing it as top level."""
        if not self._tables:
            return
        self._close_own_cell()      # a <td> left unclosed by truncated HTML
        self._close_row()
        tf = self._tables.pop()
        if not tf.table:
            return
        if self._cells:
            # Nested: becomes a structured block of the enclosing cell, in
            # source order relative to that cell's surrounding text.
            self._cells[-1].blocks.append(("table", tf.table))
        else:
            self.tables.append(tf.table)

    # ── SAX handlers ─────────────────────────────────────────────────────────

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "table":
            # A nested table interrupts the enclosing cell's text run; flush it
            # first so caption/table/caption ordering is preserved.
            if self._cells:
                if len(self._tables) >= _MAX_NEST_DEPTH:
                    return          # too deep: treat as plain text (tags dropped)
                self._flush_text()
            self._tables.append(_TableFrame())
        elif tag == "tr":
            if not self._tables:
                return
            # An unclosed <td> before this <tr> is closed implicitly rather
            # than discarded (the old code silently dropped it) — but only if
            # it belongs to THIS table, not to an enclosing one.
            self._close_own_cell()
            self._close_row()
            self._tables[-1].in_row = True
        elif tag in ("td", "th"):
            if not self._tables:
                return
            # Implicitly close a previous unclosed cell in the same row.
            self._close_own_cell()
            if tag == "th":
                self._tables[-1].in_header = True
            self._cells.append(_CellFrame(
                self._attr_int(attrs, "colspan", 1),
                self._attr_int(attrs, "rowspan", 1),
                tag == "th",
                len(self._tables),
            ))
        elif tag == "br" and self._cells:
            self._append_text("\n")
        elif tag == "img" and self._cells:
            # <img> placeholders (e.g. table-cell diagrams Chandra emits as a
            # bare <img alt=...>) carry no visible text but authoritatively mark
            # WHICH cell holds a picture and HOW MANY. Count them per cell so the
            # recovered Image entries can be assigned to the right cell later.
            self._cells[-1].imgs += 1
        elif tag == "input" and self._cells:
            # OCR marks form checkboxes as <input type="checkbox" [checked]>.
            # Render the state as a Unicode box glyph so it survives (the parser
            # otherwise drops the tag and the checked/unchecked state is lost).
            attr = dict(attrs)
            # Strip a stray trailing slash the parser can leave on an unquoted
            # attr (e.g. `type=checkbox/` → `checkbox/`).
            itype = (attr.get("type") or "checkbox").lower().rstrip("/").strip()
            if itype in ("checkbox", "radio"):
                checked = "checked" in attr  # value is "" but key present
                self._append_text("☑" if checked else "☐")
        elif tag == "p" and self._cells:
            # A <p> inside a cell is a hard break between blocks (e.g. the
            # caption above each nested sub-table).
            self._append_text("\n")

    def handle_startendtag(self, tag, attrs):
        # e.g. self-closing ``<br/>`` / ``<input .../>`` / ``<img .../>``
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "table":
            self._close_table()
        elif tag == "tr":
            self._close_own_cell()
            self._close_row()
        elif tag in ("td", "th"):
            self._close_own_cell()
        elif tag == "p" and self._cells:
            # END of a paragraph closes its BLOCK, so a cell holding several
            # <p>s keeps them as separate ("text", …) blocks instead of one
            # newline-joined run. That structure is what lets a sub-table the
            # OCR flattened into parallel <p> columns be recognised and rebuilt
            # (see `_promote_flattened_subtables`). `Cell.text` is unchanged —
            # `flatten_block_text` re-joins the blocks with newlines exactly as
            # the old single-buffer path produced them.
            self._flush_text()

    def handle_data(self, data):
        if self._cells:
            self._append_text(data)

    def finalize(self):
        """Flush any cell/row/table left open because the HTML was truncated.

        OCR/VLM output is frequently cut off mid-table (no closing ``</td>`` /
        ``</tr>`` / ``</table>``). Without this, an unterminated ``<table>`` is
        dropped entirely and the whole table silently falls back to raw-text
        rendering. We close the open cell, row and table in order so the
        content is still recovered as a table.

        Unwinds to arbitrary depth, innermost first, so a truncated NESTED
        table is still attached to its enclosing cell exactly as a well-formed
        ``</table>`` would have attached it.
        """
        while self._tables:
            self._close_table()     # closes its own open cell + row first
        # A stray cell with no enclosing table has nowhere to go; drop its
        # frame so the parser is left in a clean state.
        self._cells.clear()


# Sanity bounds. OCR/VLM output on unparseable pages (rotated CAD drawings,
# scans) sometimes hallucinates tables with hundreds of near-identical rows or
# dozens of repeated header columns. These caps keep such garbage bounded so it
# can't overflow the page or explode render time; they're far above any real
# table seen in these documents.
MAX_TABLE_ROWS = 200
MAX_TABLE_COLS = 40

log = logging.getLogger("ocr_reconstruction.table")





def _cell_parts(cell) -> Tuple[str, int, int, int]:
    """Unpack a parsed cell tuple tolerantly as (text, colspan, rowspan,
    img_count). Cells produced by `TableHTMLParser` carry 4 fields, but
    Markdown-derived and rebuilt cells may still be 3-tuples — default
    img_count to 0 for those."""
    if len(cell) >= 4:
        t, cs, rs, imgs = cell[0], cell[1], cell[2], cell[3]
    else:
        t, cs, rs = cell[0], cell[1], cell[2]
        imgs = 0
    return t, cs, rs, imgs


def parse_html_table_rows(
    html_string: str,
) -> List[Tuple[List[Tuple[str, int, int, int]], bool]]:
    """Return a flat list of (cells, is_header) pairs across all tables found.

    Each cell is `(text, colspan, rowspan, img_count)` — `img_count` is the
    number of `<img>` placeholders inside the cell (0 for text-only cells).
    Robust to truncated HTML (missing closing tags) and to runaway/degenerate
    OCR output: consecutive byte-identical rows are collapsed and the total row
    count is capped.
    """
    parser = TableHTMLParser()
    try:
        parser.feed(html_string)
    except Exception:
        pass
    parser.finalize()   # recover any table left open by truncated HTML

    rows: List[Tuple[List[Tuple[str, int, int, int]], bool]] = []
    prev_sig = None
    for table in parser.tables:
        for row, is_header in table:
            # Collapse runs of identical rows (a common hallucination shape).
            sig = tuple(_cell_parts(c) for c in row)
            if sig == prev_sig:
                continue
            prev_sig = sig
            rows.append((row, is_header))
            if len(rows) >= MAX_TABLE_ROWS:
                return rows
    return rows


# ── Non-table HTML around a table ─────────────────────────────────────────────
# A single OCR "Table" entry frequently carries MORE than the ``<table>`` grid:
# form checkboxes, review paragraphs, signatures, comments, etc. emitted as
# ``<p>``/``<br>``/``<input>``/``<u>`` after (or before) the table. Rendering
# only the grid silently drops all of it. These helpers extract that surrounding
# HTML and flatten it to plain text so it can be rendered below the table. The
# flattening is GENERAL — it is not tied to any particular document, language or
# checkbox wording.

# Every ``<table>`` / ``</table>`` tag, so the spans below can be found by DEPTH
# COUNTING. A regex cannot balance-match nested constructs: the old
# ``<table\b.*?</table\s*>`` was NON-GREEDY, so on a table containing a nested
# table it matched from the OUTER ``<table>`` to the FIRST (inner) ``</table>``.
# Everything after that inner close — the remaining sub-tables' captions AND the
# whole rest of the outer table — was then treated as "outside" text and leaked
# into the trailing-text block.
_TABLE_TAG_RE = re.compile(r"<(/?)table\b[^>]*>", re.IGNORECASE)


def _top_level_table_spans(text: str) -> List[Tuple[int, int]]:
    """``(start, end)`` offsets of each TOP-LEVEL ``<table>…</table>`` block.

    Nesting-aware: an inner ``<table>`` inside an outer one does NOT start a new
    span, it just increments the depth, so the span returned covers the whole
    outer table including all of its descendants.

    Two robustness rules for the truncated HTML the OCR emits:
      * an unclosed outer ``<table>`` runs to end-of-string (mirrors
        `TableHTMLParser.finalize`, and stops the truncated tail leaking as prose);
      * a stray ``</table>`` while at depth 0 is ignored rather than corrupting
        the scan.
    """
    spans: List[Tuple[int, int]] = []
    depth = 0
    start: Optional[int] = None
    for m in _TABLE_TAG_RE.finditer(text):
        if not m.group(1):                    # opening <table>
            if depth == 0:
                start = m.start()
            depth += 1
        elif depth:                           # closing </table>
            depth -= 1
            if depth == 0 and start is not None:
                spans.append((start, m.end()))
                start = None
    if depth and start is not None:
        spans.append((start, len(text)))      # truncated → swallow the remainder
    return spans


class _HTMLToTextParser(HTMLParser):
    """Flatten an arbitrary HTML fragment to plain text.

    Handles the constructs OCR emits around tables generically:
      * ``<input type=checkbox|radio [checked]>`` → ``☑`` / ``☐`` (state kept);
      * ``<br>`` and block boundaries (``</p>``, ``</div>``, ``<li>``, ``</tr>``)
        → newlines so the layout survives;
      * every other tag is dropped but its text is kept;
      * HTML entities are unescaped (handled by HTMLParser.handle_data for
        charrefs via ``convert_charrefs=True``).
    Runs of blank lines are collapsed and leading/trailing space trimmed.
    """
    _BLOCK_TAGS = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._buf: List[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "br":
            self._buf.append("\n")
        elif tag == "input":
            attr = dict(attrs)
            itype = (attr.get("type") or "checkbox").lower().rstrip("/").strip()
            if itype in ("checkbox", "radio"):
                self._buf.append("☑" if "checked" in attr else "☐")
        elif tag in self._BLOCK_TAGS:
            self._buf.append("\n")

    def handle_startendtag(self, tag, attrs):
        # e.g. self-closing ``<input .../>`` / ``<br/>``
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag.lower() in self._BLOCK_TAGS:
            self._buf.append("\n")

    def handle_data(self, data):
        self._buf.append(data)

    def get_text(self) -> str:
        raw = "".join(self._buf)
        # Collapse intra-line whitespace, keep newlines, drop blank-line runs.
        lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in raw.split("\n")]
        out: List[str] = []
        for ln in lines:
            if ln or (out and out[-1]):
                out.append(ln)
        return "\n".join(out).strip()


def html_fragment_to_text(fragment: str) -> str:
    """Plain-text rendering of a non-table HTML ``fragment`` (checkbox glyphs,
    line breaks preserved). Returns ``""`` when the fragment has no visible
    text."""
    if not fragment or not fragment.strip():
        return ""
    p = _HTMLToTextParser()
    try:
        p.feed(fragment)
    except Exception:
        pass
    return p.get_text()


def extract_non_table_text(text: str) -> str:
    """Return the plain text of everything in ``text`` OUTSIDE its ``<table>``
    block(s), in source order (content before the first table, between tables,
    and after the last), flattened via `html_fragment_to_text`.

    Empty when the entry is a bare table with no surrounding content — so the
    common case stays a pure table render with zero behavioural change.

    Uses `_top_level_table_spans` (depth-counting) rather than a regex so a
    NESTED table is treated as part of its enclosing table, not as a boundary
    that turns the rest of the outer table into "outside" prose.
    """
    if not text or "<table" not in text.lower():
        return ""
    spans = _top_level_table_spans(text)
    if not spans:
        return html_fragment_to_text(text)
    # Join the complement of the table spans: before the first, between each
    # pair, and after the last.
    parts: List[str] = []
    prev = 0
    for s, e in spans:
        parts.append(text[prev:s])
        prev = e
    parts.append(text[prev:])
    return html_fragment_to_text("\n".join(parts))


def parse_markdown_table(md_text: str) -> Optional[List[List[str]]]:
    lines = md_text.strip().split("\n")
    rows = []
    for line in lines:
        if re.match(r"^[\|\s:\-]+$", line):
            continue
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells and any(c for c in cells):
            rows.append(cells)
    return rows or None


def _display_width(s: str) -> int:
    n = 0
    for ch in (s or ""):
        if is_cjk_char(ch):
            n += 2
        else:
            n += 1
    return n


def _longest_token_width(s: str) -> int:
    """Display width of the longest unbreakable run in `s`.

    CJK breaks freely between chars (so a long CJK string isn't an
    unbreakable token), but Latin words and digit runs can't be broken
    mid-word — those set a min column width below which the word would
    visibly overflow or force odd wrapping. Newlines from <br> also force
    a break, so split on them.
    """
    if not s:
        return 0
    best = 0
    for line in s.split("\n"):
        run = 0
        for ch in line:
            if is_cjk_char(ch) or ch.isspace():
                if run > best:
                    best = run
                run = 0
            else:
                run += 1
        if run > best:
            best = run
    return best


def _longest_word_px(s: str, font, size_px: int) -> float:
    """Pixel width of the longest UNBREAKABLE run in `s` at the render font.

    Sibling of `_longest_token_width` but measured in pixels (not display-unit
    char counts) so it can be compared directly to a column's usable width. Word
    can only break at whitespace, newlines, or between CJK glyphs, so we split on
    exactly those and measure each maximal non-CJK sub-run. A pure-CJK string has
    no unbreakable run wider than a single glyph, so it is never flagged as
    overflowing (CJK columns stay freely squeezable)."""
    if not s:
        return 0.0
    best = 0.0
    for line in s.split("\n"):
        for run in re.split(r"\s+", line):
            if not run:
                continue
            sub = ""
            for ch in run:
                if is_cjk_char(ch):
                    if sub:
                        best = max(best, measure_width_px(sub, font, size_px))
                        sub = ""
                else:
                    sub += ch
            if sub:
                best = max(best, measure_width_px(sub, font, size_px))
    return best


# A short "label" (header, code, category) should sit on at most this many
# lines instead of collapsing into a tall vertical strip.
_LABEL_MAX_LINES = 2
# Cells at/under this display width are treated as labels (headers / short
# values). ~16 display units ≈ 8 CJK glyphs.
_LABEL_MAX_DISPLAY_WIDTH = 16
# NO cell — however long — should be forced to wrap to more than this many
# lines. Beyond this a cell reads as a 1-glyph-per-line vertical strip and
# clips. A long paragraph cell therefore demands at least
# ceil(display_width / _CELL_MAX_LINES) width from its column.
_CELL_MAX_LINES = 16
# But a single cell's demand is capped here (display units) so one very long
# paragraph can't claim more than a fair slice; the column's proportional
# weight (75th percentile) covers the rest, and the redistribution loop
# balances against the other columns.
_CELL_MIN_WIDTH_CAP = 22

# Word's DEFAULT table-cell margin is 108 twips = 5.4 pt on the left and right of
# every cell (top/bottom default to 0). We pin exactly these margins in the
# emitted `<w:tblCellMar>` so the render is deterministic, and subtract them here
# as a FIXED amount when computing a cell's usable text width. The old
# proportional `cell_w_pt * 0.93` under-reserved space for any column narrower
# than ~150 pt (7% of a narrow column is less than the fixed 10.8 pt Word
# removes), so those columns wrapped to one more line than the fitter predicted
# and clipped under `hRule="exact"`.
_CELL_MARGIN_PT = 5.4

# ── EMU → twips, for the OOXML emission boundary ONLY ────────────────────────
# WordprocessingML table widths (`w:tblW`, `w:tcW`, `w:gridCol`) are `dxa` —
# TWENTIETHS OF A POINT. The whole internal fitter works in EMU. Emitting a raw
# EMU value under `w:type="dxa"` overstates every width by exactly
# EMU_PER_PT/20 = 635x, which made a single column wider than the page: with
# `tblLayout="fixed"` Word takes `tblGrid` verbatim (no shrink-to-fit) and
# relocates the columns that cannot be hosted, landing them on the next page.
#
# `w:trHeight` and `w:tblCellMar` in this file were already correct (`pt * 20`);
# only the widths were wrong. Convert HERE, at the attribute, and never in the
# fitter — its EMU-vs-EMU and pt-vs-pt comparisons are consistent and correct.
#
# NOTE: DrawingML (`wp:extent`, `wp:posOffset`, `a:ext`) is EMU and must NOT be
# converted — those stay exactly as they are.
EMU_PER_TWIP = EMU_PER_PT // 20        # 635


def _emu_tw(emu: int) -> int:
    """One EMU width as an integer twips value for a `dxa` attribute."""
    return max(1, int(round(emu / EMU_PER_TWIP)))


def _emu_tw_split(emu_widths: List[int]) -> List[int]:
    """Column widths in twips whose SUM EQUALS the twips width of their sum.

    Rounding each column independently drifts by up to n/2 twips, which breaks
    the ``sum(gridCol) == tblW`` invariant the EMU side maintains exactly. Under
    ``tblLayout="fixed"`` Word trusts ``tblGrid`` and silently re-derives the
    table width from it, so that drift would show up as a table a few twips
    wider or narrower than its shape.

    Largest-remainder (Hamilton) apportionment: floor every column, then hand
    the leftover twips one each to the columns with the largest discarded
    fractions. Flooring also guarantees the table is never WIDER than the EMU
    intent (at most ~1 twip = 1/1440 in narrower), which is the safe direction
    for page fidelity — it can never nudge the last column past the box edge.
    """
    if not emu_widths:
        return []
    total_tw = _emu_tw(sum(emu_widths))
    floors = [max(1, emu // EMU_PER_TWIP) for emu in emu_widths]
    short = total_tw - sum(floors)
    if short > 0:
        order = sorted(range(len(emu_widths)),
                       key=lambda i: emu_widths[i] % EMU_PER_TWIP, reverse=True)
        for i in range(short):
            floors[order[i % len(order)]] += 1
    elif short < 0:
        # Only reachable when the `max(1, ...)` floor lifted narrow columns.
        order = sorted(range(len(floors)), key=lambda i: floors[i], reverse=True)
        for i in range(-short):
            j = order[i % len(order)]
            if floors[j] > 1:
                floors[j] -= 1
    return floors


def _usable_w_pt(cell_w_pt: float) -> float:
    """Usable text width (pt) inside a cell whose grid width is `cell_w_pt`,
    after Word's fixed left+right cell margins. Used by EVERY wrap measurement
    (overflow gate, column balancer, initial row-line estimate, final emission)
    so they can never disagree about how wide a cell's text area is."""
    return max(1.0, cell_w_pt - 2.0 * _CELL_MARGIN_PT)


def _content_min_width(s: str) -> int:
    """Content-aware minimum column width (in display units) for cell text `s`.

    Unlike `_longest_token_width` (which is ~1 for pure CJK because CJK breaks
    between glyphs, leaving CJK columns freely squeezable), this returns a width
    that stops a cell from wrapping into a tall vertical strip:
      * a SHORT label (≤ _LABEL_MAX_DISPLAY_WIDTH) should fit on
        ≤ _LABEL_MAX_LINES lines → needs ceil(D / _LABEL_MAX_LINES);
      * a LONG paragraph should still not exceed _CELL_MAX_LINES lines →
        needs ceil(D / _CELL_MAX_LINES), capped at _CELL_MIN_WIDTH_CAP so a
        lone huge cell can't claim the whole table.
    The max over the cell's newline-separated segments is returned.
    """
    if not s or not s.strip():
        return 0
    best = 0
    for seg in s.split("\n"):
        seg = seg.strip()
        if not seg:
            continue
        dw = _display_width(seg)
        if dw <= _LABEL_MAX_DISPLAY_WIDTH:
            need = -(-dw // _LABEL_MAX_LINES)          # short label: ≤2 lines
        else:
            need = min(
                _CELL_MIN_WIDTH_CAP, -(-dw // _CELL_MAX_LINES)  # long: ≤16 lines
            )
        if need > best:
            best = need
    return best


# ── Rebuilding a sub-table the OCR flattened ─────────────────────────────────
# The renderer draws a real nested table only when the source HTML actually
# nests one (`Cell.blocks` gains a ("table", rows) item solely from a literal
# `<table>` inside a `<td>`). OCR/VLM output frequently does NOT nest: a
# sub-table gets emitted as ordinary top-level structure instead, and the
# sub-table's own grid is lost.
#
# Two shapes are recognised, both keyed ONLY on structure — no table-, column-,
# document- or language-specific assumptions, and each is a strict no-op unless
# its shape is present:
#
#   A. ROW-RUN CONFINEMENT. A run of consecutive rows whose non-empty cells all
#      sit inside the SAME contiguous column span, with at least one column
#      outside that span empty across the whole run. Those rows are the
#      sub-table's rows promoted to top level; the run collapses to ONE outer
#      row carrying them as a nested table. (A mid-table continuation page is
#      the usual producer: the outer row has no interior rules at all, so every
#      outer column is blank beside the sub-table.)
#
#   B. PARALLEL BLOCK PAIRING. Within ONE row, k >= 2 horizontally adjacent
#      cells each holding the SAME number N >= 2 of paragraph blocks. Those are
#      the sub-table's columns emitted side by side, so the i-th block of each
#      cell is the i-th sub-row.
#
# Both emit exactly the ("table", rows) block shape `TableHTMLParser` produces,
# so the existing `_extract_blocks_by_anchor` / `_cell_blocks_xml` /
# `render_nested_table_xml` path renders the result with no emitter changes.

# A run must be at least this many rows to read as a sub-table. Two is the
# smallest number that shows a repeated pattern; one confined row is just a row
# with empty cells, which is ordinary and must not be touched.
_MIN_PROMOTE_ROWS = 2
# Likewise for parallel blocks: a single block per cell is a plain cell.
_MIN_PROMOTE_BLOCKS = 2
# Longest a paired block may be (characters) for signal B to read the cells as
# sub-table CELLS rather than as prose.
#
# Equal block counts are a weak signal on their own: two neighbouring cells of
# ordinary running prose can trivially hold the same number of paragraphs. The
# separation that actually holds is what those blocks ARE — a sub-table's cells
# are labels and values (a line or two), whereas prose blocks are paragraphs.
# Measured on this corpus: a real flattened sub-table's longest block is 74
# chars, while the false positive (two columns of numbered prose, no nested
# ruling anywhere in the source) runs to 476. The gap is wide enough that the
# exact cut hardly matters; 120 sits comfortably inside it.
_MAX_PROMOTE_BLOCK_CHARS = 120


def _cell_text_blocks(cell) -> List[str]:
    """The cell's paragraph blocks, when it carries structure worth pairing.

    Returns [] for anything that is not a plain multi-block text cell — a cell
    that already holds a nested table is left alone (never re-promoted), and a
    bare tuple from a caller that hand-built its rows simply has no blocks.
    """
    blocks = getattr(cell, "blocks", None) or ()
    if any(k == "table" for k, _ in blocks):
        return []
    out = [v for k, v in blocks if k == "text" and (v or "").strip()]
    if len(out) < _MIN_PROMOTE_BLOCKS:
        return []
    # Paragraph-sized blocks mean this is prose, not a column of sub-table cells.
    if any(len(v) > _MAX_PROMOTE_BLOCK_CHARS for v in out):
        return []
    return out


def _has_nested_table(rows) -> bool:
    """True when any cell already carries real nested-table structure."""
    for cells, _is_header in rows:
        for c in cells:
            if any(k == "table" for k, _ in (getattr(c, "blocks", None) or ())):
                return True
    return False


def _promote_parallel_blocks(cells: List) -> Optional[List]:
    """Signal B on ONE row: pair k>=2 adjacent multi-block cells into a nested
    table. Returns the rewritten cell list, or None when the shape is absent.

    The cells must be ADJACENT and agree on their block count — that agreement
    is the evidence they are columns of one sub-table rather than unrelated
    multi-paragraph cells. Disagreement means we cannot map blocks to sub-rows
    unambiguously, so we leave the row untouched rather than guess (the same
    rule `_normalize_column_granularity` applies to its own re-span).

    CONFINEMENT IS ALSO REQUIRED. Equal block counts alone are far too weak: a
    perfectly ordinary full-width data row can have two neighbouring cells that
    happen to hold the same number of paragraphs (measured: two real rows in this
    corpus at 8+8 and 7+7 blocks, both with every outer column filled and no
    nested ruling in the source at all). Demanding that some column in the row be
    EMPTY — the same "the outer grid is blank beside the sub-table" evidence
    signal A relies on — separates a genuine sub-table from that coincidence.
    """
    # A cell holding only a picture is CONTENT, not a blank: a picture column
    # with no text must not read as the blank margin beside a sub-table.
    def _has_content(c) -> bool:
        t, _cs, _rs, imgs = _cell_parts(c)
        return bool((t or "").strip()) or imgs > 0

    if all(_has_content(c) for c in cells):
        return None                      # full-width data row, not a sub-table
    blocks_per = [_cell_text_blocks(c) for c in cells]
    best: Optional[Tuple[int, int, int]] = None      # (start, end, n_blocks)
    i = 0
    while i < len(cells):
        n = len(blocks_per[i])
        if n >= _MIN_PROMOTE_BLOCKS:
            j = i
            while j + 1 < len(cells) and len(blocks_per[j + 1]) == n:
                j += 1
            if j > i and (best is None or (j - i) > (best[1] - best[0])):
                best = (i, j, n)
            i = j + 1
        else:
            i += 1
    if best is None:
        return None
    start, end, n_blocks = best
    # Row i of the sub-table = the i-th block of each paired cell.
    sub_rows = [
        ([Cell(blocks_per[c][r], 1, 1, 0) for c in range(start, end + 1)], False)
        for r in range(n_blocks)
    ]
    src = cells[start]
    span = sum(max(1, _cell_parts(c)[1]) for c in cells[start:end + 1])
    _t, _cs, rs, imgs = _cell_parts(src)
    holder = Cell(
        flatten_block_text((("table", sub_rows),)).strip(),
        span, rs, imgs, (("table", sub_rows),),
    )
    return list(cells[:start]) + [holder] + list(cells[end + 1:])


def _promote_flattened_subtables(rows):
    """Rebuild sub-tables the OCR flattened (signals A and B above).

    Returns the SAME ``rows`` object when nothing matched, so callers can detect
    the no-op cheaply and well-formed tables stay byte-identical.
    """
    from collections import Counter

    if not rows or _has_nested_table(rows):
        return rows

    # ── Three-state column classification ───────────────────────────────────
    # A column can only be EVIDENCE about the row it actually belongs to. Each
    # grid column of each row is exactly one of:
    #
    #   CONTENT  the row's own cell, carrying text OR a picture (`img_count`).
    #            An <img>-only cell is content: a picture column that happens to
    #            hold no text is not a blank left margin.
    #   BLANK    the row's own cell, actually emitted and empty. THIS is the
    #            evidence that a sub-table is inset from the outer grid.
    #   COVERED  owned by a `rowspan` cell anchored in a DIFFERENT row. Carries
    #            no evidence either way — it is not this row's to speak for.
    #
    # Conflating COVERED with BLANK is what made this promoter destroy ordinary
    # flat tables: a row under a `rowspan` legitimately emits fewer <td>s, which
    # looked exactly like an inset sub-table and collapsed real data rows.
    #
    # Occupancy is walked exactly as `parse_table_grid._place` does it, so the
    # classification agrees with the grid the renderer will actually build.
    widths = [sum(max(1, _cell_parts(c)[1]) for c in cells) for cells, _h in rows]
    grid_w = max(widths) if widths else 0
    n_rows_in = len(rows)
    if grid_w <= 0:
        return rows

    _CONTENT, _BLANK = "content", "blank"
    state: List[List[Optional[str]]] = [
        [None] * grid_w for _ in range(n_rows_in)
    ]
    # Occupancy is tracked SEPARATELY from `state`. `state[r][c]` records only
    # what row `r`'s OWN cell at column `c` is (content / blank / covered), while
    # `taken[r][c]` records whether ANY cell — including one anchored in an
    # earlier row via `rowspan` — already owns that slot.
    #
    # These were one array, and conflating them mis-placed every row under a tall
    # `rowspan`. A `rowspan=16` label in column 0 left rows 1..15 free at column 0
    # (only the anchor row was written), so each of those rows' leading blank
    # <td> slid LEFT into column 0 and `_confined` read it as a sub-table inset —
    # collapsing six real data rows into one flattened cell and losing the first.
    # Marking the covered slots taken keeps every later row in its true column.
    taken: List[List[bool]] = [[False] * grid_w for _ in range(n_rows_in)]
    for r, (cells, _h) in enumerate(rows):
        col = 0
        for c in cells:
            t, cs, rs, imgs = _cell_parts(c)
            cs = max(1, cs)
            rs = max(1, rs)
            while col < grid_w and taken[r][col]:
                col += 1
            if col >= grid_w:
                break
            cs = min(cs, grid_w - col)
            rs = min(rs, n_rows_in - r)
            kind = _CONTENT if ((t or "").strip() or imgs > 0) else _BLANK
            for rr in range(r, r + rs):
                for cc in range(col, col + cs):
                    taken[rr][cc] = True
                    # Only the anchor row gets the cell's own state; the rows it
                    # merely spans into are COVERED (left as None).
                    if rr == r:
                        state[rr][cc] = kind
            col += cs

    spans: List[Optional[Tuple[int, int]]] = []
    for r in range(n_rows_in):
        content = [c for c in range(grid_w) if state[r][c] == _CONTENT]
        spans.append((min(content), max(content) + 1) if content else None)

    # ── Signal A: maximal runs of rows confined to the same inner span ───────
    consumed = [False] * len(rows)
    out: List = []
    i = 0
    changed = False
    def _confined(idx: int) -> bool:
        """Row `idx` has a genuine BLANK cell of its own left of its content.

        A LEFT inset is required, not merely any blank column: trailing blanks
        are ordinary row padding (a ragged last row, an over-segmented row) and
        carry no evidence of nesting. And the inset must be made of the row's
        OWN blank cells — a leading column that is merely `rowspan`-COVERED
        belongs to another row and says nothing about this one.
        """
        sp = spans[idx]
        if sp is None:
            return False
        return any(state[idx][c] == _BLANK for c in range(sp[0]))

    def _run_is_bounded(start: int, end: int, lo: int, hi: int) -> bool:
        """True when the run `start..end` really is an ENCLOSED block.

        Two independent requirements, both keyed only on grid shape:

        1. ENCLOSED HORIZONTALLY. A nested sub-table sits inside the outer grid,
           so it must leave a column free on BOTH sides. A run that is merely
           inset on the left but runs out to the grid's right edge is an ordinary
           row group under a `rowspan` label, not a sub-table. Without this, a
           tall label in column 0 (`rowspan=16`) made every row beneath it look
           inset, and six real data rows were collapsed into one flattened cell —
           losing the first row and shifting the rest.

        2. SEPARABLE VERTICALLY. The rows immediately outside the run must not
           continue its column pattern. When the neighbour's content starts in
           the same column the run does, the run is that row's own continuation.
        """
        # (1) must be inset on the left AND stop short of the right edge.
        if lo <= 0 or hi >= grid_w:
            return False
        # (2) neighbours must not share the run's starting column.
        for nb in (start - 1, end + 1):
            if not (0 <= nb < n_rows_in):
                continue
            sp_nb = spans[nb]
            if sp_nb is not None and sp_nb[0] == lo:
                return False
        return True

    while i < len(rows):
        sp = spans[i]
        # Confined = leaves at least one whole column outside the span, on
        # either side. A full-width row is ordinary content, never a sub-table.
        if _confined(i):
            j = i
            lo, hi = sp
            # The span the run's rows AGREE on. A single row that spans wider
            # (an OCR `colspan` stretched over the blank outer columns) must not
            # define the sub-table's width — that is what the modal span below
            # protects against.
            narrow_lo, narrow_hi = lo, hi
            while j + 1 < len(rows) and _confined(j + 1):
                # EVERY row of the run must be confined ON ITS OWN (see above);
                # the merged span must still leave an outside column free.
                nlo, nhi = spans[j + 1]
                mlo, mhi = min(lo, nlo), max(hi, nhi)
                if mlo > 0 or mhi < grid_w:
                    lo, hi = mlo, mhi
                    narrow_lo = min(narrow_lo, nlo)
                    narrow_hi = max(narrow_hi, nhi)
                    j += 1
                else:
                    break
            # Prefer the extent the MAJORITY of the run's rows occupy, so one
            # over-wide row cannot hand the empty outer columns a share of the
            # width. Falls back to the merged span when there is no majority.
            _ext = Counter(
                spans[r] for r in range(i, j + 1) if spans[r] is not None
            )
            if _ext:
                (_mlo, _mhi), _n = _ext.most_common(1)[0]
                if _n * 2 > (j - i + 1):
                    lo, hi = _mlo, _mhi
            if (j - i + 1) >= _MIN_PROMOTE_ROWS and _run_is_bounded(i, j, lo, hi):
                # Sub-table rows = the run's rows, keeping only the cells that
                # fall inside the span so the outer blanks are dropped.
                sub_rows = []
                for r in range(i, j + 1):
                    col = 0
                    keep = []
                    for c in rows[r][0]:
                        t, cs, rs_, imgs = _cell_parts(c)
                        cs = max(1, cs)
                        # Keep every cell that STARTS inside the span. Requiring
                        # it to also END inside (`col + cs <= hi`) silently
                        # DISCARDED a wider one — the run's own caption row,
                        # emitted with a colspan reaching over the blank outer
                        # columns, vanished along with its text.
                        if lo <= col < hi and (
                            (t or "").strip() or col + cs <= hi
                        ):
                            # Clamp the span to the sub-table's own width: a cell
                            # that spanned the OUTER grid (a `colspan` running
                            # over the blank outer columns) would otherwise widen
                            # the sub-table and drag the holder's colspan out with
                            # it, handing the empty outer columns a share of the
                            # width they must not get.
                            span = max(1, min(cs, hi - col))
                            keep.append(
                                c.with_span(t, span, rs_, imgs)
                                if isinstance(c, Cell)
                                else Cell(t, span, rs_, imgs)
                            )
                        col += cs
                    if keep:
                        sub_rows.append((keep, rows[r][1]))
                if sub_rows:
                    blocks = (("table", sub_rows),)
                    holder = Cell(
                        flatten_block_text(blocks).strip(),
                        max(1, hi - lo), 1, 0, blocks,
                    )
                    # One outer row: blanks left of the span, the sub-table, then
                    # blanks right of it — preserving the outer grid width.
                    new_cells = [Cell("", 1, 1, 0) for _ in range(lo)]
                    new_cells.append(holder)
                    new_cells.extend(Cell("", 1, 1, 0)
                                     for _ in range(grid_w - hi))
                    out.append((new_cells, rows[i][1]))
                    for r in range(i, j + 1):
                        consumed[r] = True
                    changed = True
                    log.info(
                        "[table] rebuilt a flattened sub-table: rows %d-%d were "
                        "confined to columns %d-%d — nesting them in one row",
                        i, j, lo, hi - 1,
                    )
                    i = j + 1
                    continue
        out.append(rows[i])
        i += 1

    # ── Signal B: parallel block pairing, on rows signal A did not consume ───
    final: List = []
    for idx, (cells, is_header) in enumerate(out):
        rebuilt = _promote_parallel_blocks(list(cells))
        if rebuilt is not None:
            final.append((rebuilt, is_header))
            changed = True
            log.info(
                "[table] rebuilt a flattened sub-table: adjacent cells carried "
                "equal parallel blocks — pairing them as sub-rows",
            )
        else:
            final.append((cells, is_header))

    return final if changed else rows


def _dropped_column_from_pictures(
    pictures: List[Dict],
    bbox,
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    img_by_anchor: Dict[Tuple[int, int], int],
    max_cols: int,
    n_rows: int,
    col_weight_hint=None,
) -> Optional[int]:
    """Index at which the OCR dropped a blank column, or None.

    Pairs each `<img>` placeholder with the picture the assigner gives it — so
    the two stay consistent by construction — then defers the geometric decision
    to `table_geometry.dropped_column_index`.
    """
    if not pictures or not img_by_anchor or max_cols <= 0 or n_rows <= 0:
        return None
    if not bbox or len(bbox) != 4:
        return None
    assigned = _assign_pictures_by_img_structure(
        pictures, bbox, cell_anchors, img_by_anchor, n_rows,
        col_weight_hint, max_cols,
    )
    if not assigned:
        return None
    pics: List[Dict] = []
    cols: List[int] = []
    for (_r, c), pl in assigned.items():
        for pic in pl:
            pics.append(pic)
            cols.append(c)
    if not pics:
        return None
    return dropped_column_index(pics, bbox, cols, max_cols, col_weight_hint)

def _insert_blank_column(rows, index: int):
    """Return `rows` with one empty cell inserted at grid column `index`.

    Walks each row's cells accumulating grid width, so the blank lands on the
    same GRID boundary in every row regardless of colspans, and rows that never
    reach `index` simply get the blank appended. Text is never touched.
    """
    out = []
    for cells, is_header in rows:
        new_cells = []
        col = 0
        placed = False
        for c in cells:
            if not placed and col >= index:
                new_cells.append(Cell("", 1, 1, 0))
                placed = True
            new_cells.append(c)
            col += max(1, _cell_parts(c)[1])
        if not placed:
            new_cells.append(Cell("", 1, 1, 0))
        out.append((new_cells, is_header))
    return out

def _shorten_contradicted_rowspans(
    rows: List[Tuple[List[Tuple[str, int, int]], bool]],
) -> List[Tuple[List[Tuple[str, int, int]], bool]]:
    """Shorten a ``rowspan`` that a later row's own cells contradict.

    THE PROBLEM. When OCR over-states a `rowspan`, the covered row's cells have
    nowhere to start: the walker skips past the wrongly-occupied columns and
    every cell in that row lands one or more columns too far RIGHT, leaving a
    hole at the end. Measured on a real page, a column given ``rowspan="2"``
    where the source rules divide it pushed the final row's image, model name and
    dimensions each one column right.

    THE SIGNAL, keyed only on shape. A row is contradicted when it supplies
    FEWER cells than it has free columns AND the shortfall leaves a hole of more
    than one column. Retracting one of the spans that reach into it by one row
    frees exactly the columns the row's own cells need.

    WHY THE ">1 column" TEST MATTERS. A hole of exactly one column at the right
    edge is the ORDINARY under-counted-rowspan shape that
    `_close_trailing_rowspan_gaps` already repairs by EXTENDING a span — the
    inverse operation. Measured across both reference documents, every
    single-column hole was that benign case and every genuine contradiction left
    a hole of 3. Restricting to multi-column holes keeps the two repairs from
    fighting over the same row.

    Only ever RETRACTS a span by the minimum needed, never moves a cell to a
    different column and never changes any text, so it cannot lose content.
    Returns the SAME ``rows`` object when nothing changed.
    """
    n_rows = len(rows)
    if n_rows < 2:
        return rows
    widths = [sum(max(1, _cell_parts(c)[1]) for c in cells) for cells, _h in rows]
    max_cols = max(widths, default=0)
    if max_cols <= 0:
        return rows

    # Place top-to-bottom, remembering which anchor owns each covered column so a
    # contradicted row can name the span to retract.
    occ = [[False] * max_cols for _ in range(n_rows)]
    owner: Dict[Tuple[int, int], Tuple[int, int]] = {}   # (row, col) -> anchor
    spans: Dict[Tuple[int, int], int] = {}               # anchor -> cell index
    retract: List[Tuple[int, int, int]] = []             # (row, idx, new_rs)
    for r, (cells, _h) in enumerate(rows):
        col = 0
        for idx, c in enumerate(cells):
            _t, cs, rs, _i = _cell_parts(c)
            while col < max_cols and occ[r][col]:
                col += 1
            if col >= max_cols:
                break
            cs = min(max(1, cs), max_cols - col)
            rs = min(max(1, rs), n_rows - r)
            if rs > 1:
                spans[(r, col)] = idx
            for rr in range(r, r + rs):
                for cc in range(col, col + cs):
                    occ[rr][cc] = True
                    if rr > r:
                        owner[(rr, cc)] = (r, col)
            col += cs
        # Does this row's own content fit the columns left free for it?
        free = [c for c in range(max_cols) if not occ[r][c]]
        # `free` is measured AFTER placement, so a hole is what nothing filled.
        if not free or len(free) <= 1:
            continue
        # Retract the span whose covered column sits immediately LEFT of this
        # row's first placed cell — that is the one crowding the row rightwards.
        first_free_before = None
        for cc in range(max_cols):
            if (r, cc) in owner:
                first_free_before = cc
        if first_free_before is None:
            continue
        anchor = owner.get((r, first_free_before))
        if anchor is None or anchor not in spans:
            continue
        a_r, a_c = anchor
        _t, _cs, a_rs, _i = _cell_parts(rows[a_r][0][spans[anchor]])
        new_rs = r - a_r
        if new_rs < 1 or new_rs >= max(1, a_rs):
            continue
        retract.append((a_r, spans[anchor], new_rs))

    if not retract:
        return rows

    out = [(list(cells), is_h) for cells, is_h in rows]
    for a_r, idx, new_rs in retract:
        cell = out[a_r][0][idx]
        txt, cs, rs, imgs = _cell_parts(cell)
        if hasattr(cell, "with_span"):
            out[a_r][0][idx] = cell.with_span(txt, cs, new_rs, imgs)
        else:
            out[a_r][0][idx] = (txt, cs, new_rs, imgs)
        log.info(
            "[table] rowspan at row %d ended too late (%d rows) — retracting to "
            "%d so the contradicted row's own cells can be placed",
            a_r, rs, new_rs,
        )
    return [(cells, is_h) for cells, is_h in out]

def _normalize_column_granularity(
    rows: List[Tuple[List[Tuple[str, int, int]], bool]],
) -> List[Tuple[List[Tuple[str, int, int]], bool]]:
    """Re-span under-segmented rows onto the table body's column boundaries.

    OCR/VLM output for merged-cell tables is often inconsistent: the body rows
    encode each logical column as several grid columns via ``colspan`` (e.g.
    every cell ``colspan=2``), but the header row — or a stray name cell — is
    emitted as single-width ``<td>``s, sometimes with empty separator ``<td>``s
    between them. Placed as-is, those finer cells land on the wrong grid columns
    and the header stops lining up with the body (and, because the leading cells
    are too narrow, downstream rowspan cells get shifted a column over too).

    We fix this BEFORE grid placement so the correction propagates: rows that
    are narrower than the table's full width get their non-empty cells re-spanned
    across the body's canonical column boundaries, absorbing the empty artifact
    cells.

    Crucially the re-span is ROWSPAN-AWARE. A narrow row is not always a full
    row minus artifacts — it can be a sub-header sitting *under* a ``colspan``
    header while the outer columns are already covered by ``rowspan`` cells from
    the row above (e.g. a ``样品组别 / 旋入扭矩 / [失效扭矩 spanning 扭矩值|失效模式] /
    比值`` header block). There the narrow row's cells belong only in the free
    middle columns, NOT stretched across the whole width. We therefore place
    rows top-to-bottom tracking rowspan occupancy, and re-span each narrow row's
    content cells across only the canonical segments still FREE in that row. If
    the content-cell count doesn't match the free-segment count, we leave the
    row untouched rather than guess.

    This is a strict no-op for well-formed tables — uniform-width tables,
    all-``colspan`` tables with nothing under-segmented, ragged data tables
    whose short rows have genuine trailing empties, and sub-header rows that
    already sit correctly under a rowspan block are all left byte-identical.

    Returns the SAME ``rows`` object when nothing changed, so callers can detect
    the no-op cheaply.
    """
    def _row_width(cells):
        return sum(max(1, _cell_parts(c)[1]) for c in cells)

    def _boundaries(cells):
        b = {0}
        col = 0
        for c in cells:
            col += max(1, _cell_parts(c)[1])
            b.add(col)
        return b

    widths = [_row_width(cells) for cells, _ in rows]
    if not widths:
        return rows
    maxw = max(widths)

    # The body granularity is defined by the rows that fill the full width. Need
    # a stable majority (>= 2) so one over-segmented outlier can't define it.
    full = [cells for (cells, _), w in zip(rows, widths) if w == maxw]
    if len(full) < 2:
        return rows

    # Canonical boundaries: column edges present in EVERY full-width row.
    canon = set.intersection(*[_boundaries(c) for c in full])
    # If the body is already one-cell-per-column there's no coarser grid to
    # align a finer row to — nothing to normalize.
    if canon == set(range(maxw + 1)):
        return rows
    bounds = sorted(canon)          # e.g. [0, 2, 4, 6]
    n_seg = len(bounds) - 1

    # Track rowspan occupancy exactly as parse_table_grid will: place each row's
    # (possibly rewritten) cells into a sparse grid so later rows can see which
    # columns are already taken by a rowspan descending from above.
    occ = [[False] * maxw for _ in range(len(rows))]

    new_rows: List[Tuple[List[Tuple[str, int, int]], bool]] = []
    changed = False
    for row_idx, ((cells, is_header), w) in enumerate(zip(rows, widths)):
        emit_cells = cells

        if w < maxw:
            # Which canonical segments are FREE in this row (no column covered
            # by a rowspan from above)? A segment must be wholly free to count;
            # a segment straddling free + occupied columns is ambiguous -> bail.
            free_segs: List[int] = []
            straddle = False
            for si in range(n_seg):
                cols = range(bounds[si], bounds[si + 1])
                covered = [occ[row_idx][c] for c in cols]
                if not any(covered):
                    free_segs.append(si)
                elif not all(covered):
                    straddle = True
                    break

            if not straddle:
                # Keep the SOURCE cell alongside its parts so the re-span below
                # can preserve any nested-table `blocks` it carries (a rebuilt
                # bare tuple would silently drop them).
                content = []
                for c in cells:
                    t, _cs, rs, imgs = _cell_parts(c)
                    if (t or "").strip():
                        content.append((t, rs, imgs, c))
                # Re-span content cells across the free segments, one per
                # segment. Only when the counts line up exactly — otherwise we
                # can't unambiguously map cells to segments, so leave as-is.
                if content and len(content) == len(free_segs):
                    rebuilt = []
                    for (t, rs, imgs, src), si in zip(content, free_segs):
                        b0, b1 = bounds[si], bounds[si + 1]
                        span = max(1, b1 - b0)
                        rebuilt.append(
                            src.with_span(t, span, rs, imgs)
                            if isinstance(src, Cell)
                            else (t, span, rs, imgs)
                        )
                    if rebuilt != list(cells):
                        emit_cells = rebuilt
                        changed = True

        new_rows.append((emit_cells, is_header))

        # Place emit_cells into `occ` (skipping already-occupied columns) so
        # the rowspans this row starts are visible to subsequent rows.
        col = 0
        for c in emit_cells:
            _t, cs, rs, _imgs = _cell_parts(c)
            while col < maxw and occ[row_idx][col]:
                col += 1
            if col >= maxw:
                break
            cs = min(max(1, cs), maxw - col)
            rs = min(max(1, rs), len(rows) - row_idx)
            for rr in range(row_idx, row_idx + rs):
                for cc in range(col, col + cs):
                    occ[rr][cc] = True
            col += cs

    return new_rows if changed else rows


# Nested-table structure from the most recent `parse_table_grid` call, keyed by
# cell anchor. A side-channel rather than a 6th return value because
# `parse_table_grid`'s 5-value shape is unpacked verbatim by 7 external call
# sites (ocr_service's table_validate / table_reocr / table_derotate_reocr /
# cell_picture_recovery, and both json_to_docx copies). Read it with
# `pop_blocks_by_anchor()` IMMEDIATELY after the `parse_table_grid` call.
_LAST_BLOCKS_BY_ANCHOR: Dict[Tuple[int, int], Tuple] = {}


def pop_blocks_by_anchor() -> Dict[Tuple[int, int], Tuple]:
    """Nested-table blocks from the last `parse_table_grid` call, as
    ``{(row, col): blocks}``. Empty for every table without nested tables."""
    return dict(_LAST_BLOCKS_BY_ANCHOR)


def _extract_blocks_by_anchor(html_text: str) -> Dict[Tuple[int, int], Tuple]:
    """Fresh re-parse of ``html_text`` returning ``{(row, col): blocks}``
    for every top-level cell that contains at least one nested table.

    Self-contained: does not depend on any `Cell._blocks` field surviving
    the caller pipeline, and does not touch the `_LAST_BLOCKS_BY_ANCHOR`
    side-channel. This is the safety net that keeps nested tables rendering
    even when row/cell objects were round-tripped as bare tuples through
    an external step (json_to_docx, table_validate, table_reocr and so on)
    that stripped the `_blocks` attribute.

    Returns an empty dict on any parse failure or when the source has no
    nested tables — callers can then fall back to `pop_blocks_by_anchor()`.
    """
    if not html_text or "<table" not in html_text.lower():
        return {}
    parser = TableHTMLParser()
    try:
        parser.feed(html_text)
    except Exception:
        return {}
    try:
        parser.finalize()
    except Exception:
        pass
    if not parser.tables:
        return {}

    out: Dict[Tuple[int, int], Tuple] = {}
    # Only the first top-level table — that is what render_table renders.
    top_rows = parser.tables[0]
    n_rows = len(top_rows)
    if not n_rows:
        return {}
    widths = [sum(max(1, c.colspan) for c in cells) for cells, _ in top_rows]
    max_cols = max(widths) if widths else 0
    if not max_cols:
        return {}
    occ = [[False] * max_cols for _ in range(n_rows)]
    for r, (cells, _is_header) in enumerate(top_rows):
        col = 0
        for c in cells:
            while col < max_cols and occ[r][col]:
                col += 1
            if col >= max_cols:
                break
            cs = min(max(1, c.colspan), max_cols - col)
            rs = min(max(1, c.rowspan), n_rows - r)
            blocks = getattr(c, "blocks", ()) or ()
            if blocks and any(k == "table" for k, _ in blocks):
                out[(r, col)] = blocks
            for rr in range(r, r + rs):
                for cc in range(col, col + cs):
                    occ[rr][cc] = True
            col += cs
    return out


def parse_table_grid(
    rows: List[Tuple[List[Tuple[str, int, int, int]], bool]],
    insert_blank_col: Optional[int] = None,
) -> Tuple[
    int,
    int,
    Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    List[List[bool]],
    Dict[Tuple[int, int], int],
]:
    """Walk parsed (cells, is_header) rows into a sparse grid.

    Returns (max_cols, n_rows, cell_anchors, occupied, img_by_anchor), where
    `img_by_anchor` maps a cell anchor `(row, col)` to the number of `<img>`
    placeholders that cell contained (0-valued anchors are omitted).

    The grid width is the MODAL row width, not the maximum. OCR frequently
    mis-segments a single row (an extra spurious split, a stray colspan) so
    that one row is wider than all the others; taking the max would let that
    outlier stretch every other row and shift merged headers out of alignment.
    Using the most common width instead is robust to such outliers. As a
    safeguard we never make the grid narrower than the widest row's last
    NON-EMPTY cell, so real content is never clipped — only trailing empty
    padding from an over-segmented row is dropped. A hard cap keeps
    hallucinated many-column headers bounded.
    """
    from collections import Counter

    # Rebuild any sub-table the OCR flattened into top-level rows/columns FIRST,
    # so the grid below sees the real outer shape (a flattened sub-table
    # otherwise inflates both the column count and the row count).
    rows = _promote_flattened_subtables(rows)

    # Re-insert a column the OCR dropped (see `_dropped_column_from_pictures`).
    # AFTER promotion deliberately: a fresh blank leading column is exactly the
    # left-inset shape `_promote_flattened_subtables` reads as a nested
    # sub-table, and running it first collapsed a real 4-row table into one row.
    if insert_blank_col is not None:
        rows = _insert_blank_column(rows, insert_blank_col)

    # Repair header/body column-granularity mismatch from OCR under-segmentation
    # before placing cells, so the correction propagates to rowspan placement.
    rows = _normalize_column_granularity(rows)

    # Retract a rowspan that a later row's own cells contradict, so that row's
    # cells start in their true column instead of being pushed rightwards.
    rows = _shorten_contradicted_rowspans(rows)

    n_rows = len(rows)

    def _row_width(row_cells):
        return sum(max(1, _cell_parts(c)[1]) for c in row_cells)

    widths = [_row_width(r[0]) for r in rows]
    raw_max = min(max(widths, default=1) or 1, MAX_TABLE_COLS)

    def _place(target_cols: int):
        """Place cells into a target-width grid; report last non-empty col."""
        anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]] = {}
        img_map: Dict[Tuple[int, int], int] = {}
        blk_map: Dict[Tuple[int, int], Tuple] = {}
        occ = [[False] * target_cols for _ in range(n_rows)]
        last_nonempty = 0
        for row_idx, (cells, is_header) in enumerate(rows):
            col = 0
            for c in cells:
                cell_text, cs, rs, imgs = _cell_parts(c)
                while col < target_cols and occ[row_idx][col]:
                    col += 1
                if col >= target_cols:
                    break
                cs = min(max(1, cs), target_cols - col)
                rs = min(max(1, rs), n_rows - row_idx)
                anchors[(row_idx, col)] = (cell_text, cs, rs, is_header)
                if imgs > 0:
                    img_map[(row_idx, col)] = imgs
                # Carry any nested-table structure on a PARALLEL map keyed by
                # anchor (same pattern as `img_map`). The anchor tuple itself
                # stays 4 fields: 14 internal sites destructure exactly four,
                # and `parse_table_grid`'s 5-value return is unpacked by 7
                # external call sites that must keep working untouched.
                blocks = getattr(c, "blocks", ())
                if blocks and any(k == "table" for k, _ in blocks):
                    blk_map[(row_idx, col)] = blocks
                for rr in range(row_idx, row_idx + rs):
                    for cc in range(col, col + cs):
                        occ[rr][cc] = True
                if (cell_text or "").strip():
                    last_nonempty = max(last_nonempty, col + cs)
                col += cs
        return anchors, occ, last_nonempty, img_map, blk_map

    # First pass at the widest observed width to learn where real content ends.
    _a0, _o0, content_extent, _i0, _b0 = _place(raw_max)

    # Modal width, measured on EFFECTIVE occupancy rather than raw <td> counts.
    #
    # A row's own cell count understates how many grid columns it really covers
    # whenever a `rowspan` anchored ABOVE it reaches down into it. On a table
    # whose outer vertical rules are emitted as tall blank spacer cells
    # (`<td rowspan="12"></td>`), the anchor row counts 10 while every row below
    # counts only 4 — so the raw modal was 4, the grid was built 8 wide, and the
    # anchor row's last cells were dropped while every following row was shifted
    # into the wrong column.
    #
    # `_o0` is the occupancy the trial placement already computed at full width,
    # so this costs nothing extra and, by construction, agrees with the grid the
    # renderer goes on to build.
    eff_widths = [sum(1 for v in orow if v) for orow in _o0] or widths
    modal = Counter(eff_widths).most_common(1)[0][0] if eff_widths else 1
    max_cols = max(1, min(raw_max, max(modal, content_extent)))

    # Report cells this grid CANNOT hold. A row wider than `max_cols` has its
    # overflow silently dropped, and that used to happen with no diagnostic at
    # all: one measured page whose 12 data rows ran together into a single
    # 701-cell row placed just 67 of 728 cells. The OCR side repairs this shape
    # (`table_validate`, which measures placeable cells); this log is what makes
    # a leak visible if any slips through.
    overflow = sum(max(0, w - max_cols) for w in widths)
    if overflow:
        log.warning(
            "[table] grid is %d cols but the widest row has %d cells — "
            "DROPPING %d cell(s) that cannot be placed",
            max_cols, max(widths, default=0), overflow,
        )

    cell_anchors, occupied, _, img_by_anchor, blocks_by_anchor = _place(max_cols)
    _merge_duplicate_rowspan_labels(cell_anchors, occupied, max_cols, n_rows)
    _close_trailing_rowspan_gaps(cell_anchors, occupied, max_cols, n_rows)
    # Side-channel for the nested-table structure: `parse_table_grid`'s 5-value
    # return shape is fixed by 7 external unpack sites, so `render_table` reads
    # the blocks back through `pop_blocks_by_anchor()` instead.
    _LAST_BLOCKS_BY_ANCHOR.clear()
    _LAST_BLOCKS_BY_ANCHOR.update(blocks_by_anchor)
    return max_cols, n_rows, cell_anchors, occupied, img_by_anchor


def _close_trailing_rowspan_gaps(
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    occupied: List[List[bool]],
    max_cols: int,
    n_rows: int,
) -> None:
    """Extend a ``rowspan`` that stops short of the table's last row.

    OCR routinely under-counts a `rowspan` that runs to the bottom of a table,
    emitting ``rowspan="2"`` where the source spans three rows. The grid then has
    a HOLE in that column's final rows, which renders as a stray empty cell —
    the reported "last column has two rows but reconstruction built three".

    The repair keys ONLY on the occupancy shape: a column whose occupancy ends
    before the last row, where the last occupied cell is a ``rowspan`` anchor and
    every row below it is free. That anchor is then stretched to the bottom. It
    is a strict no-op for a fully-occupied column, a fully-empty one, and for a
    hole that sits under a single-row cell (which is a genuinely blank cell, not
    a truncated span). Mutates both maps in place.
    """
    for c in range(max_cols):
        # Find the last occupied row in this column.
        last = -1
        for r in range(n_rows):
            if occupied[r][c]:
                last = r
        # Nothing to do when the column is empty or already reaches the bottom.
        if last < 0 or last >= n_rows - 1:
            continue
        # Every row below must be free — a hole with content under it is a real
        # gap in the grid, not a truncated span, and must be left alone.
        if any(occupied[r][c] for r in range(last + 1, n_rows)):
            continue
        # The covering cell must be a ROWSPAN anchor: walk up to its anchor row.
        anchor_r = None
        for r in range(last, -1, -1):
            if (r, c) in cell_anchors:
                anchor_r = r
                break
            if not occupied[r][c]:
                break
        if anchor_r is None:
            continue
        txt, cs, rs, is_h = cell_anchors[(anchor_r, c)]
        # Only extend a cell that already spans rows AND whose span ends exactly
        # at the hole; a plain 1-row cell followed by blanks is ordinary.
        if rs <= 1 or anchor_r + rs != last + 1:
            continue
        new_rs = n_rows - anchor_r
        cell_anchors[(anchor_r, c)] = (txt, cs, new_rs, is_h)
        for r in range(last + 1, n_rows):
            for cc in range(c, min(max_cols, c + max(1, cs))):
                occupied[r][cc] = True
        log.info(
            "[table] column %d: rowspan at row %d ended at row %d but the table "
            "has %d rows — extending it to the bottom",
            c, anchor_r, last, n_rows,
        )


def _merge_duplicate_rowspan_labels(
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    occupied: List[List[bool]],
    max_cols: int,
    n_rows: int,
) -> None:
    """Merge a section label the OCR duplicated across a header row and the
    following data row into a single vertically-spanning cell.

    Chandra sometimes emits a repeated sub-header row whose FIRST (row-label)
    cell duplicates the label of the data row right below it, e.g.::

        <tr><th>Aprovação</th><th>Assinatura</th>…</tr>   (sub-header)
        <tr><td>Aprovação</td><td>Wang…</td>…</tr>         (data)

    In the source that label is ONE cell spanning both rows (rowspan=2); the
    duplicate makes the first column read ``Aprovação / Aprovação``. We DON'T
    change the OCR JSON — we fix it here at render time: when row ``r`` col 0 is a
    header cell (cs==1) whose text equals row ``r+1`` col 0 (cs==1, non-header),
    drop the lower cell and give the upper one ``rowspan=2``. Keyed only on the
    duplication pattern — no table/column/document/language assumptions."""
    for r in range(n_rows - 1):
        top = cell_anchors.get((r, 0))
        bot = cell_anchors.get((r + 1, 0))
        if not top or not bot:
            continue
        t_txt, t_cs, t_rs, t_h = top
        b_txt, b_cs, b_rs, b_h = bot
        label = (t_txt or "").strip()
        if not label or label != (b_txt or "").strip():
            continue
        # Only collapse a header→data duplication, single-column, no existing
        # rowspans that would be disturbed.
        if t_cs != 1 or b_cs != 1 or t_rs != 1 or b_rs != 1 or not t_h:
            continue
        # Merge: keep the upper cell, span it over both rows, drop the lower.
        cell_anchors[(r, 0)] = (t_txt, t_cs, 2, t_h)
        del cell_anchors[(r + 1, 0)]
        if r + 1 < len(occupied) and occupied[r + 1]:
            occupied[r + 1][0] = True


def _pic_sort_key(pic: Dict) -> Tuple[float, float]:
    """Reading-order key for a recovered picture: top-to-bottom then
    left-to-right by bbox centroid."""
    pb = pic.get("bbox") or []
    if len(pb) != 4:
        return (0.0, 0.0)
    return ((pb[1] + pb[3]) / 2.0, (pb[0] + pb[2]) / 2.0)


def _assign_pictures_by_img_structure(
    pictures: List[Dict],
    bbox: List[int],
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    img_by_anchor: Dict[Tuple[int, int], int],
    n_rows: int,
    col_weight: Optional[List[int]] = None,
    max_cols: int = 0,
) -> Optional[Dict[Tuple[int, int], List[Dict]]]:
    """Assign recovered pictures to the `<img>` cells parsed from the table HTML,
    by each picture's LAYOUT BBOX POSITION IN BOTH AXES.

    `img_by_anchor` maps a cell anchor to how many `<img>` placeholders that cell
    held, which authoritatively marks WHICH cells hold pictures and HOW MANY.

    Each image cell owns the RECTANGLE its span covers: the columns from
    `col_weight` (the same proportional edges the geometric fallback builds) and
    the rows from its row-span. A picture goes to the cell whose rectangle is
    nearest its centroid, measured as a true point-to-rectangle distance — zero
    when the centroid is inside.

    WHY BOTH AXES. This used to compare Y ONLY, so two image cells sharing a row
    band were indistinguishable and a picture from a right-hand column landed in
    a left-hand one: measured on a real page, the diagram at x=1771..2274
    (plainly column 5) was placed in column 1, giving that cell three pictures
    while column 5 rendered empty. The x coordinate is exactly the information
    that separates them.

    CAPACITY IS A HARD LIMIT, and surplus pictures are DROPPED. The `<img>`
    placeholders state how many pictures the table can show; a layout model that
    emits more `Image` entries than that is emitting artifacts (measured: 3-27pt
    slivers carrying alt-text beside 50-150pt diagrams). Keeping the LARGEST
    `capacity` pictures is a size ordering rather than a threshold, so it adapts
    to any document: whatever the real diagrams are, they are the big ones.

    Assignment is globally greedy over (distance, picture, cell) so the closest
    pairing wins overall rather than being claimed by whichever picture happened
    to be considered first.

    Returns None only when there is no usable `<img>` structure at all, so the
    caller falls back to the geometric scorer.
    """
    if not img_by_anchor or n_rows <= 0:
        return None
    tbl_x1 = float(bbox[0])
    tbl_y1 = float(bbox[1])
    tbl_w = max(1.0, float(bbox[2]) - float(bbox[0]))
    tbl_h = max(1.0, float(bbox[3]) - float(bbox[1]))

    # Column edges in the same proportional space the geometric fallback uses.
    # Without weights every column is equal-width, which is still far better than
    # ignoring x entirely.
    if col_weight and max_cols > 0:
        w_total = sum(col_weight) or max_cols
        col_edges = [tbl_x1]
        for cw in col_weight:
            col_edges.append(col_edges[-1] + tbl_w * (cw / w_total))
    else:
        n_c = max(1, max_cols)
        col_edges = [tbl_x1 + tbl_w * (i / n_c) for i in range(n_c + 1)]

    def _edge(i: int) -> float:
        return col_edges[min(max(i, 0), len(col_edges) - 1)]

    cells: List[Dict] = []
    for rc, n in sorted(img_by_anchor.items()):
        if n <= 0 or rc not in cell_anchors:
            continue
        _t, cs, rs, _h = cell_anchors[rc]
        r0, c0 = rc
        r1 = min(n_rows, r0 + max(1, rs))
        cells.append({
            "anchor": rc, "cap": n, "count": 0,
            "x0": _edge(c0), "x1": _edge(c0 + max(1, cs)),
            "y0": tbl_y1 + tbl_h * (r0 / n_rows),
            "y1": tbl_y1 + tbl_h * (r1 / n_rows),
        })
    if not cells:
        return None

    def _centroid(pic: Dict) -> Tuple[float, float]:
        pb = pic.get("_orig_bbox") or pic.get("bbox") or []
        if len(pb) != 4:
            return (tbl_x1, tbl_y1)
        return ((float(pb[0]) + float(pb[2])) / 2.0,
                (float(pb[1]) + float(pb[3])) / 2.0)

    def _area(pic: Dict) -> float:
        pb = pic.get("_orig_bbox") or pic.get("bbox") or []
        if len(pb) != 4:
            return 0.0
        return abs(float(pb[2]) - float(pb[0])) * abs(float(pb[3]) - float(pb[1]))

    # Keep only as many pictures as the table has slots, largest first.
    capacity = sum(c["cap"] for c in cells)
    keep = sorted(pictures, key=_area, reverse=True)[:capacity]
    if len(keep) < len(pictures):
        log.warning(
            "[table] %d picture(s) recovered but the table's <img> cells hold "
            "%d — keeping the %d largest and dropping the rest as artifacts",
            len(pictures), capacity, len(keep),
        )
    # Reading order among the kept pictures keeps ties deterministic.
    keep = sorted(keep, key=_pic_sort_key)

    def _dist(pic: Dict, cell: Dict) -> float:
        cx, cy = _centroid(pic)
        dx = max(cell["x0"] - cx, 0.0, cx - cell["x1"])
        dy = max(cell["y0"] - cy, 0.0, cy - cell["y1"])
        return (dx * dx + dy * dy) ** 0.5

    pairs = sorted(
        (_dist(p, c), pi, ci)
        for pi, p in enumerate(keep)
        for ci, c in enumerate(cells)
    )
    out: Dict[Tuple[int, int], List[Dict]] = {}
    placed: set = set()
    for _d, pi, ci in pairs:
        if pi in placed or cells[ci]["count"] >= cells[ci]["cap"]:
            continue
        placed.add(pi)
        cells[ci]["count"] += 1
        out.setdefault(cells[ci]["anchor"], []).append(keep[pi])
    # Keep each cell's pictures in reading order for the stacked-image case.
    for anchor in out:
        out[anchor].sort(key=_pic_sort_key)
    return out


def _assign_pictures_to_cells(
    pictures: List[Dict],
    bbox: List[int],
    col_weight: List[int],
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    max_cols: int,
    n_rows: int,
    img_by_anchor: Optional[Dict[Tuple[int, int], int]] = None,
) -> Dict[Tuple[int, int], List[Dict]]:
    """Decide which (row, col) anchor each picture lives in.

    When the table HTML carried `<img>` placeholders (``img_by_anchor``), that
    structure is authoritative and used first (see
    `_assign_pictures_by_img_structure`). Only when there is no usable `<img>`
    structure do we fall back to the geometric scorer below.

    Why not just bbox-centroid → column edges? Two failure modes that hit
    real PDFs:
      1. One column has very long text and steals geometric width from a
         neighbouring image column (`备注` cell expands past the picture).
      2. The ocr layout returns pictures in cells the OCR labelled
         empty (the cell text is `""`), so emptiness is a strong signal
         that's lost if we only look at x-positions.

    The scoring combines:
      * row band: rows are split into equal vertical bands inside the
        table bbox (we don't know real row heights yet — column-weight
        widths are derived first). Each picture lands in the row whose
        band contains its y-centroid.
      * cell emptiness in that row: empty cells score highest, short
        labels (≤ 3 chars) next, long text cells last.
      * horizontal proximity: among candidate cells with the best
        emptiness tier, the cell whose column midpoint is closest to the
        picture x-centroid wins.

    The returned mapping keys on the (row, col) of the CELL ANCHOR
    (rowspan/colspan accounted for), so downstream code can index into
    `cell_anchors` directly.
    """
    out: Dict[Tuple[int, int], List[Dict]] = {}
    if not pictures or n_rows <= 0 or max_cols <= 0:
        return out

    # Authoritative <img>-structure assignment first (by picture bbox y-position);
    # geometry is the fallback only when there are no <img> cells at all.
    structural = _assign_pictures_by_img_structure(
        pictures, bbox, cell_anchors, img_by_anchor or {}, n_rows,
        col_weight, max_cols,
    )
    if structural is not None:
        return structural

    tbl_x1, tbl_y1, tbl_x2, tbl_y2 = bbox
    tbl_w_px = max(1.0, tbl_x2 - tbl_x1)
    tbl_h_px = max(1.0, tbl_y2 - tbl_y1)
    # Pre-row vertical band: we don't have real row heights yet, so split
    # the bbox into equal-height bands. This is good enough to identify the
    # row a picture lives in for the dominant "image-in-its-own-row" case.
    row_band_h = tbl_h_px / n_rows
    # Text-derived column boundaries (same formula as the live rendering
    # below — we only use them as a tiebreaker for horizontal proximity).
    weight_total = sum(col_weight) or max_cols
    col_edges_px = [tbl_x1]
    for cw in col_weight:
        col_edges_px.append(col_edges_px[-1] + tbl_w_px * (cw / weight_total))

    # For each row, list every cell anchor whose VERTICAL SPAN covers that
    # row — including anchors that started earlier and continue via rowspan.
    # This lets a picture sitting in a rowspan-merged cell pick the anchor
    # at the rowspan's top, not a numeric continuation cell that happens to
    # land in the same equal-height row band.
    cells_by_row: Dict[int, List[Tuple[int, int, int, str]]] = {}
    for (r, c), (txt, cs, rs, _h) in cell_anchors.items():
        for rr in range(r, min(n_rows, r + max(1, rs))):
            cells_by_row.setdefault(rr, []).append((r, c, cs, txt or ""))

    def _emptiness_tier(text: str) -> int:
        """0 = empty, 1 = very short label, 2 = anything longer.
        Lower tier wins (more likely to hold a picture)."""
        s = (text or "").strip()
        if not s:
            return 0
        # CJK display width counts double; keep the threshold conservative
        if _display_width(s) <= 6:
            return 1
        return 2

    for pic in pictures:
        pb = pic.get("bbox") or []
        if len(pb) != 4:
            continue
        cx_px = (pb[0] + pb[2]) / 2.0
        cy_px = (pb[1] + pb[3]) / 2.0

        # Row band: clamp into [0, n_rows-1]
        row_idx = int((cy_px - tbl_y1) / row_band_h)
        row_idx = max(0, min(n_rows - 1, row_idx))

        # Walk rows outward from row_idx looking for the row whose anchors
        # have a candidate cell. The picture may visually overlap two row
        # bands, so we don't insist on the band match alone.
        candidates: List[Tuple[int, int, int, float]] = []
        for dr in range(n_rows):  # 0, then 1, -1, 2, -2, ...
            offsets = (0,) if dr == 0 else (dr, -dr)
            for off in offsets:
                r = row_idx + off
                if r < 0 or r >= n_rows:
                    continue
                row_cells = cells_by_row.get(r) or []
                for (anchor_r, c, cs, txt) in row_cells:
                    tier = _emptiness_tier(txt)
                    # Span midpoint accounts for colspan>1 cells.
                    mid = (col_edges_px[c] + col_edges_px[min(max_cols, c + cs)]) / 2.0
                    dist = abs(cx_px - mid)
                    candidates.append((anchor_r, c, tier, dist))
                if candidates:
                    break
            if candidates:
                break

        if not candidates:
            continue
        # Best = lowest tier, then lowest x-distance.
        candidates.sort(key=lambda t: (t[2], t[3]))
        anchor_r, anchor_c, _tier, _dist = candidates[0]
        out.setdefault((anchor_r, anchor_c), []).append(pic)
    return out


def compute_col_weights(
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    max_cols: int,
) -> List[int]:
    """Per-column weights inferred from cell text.

    A column's soft weight is a HIGH PERCENTILE (75th) of its cells' width
    contributions — not the max, and not the median. The max let a single
    outlier cell (one long ``H1, H2, … H15`` list among empty cells) blow a
    column out. But the median under-serves the common real case where a
    column is short in MOST rows yet holds long paragraphs in several rows
    (e.g. a '特征判定' answer column: many '否'/'固定。' plus a few 60-180 char
    answers). The median of that column is ~one glyph, so it was squeezed to a
    1-glyph-per-line vertical strip and its long cells clipped. The 75th
    percentile gives such a column enough width for its long cells while still
    ignoring a lone outlier and empty placeholder cells. A hard floor from the
    longest unbreakable token AND a content-aware label minimum are still
    applied per column (via max) so no column is squeezed below what it needs.
    """
    contribs: List[List[int]] = [[] for _ in range(max_cols)]
    col_min = [1] * max_cols
    for (_r, c), (txt, cs, _rs, _h) in cell_anchors.items():
        if c >= max_cols:
            continue
        cs = max(1, min(cs, max_cols - c))
        # Per-column floor = the larger of the longest unbreakable Latin/digit
        # token and the content-aware label minimum. The latter keeps short CJK
        # label columns (headers like '可能的危险(源)') from being squeezed so
        # narrow they wrap into a 1-glyph-per-line vertical strip and clip.
        if cs == 1:
            contribs[c].append(_display_width(txt))
            col_min[c] = max(
                col_min[c], _longest_token_width(txt), _content_min_width(txt)
            )
        else:
            share = max(1, _display_width(txt) // cs)
            tok_share = max(1, _longest_token_width(txt) // cs)
            content_share = max(1, _content_min_width(txt) // cs)
            for cc in range(c, c + cs):
                contribs[cc].append(share)
                col_min[cc] = max(col_min[cc], tok_share, content_share)

    def _percentile(vals: List[int], q: float) -> float:
        """Linear-interpolated q-quantile (0..1) of a non-empty list."""
        s = sorted(vals)
        if len(s) == 1:
            return float(s[0])
        import math
        k = (len(s) - 1) * q
        lo = math.floor(k)
        hi = math.ceil(k)
        if lo == hi:
            return float(s[int(k)])
        return s[lo] * (hi - k) + s[hi] * (k - lo)

    col_weight = []
    for c in range(max_cols):
        # 75th percentile of the non-trivial contributions: wide enough for a
        # column that is long in a MINORITY of rows (its long cells fit without
        # a vertical strip), while empty placeholders and a single outlier
        # don't distort it.
        nontrivial = [v for v in contribs[c] if v > 1]
        col_weight.append(
            max(1, int(round(_percentile(nontrivial, 0.75)))) if nontrivial else 1
        )

    # Lift each column to its longest-unbreakable-token floor so narrow-but-
    # text-bearing columns (e.g. an 'N/ACC' column) stay readable.
    return [max(w, m) for w, m in zip(col_weight, col_min)]


def _balance_col_widths(
    col_w_emus: List[int],
    min_col_emus: List[int],
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    max_cols: int,
    w: int,
    meas_font,
    meas_font_bold,
    cell_size_px: int,
) -> List[int]:
    """Redistribute width between columns to un-crush the worst column.

    The proportional allocation is feed-forward: widths are frozen before we
    know how many lines each cell wraps to. So a column can end up so narrow its
    text wraps into a tall 1-glyph-per-line strip (which then clips) while a
    sibling column is over-wide. This pass fixes that WITHOUT growing the table:
    it moves EMU from low line-pressure (over-wide) columns to the highest
    line-pressure (crushed) column, holding ``sum(col_w_emus) == w`` and never
    dropping a column below its floor ``min_col_emus[c]``.

    Objective: minimise the MAXIMUM column line-pressure (the crushed column is
    exactly the max). Strict-improvement only — a move is kept solely when it
    lowers the worst column's line count without pushing another column's line
    count above the old maximum. Converges to a no-op on already-balanced tables
    (so tables that render fine today are untouched).
    """
    if max_cols < 2 or meas_font is None:
        return col_w_emus

    # Cells grouped by their anchor column, with the weight (bold?) they render
    # at, so line-count measurement matches the row-height / emission code.
    col_cells: Dict[int, List[Tuple[str, int, bool]]] = {c: [] for c in range(max_cols)}
    for (r, c), (txt, cs, _rs, is_header) in cell_anchors.items():
        if c >= max_cols or not (txt or "").strip():
            continue
        col_cells[c].append((txt, max(1, cs), bool(is_header or r == 0)))

    def _col_pressure(widths: List[int], c: int) -> float:
        """Pressure of column c at the given widths: wrapped-line count plus a
        WORD-OVERFLOW penalty. Only counts cells the column ANCHORS (colspan
        cells contribute to their anchor column so a wide merged header doesn't
        inflate a data column).

        `wrap_to_width` never breaks a single over-long word — it places it on
        one line even when it exceeds the cell, so a column whose longest word
        overflows still reports line-count = 1 and the balancer leaves it too
        narrow (Word then breaks that word char-by-char, e.g. 'Aprovad'/'o').
        The penalty (0, 1] adds how far the longest word overruns the usable
        width, so an overflowing 1-line column outranks a clean 1-line column
        and the loop steals width for it from an over-wide sibling. Capped at 1
        so it can never outrank a genuinely multi-line crushed column."""
        best = 1.0
        for (txt, cs, bold) in col_cells.get(c, []):
            cell_w_emu = sum(widths[c:c + cs])
            cell_w_pt = cell_w_emu / EMU_PER_PT
            usable_pt = _usable_w_pt(cell_w_pt)
            font = meas_font_bold if bold else meas_font
            n = len(wrap_to_width(txt, font, usable_pt, cell_size_px))
            lw_px = _longest_word_px(txt, font, cell_size_px)
            penalty = (
                min(lw_px / usable_pt - 1.0, 1.0)
                if (usable_pt > 0 and lw_px > usable_pt) else 0.0
            )
            best = max(best, float(n) + penalty)
        return best

    def _pressures(widths: List[int]) -> List[float]:
        return [_col_pressure(widths, c) for c in range(max_cols)]

    widths = list(col_w_emus)
    min_step = max(1, int(round(w * 0.005)))   # 0.5% of table width, granularity

    def _col_word_overflow_emu(widths: List[int], c: int) -> int:
        """EMU by which column c's widest unbreakable word overruns its usable
        width (0 = no word overflow). A word wider than the cell is what makes
        Word break it mid-word ('Aprovad'/'o'); this measures that deficit in
        the SAME EMU space as the column widths so we can close it exactly."""
        worst = 0.0
        for (txt, cs, bold) in col_cells.get(c, []):
            cell_w_emu = sum(widths[c:c + cs])
            usable_pt = _usable_w_pt(cell_w_emu / EMU_PER_PT)
            font = meas_font_bold if bold else meas_font
            lw_px = _longest_word_px(txt, font, cell_size_px)
            if lw_px > usable_pt:
                # px is measured at this size == pt budget elsewhere; convert the
                # (word − usable) pt deficit back to EMU for the transfer.
                worst = max(worst, (lw_px - usable_pt) * EMU_PER_PT)
        return int(round(worst))

    # ── Word-overflow pre-pass ────────────────────────────────────────────────
    # HARD failure (a mid-word char break by Word) takes priority over the SOFT
    # line-count wrapping the main loop balances: a paragraph gaining a line is
    # fine, a word splitting across lines is not. So first, while any column has
    # a word wider than its cell, pull width into it from the column with the
    # most slack above its floor — even a multi-line paragraph column (exactly
    # the "steal from the widest column" the user asked for) — provided the
    # donor doesn't itself start overflowing a word. Runs to a no-op when no
    # column word-overflows (untouched for tables that already fit).
    for _ in range(128):
        overflow = [_col_word_overflow_emu(widths, c) for c in range(max_cols)]
        need = max(overflow)
        if need <= 0:
            break
        receiver = max(range(max_cols), key=lambda c: (overflow[c], -c))
        donors = [
            c for c in range(max_cols)
            if c != receiver
            and widths[c] - min_col_emus[c] >= min_step
            and _col_word_overflow_emu(widths, c) == 0
        ]
        if not donors:
            break
        donor = max(donors, key=lambda c: widths[c] - min_col_emus[c])
        donor_slack = widths[donor] - min_col_emus[donor]
        # Move just enough to close the receiver's deficit, bounded by the
        # donor's slack and never so much the donor starts breaking a word.
        move = min(donor_slack, max(min_step, overflow[receiver]))
        trial = list(widths)
        trial[donor] -= move
        trial[receiver] += move
        if _col_word_overflow_emu(trial, donor) > 0:
            # Donor would overflow — shrink the move to its own slack-to-overflow.
            move = min_step
            trial = list(widths)
            trial[donor] -= move
            trial[receiver] += move
            if _col_word_overflow_emu(trial, donor) > 0:
                break  # even one step breaks the donor — stop
        # Keep only if the receiver's overflow strictly decreased.
        if _col_word_overflow_emu(trial, receiver) >= overflow[receiver]:
            break
        widths = trial

    pressures = _pressures(widths)

    for _ in range(64):
        cur_max = max(pressures)
        if cur_max <= 1:
            break  # nothing is wrapping badly
        # Receiver: the most-crushed column (highest line pressure).
        receiver = max(range(max_cols), key=lambda c: (pressures[c], -c))
        # Donors: columns with slack above their floor AND lower pressure than
        # the receiver (over-wide relative to their content). Pick most slack.
        donors = [
            c for c in range(max_cols)
            if c != receiver
            and widths[c] - min_col_emus[c] >= min_step
            and pressures[c] < cur_max
        ]
        if not donors:
            break
        donor = max(donors, key=lambda c: widths[c] - min_col_emus[c])
        donor_slack = widths[donor] - min_col_emus[donor]

        # Find the SMALLEST transfer that reduces the receiver's line count
        # (crossing a wrap boundary), bounded by the donor's slack. Widening in
        # tiny fixed steps stalls because one step rarely crosses a boundary;
        # instead grow the trial transfer geometrically until the receiver's
        # pressure drops or the donor runs out / would itself become the worst.
        best_trial = None
        move = min_step
        while move <= donor_slack:
            trial = list(widths)
            trial[donor] -= move
            trial[receiver] += move
            # Never shrink a donor so far its own longest word overflows — that
            # would trade the receiver's soft line-wrap for a hard mid-word break
            # in the donor and undo the word-overflow pre-pass above.
            if _col_word_overflow_emu(trial, donor) > 0:
                break
            trial_p = _pressures(trial)
            # Reject if the donor became as crushed as the receiver was — that
            # just relocates the problem.
            if trial_p[donor] >= cur_max:
                break
            improved = (
                max(trial_p) < cur_max
                or (max(trial_p) == cur_max
                    and trial_p.count(cur_max) < pressures.count(cur_max))
            )
            if improved:
                best_trial = (trial, trial_p)
                break
            move *= 2
        if best_trial is None:
            break
        widths, pressures = best_trial

    # Preserve exact-sum invariant (rounding-safe): any residual goes to the
    # widest column, mirroring the caller's drift fixup.
    drift = sum(widths) - w
    if drift != 0:
        widest = max(range(max_cols), key=lambda i: widths[i])
        widths[widest] = max(1, widths[widest] - drift)
    return widths


# ── No-clip fitting: redistribute width+height, then shrink font ──────────────

# Absolute minimum font (pt) the uniform-shrink fallback may reach. Smaller than
# the normal 6pt readable floor: a tiny-but-fully-visible cell beats a clipped
# one when the text genuinely can't fit the original table size.
#
# RAISED from 3.5pt to `TABLE_MIN_FONT_PT`. 3.5pt is not legible at print size,
# and it was being reached routinely rather than exceptionally: the shrink loop
# is the FIRST thing that gives when a bbox is tight, so a dense table would
# silently land on the floor. The table is allowed to grow past its source bbox
# instead — a table one row taller than its OCR rectangle reads far better than
# one nobody can read. Override per deployment with DOCX_TABLE_MIN_FONT_PT.
#
# TWO TIERS, in order of preference:
#
#   TABLE_MIN_FONT_PT (10pt)       the size a table STARTS at and the floor the
#                                  size-ceiling heuristics may never cut below.
#   TABLE_SOFT_MIN_FONT_PT (8pt)   how far the uniform-shrink loop may go while
#                                  still holding the table inside its source
#                                  bbox. Only when 8pt STILL overflows does the
#                                  table grow taller instead of shrinking on.
#
# The middle tier matters because the two failure modes are asymmetric: shrinking
# from 10pt to 8pt costs a little readability, whereas growing past the source
# rectangle costs layout fidelity on every page the table shares. Trying the
# cheap remedy first keeps well-proportioned pages untouched and reserves growth
# for tables that genuinely cannot be made to fit.
TABLE_MIN_FONT_PT = float(os.environ.get("DOCX_TABLE_MIN_FONT_PT", "10.0"))
TABLE_SOFT_MIN_FONT_PT = min(
    TABLE_MIN_FONT_PT,
    float(os.environ.get("DOCX_TABLE_SOFT_MIN_FONT_PT", "8.0")),
)
# The shrink loop's real floor is the SOFT minimum; `TABLE_MIN_FONT_PT` remains
# the starting size and the ceiling floor.
_HARD_MIN_FONT_PT = TABLE_SOFT_MIN_FONT_PT

# Ceiling floor: the per-cell size cap is derived from the narrowest column
# (`narrowest_col_pt / 5.0`), which on a many-column table collapses to a few
# points regardless of how much text the cells actually hold. Never let that
# derived ceiling fall below the target size — the real no-clip guarantee is the
# shrink loop below, not this heuristic cap.
_TABLE_TARGET_FONT_PT = TABLE_MIN_FONT_PT


def _line_h_pt(size_px: int, bold: bool, cjk: bool) -> float:
    """Height (pt) of one wrapped line at `size_px`, matching the measurement
    used everywhere else (getmetrics + the CJK floor + 6% leading)."""
    font = get_font(max(1, size_px), bold=bold)
    if font is None:
        return size_px * 1.2
    asc, desc = font.getmetrics()
    nat = asc + desc
    if cjk:
        nat = max(nat, size_px * 1.2)
    return nat * 1.06


def _needed_lines(text: str, cell_w_pt: float, size_px: int, bold: bool) -> int:
    """Wrapped-line count for `text` in a cell `cell_w_pt` wide at `size_px`."""
    if not (text or "").strip():
        return 1
    font = get_font(max(1, size_px), bold=bold)
    if font is None:
        return 1
    return max(1, len(wrap_to_width(
        text, font, _usable_w_pt(cell_w_pt), size_px,
    )))


def _cell_overflows(
    text: str, cell_w_pt: float, cell_h_pt: float,
    size_px: int, bold: bool, reserve_pt: float = 0.0,
) -> bool:
    """True when `text` needs more vertical room than the cell provides.

    reserve_pt is height taken by a picture in the same cell (fit text above it).
    """
    if not (text or "").strip():
        return False
    avail_h = max(0.0, cell_h_pt - reserve_pt)
    n = _needed_lines(text, cell_w_pt, size_px, bold)
    return n * _line_h_pt(size_px, bold, has_cjk(text)) > avail_h + 0.5


def _any_cell_overflows(
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    col_w_emus: List[int],
    row_h_pts: List[float],
    max_cols: int,
    n_rows: int,
    size_px: int,
    reserve_pt_by_cell: Dict[Tuple[int, int], float],
) -> bool:
    for (r, c), (txt, cs, rs, is_h) in cell_anchors.items():
        if c >= max_cols:
            continue
        cell_w_pt = sum(col_w_emus[c:c + max(1, cs)]) / EMU_PER_PT
        cell_h_pt = sum(
            row_h_pts[r:r + max(1, rs)]
        ) if r < len(row_h_pts) else row_h_pts[-1]
        bold = bool(is_h or r == 0)
        if _cell_overflows(
            txt, cell_w_pt, cell_h_pt, size_px, bold,
            reserve_pt_by_cell.get((r, c), 0.0),
        ):
            return True
    return False


def _balance_row_heights(
    row_h_pts: List[float],
    row_min_pts: List[float],
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    col_w_emus: List[int],
    max_cols: int,
    n_rows: int,
    size_px: int,
    reserve_pt_by_cell: Dict[Tuple[int, int], float],
) -> List[float]:
    """Move height from rows with vertical slack to overflowing rows, holding
    ``sum(row_h_pts)`` constant and never dropping a row below ``row_min_pts``.

    Mirror of `_balance_col_widths` for the vertical axis: the row "pressure" is
    the worst (needed − available) line deficit among its anchored cells; height
    flows from the row with the most slack (available ≫ needed) to the row with
    the largest deficit. Strict-improvement, converges to a no-op when balanced.
    """
    if n_rows < 2:
        return row_h_pts

    def _row_deficit(heights: List[float], r: int) -> float:
        """Max (needed_h − avail_h) in pt over row r's anchored cells (>0 = clip)."""
        worst = 0.0
        for c in range(max_cols):
            anc = cell_anchors.get((r, c))
            if anc is None:
                continue
            txt, cs, rs, is_h = anc
            if rs != 1 or not (txt or "").strip():
                # Only single-row cells pin a single row's height; multi-row
                # cells are handled by their spanned rows collectively.
                continue
            cell_w_pt = sum(col_w_emus[c:c + max(1, cs)]) / EMU_PER_PT
            bold = bool(is_h or r == 0)
            n = _needed_lines(txt, cell_w_pt, size_px, bold)
            need_h = n * _line_h_pt(size_px, bold, has_cjk(txt))
            need_h += reserve_pt_by_cell.get((r, c), 0.0)
            worst = max(worst, need_h - heights[r])
        return worst

    heights = list(row_h_pts)
    min_step = max(0.5, sum(heights) * 0.005)
    for _ in range(64):
        deficits = [_row_deficit(heights, r) for r in range(n_rows)]
        cur_max = max(deficits)
        if cur_max <= 0.5:
            break
        receiver = max(range(n_rows), key=lambda r: (deficits[r], -r))
        donors = [
            r for r in range(n_rows)
            if r != receiver
            and heights[r] - row_min_pts[r] >= min_step
            and deficits[r] < cur_max
        ]
        if not donors:
            break
        donor = max(donors, key=lambda r: heights[r] - row_min_pts[r])
        donor_slack = heights[donor] - row_min_pts[donor]
        move = min(donor_slack, max(min_step, cur_max))
        trial = list(heights)
        trial[donor] -= move
        trial[receiver] += move
        new_def = [_row_deficit(trial, r) for r in range(n_rows)]
        if new_def[donor] >= cur_max:
            # Would just relocate the problem — try a smaller move.
            move = min_step
            trial = list(heights)
            trial[donor] -= move
            trial[receiver] += move
            new_def = [_row_deficit(trial, r) for r in range(n_rows)]
            if new_def[donor] >= cur_max or max(new_def) >= cur_max:
                break
        if max(new_def) < cur_max or (
            max(new_def) == cur_max
            and new_def.count(cur_max) < deficits.count(cur_max)
        ):
            heights = trial
        else:
            break

    # Preserve exact-sum invariant: any residual goes to the tallest row.
    drift = sum(heights) - sum(row_h_pts)
    if abs(drift) > 1e-6:
        tallest = max(range(n_rows), key=lambda i: heights[i])
        heights[tallest] = max(row_min_pts[tallest], heights[tallest] - drift)
    return heights


def _measure_text_block(
    text: str, box_w_pt: float, size_pt: float,
) -> Tuple[List[str], float]:
    """Wrap `text` to `box_w_pt` at `size_pt` and return (wrapped_lines,
    needed_height_pt). Newlines in `text` are honoured as hard breaks."""
    size_px = max(1, int(round(size_pt)))
    font = get_font(size_px)
    if font is None:
        lines = text.split("\n")
        return lines, len(lines) * size_pt * 1.3
    usable_pt = max(1.0, box_w_pt - 2.0 * _CELL_MARGIN_PT)
    wrapped: List[str] = []
    for para in text.split("\n"):
        if not para:
            wrapped.append("")
            continue
        wrapped.extend(wrap_to_width(para, font, usable_pt, size_px))
    asc, desc = font.getmetrics()
    nat = asc + desc
    if has_cjk(text):
        nat = max(nat, size_px * 1.2)
    line_h_pt = nat * 1.15
    return wrapped, max(1, len(wrapped)) * line_h_pt


def _emit_trailing_text_block(
    ctx, text: str, x: int, y: int, w: int, h: int, style: Dict,
) -> None:
    """Render a flattened non-table text block (checkboxes, review paragraphs,
    signatures, …) as a positioned text box at (x, y, w, h). Shrinks the font
    only as far as needed to fit the reserved height; never clips."""
    if not text.strip():
        return
    box_w_pt = max(1.0, w / EMU_PER_PT)
    box_h_pt = max(1.0, h / EMU_PER_PT)
    base_size_pt = float(style.get("size") or 11.0)
    size_pt = base_size_pt
    lines, need_pt = _measure_text_block(text, box_w_pt, size_pt)
    while need_pt > box_h_pt + 0.5 and size_pt > 5.0:
        size_pt = max(5.0, size_pt - 0.5)
        lines, need_pt = _measure_text_block(text, box_w_pt, size_pt)
    t_style = dict(style)
    t_style["size"] = size_pt
    line_pt = None
    if len(lines) >= 2:
        line_pt = min(box_h_pt / len(lines), size_pt * 1.5)
    # Keep the block on the page: if it would extend past the page bottom, lift
    # it up by the overflow (never above the page top).
    page_h_emu = int(round(ctx.page_h_pt * EMU_PER_PT))
    overflow = (y + h) - page_h_emu
    if overflow > 0:
        y = max(0, y - overflow)
    runs_xml = build_run_xml("\n".join(lines), t_style)
    para_xml = build_paragraph_xml(runs_xml, line_pt=line_pt)
    ctx.xml_chunks.append(
        build_anchored_textbox_xml(
            x, y, w, h, para_xml, ctx._next_id(), body_auto_fit=False,
        )
    )


def _pic_intrinsic_h(pic: Dict) -> float:
    """A recovered picture's source height (bbox height in pixels; falls back to
    the raster's own pixel height). Used only to split a shared rowspan between
    stacked pictures in proportion to their sizes — units cancel, so the raw
    pixel height is fine."""
    pb = pic.get("bbox") or []
    if len(pb) == 4 and pb[3] > pb[1]:
        return float(pb[3] - pb[1])
    img = pic.get("image_obj")
    if img is not None:
        try:
            return float(img.size[1])
        except Exception:
            pass
    return 1.0


def _split_multi_picture_image_cells(
    pic_assignments: Dict[Tuple[int, int], List[Dict]],
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    occupied: List[List[bool]],
    n_rows: int,
) -> None:
    """Split any image cell that was assigned MORE THAN ONE picture into one
    stacked sub-cell PER picture, so each diagram becomes its own row instead of
    several diagrams crammed into a single merged cell.

    Chandra sometimes emits a single ``<td rowspan=N><img></td>`` for a product
    group that actually holds several stacked diagrams (it under-counts the
    ``<img>`` tags and merges the rows). ``_assign_pictures_by_img_structure``
    still recovers every picture by y-position, so that one rowspan image cell
    ends up owning >1 picture; the emission loop then stacks them inside ONE
    vMerge cell (dividing its height by the picture count). We fix the STRUCTURE
    here instead: the anchor's ``rs`` rows are partitioned into ``n`` contiguous
    sub-spans — sized in proportion to each picture's own height so a taller
    diagram gets more rows — and each sub-cell is given exactly one picture. The
    existing reserve loop then grows each sub-span to fit its picture, and the
    emission loop renders each as its own ``vMerge`` cell (no height division).

    Mutates ``pic_assignments`` and ``cell_anchors`` in place. ``occupied`` is
    unchanged (the sub-spans partition exactly the rectangle the original anchor
    already occupied). A strict no-op for the normal one-picture-per-cell case,
    so well-formed picture tables render byte-identically.
    """
    for (a_r, a_c), pics in list(pic_assignments.items()):
        if len(pics) <= 1:
            continue
        anchor = cell_anchors.get((a_r, a_c))
        if anchor is None:
            continue
        text, cs, rs, is_header = anchor
        pics = sorted(pics, key=_pic_sort_key)   # top-to-bottom
        n = len(pics)

        if rs < n:
            # Not enough rows for one picture each: give the first rs-1 pictures a
            # single row and pile the remainder into the last sub-cell (the
            # reserve loop grows it and the emission scale clamp shrinks the
            # stacked pictures so nothing clips). Rare in practice.
            spans = [1] * (rs - 1) + [1]
            groups: List[List[Dict]] = [[p] for p in pics[:rs - 1]]
            groups.append(pics[rs - 1:])
        else:
            # Partition rs rows across n pictures ∝ picture height, min 1 each.
            heights = [max(1.0, _pic_intrinsic_h(p)) for p in pics]
            htot = sum(heights)
            spans = [max(1, int(rs * hh / htot)) for hh in heights]
            # Fix rounding so the sub-spans sum to exactly rs.
            drift = rs - sum(spans)
            i = 0
            while drift != 0 and n > 0:
                if drift > 0:
                    spans[i % n] += 1
                    drift -= 1
                else:
                    if spans[i % n] > 1:
                        spans[i % n] -= 1
                        drift += 1
                i += 1
            groups = [[p] for p in pics]

        # Rewrite the anchor into contiguous sub-anchors, one group each.
        del pic_assignments[(a_r, a_c)]
        r = a_r
        for idx, (span_i, grp) in enumerate(zip(spans, groups)):
            cell_anchors[(r, a_c)] = (
                text if idx == 0 else "", cs, span_i, is_header,
            )
            pic_assignments[(r, a_c)] = grp
            r += span_i



# ── Nested sub-tables inside a cell (flow mode) ──────────────────────────────
# An empty paragraph, needed in two places by the OOXML schema (`w:tc` is
# EG_BlockLevelElts): a `w:tc` MUST end with a `w:p`, and two adjacent `w:tbl`s
# MUST be separated by one or Word merges them into a single table. LibreOffice
# tolerates both violations, Word does not — it shows a repair prompt.
#
# `w:sz` must sit in a `w:rPr`, NOT directly in `w:pPr` — it is a RUN property
# (compare `ooxml.build_run_xml`). Emitted loose in `w:pPr` it is simply ignored,
# so this "1pt" spacer actually rendered at the inherited body size and pushed a
# visible gap into every cell that holds a sub-table. Wrapping it in the
# paragraph-mark `w:rPr` makes the height what the name claims: half a point.
# 2.0pt. At the previous 1.0pt the sub-table's last row border sat close enough
# to the parent cell's border to read as touching. The SAME constant drives the
# leading and the trailing spacer, so the gap above and below a sub-table match.
# `_SPACER_PT` derives from it, so `_cell_blocks_xml` reserves the larger gap in
# `natural_pt` automatically and the sub-table shrinks to fit rather than
# overflowing its cell.
_SPACER_HALF_POINTS = 4                  # w:sz is in half-points → 2.0pt
_EMPTY_P = (
    f'<w:p xmlns:w="{NS_W}"><w:pPr>'
    f'<w:spacing w:before="0" w:after="0" w:line="{_SPACER_HALF_POINTS * 10}"'
    f' w:lineRule="exact"/>'
    f'<w:rPr><w:sz w:val="{_SPACER_HALF_POINTS}"/>'
    f'<w:szCs w:val="{_SPACER_HALF_POINTS}"/></w:rPr>'
    f'</w:pPr></w:p>'
)
# Height a single `_EMPTY_P` actually occupies, in points. `w:line` is in
# twentieths of a point and `lineRule="exact"` pins it, so this is exact rather
# than an estimate — which is what lets the fill maths below stay honest.
_SPACER_PT = _SPACER_HALF_POINTS / 2.0


def render_nested_table_xml(rows, avail_w_emu: int, style: Dict,
                            size_pt: float,
                            target_h_pt: Optional[float] = None,
                            ) -> Tuple[str, float]:
    """Render a NESTED table in FLOW mode: ``(<w:tbl> xml, needed_height_pt)``.

    Deliberately NOT `render_table`: that one is bbox/anchor oriented (it
    computes absolute page geometry and terminates by appending an anchored
    text box to `ctx.xml_chunks`), and a sub-table inside a cell must not be
    positioned independently — it flows with its parent cell.

    So this drops the whole feed-forward fitting apparatus (column/row
    balancers, coupled redistribution, uniform font fallback) and uses the
    simple flow rule instead:

      * columns proportional to `compute_col_weights`, normalised to
        ``avail_w_emu``;
      * each row as tall as its tallest wrapped cell;
      * rows emit ``hRule="exact"`` — a sub-table is as fixed as its parent.

    Height is an OUTPUT, not a constraint — that is what makes this tractable.
    The caller reserves the returned height in the parent cell.

    ``target_h_pt`` is the one exception, and it is deliberately OPTIONAL so the
    rule above still holds everywhere it matters. When ``None`` (every
    MEASUREMENT call) the natural height is returned and nothing changes. When
    given (the EMISSION call, once the parent's final row heights are known) any
    surplus over the natural height is distributed across the rows IN PROPORTION
    to their natural heights, so the sub-table covers its parent cell instead of
    hugging the top and leaving the parent's inflation as dead space below.
    Proportional (rather than all-on-the-last-row) keeps the row rhythm intact.

    The target is TWO-WAY. Under the natural height the sub-table is shrunk to
    fit — font first (down to `_HARD_MIN_FONT_PT`), then row compression — because
    a nested table may no more overflow its cell than the outer table may overflow
    its bbox. Over it, the rows grow proportionally to cover the cell.
    """
    if not rows:
        return "", 0.0
    max_cols, n_rows, anchors, occupied, _img = parse_table_grid(rows)
    if max_cols <= 0 or n_rows <= 0 or not anchors:
        return "", 0.0

    weights = compute_col_weights(anchors, max_cols)
    total_w = sum(weights) or max_cols
    col_w = [max(1, int(round(avail_w_emu * cw / total_w))) for cw in weights]
    drift = sum(col_w) - avail_w_emu
    if drift != 0:
        widest = max(range(max_cols), key=lambda i: col_w[i])
        col_w[widest] = max(1, col_w[widest] - drift)
    # Emission widths in twips (`dxa`). `col_w` is final here — this emitter has
    # no rebalancing loop — so the conversion can happen once, up front. Every
    # emitted width below is derived by SUMMING these twips, never by converting
    # an EMU sum, so a colspan cell always equals the columns it covers exactly.
    col_w_tw = _emu_tw_split(col_w)

    def _natural_rows(spx: int) -> List[float]:
        """Row heights this table needs at font size `spx` — the natural layout."""
        rh = [0.0] * n_rows
        for (r, c), (txt, cs, rs, is_h) in anchors.items():
            if rs != 1:
                continue
            cw_pt = sum(col_w[c:c + max(1, cs)]) / EMU_PER_PT
            bold = bool(is_h or r == 0)
            n = _needed_lines(txt, cw_pt, spx, bold)
            rh[r] = max(rh[r], n * _line_h_pt(spx, bold, has_cjk(txt)))
        floor_h = _line_h_pt(spx, False, False)
        rh = [max(h, floor_h) for h in rh]
        # Multi-row cells: make sure their spanned rows collectively fit.
        for (r, c), (txt, cs, rs, is_h) in anchors.items():
            if rs <= 1 or not (txt or "").strip():
                continue
            cw_pt = sum(col_w[c:c + max(1, cs)]) / EMU_PER_PT
            bold = bool(is_h or r == 0)
            need = _needed_lines(txt, cw_pt, spx, bold) * _line_h_pt(
                spx, bold, has_cjk(txt))
            span = list(range(r, min(n_rows, r + rs)))
            have = sum(rh[i] for i in span)
            if need > have and span:
                add = (need - have) / len(span)
                for i in span:
                    rh[i] += add
        return rh

    size_px = max(1, int(round(size_pt)))
    row_h = _natural_rows(size_px)

    # ── Fit the sub-table to its cell, in BOTH directions ───────────────────
    # A nested table is as fixed as its parent: it may not spill past the cell
    # it lives in. So when the natural layout is TALLER than the target, shrink
    # the FONT first (same `_HARD_MIN_FONT_PT` floor the outer table uses, so
    # text stays as readable as possible), and only compress rows if the floor
    # is reached. When it is SHORTER, grow proportionally to cover the cell —
    # that is the fill behaviour, unchanged.
    if target_h_pt is not None and float(target_h_pt) > 0.5:
        target = float(target_h_pt)
        if sum(row_h) > target + 0.5:
            trial_pt = float(size_pt)
            while (sum(row_h) > target + 0.5
                   and trial_pt > _HARD_MIN_FONT_PT + 1e-6):
                trial_pt = max(_HARD_MIN_FONT_PT, trial_pt - 0.5)
                size_px = max(1, int(round(trial_pt)))
                row_h = _natural_rows(size_px)
            size_pt = trial_pt
        natural_total = sum(row_h)
        if natural_total > 1e-6 and abs(natural_total - target) > 0.5:
            # Grow to fill, or (font floor reached) compress to fit.
            row_h = [h * (target / natural_total) for h in row_h]

    cell_style = dict(style)
    cell_style["size"] = size_pt
    _mar_tw = int(round(_CELL_MARGIN_PT * 20))
    tbl_pr = (
        '<w:tblPr>'
        f'<w:tblW w:w="{sum(col_w_tw)}" w:type="dxa"/>'
        '<w:tblLayout w:type="fixed"/>'
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '</w:tblBorders>'
        '<w:tblCellMar>'
        '<w:top w:w="0" w:type="dxa"/>'
        f'<w:left w:w="{_mar_tw}" w:type="dxa"/>'
        '<w:bottom w:w="0" w:type="dxa"/>'
        f'<w:right w:w="{_mar_tw}" w:type="dxa"/>'
        '</w:tblCellMar>'
        '</w:tblPr>'
    )
    grid = "<w:tblGrid>" + "".join(
        f'<w:gridCol w:w="{cw}"/>' for cw in col_w_tw) + "</w:tblGrid>"

    rows_xml: List[str] = []
    for r in range(n_rows):
        cells_xml: List[str] = []
        skip = 0
        for c in range(max_cols):
            if skip > 0:
                skip -= 1
                continue
            anc = anchors.get((r, c))
            if anc is not None:
                txt, cs, rs, is_h = anc
                cw_emu = sum(col_w[c:c + cs])
                bold = bool(is_h or r == 0)
                st = dict(cell_style)
                if bold:
                    st["bold"] = True
                font = get_font(size_px, bold=bold)
                shown = "\n".join(wrap_to_width(
                    txt, font, _usable_w_pt(cw_emu / EMU_PER_PT), size_px,
                )) if (txt and font is not None) else (txt or "")
                para = build_paragraph_xml(
                    build_run_xml(shown, st), alignment="center",
                    line_pt=_line_h_pt(size_px, bold, has_cjk(shown))
                    if shown.strip() else None,
                )
                pr = [f'<w:tcW w:w="{sum(col_w_tw[c:c + cs])}" w:type="dxa"/>']
                if cs > 1:
                    pr.append(f'<w:gridSpan w:val="{cs}"/>')
                if rs > 1:
                    pr.append('<w:vMerge w:val="restart"/>')
                pr.append('<w:vAlign w:val="center"/>')
                cells_xml.append(
                    f'<w:tc><w:tcPr>{"".join(pr)}</w:tcPr>{para}</w:tc>')
                skip = cs - 1
            elif occupied[r][c]:
                # vMerge continuation (or colspan continuation, skipped).
                above_vm, vm_cs = False, 1
                rr = r - 1
                while rr >= 0:
                    if (rr, c) in anchors:
                        _t, _cs, _rs, _h = anchors[(rr, c)]
                        if _rs > 1 and rr + _rs > r:
                            above_vm, vm_cs = True, max(1, _cs)
                        break
                    if not occupied[rr][c]:
                        break
                    rr -= 1
                if above_vm:
                    cont_w = sum(col_w[c:c + vm_cs])
                    span_xml = f'<w:gridSpan w:val="{vm_cs}"/>' if vm_cs > 1 else ""
                    cells_xml.append(
                        f'<w:tc><w:tcPr>'
                        f'<w:tcW w:w="{sum(col_w_tw[c:c + vm_cs])}" w:type="dxa"/>'
                        f'{span_xml}<w:vMerge w:val="continue"/>'
                        f'<w:vAlign w:val="center"/></w:tcPr>'
                        f'{build_paragraph_xml("", alignment="center")}</w:tc>')
                    skip = vm_cs - 1
            else:
                cells_xml.append(
                    f'<w:tc><w:tcPr><w:tcW w:w="{col_w_tw[c]}" w:type="dxa"/>'
                    f'<w:vAlign w:val="center"/></w:tcPr>'
                    f'{build_paragraph_xml("", alignment="center")}</w:tc>')
        if not cells_xml:
            continue
        # `exact`, never `atLeast`: a nested table is as FIXED as its parent and
        # must not spill past the cell. `atLeast` let Word grow a sub-table row
        # beyond the declared height, which pushed the parent cell out. The
        # font-shrink + compression above is what makes the content fit these
        # heights, so pinning them clips nothing that the fit could have saved.
        tw = max(1, int(round(row_h[r] * 20)))
        # `cantSplit` FIRST in w:trPr (it is a sequence type): a row must never
        # break across a page boundary — that is exactly the "columns went to
        # the next page" symptom this guards against.
        rows_xml.append(
            f'<w:tr><w:trPr><w:cantSplit/>'
            f'<w:trHeight w:val="{tw}" w:hRule="exact"/>'
            f'</w:trPr>{"".join(cells_xml)}</w:tr>')

    if not rows_xml:
        return "", 0.0
    xml = f'<w:tbl xmlns:w="{NS_W}">{tbl_pr}{grid}{"".join(rows_xml)}</w:tbl>'
    return xml, sum(row_h)


def _cell_blocks_xml(blocks, cell_w_emu: int, style: Dict,
                     size_pt: float,
                     target_h_pt: Optional[float] = None,
                     ) -> Tuple[str, float]:
    """Render a cell's ordered ``blocks`` (see `Cell.blocks`) as OOXML.

    Emits one ``<w:p>`` per text block and one ``<w:tbl>`` per nested table, IN
    SOURCE ORDER, so ``caption / table / caption / table`` renders the way the
    source document reads. Returns ``(xml, needed_height_pt)``.

    Enforces the two `w:tc` content rules: an empty ``<w:p>`` between adjacent
    tables, and a trailing ``<w:p>`` as the cell's last child. A LEADING spacer
    is emitted before a first-child ``<w:tbl>`` too: the parent pins
    ``<w:top w:w="0">`` cell margins and ``vAlign=top``, so without it the
    sub-table's first row border lands exactly ON the parent cell's border and
    the two lines visibly collide.

    ``target_h_pt`` fills the cell. It is apportioned by ONE general rule:

        text blocks keep their natural height; the surplus is divided among the
        TABLE blocks in proportion to their natural heights.

    That degenerates correctly over the whole space — one table takes it all, N
    tables share it, a caption+table grows only the table, and a cell with no
    table block gets no target at all and renders exactly as before. Computing
    it needs every block's natural height up front, hence the two passes.
    """
    size_px = max(1, int(round(size_pt)))
    inner_w = max(1, cell_w_emu - int(round(2 * _CELL_MARGIN_PT * EMU_PER_PT)))

    # ── Pass 1: measure every block at its natural size ──────────────────────
    # `plan` mirrors the emission order exactly so pass 2 can walk it directly.
    plan: List[Tuple[str, object, float]] = []   # (kind, payload_or_xml, nat_h)
    prev_was_table = False
    n_spacers = 1                                # the mandatory trailing w:p
    for kind, payload in blocks:
        if kind == "text":
            txt = (payload or "").strip()
            if not txt:
                continue
            font = get_font(size_px)
            shown = "\n".join(wrap_to_width(
                txt, font, _usable_w_pt(cell_w_emu / EMU_PER_PT), size_px,
            )) if font is not None else txt
            lh = _line_h_pt(size_px, False, has_cjk(shown))
            xml = build_paragraph_xml(
                build_run_xml(shown, {**style, "size": size_pt}), line_pt=lh,
            )
            plan.append(("text", xml, max(1, len(shown.split("\n"))) * lh))
            prev_was_table = False
        else:                                    # ("table", rows)
            _xml, sub_h = render_nested_table_xml(
                payload, inner_w, style, size_pt)
            if not _xml:
                continue
            if prev_was_table or not plan:
                # Between two tables Word would MERGE them without a separator;
                # before a leading table the spacer is what keeps the sub-table
                # off the parent cell's top border.
                n_spacers += 1
            plan.append(("table", payload, sub_h))
            prev_was_table = True
    if not plan:
        return "", 0.0

    natural_pt = sum(h for _k, _p, h in plan) + n_spacers * _SPACER_PT

    # ── Apportion the difference across the TABLE blocks only ────────────────
    # Two-way, matching `render_nested_table_xml`: a SURPLUS is shared out so the
    # tables cover the cell, and a DEFICIT (the blocks want more room than the
    # cell has) is shared out too, so each table shrinks to fit rather than
    # overflowing. Text blocks keep their natural height either way — shrinking
    # prose to make room for a table would be the wrong trade.
    targets: Dict[int, float] = {}
    if target_h_pt is not None:
        delta = float(target_h_pt) - natural_pt
        tbl_idx = [i for i, (k, _p, _h) in enumerate(plan) if k == "table"]
        tbl_total = sum(plan[i][2] for i in tbl_idx)
        if abs(delta) > 0.5 and tbl_idx and tbl_total > 1e-6:
            for i in tbl_idx:
                share = plan[i][2] / tbl_total
                # Never below a hairline: a table with zero height is invalid.
                targets[i] = max(1.0, plan[i][2] + delta * share)

    # ── Pass 2: emit, re-rendering each table at its apportioned height ──────
    parts: List[str] = []
    total_pt = n_spacers * _SPACER_PT
    prev_was_table = False
    for i, (kind, payload, nat_h) in enumerate(plan):
        if kind == "text":
            parts.append(payload)                # already-built paragraph XML
            total_pt += nat_h
            prev_was_table = False
            continue
        if prev_was_table or not parts:
            parts.append(_EMPTY_P)
        sub_xml, sub_h = render_nested_table_xml(
            payload, inner_w, style, size_pt, target_h_pt=targets.get(i),
        )
        if not sub_xml:
            continue
        parts.append(sub_xml)
        total_pt += sub_h
        prev_was_table = True
    if not parts:
        return "", 0.0
    # A w:tc must END with a w:p.
    parts.append(_EMPTY_P)
    return "".join(parts), total_pt


def _cell_text_direction(cell_rotations, row_idx: int, col_idx: int):
    """OOXML `w:textDirection` value for a cell, or None when upright.

    `cell_rotations` maps "row,col" -> degrees CCW-positive, carried on the
    Table entry so the (text, colspan, rowspan) cell tuple — which threads
    through every width/height balancer — stays untouched.

    btLr reads bottom-to-top (the usual rotated-header look); tbRl reads
    top-to-bottom.
    """
    if not cell_rotations:
        return None
    try:
        rot = float(cell_rotations.get("%d,%d" % (row_idx, col_idx)) or 0.0)
    except (AttributeError, TypeError, ValueError):
        return None
    rot %= 360.0
    if abs(rot - 90.0) <= 5.0:
        return "btLr"
    if abs(rot - 270.0) <= 5.0:
        return "tbRl"
    return None


def render_table(
    ctx,
    entry: Dict,
    pictures_for_table: Optional[List[Dict]] = None,
) -> None:
    text = (entry.get("text") or "").strip()
    if not text:
        return
    bbox = entry.get("bbox")
    if not bbox or len(bbox) != 4:
        return
    # Rotation, in two independent flavours:
    #   * `cell_rotations` {"row,col": deg} turns individual cells' text while
    #     the grid stays upright (rotated headers) — Case A.
    #   * `rotation` on the entry itself turns the WHOLE table, container and
    #     all (a landscape table on a portrait page) — Case B.
    cell_rotations = entry.get("cell_rotations") or {}
    table_rotation = normalize_rotation(entry.get("rotation"))
    if is_quarter_turn(table_rotation):
        # The grid is built in transposed space and the container is spun, so
        # the shape's unrotated extent is the bbox with axes swapped.
        x, y, w, h = rotated_shape_geometry(
            bbox, table_rotation, ctx.zoom, ctx.page_w_pt, ctx.page_h_pt,
        )
    else:
        x, y, w, h = bbox_px_to_emu(
            bbox, ctx.zoom, ctx.page_w_pt, ctx.page_h_pt,
        )

    # ── Obstacle-bounded WIDTH (axis 1 of 2; the height bound is applied at the
    # PAGE CONTAINMENT block below, once the grown row heights are known) ──────
    # `w` is an AXIOM for every width computation that follows — the proportional
    # split, the floors, the trim loop, the drift fixup and `_balance_col_widths`
    # all conserve it exactly — so a `w` that already overlaps a right-hand
    # neighbour can never be recovered from later. Bound it HERE, at the source.
    #
    # Skipped for quarter-turn tables: `w` is the TRANSPOSED extent there, so a
    # page-frame comparison against it would be measuring the wrong axis.
    _obstacle_w_emu = 0
    if not is_quarter_turn(table_rotation):
        _bx1, _by1, _bx2, _by2 = [float(v) for v in bbox]
        _room_w_pt = rightward_room_pt(
            getattr(ctx, "page_entries", None) or (), entry,
            _bx1, _by1, _by2, ctx.zoom,
        )
        if _room_w_pt is not None:
            _obstacle_w_emu = max(1, int(round(_room_w_pt * EMU_PER_PT)))
            if _obstacle_w_emu < w:
                log.warning(
                    "[table] width %.0fpt would overlap the neighbour to its "
                    "right — capping at %.0fpt",
                    w / EMU_PER_PT, _obstacle_w_emu / EMU_PER_PT,
                )
                w = _obstacle_w_emu

    # Parse table cells (HTML preferred; markdown fallback). Cells carry
    # (text, colspan, rowspan) so we can honour merged headers.
    rows: List[Tuple[List[Tuple[str, int, int]], bool]] = []
    if "<table" in text.lower():
        rows = parse_html_table_rows(text)
    else:
        md_rows = parse_markdown_table(text)
        if md_rows:
            rows = [([(c, 1, 1) for c in r], False) for r in md_rows]

    style = entry.get("style") or {}

    # A "Table" entry often carries non-table HTML around the grid (form
    # checkboxes, review paragraphs, signatures — emitted as <p>/<input>/<br>).
    # Rendering only the grid drops all of it. Extract that surrounding content
    # now; below we reserve vertical room for it and render it beneath the table
    # so nothing is lost. Empty for a bare table → zero behavioural change.
    trailing_text = extract_non_table_text(text) if rows else ""
    trailing_h_emu = 0
    trailing_style = dict(style)
    if trailing_text:
        box_w_pt = max(1.0, w / EMU_PER_PT)
        _size = float(style.get("size") or 11.0)
        _lines, _need_pt = _measure_text_block(trailing_text, box_w_pt, _size)
        trailing_h_emu = int(round(_need_pt * EMU_PER_PT))
        # Never let the trailing block claim more than 60% of the bbox — the
        # table must keep enough room to render its own rows.
        trailing_h_emu = min(trailing_h_emu, int(h * 0.6))
        # Shrink the table's allotted height so the block fits below it.
        h = max(1, h - trailing_h_emu)

    if not rows:
        # Fall back to text rendering inside a positioned box.
        x1, y1, x2, y2 = bbox
        box_w_pt = max(1.0, (x2 - x1) / ctx.zoom)
        box_h_pt = max(1.0, (y2 - y1) / ctx.zoom)
        t_style = dict(style)
        base_size_pt = float(t_style.get("size") or 11.0)
        fitted, lines = fit_multiline(
            text, box_w_pt, box_h_pt, max_size_pt=base_size_pt,
        )
        if fitted < base_size_pt:
            t_style["size"] = fitted
        line_pt = None
        render_text = text
        if lines:
            render_text = "\n".join(lines)
            if len(lines) >= 2:
                line_pt = box_h_pt / len(lines)
        runs_xml = build_run_xml(render_text, t_style)
        para_xml = build_paragraph_xml(runs_xml, line_pt=line_pt)
        ctx.xml_chunks.append(
            build_anchored_textbox_xml(
                x, y, w, h, para_xml, ctx._next_id(),
                body_auto_fit=True,
                rotation=table_rotation,
                # `x/y/w/h` came from rotated_shape_geometry when turned, which
                # is the transposed extent the shape-rotating path expects.
                rotate_shape=True,
            )
        )
        return

    max_cols, n_rows, cell_anchors, occupied, img_by_anchor = parse_table_grid(rows)

    # A layout model can omit a whole table column — typically a blank label
    # column, which carries no text for it to notice. Every row is then uniformly
    # one cell too narrow, so the grid above has nothing to disagree with and
    # every cell is placed one column off. Pictures are the only in-table content
    # with page geometry of their own, so they are the one available witness:
    # when every picture sits a full column-width right of where its `<img>` was
    # parsed, insert the missing blank column and re-place. Inert for tables with
    # no pictures (see `dropped_column_index` for the full trigger conditions).
    if pictures_for_table and img_by_anchor:
        _ins = _dropped_column_from_pictures(
            pictures_for_table, bbox, cell_anchors, img_by_anchor,
            max_cols, n_rows, col_weight_hint=None,
        )
        if _ins is not None:
            log.info(
                "[table] picture geometry disagrees with the parsed grid in "
                "every case — inserting the blank column the OCR dropped at "
                "index %d and re-placing", _ins,
            )
            (max_cols, n_rows, cell_anchors, occupied,
             img_by_anchor) = parse_table_grid(rows, insert_blank_col=_ins)

    # Nested sub-tables (if any) for cells that hold them, keyed by anchor.
    # Prefer a fresh re-parse of the source HTML over the side-channel: the
    # side-channel breaks whenever an upstream step (json_to_docx round-trip,
    # table_validate, table_reocr, cell_picture_recovery) rebuilds cells as
    # bare 4-tuples and drops the `_blocks` attribute. Re-parsing `text` is
    # self-contained and always yields Cell instances with `_blocks` intact.
    # Falls back to the side-channel if the re-parse finds nothing, so tables
    # produced by callers that hand-build `rows` without a source string
    # keep working.
    blocks_by_anchor = _extract_blocks_by_anchor(text) or pop_blocks_by_anchor()

    # Column widths: inferred from content, normalized to bbox width so the
    # total table width = source bbox width. When the entry carries
    # `_shared_col_weights` (set by the chain detector in json_to_docx for
    # tables that continue across pages), prefer those — that keeps the
    # column proportions identical across the head and continuation rows of
    # the same logical table.
    shared = entry.get("_shared_col_weights")
    if shared and len(shared) == max_cols:
        col_weight = [max(1, int(w)) for w in shared]
    else:
        col_weight = compute_col_weights(cell_anchors, max_cols)

    # ── Container-bbox guard (part 1 of 2: eligibility + natural width) ──────
    # Some table entries carry the rectangle of something ENCLOSING the table —
    # a page frame, a form panel, a mislabelled `Form` block rescued by
    # `_looks_like_table` — instead of the table's own extent. `w` is an axiom
    # for everything below, so such a table is stretched across the whole
    # container and `_normalise_to` inflates its rows to match.
    #
    # The DECISION needs wrapped line counts, which do not exist until the
    # columns are sized, so it is deferred to part 2 below. Only the natural
    # width is computed here, where `col_weight` is fresh and untouched.
    #
    # Skipped for quarter-turn tables (`w` is the TRANSPOSED extent there, and
    # the trailing-text block does its own footprint arithmetic off x/y/w/h) and
    # for tables holding pictures (picture columns are sized in real points from
    # raster dimensions, which the display-unit natural width does not model).
    _bbox_clamped = False
    _guard_eligible = (
        not is_quarter_turn(table_rotation) and not pictures_for_table
    )
    _natural_w = natural_table_width_emu(
        cell_anchors, max_cols, col_weight,
        float(style.get("size") or 11.0),
        _display_width, _longest_token_width,
    ) if _guard_eligible else 0

    # Weight columns by any pictures contained in the table. The cell-based
    # detector (`_assign_pictures_to_cells`, used later for inline placement)
    # is the source of truth for "which cell does this picture belong to" —
    # it scores each candidate cell by emptiness + horizontal proximity, so
    # it works even when a text-heavy neighbouring column would otherwise
    # swallow the picture under naïve centroid-in-edges geometry.
    #
    # We pre-compute that assignment here so the column it picks can be
    # weight-boosted; then the same map is reused inline below to actually
    # place the picture XML.
    pic_assignments: Dict[Tuple[int, int], List[Dict]] = {}
    # Per-column minimum width (pt) demanded by any picture assigned to that
    # column. Zero for text-only columns / picture-free tables.
    pic_col_min_pt = [0.0] * max_cols
    if pictures_for_table:
        pic_assignments = _assign_pictures_to_cells(
            pictures_for_table, bbox, col_weight, cell_anchors,
            max_cols, n_rows, img_by_anchor,
        )
        # A picture column's need is a MINIMUM WIDTH (the fitted image width),
        # not a proportional weight. Record the widest picture width per column
        # here and fold it into the per-column floor below. Expressing it as a
        # weight instead (the old behaviour) let the image column's huge pixel
        # count dominate the proportional split and squeeze every numeric
        # column down to a few points.
        zoom = max(1e-6, float(getattr(ctx, "zoom", 1.0) or 1.0))
        for (_r, c), pics in pic_assignments.items():
            for pic in pics:
                pb = pic.get("bbox") or []
                if len(pb) != 4 or c >= max_cols:
                    continue
                pic_w_pt = max(1.0, (pb[2] - pb[0]) / zoom)
                pic_col_min_pt[c] = max(pic_col_min_pt[c], pic_w_pt)

        # Split any image cell that owns >1 picture into one stacked sub-cell per
        # picture, so multiple diagrams in a merged cell become separate rows
        # (matching the source) instead of being crammed together. No-op when
        # every image cell has a single picture.
        _split_multi_picture_image_cells(
            pic_assignments, cell_anchors, occupied, n_rows,
        )

    # Allocate width with a floor per column: a column with a short label
    # like 'P3' still needs enough room for that label, otherwise narrow
    # columns get squeezed to a few millimetres and the text either clips
    # or wraps awkwardly. The floor is the larger of: ~4 chars at the
    # base font size, or the column's longest unbreakable token. Floors are
    # capped so they can never consume more than 60% of the table — that
    # would leave nothing for the long-text columns.
    declared_size_pt = float(style.get("size") or 11.0)
    char_w_pt = max(4.0, declared_size_pt * 0.55)
    min_col_pt = char_w_pt * 4.0  # ~"N/ACC" width with padding
    min_col_emu_default = int(round(min_col_pt * EMU_PER_PT))
    # Cap the floor so the floors collectively never exceed 60% of `w` —
    # otherwise a wide-column-heavy table would have nothing left to spend.
    min_col_cap = max(1, int(w * 0.6 / max_cols))
    floor_per_col = min(min_col_emu_default, min_col_cap)
    min_col_emus = [floor_per_col] * max_cols

    # Per-column CONTENT-AWARE floor: a short-label column (e.g. a CJK header
    # like '可能的危险(源)') must be wide enough that its label wraps to at most
    # ~2 lines instead of collapsing into a 1-glyph-per-line vertical strip that
    # then clips. Compute each column's content-min in DISPLAY units, convert to
    # pt at roughly half the declared glyph advance per display unit (1 CJK glyph
    # = 2 display units ≈ declared_size_pt wide), and fold it into the floor.
    # Each such floor is individually capped so one column can't eat the table;
    # the redistribution loop below steals the extra width from long-text columns.
    pt_per_display = max(1.0, declared_size_pt * 0.5)
    content_min_disp = [0] * max_cols
    for (_r, c), (txt, cs, _rs, _h) in cell_anchors.items():
        if c >= max_cols:
            continue
        cs = max(1, min(cs, max_cols - c))
        cm = _content_min_width(txt)
        if cm <= 0:
            continue
        share = max(1, cm // cs)
        for cc in range(c, min(max_cols, c + cs)):
            content_min_disp[cc] = max(content_min_disp[cc], share)
    single_col_cap_emu = max(floor_per_col, int(w * 0.5))
    for c in range(max_cols):
        if content_min_disp[c] > 0:
            # +2 display units of padding so the label isn't flush to the border.
            need_emu = int(round((content_min_disp[c] + 2) * pt_per_display * EMU_PER_PT))
            min_col_emus[c] = max(min_col_emus[c], min(need_emu, single_col_cap_emu))
    # Raise the floor of any picture column to the picture's own width (plus a
    # little padding), so the column is guaranteed wide enough to show the
    # image at full size without upscaling the whole column out of proportion.
    # Cap a single picture column at 55% of the table so the data columns keep
    # a usable share.
    if any(pic_col_min_pt):
        pic_col_cap_emu = int(round(w * 0.55))
        for c in range(max_cols):
            if pic_col_min_pt[c] > 0:
                need_emu = int(round((pic_col_min_pt[c] + 4.0) * EMU_PER_PT))
                min_col_emus[c] = max(
                    min_col_emus[c], min(need_emu, pic_col_cap_emu)
                )

    total_weight = sum(col_weight) or max_cols
    raw_emus = [
        max(1, int(round(w * cw / total_weight))) for cw in col_weight
    ]
    # Lift columns to their floor, then re-balance: take the over-allocation
    # proportionally from columns that are above their floor.
    col_w_emus = [max(rw, mn) for rw, mn in zip(raw_emus, min_col_emus)]
    delta = sum(col_w_emus) - w
    if delta > 0:
        # Trim from columns that have headroom above their floor.
        for _ in range(8):  # at most a few passes — converges fast
            slack = [c - mn for c, mn in zip(col_w_emus, min_col_emus)]
            slack_total = sum(s for s in slack if s > 0)
            if slack_total <= 0 or delta <= 0:
                break
            new_emus = []
            for c, s in zip(col_w_emus, slack):
                if s <= 0:
                    new_emus.append(c)
                    continue
                take = int(round(delta * s / slack_total))
                take = min(take, s)
                new_emus.append(c - take)
            new_delta = sum(new_emus) - w
            if new_delta == delta:
                break
            col_w_emus = new_emus
            delta = new_delta
    # Final fixup so the row sums exactly to `w`.
    drift = sum(col_w_emus) - w
    if drift != 0:
        # Adjust the widest column to absorb the rounding drift.
        widest = max(range(max_cols), key=lambda i: col_w_emus[i])
        col_w_emus[widest] = max(1, col_w_emus[widest] - drift)

    # WIDTH-FIRST rebalancing: the proportional split above is feed-forward, so
    # a column can be left crushed (its text wraps to a 1-glyph-per-line strip)
    # while a sibling is over-wide. Move width from low-line-pressure columns to
    # the most-crushed one — within the SAME total `w`, respecting floors — so
    # text stays visible before we ever grow rows or shrink fonts. A no-op when
    # widths are already balanced. Measured at a font derived from the current
    # narrowest column (relative line counts are what the balancer compares).
    _bal_narrow_pt = min(col_w_emus) / EMU_PER_PT
    # Never below the target size — see `_TABLE_TARGET_FONT_PT`. The balancer
    # compares RELATIVE line counts, so measuring at the size the cells will
    # actually render at keeps its decisions consistent with the final emission.
    _bal_size_pt = min(
        max(declared_size_pt, _TABLE_TARGET_FONT_PT),
        max(_TABLE_TARGET_FONT_PT, _bal_narrow_pt / 5.0),
        12.0,
    )
    _bal_size_px = max(1, int(round(_bal_size_pt)))
    _bal_font = get_font(_bal_size_px)
    _bal_font_bold = get_font(_bal_size_px, bold=True) or _bal_font
    col_w_emus = _balance_col_widths(
        col_w_emus, min_col_emus, cell_anchors, max_cols, w,
        _bal_font, _bal_font_bold, _bal_size_px,
    )

    # NOTE: the <w:tblGrid> is built LATER (just before emission) from the FINAL
    # `col_w_emus`, because the coupled redistribution + font-fallback loop below
    # re-balances columns at the actually-rendered font size. Building it here
    # would freeze the grid at `_bal_size_px`, which can differ from the final
    # cell font — leaving a column word-overflowing at render time even though it
    # fit at the balance size.

    borders_xml = (
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '</w:tblBorders>'
    )
    # Pin the cell margins to EXACTLY what the fitter models (`_usable_w_pt`
    # subtracts `_CELL_MARGIN_PT` per side; top/bottom = 0). Emitting them
    # explicitly stops Word from applying its own default (108 twips L/R) that
    # the fitter would otherwise have to guess at.
    _mar_tw = int(round(_CELL_MARGIN_PT * 20))   # 5.4 pt → 108 twips
    cell_mar_xml = (
        '<w:tblCellMar>'
        '<w:top w:w="0" w:type="dxa"/>'
        f'<w:left w:w="{_mar_tw}" w:type="dxa"/>'
        '<w:bottom w:w="0" w:type="dxa"/>'
        f'<w:right w:w="{_mar_tw}" w:type="dxa"/>'
        '</w:tblCellMar>'
    )
    # `tblW` is filled in at ASSEMBLY time (below, beside the tblGrid) from the
    # FINAL twips column widths. `col_w_emus` is rebalanced several times after
    # this point, so freezing the table width here would disagree with the grid
    # — and under `tblLayout="fixed"` Word trusts the grid and silently
    # re-derives the width from it.
    tbl_pr_tmpl = (
        '<w:tblPr>'
        '<w:tblW w:w="{tbl_w_tw}" w:type="dxa"/>'
        '<w:tblLayout w:type="fixed"/>'
        f'{borders_xml}'
        f'{cell_mar_xml}'
        '</w:tblPr>'
    )

    # Per-cell font cap based on the narrowest column to keep cells readable.
    narrowest_col_pt = min(col_w_emus) / EMU_PER_PT
    size_ceiling_pt = min(
        max(declared_size_pt, _TABLE_TARGET_FONT_PT),
        max(_TABLE_TARGET_FONT_PT, narrowest_col_pt / 5.0),
        12.0,
    )
    cell_size_pt = size_ceiling_pt
    cell_size_px = max(1, int(round(cell_size_pt)))
    meas_font = get_font(cell_size_px)
    meas_font_bold = get_font(cell_size_px, bold=True) or meas_font
    natural_h_px = (
        (sum(meas_font.getmetrics()) if meas_font else cell_size_px * 1.2)
    )

    def _line_count_in_col(cell: str, col_idx: int, cs: int = 1,
                           bold: bool = False) -> int:
        # Measure with the SAME weight the cell renders at — header / row-0
        # cells render bold, and bold glyphs are ~12-15% wider, so measuring
        # non-bold would under-count lines and leave the row too short (clip).
        font = meas_font_bold if bold else meas_font
        if not cell or font is None:
            return 1
        cell_w_emu = sum(col_w_emus[col_idx:col_idx + max(1, cs)])
        cell_w_pt = cell_w_emu / EMU_PER_PT
        wrapped = wrap_to_width(
            cell, font, _usable_w_pt(cell_w_pt), cell_size_px,
        )
        return max(1, len(wrapped))

    row_lines_arr = [1] * n_rows
    for (row_idx, col_idx), (cell_text, cs, _rs, is_header) in cell_anchors.items():
        row_lines_arr[row_idx] = max(
            row_lines_arr[row_idx],
            _line_count_in_col(
                cell_text, col_idx, cs, bold=bool(is_header or row_idx == 0),
            ),
        )

    bbox_h_pt = h / EMU_PER_PT

    # ── Container-bbox guard (part 2 of 2: decide, then apply) ──────────────
    # Now that the rows have been wrapped, ask the question that actually
    # separates a container rectangle from a table's own: how many rows of text
    # would fit in this box, versus how many it holds?
    #
    # This is the HEIGHT test, not a width one. `bbox_px_to_emu` has already
    # clamped the bbox to the page, so a runaway box arrives no wider than one
    # page and its width ratio is unremarkable — measured at 1.2-2.0x for the
    # reported regression, BELOW ordinary healthy tables. Height carries no such
    # ceiling, and the same regression measures 12-22x on this test.
    if _guard_eligible and _natural_w > 0:
        _line_h = natural_h_px * 1.06
        if bbox_is_container(bbox_h_pt, sum(row_lines_arr), _line_h):
            _w_before = w
            x, w, _bbox_clamped = clamp_to_natural_width(x, w, _natural_w)
            if _bbox_clamped:
                # `col_w_emus` was split out of the OLD `w`; rescale it to the
                # new budget so the downstream floors, balancers and the drift
                # fixup all keep working against a consistent total.
                _scale = w / _w_before
                col_w_emus = [max(1, int(round(c * _scale)))
                              for c in col_w_emus]
                _drift = sum(col_w_emus) - w
                if _drift != 0:
                    _widest = max(range(max_cols), key=lambda i: col_w_emus[i])
                    col_w_emus[_widest] = max(1, col_w_emus[_widest] - _drift)
                # The height is no more the table's own than the width was, and
                # `_normalise_to` treats `bbox_h_pt` as a TARGET — it inflates
                # rows to fill it ("no gap under the last row"). Left alone the
                # table would still be stretched down the container. `min` keeps
                # this a one-way valve: a bbox TIGHTER than content still wins,
                # and the fitting loop handles that case as it always has.
                _natural_h_pt = max(1e-6, sum(row_lines_arr) * _line_h)
                _h_ratio = bbox_h_pt / _natural_h_pt
                if _natural_h_pt < bbox_h_pt:
                    bbox_h_pt = _natural_h_pt
                log.warning(
                    "[table] bbox is %.1fx taller than its %d wrapped lines "
                    "need — treating it as a container: narrowing %.0fpt to "
                    "%.0fpt, centring, and sizing height to content (%.0fpt)",
                    _h_ratio, sum(row_lines_arr), _w_before / EMU_PER_PT,
                    w / EMU_PER_PT, bbox_h_pt,
                )

    weight_sum = sum(row_lines_arr) or n_rows or 1
    row_h_pts = [
        (bbox_h_pt * rl / weight_sum) for rl in row_lines_arr
    ]
    # Floor each row by ITS OWN line count as an initial guess. The coupled
    # redistribution + font fallback below then reshapes these so no cell clips
    # while the row total is pinned to the source bbox height (rows emit with
    # hRule="exact", so their final heights must actually hold their content).
    line_h_pt = natural_h_px * 1.06
    row_h_pts = [
        max(rh, line_h_pt * rl) for rh, rl in zip(row_h_pts, row_lines_arr)
    ]

    # Reserve vertical room for pictures. A picture cell (often a rowspan block
    # holding one or more stacked illustrations) needs its merged height to be
    # at least the SUM of its pictures' fitted heights, else the images spill
    # past the cell border (rows use hRule="exact", a hard clip). We compute
    # each picture's height when scaled to the cell's column width, add a small
    # per-picture pad, and if the anchor's rowspan rows don't already provide
    # that much height, grow those rows evenly to cover the deficit. Text-only
    # cells and picture-free tables are untouched.
    #
    # Per-row minimum kept for the bbox re-fit below: a row must never shrink
    # below its own text line count, otherwise text clips.
    row_min_pts = [line_h_pt * rl for rl in row_lines_arr]
    # Height (EMU) each picture cell reserves for its images, so the per-cell
    # text fit below can subtract it and fit the label into the space above.
    _pic_reserve_emu_by_cell: Dict[Tuple[int, int], int] = {}
    if pic_assignments:
        pic_pad_pt = 3.0
        for (a_r, a_c), pics in pic_assignments.items():
            anchor = cell_anchors.get((a_r, a_c))
            if not anchor:
                continue
            _t, a_cs, a_rs, _h = anchor
            col_w_pt = sum(
                col_w_emus[a_c:a_c + max(1, a_cs)]
            ) / EMU_PER_PT
            avail_w_pt = max(1.0, col_w_pt - 4.0)   # ~2pt padding each side
            need_pt = 0.0
            for pic in pics:
                pb = pic.get("bbox") or []
                if len(pb) != 4 or pb[2] <= pb[0] or pb[3] <= pb[1]:
                    continue
                iw_pt = (pb[2] - pb[0]) / ctx.zoom
                ih_pt = (pb[3] - pb[1]) / ctx.zoom
                # height once the picture is scaled down to fit the column
                # width (never upscaled)
                scale = min(1.0, avail_w_pt / max(1.0, iw_pt))
                need_pt += ih_pt * scale + pic_pad_pt
            _pic_reserve_emu_by_cell[(a_r, a_c)] = int(
                round(need_pt * EMU_PER_PT)
            )
            span = list(range(a_r, min(n_rows, a_r + max(1, a_rs))))
            have_pt = sum(row_h_pts[r] for r in span)
            deficit = need_pt - have_pt
            if deficit > 0:
                add = deficit / len(span)
                for r in span:
                    row_h_pts[r] += add
                    # Deliberately NOT raising `row_min_pts` here. `_normalise_to`
                    # treats those floors as HARD, so baking a grown height into
                    # them made the sum un-reclaimable and the function returned
                    # silently OVER the bbox — the table then rendered taller than
                    # its own rectangle with no clamp and no log. The reserve is
                    # still honoured as a preference (rows start taller and
                    # `_balance_row_heights` protects them); it is simply no
                    # longer allowed to overrule the bbox.

    # ── No-clip fitting: keep the table EXACTLY its source size, redistribute
    # internal column widths + row heights (whole column / whole row, totals
    # fixed) from slack cells to crushed cells, and only if that is exhausted
    # shrink the font uniformly. The table box never grows past its bbox (that
    # would overlap the content below) and no cell is ever clipped. ──────────────

    # Normalise rows to sum EXACTLY to the bbox height. Picture rows keep their
    # reserved minimum; the extra/deficit is spread over the other rows' slack.
    reserve_pt_by_cell: Dict[Tuple[int, int], float] = {
        k: v / EMU_PER_PT for k, v in _pic_reserve_emu_by_cell.items()
    }

    # Nested sub-tables reserve height in the SAME channel as pictures: the
    # fitting machinery (`_cell_overflows`, `_balance_row_heights`,
    # `_min_rows_for`, the grow-backstop) already subtracts/honours it, so
    # sub-tables need no new sizing concept.
    #
    # `_cell_blocks_xml` is deliberately NOT called inside the 24-iteration
    # coupled loop below: it would re-lay out every sub-table ~24x per table,
    # and a height function that shifts under the loop breaks its
    # strict-improvement convergence. We measure once here (at the current
    # widths) and once more after the loop converges, taking the larger — which
    # errs toward over-reserving and therefore never clips.
    def _measure_sub_reserves(size_pt: float) -> Dict[Tuple[int, int], float]:
        out: Dict[Tuple[int, int], float] = {}
        for (a_r, a_c), blocks in blocks_by_anchor.items():
            anc = cell_anchors.get((a_r, a_c))
            if anc is None:
                continue
            _t, a_cs, _rs, _h = anc
            cw_emu = sum(col_w_emus[a_c:a_c + max(1, a_cs)])
            _xml, need_pt = _cell_blocks_xml(blocks, cw_emu, style, size_pt)
            if need_pt > 0:
                out[(a_r, a_c)] = need_pt
        return out

    def _apply_sub_reserves(subs: Dict[Tuple[int, int], float]) -> None:
        """Fold sub-table demands into `reserve_pt_by_cell` and grow the
        spanned rows to cover any deficit (mirrors the picture-reserve pass)."""
        for (a_r, a_c), need_pt in subs.items():
            anc = cell_anchors.get((a_r, a_c))
            if anc is None:
                continue
            _t, _cs, a_rs, _h = anc
            reserve_pt_by_cell[(a_r, a_c)] = max(
                reserve_pt_by_cell.get((a_r, a_c), 0.0), need_pt)
            span = list(range(a_r, min(n_rows, a_r + max(1, a_rs))))
            if not span:
                continue
            have = sum(row_h_pts[r] for r in span)
            deficit = need_pt - have
            if deficit > 0:
                add = deficit / len(span)
                for r in span:
                    row_h_pts[r] += add
                    # Deliberately NOT raising `row_min_pts` here. `_normalise_to`
                    # treats those floors as HARD, so baking a grown height into
                    # them made the sum un-reclaimable and the function returned
                    # silently OVER the bbox — the table then rendered taller than
                    # its own rectangle with no clamp and no log. The reserve is
                    # still honoured as a preference (rows start taller and
                    # `_balance_row_heights` protects them); it is simply no
                    # longer allowed to overrule the bbox.

    if blocks_by_anchor:
        _apply_sub_reserves(_measure_sub_reserves(cell_size_pt))

    def _normalise_to(target_pt: float) -> None:
        """Scale row_h_pts so they sum to target_pt, honouring row_min_pts as
        hard floors and putting any residual on the tallest slack row."""
        nonlocal row_h_pts
        cur = sum(row_h_pts)
        if cur <= 1e-6:
            return
        if abs(cur - target_pt) < 0.5:
            return
        if cur > target_pt:
            # Over budget: reclaim from rows with slack above their minimum.
            excess = cur - target_pt
            for _ in range(12):
                slack = [rh - mn for rh, mn in zip(row_h_pts, row_min_pts)]
                slack_total = sum(s for s in slack if s > 0)
                if slack_total <= 1e-6 or excess <= 0.5:
                    break
                new_h = []
                for rh, s in zip(row_h_pts, slack):
                    if s <= 0:
                        new_h.append(rh)
                    else:
                        new_h.append(rh - min(s, excess * s / slack_total))
                excess = sum(new_h) - target_pt
                row_h_pts = new_h
        else:
            # Under budget: hand the spare height to every row proportionally so
            # the table fills its bbox exactly (no gap under the last row).
            add = target_pt - cur
            base = sum(row_h_pts) or n_rows
            row_h_pts = [rh + add * (rh / base) for rh in row_h_pts]

    _normalise_to(bbox_h_pt)

    # Coupled redistribution: alternate column-width and row-height balancing,
    # re-measuring overflow, until nothing overflows or no improving move remains.
    fit_size_px = cell_size_px
    for _ in range(24):
        if not _any_cell_overflows(
            cell_anchors, col_w_emus, row_h_pts, max_cols, n_rows,
            fit_size_px, reserve_pt_by_cell,
        ):
            break
        # (a) whole-column: give width to the most line-pressured column.
        _bf = get_font(fit_size_px)
        _bfb = get_font(fit_size_px, bold=True) or _bf
        new_cols = _balance_col_widths(
            col_w_emus, min_col_emus, cell_anchors, max_cols, w,
            _bf, _bfb, fit_size_px,
        )
        # (b) whole-row: give height to the most clipped row.
        new_rows = _balance_row_heights(
            row_h_pts, row_min_pts, cell_anchors, new_cols,
            max_cols, n_rows, fit_size_px, reserve_pt_by_cell,
        )
        if new_cols == col_w_emus and all(
            abs(a - b) < 1e-6 for a, b in zip(new_rows, row_h_pts)
        ):
            break  # converged — redistribution can't help further
        col_w_emus = new_cols
        row_h_pts = new_rows

    # Font fallback: if redistribution alone can't fit every cell WITHIN THE
    # FIXED bbox height, shrink the whole-table font in 0.5pt steps (down to a
    # hard floor) until nothing overflows. Each step ALSO re-balances columns at
    # the current font (so the grid we emit is verified at the size it renders —
    # a word that fit at the balance size can overflow at a larger render size),
    # recomputes the tighter per-row minimums, forces the rows back to sum
    # EXACTLY to the bbox height (never grow), re-balances row heights, and
    # re-measures. Uniform font keeps the table visually consistent;
    # smaller-but-fully-visible always beats clipped.
    def _min_rows_for(size_px: int) -> List[float]:
        lh = _line_h_pt(size_px, False, False)
        return [
            max(lh * 1, reserve_pt_by_cell.get((r, 0), 0.0))
            for r in range(n_rows)
        ]

    # LAST RESORT ONLY. A table's layout is normally FIXED: it renders at exactly
    # its source bbox and does not extend downward. Unconditional growth was the
    # old "never clip" contract, and it ran tables over the layout beneath them
    # and off the page — measured on a real page, a 349pt table rendered 665pt
    # tall with its bottom at 752pt on a 595pt page.
    #
    # It is now reachable again, but only after the cheaper remedy is exhausted:
    # the loop below first shrinks from `TABLE_MIN_FONT_PT` down to
    # `TABLE_SOFT_MIN_FONT_PT` inside the fixed bbox, and only sets this flag if
    # the content STILL overflows at that size. Growth is then bounded by the
    # page edge and the nearest entry below (the obstacle clamp further down), so
    # it can no longer run off the page the way the old contract did.
    #
    # Pictures are scaled by `min(1.0, scale_w, scale_h)` at emission against the
    # FINAL row heights, so an oversized picture renders smaller rather than
    # forcing the row taller.
    _grow_to_fit = False
    # Height budget the fitting loop normalises rows to. Starts at the source
    # bbox; the loop may RAISE it (never lower it) when holding the table to its
    # source rectangle would force the font below `_HARD_MIN_FONT_PT`.
    #
    # Readability is now the binding constraint, not the rectangle. The bbox is
    # still preferred and still wins whenever the content fits inside it, so a
    # well-sized table is bit-for-bit unchanged; only tables that would otherwise
    # have rendered unreadably small are allowed to grow, and then only as far as
    # the room actually available below them (the obstacle/page clamp further
    # down still binds).
    _fit_budget_pt = bbox_h_pt
    while True:
        _spx = max(1, int(round(fit_size_px)))
        # Re-balance columns AT THE CURRENT FONT so the emitted grid is the one
        # this iteration verifies (fixes the ordering bug where a post-loop
        # re-balance could invalidate the fit).
        _ff = get_font(_spx)
        _ffb = get_font(_spx, bold=True) or _ff
        col_w_emus = _balance_col_widths(
            col_w_emus, min_col_emus, cell_anchors, max_cols, w,
            _ff, _ffb, _spx,
        )
        row_min_pts = _min_rows_for(_spx)
        _normalise_to(_fit_budget_pt)
        row_h_pts = _balance_row_heights(
            row_h_pts, row_min_pts, cell_anchors, col_w_emus,
            max_cols, n_rows, _spx, reserve_pt_by_cell,
        )
        if not _any_cell_overflows(
            cell_anchors, col_w_emus, row_h_pts, max_cols, n_rows,
            _spx, reserve_pt_by_cell,
        ):
            break
        if fit_size_px <= _HARD_MIN_FONT_PT + 1e-6:
            # Shrinking is exhausted: we are at `TABLE_SOFT_MIN_FONT_PT` and the
            # content still overflows the source bbox. Rather than shrinking
            # further (illegible) or clipping (content lost), GROW the height
            # budget to exactly what the rows need at this size and re-fit once.
            _needed_pt = sum(row_min_pts)
            if _needed_pt > _fit_budget_pt + 0.5:
                log.info(
                    "[table] content needs %.0fpt at the %.1fpt readable floor "
                    "but the bbox is %.0fpt — growing the table to keep the "
                    "text legible",
                    _needed_pt, _HARD_MIN_FONT_PT, _fit_budget_pt,
                )
                _fit_budget_pt = _needed_pt
                _grow_to_fit = True
                _normalise_to(_fit_budget_pt)
                row_h_pts = _balance_row_heights(
                    row_h_pts, row_min_pts, cell_anchors, col_w_emus,
                    max_cols, n_rows, _spx, reserve_pt_by_cell,
                )
            break
        fit_size_px = max(_HARD_MIN_FONT_PT, fit_size_px - 0.5)

    # Lock the per-cell emission font to the fitted size and keep totals exact.
    cell_size_pt = float(fit_size_px)
    cell_size_px = max(1, int(round(fit_size_px)))
    row_min_pts = _min_rows_for(cell_size_px)

    # Re-measure sub-tables at the FINAL font and column widths (both may have
    # moved during the fitting loop) and re-apply. `_apply_sub_reserves` takes
    # the max against what is already reserved, so this can only grow — the
    # emitted sub-table can never need more room than we reserved for it.
    if blocks_by_anchor:
        _apply_sub_reserves(_measure_sub_reserves(cell_size_pt))
        # A sub-table demanding more than the bbox does NOT grow the table; the
        # sub-table is shrunk to its cell instead (see `render_nested_table_xml`,
        # whose target is two-way).

    if not _grow_to_fit:
        # `_fit_budget_pt` is the bbox height unless the loop above had to grow
        # it to keep the font at the readable floor; normalising back to the raw
        # bbox here would silently undo that growth and re-clip the text.
        _normalise_to(_fit_budget_pt)
    else:
        # Grow-backstop: give every row at least the height its tallest cell
        # needs at the floor font, so NO cell clips. Rows below emit with
        # hRule="atLeast" and the outer box takes the grown height. Rowspan
        # deficits are spread across the spanned rows (mirrors the picture
        # reserve pass above).
        for r in range(n_rows):
            need_here = row_min_pts[r]
            for c in range(max_cols):
                anc = cell_anchors.get((r, c))
                if anc is None:
                    continue
                txt, cs, rs, is_h = anc
                if rs != 1 or not (txt or "").strip():
                    continue
                cw_pt = sum(col_w_emus[c:c + max(1, cs)]) / EMU_PER_PT
                bold = bool(is_h or r == 0)
                n = _needed_lines(txt, cw_pt, cell_size_px, bold)
                nh = n * _line_h_pt(cell_size_px, bold, has_cjk(txt))
                nh += reserve_pt_by_cell.get((r, c), 0.0)
                need_here = max(need_here, nh)
            row_h_pts[r] = max(row_h_pts[r], need_here)
        # Rowspan cells: ensure their spanned rows collectively hold the content.
        for (r, c), (txt, cs, rs, is_h) in cell_anchors.items():
            if rs <= 1 or not (txt or "").strip():
                continue
            cw_pt = sum(col_w_emus[c:c + max(1, cs)]) / EMU_PER_PT
            bold = bool(is_h or r == 0)
            n = _needed_lines(txt, cw_pt, cell_size_px, bold)
            need = n * _line_h_pt(cell_size_px, bold, has_cjk(txt))
            need += reserve_pt_by_cell.get((r, c), 0.0)
            span = list(range(r, min(n_rows, r + rs)))
            have = sum(row_h_pts[rr] for rr in span)
            if need > have and span:
                add = (need - have) / len(span)
                for rr in span:
                    row_h_pts[rr] += add

    # Picture placement reuses the cell assignment already computed above
    # (the same one that fed into col_weight). Recomputing here from final
    # `col_w_emus` would diverge from the placement we promised the weight
    # boost was based on, and the assignment doesn't need final widths
    # because it scores by emptiness + horizontal proximity, not strict
    # column boundaries.
    pic_inline_by_cell: Dict[Tuple[int, int], List[Dict]] = pic_assignments

    # ── FIXED-HEIGHT BACKSTOP ────────────────────────────────────────────────
    # Unconditional, and deliberately the LAST word on height. Several passes
    # above raise rows to cover a demand (pictures, sub-tables, per-row floors),
    # and `_normalise_to` cannot always reclaim that — when the floors alone sum
    # above the target it returns silently over budget. A table must be exactly
    # its source rectangle, so rescale here rather than trusting every upstream
    # path to have behaved.
    #
    # Rescaling proportionally (rather than trimming the last row) keeps the row
    # rhythm the fitter chose. Pictures re-scale against these final heights at
    # emission, so they shrink with the rows instead of overflowing them.
    # The target is `_fit_budget_pt`, not the raw bbox: when the loop above grew
    # the budget to hold the text at the readable floor, compressing back to the
    # bbox here would re-create exactly the clipping that growth prevented.
    _rows_sum = sum(row_h_pts)
    if _rows_sum > _fit_budget_pt + 0.5 and _rows_sum > 1e-6:
        _shrink = _fit_budget_pt / _rows_sum
        row_h_pts = [rh * _shrink for rh in row_h_pts]
        log.warning(
            "[table] content demanded %.0fpt in a %.0fpt budget — compressing "
            "rows to keep the table at its intended size",
            _rows_sum, _fit_budget_pt,
        )

    # `col_w_emus` is FINAL here (every `_balance_col_widths` call and the whole
    # coupled redistribution + font-fallback loop are behind us). Convert once
    # to twips for emission; every emitted width below sums THESE values rather
    # than converting an EMU sum, so a colspan/vMerge cell always equals exactly
    # the columns it covers and `sum(gridCol) == tblW` holds.
    col_w_tw = _emu_tw_split(col_w_emus)

    # PAGE CONTAINMENT. `_grow_to_fit` emits rows as `hRule="atLeast"`, which
    # lets Word grow a row past the height we declared — a second, independent
    # way for a table to run off the page bottom and continue onto the next one.
    # The user needs the source page preserved exactly, so cap the grown total
    # at the room actually left below the table's top edge. If the cap binds we
    # go back to `hRule="exact"`: losing a few points off one cell is preferable
    # to relocating rows to another page.
    #
    # The page edge is only HALF the bound: growing to it runs the table over
    # whatever sits below (the page number is the visible symptom). So the cap is
    # the nearer of the page edge and the closest entry underneath us — the same
    # obstacle rule applied to the width axis above.
    _page_h_pt = float(getattr(ctx, "page_h_pt", 0.0) or 0.0)
    if _grow_to_fit and _page_h_pt > 0:
        _avail_pt = max(1.0, _page_h_pt - (y / EMU_PER_PT))
        if not is_quarter_turn(table_rotation):
            _bx1, _by1, _bx2, _by2 = [float(v) for v in bbox]
            _room_h_pt = downward_room_pt(
                getattr(ctx, "page_entries", None) or (), entry,
                _bx1, _bx2, _by1, ctx.zoom,
            )
            # None = nothing below; the page edge stays the only bound.
            if _room_h_pt is not None:
                _avail_pt = min(_avail_pt, max(1.0, _room_h_pt))
        if sum(row_h_pts) > _avail_pt:
            _scale = _avail_pt / sum(row_h_pts)
            row_h_pts = [rh * _scale for rh in row_h_pts]
            _grow_to_fit = False        # back to exact: never spill off-page
            log.warning(
                "[table] content needs %.0fpt but only %.0fpt remain on the "
                "page — compressing rows to keep the table on its own page",
                sum(row_h_pts) / max(1e-6, _scale), _avail_pt,
            )

    # Cell + row emission.
    rows_xml_parts: List[str] = []
    for row_idx in range(n_rows):
        cells_xml: List[str] = []
        skip_cols = 0   # columns consumed by a preceding cell's gridSpan
        for col_idx in range(max_cols):
            if skip_cols > 0:
                skip_cols -= 1
                continue
            anchor = cell_anchors.get((row_idx, col_idx))
            if anchor is not None:
                cell_text, cs, rs, is_header = anchor
                cell_w_emu = sum(col_w_emus[col_idx:col_idx + cs])
                pics_here = pic_inline_by_cell.get((row_idx, col_idx)) or []
                pic_paragraphs = []
                cell_h_emu = int(round(row_h_pts[row_idx] * EMU_PER_PT))
                if rs > 1:
                    cell_h_emu = sum(
                        int(round(row_h_pts[row_idx + dr] * EMU_PER_PT))
                        for dr in range(rs)
                    )

                # When this cell also holds pictures, the text is fit into the
                # space left ABOVE them (subtract the reserved picture height)
                # so the label and the image don't collide / clip each other.
                cell_w_pt_for_fit = cell_w_emu / EMU_PER_PT
                # The coupled redistribution + font fallback above already sized
                # the columns, rows and `cell_size_px` so that EVERY cell's full
                # text fits its cell at that size — so we render at exactly the
                # fitted size and wrap the FULL text (never an ellipsis, never a
                # smaller-than-fitted 6pt path that would clip). This is what
                # guarantees no clipped / hidden / scrollable cells.
                cell_style = dict(style)
                cell_style["size"] = cell_size_pt
                if cell_text:
                    _wf = get_font(
                        cell_size_px, bold=bool(is_header or row_idx == 0),
                    )
                    if _wf is not None:
                        rendered_cell_text = "\n".join(wrap_to_width(
                            cell_text, _wf,
                            _usable_w_pt(cell_w_pt_for_fit), cell_size_px,
                        ))
                    else:
                        rendered_cell_text = cell_text
                else:
                    rendered_cell_text = cell_text
                if is_header or row_idx == 0:
                    cell_style["bold"] = True
                run_xml = build_run_xml(rendered_cell_text, cell_style)
                # Tables are universally centered in the source docs we see
                # (engineering reports, risk matrices, parts lists). Center
                # horizontally + vertically so cells read consistently and a
                # short value in a wide column isn't pinned to the left edge.
                #
                # Pin an EXACT per-line height equal to the value the fitter
                # measured (`_line_h_pt`) so Word can't fall back to its ~1.15x
                # default single spacing, which is taller than we reserved and
                # would clip multi-line cells under `hRule="exact"`. This makes
                # measurement == render for height.
                _cell_bold = bool(is_header or row_idx == 0)
                _line_pt = (
                    _line_h_pt(
                        cell_size_px, _cell_bold, has_cjk(rendered_cell_text),
                    )
                    if rendered_cell_text.strip() else None
                )
                para_xml = build_paragraph_xml(
                    run_xml, alignment="center", line_pt=_line_pt,
                )
                # One text line's worth of reserved height above any picture(s)
                # in the cell — use the SAME per-line height the paragraph is
                # pinned to (above) so the reserve matches the render exactly.
                text_reserve_emu = (
                    int(round(
                        _line_h_pt(cell_size_px, _cell_bold, has_cjk(cell_text))
                        * EMU_PER_PT
                    ))
                    if cell_text else 0
                )
                pic_budget_h_emu = max(1, cell_h_emu - text_reserve_emu)
                pad_emu = 2 * EMU_PER_PT
                max_pic_w_emu = max(1, cell_w_emu - pad_emu)
                n_pics = sum(1 for p in pics_here if p.get("image_obj"))
                per_pic_h_emu = (
                    max(1, pic_budget_h_emu // n_pics) if n_pics else 1
                )
                for pic in pics_here:
                    img_obj = pic.get("image_obj")
                    if img_obj is None:
                        continue
                    img_obj = apply_pixel_rotation(img_obj, pic)
                    buf = BytesIO()
                    img_obj.save(buf, format="PNG")
                    buf.seek(0)
                    rel_id = add_image_relationship(ctx.doc, buf, "png")
                    iw, ih = img_obj.size
                    if iw <= 0 or ih <= 0:
                        continue
                    # Intrinsic size in the SAME point space as the cell: the
                    # crop pixels are rendered at ctx.zoom (a 2x page raster is
                    # typical), so divide by zoom to get the picture's true
                    # physical size — exactly what render_standalone_picture
                    # does via bbox_px_to_emu. Prefer the picture's own bbox
                    # when present (already the source-accurate extent); fall
                    # back to the crop pixel dims / zoom. Treating the raw crop
                    # pixels as points made every table image ~zoom-times too
                    # big, so it filled the whole column width and overflowed
                    # the cell height regardless of the fit clamp below.
                    zoom = max(1e-6, float(getattr(ctx, "zoom", 1.0) or 1.0))
                    pb = pic.get("bbox") or []
                    if len(pb) == 4 and pb[2] > pb[0] and pb[3] > pb[1]:
                        intrinsic_w_pt = (pb[2] - pb[0]) / zoom
                        intrinsic_h_pt = (pb[3] - pb[1]) / zoom
                    else:
                        intrinsic_w_pt = iw / zoom
                        intrinsic_h_pt = ih / zoom
                    intrinsic_w_emu = max(1.0, intrinsic_w_pt * EMU_PER_PT)
                    intrinsic_h_emu = max(1.0, intrinsic_h_pt * EMU_PER_PT)
                    # Fit inside the cell: never wider than the padded column,
                    # never taller than this picture's share of the cell height,
                    # and never upscale past the source size.
                    scale_w = max_pic_w_emu / intrinsic_w_emu
                    scale_h = per_pic_h_emu / intrinsic_h_emu
                    scale = min(1.0, scale_w, scale_h)
                    pic_w_emu = max(1, int(round(intrinsic_w_emu * scale)))
                    pic_h_emu = max(1, int(round(intrinsic_h_emu * scale)))
                    inline_pic_xml = build_inline_picture_xml(
                        pic_w_emu, pic_h_emu, rel_id, ctx._next_id(),
                        pic_name=pic.get("id") or "Picture",
                    )
                    pic_paragraphs.append(
                        f'<w:p xmlns:w="{NS_W}">'
                        f'<w:pPr><w:spacing w:before="0" w:after="0"/>'
                        f'<w:jc w:val="center"/></w:pPr>'
                        f'{inline_pic_xml}</w:p>'
                    )
                # A cell carrying nested sub-tables renders its ordered blocks
                # (caption / table / caption / table …) as real nested
                # <w:tbl>s instead of the flattened single paragraph. Falls
                # back to `para_xml` if the blocks produce nothing.
                _blocks = blocks_by_anchor.get((row_idx, col_idx))
                _blocks_xml = ""
                if _blocks:
                    # FILL the cell. `row_h_pts` is final by here (the fitting
                    # loop and every `_normalise_to` have run), so this is the
                    # true height the sub-table has to cover. Without a target
                    # it renders at its natural size, pins to the top
                    # (`vAlign=top`) and leaves the parent's inflation as dead
                    # space underneath.
                    _cell_target_pt = sum(
                        row_h_pts[row_idx + _dr]
                        for _dr in range(max(1, rs))
                        if row_idx + _dr < n_rows
                    )
                    _blocks_xml, _ = _cell_blocks_xml(
                        _blocks, cell_w_emu, cell_style, cell_size_pt,
                        target_h_pt=_cell_target_pt,
                    )

                tc_pr_parts = [
                    f'<w:tcW w:w="{sum(col_w_tw[col_idx:col_idx + cs])}"'
                    f' w:type="dxa"/>'
                ]
                if cs > 1:
                    tc_pr_parts.append(f'<w:gridSpan w:val="{cs}"/>')
                if rs > 1:
                    tc_pr_parts.append('<w:vMerge w:val="restart"/>')
                # Rotated cell content (e.g. turned headers over narrow
                # columns). MUST sit after gridSpan/vMerge and before vAlign —
                # w:tcPr is a sequence type, and the wrong order still opens in
                # LibreOffice but is rejected by Word.
                _td = _cell_text_direction(cell_rotations, row_idx, col_idx)
                if _td:
                    tc_pr_parts.append(f'<w:textDirection w:val="{_td}"/>')
                # A stack of sub-tables reads wrong centred vertically, and
                # top-aligning keeps the reserve maths simple.
                tc_pr_parts.append(
                    '<w:vAlign w:val="top"/>' if _blocks_xml
                    else '<w:vAlign w:val="center"/>'
                )
                _body_xml = (
                    _blocks_xml if _blocks_xml
                    else f"{para_xml}{''.join(pic_paragraphs)}"
                )
                cells_xml.append(
                    "<w:tc>"
                    f'<w:tcPr>{"".join(tc_pr_parts)}</w:tcPr>'
                    f"{_body_xml}"
                    "</w:tc>"
                )
            elif occupied[row_idx][col_idx]:
                # Continuation slot: emit vMerge continuation if the cell
                # above is a vMerge anchor; otherwise it's a colspan
                # continuation covered by the previous cell's gridSpan.
                above_is_vmerge = False
                vmerge_cs = 1
                r = row_idx - 1
                while r >= 0:
                    if (r, col_idx) in cell_anchors:
                        _t, _cs, _rs, _h = cell_anchors[(r, col_idx)]
                        if _rs > 1 and r + _rs > row_idx:
                            above_is_vmerge = True
                            vmerge_cs = max(1, _cs)
                        break
                    if not occupied[r][col_idx]:
                        break
                    r -= 1
                if above_is_vmerge:
                    # The vMerge anchor may ALSO span columns (rowspan+colspan,
                    # e.g. a corner header). The continuation cell must repeat
                    # the SAME gridSpan and consume the spanned continuation
                    # columns — otherwise this row covers fewer grid columns
                    # than the others, which makes Word/LibreOffice misalign
                    # the row and drop trailing cells further down the table.
                    cont_w_emu = sum(col_w_emus[col_idx:col_idx + vmerge_cs])
                    span_xml = (
                        f'<w:gridSpan w:val="{vmerge_cs}"/>'
                        if vmerge_cs > 1 else ""
                    )
                    empty_para = build_paragraph_xml("", alignment="center")
                    cells_xml.append(
                        "<w:tc>"
                        f'<w:tcPr>'
                        f'<w:tcW w:w="'
                        f'{sum(col_w_tw[col_idx:col_idx + vmerge_cs])}"'
                        f' w:type="dxa"/>'
                        f'{span_xml}'
                        '<w:vMerge w:val="continue"/>'
                        '<w:vAlign w:val="center"/></w:tcPr>'
                        f"{empty_para}"
                        "</w:tc>"
                    )
                    skip_cols = vmerge_cs - 1   # skip the columns we just spanned
                # else: skip — covered by previous cell's gridSpan
            else:
                # No content at this position (uncommon edge case).
                empty_para = build_paragraph_xml("", alignment="center")
                cells_xml.append(
                    "<w:tc>"
                    f'<w:tcPr><w:tcW w:w="{col_w_tw[col_idx]}" w:type="dxa"/>'
                    '<w:vAlign w:val="center"/></w:tcPr>'
                    f"{empty_para}"
                    "</w:tc>"
                )
        # Normally `exact` so a single cell with oversized content can't blow
        # past the row budget and push the table off the source bbox. In the
        # grow-backstop case (content can't fit even at the floor font) rows use
        # `atLeast` so Word keeps each row tall enough for its content instead of
        # clipping it — the row heights above were grown to hold the text and the
        # outer box takes the grown total, so nothing is lost.
        # Pictures placed in cells are pre-scaled to fit (see scale_h above).
        _hrule = "atLeast" if _grow_to_fit else "exact"
        row_h_twips = max(1, int(round(row_h_pts[row_idx] * 20)))
        # `cantSplit` FIRST (w:trPr is a sequence type): a row must never break
        # across a page boundary. Together with the page-height clamp above,
        # this is what keeps a one-page table wholly on its own page.
        tr_pr = (
            f'<w:trPr><w:cantSplit/>'
            f'<w:trHeight w:val="{row_h_twips}" w:hRule="{_hrule}"/></w:trPr>'
        )
        rows_xml_parts.append("<w:tr>" + tr_pr + "".join(cells_xml) + "</w:tr>")

    # Build the grid from the FINAL, fully-balanced column widths (see note at
    # the pre-emission balance call above).
    grid_xml = "<w:tblGrid>" + ("".join(
        f'<w:gridCol w:w="{cw}"/>' for cw in col_w_tw
    )) + "</w:tblGrid>"
    # tblW is derived from the SAME twips list as the grid, so the two can never
    # disagree (see the `tbl_pr_tmpl` note above).
    tbl_pr_xml = tbl_pr_tmpl.format(tbl_w_tw=sum(col_w_tw))
    tbl_xml = f"<w:tbl>{tbl_pr_xml}{grid_xml}{''.join(rows_xml_parts)}</w:tbl>"
    body_xml = tbl_xml + '<w:p><w:pPr><w:spacing w:before="0" w:after="0"/></w:pPr></w:p>'
    # Pin the outer text-box height to the sum of the (redistributed) row heights,
    # which the fitting loop kept EQUAL to the source bbox height. Always
    # `noAutofit`: the table must never grow past its source rectangle (that would
    # overlap the content below) and never shrink to clip the last rows. The
    # coupled redistribution + uniform font fallback above guarantee every cell's
    # full text already fits inside these fixed rows, so `hRule="exact"` clips
    # nothing.
    rows_total_emu = int(round(sum(row_h_pts) * EMU_PER_PT))
    # Safe for a rotated table too: `h` was the TRANSPOSED extent (from
    # rotated_shape_geometry) that the grid was laid out against, and the
    # fitting loop kept the row-height sum equal to it, so re-deriving `h` here
    # reproduces the same transposed value rather than reverting to the bbox.
    h = rows_total_emu
    ctx.xml_chunks.append(
        build_anchored_textbox_xml(
            x, y, w, h, body_xml, ctx._next_id(),
            body_auto_fit=False,
            # Case B: spins the whole container (grid included). No-op at 0.
            # `rotate_shape` is required — `bodyPr vert=` would only reflow the
            # text inside each cell and leave the grid itself upright.
            rotation=table_rotation,
            rotate_shape=True,
        )
    )

    # Render the non-table content (checkboxes, review paragraphs, signatures)
    # in the band reserved for it, directly beneath the table.
    if trailing_text and trailing_h_emu > 0:
        if is_quarter_turn(table_rotation):
            # "Beneath" is in the TABLE's frame, not the page's. The container is
            # spun about its centre, so its on-page footprint is the transposed
            # rectangle: the band below the table's last row lies to the LEFT of
            # that footprint at 90° (and to the RIGHT at 270°), running the full
            # footprint height. Keep the block's own text upright — it is prose,
            # and matching the table's turn here would need a second rotated
            # measurement pass for no reader benefit.
            cx = x + w // 2
            cy = y + h // 2
            foot_w, foot_h = h, w          # footprint = transposed extent
            foot_x = cx - foot_w // 2
            foot_y = cy - foot_h // 2
            if normalize_rotation(table_rotation) == 90.0:
                t_x = max(0, foot_x - trailing_h_emu)
            else:
                t_x = foot_x + foot_w
            _emit_trailing_text_block(
                ctx, trailing_text, t_x, foot_y, trailing_h_emu, foot_h,
                trailing_style,
            )
        else:
            _emit_trailing_text_block(
                ctx, trailing_text, x, y + h, w, trailing_h_emu, trailing_style,
            )