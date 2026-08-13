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
from .text_fit import (
    fit_multiline, get_font, is_cjk_char, wrap_to_width, has_cjk,
    measure_width_px,
)


class TableHTMLParser(HTMLParser):
    """Parses <table>/<tr>/<th>/<td> + colspan/rowspan.

    Each cell is captured as (text, colspan, rowspan) — defaults 1/1 when the
    attribute is missing or malformed. Caller can then build a proper OOXML
    gridSpan / vMerge layout to honour merged header rows seen in the source
    PDFs (very common in CJK technical reports).
    """
    def __init__(self):
        super().__init__()
        self.tables = []
        self.current_table = []
        self.current_row: List[Tuple[str, int, int, int]] = []
        self.current_cell = ""
        self.current_colspan = 1
        self.current_rowspan = 1
        self.current_imgs = 0
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.in_header = False

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

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
            self.current_table = []
        elif tag == "tr":
            self.in_row = True
            self.current_row = []
        elif tag in ("td", "th"):
            self.in_cell = True
            self.current_cell = ""
            self.current_colspan = self._attr_int(attrs, "colspan", 1)
            self.current_rowspan = self._attr_int(attrs, "rowspan", 1)
            self.current_imgs = 0
            if tag == "th":
                self.in_header = True
        elif tag == "br" and self.in_cell:
            self.current_cell += "\n"
        elif tag == "img" and self.in_cell:
            # <img> placeholders (e.g. table-cell diagrams Chandra emits as a
            # bare <img alt=...>) carry no visible text but authoritatively mark
            # WHICH cell holds a picture and HOW MANY. Count them per cell so the
            # recovered Image entries can be assigned to the right cell later.
            self.current_imgs += 1
        elif tag == "input" and self.in_cell:
            # OCR marks form checkboxes as <input type="checkbox" [checked]>.
            # Render the state as a Unicode box glyph so it survives (the parser
            # otherwise drops the tag and the checked/unchecked state is lost).
            attr = dict(attrs)
            # Strip a stray trailing slash the parser can leave on an unquoted
            # attr (e.g. `type=checkbox/` → `checkbox/`).
            itype = (attr.get("type") or "checkbox").lower().rstrip("/").strip()
            if itype in ("checkbox", "radio"):
                checked = "checked" in attr  # value is "" but key present
                self.current_cell += "☑" if checked else "☐"

    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table = False
            if self.current_table:
                self.tables.append(self.current_table)
        elif tag == "tr":
            self.in_row = False
            if self.current_row:
                self.current_table.append((self.current_row, self.in_header))
            self.in_header = False
        elif tag in ("td", "th"):
            self.in_cell = False
            self.current_row.append((
                self.current_cell.strip(),
                self.current_colspan,
                self.current_rowspan,
                self.current_imgs,
            ))
            self.current_cell = ""
            self.current_colspan = 1
            self.current_rowspan = 1
            self.current_imgs = 0

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell += data

    def finalize(self):
        """Flush any cell/row/table left open because the HTML was truncated.

        OCR/VLM output is frequently cut off mid-table (no closing ``</td>`` /
        ``</tr>`` / ``</table>``). Without this, an unterminated ``<table>`` is
        dropped entirely and the whole table silently falls back to raw-text
        rendering. We close the open cell, row and table in order so the
        content is still recovered as a table.
        """
        if self.in_cell:
            self.current_row.append((
                self.current_cell.strip(),
                self.current_colspan,
                self.current_rowspan,
                self.current_imgs,
            ))
            self.in_cell = False
            self.current_cell = ""
            self.current_imgs = 0
        if self.current_row:
            self.current_table.append((self.current_row, self.in_header))
            self.current_row = []
        if self.current_table and self.current_table not in self.tables:
            self.tables.append(self.current_table)
            self.current_table = []


# Sanity bounds. OCR/VLM output on unparseable pages (rotated CAD drawings,
# scans) sometimes hallucinates tables with hundreds of near-identical rows or
# dozens of repeated header columns. These caps keep such garbage bounded so it
# can't overflow the page or explode render time; they're far above any real
# table seen in these documents.
MAX_TABLE_ROWS = 200
MAX_TABLE_COLS = 40

log = logging.getLogger("translation_reconstruction.table")



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
# A single "Table" entry frequently carries MORE than the ``<table>`` grid: form
# checkboxes, review paragraphs, signatures, comments, etc. emitted as
# ``<p>``/``<br>``/``<input>``/``<u>`` after (or before) the table. Rendering
# only the grid silently drops all of it. These helpers extract that surrounding
# HTML and flatten it to plain text so it can be rendered below the table. The
# flattening is GENERAL — it is not tied to any particular document, language or
# checkbox wording.

_TABLE_BLOCK_RE = re.compile(r"<table\b.*?</table\s*>", re.IGNORECASE | re.DOTALL)


class _HTMLToTextParser(HTMLParser):
    """Flatten an arbitrary HTML fragment to plain text.

    Handles the constructs OCR emits around tables generically:
      * ``<input type=checkbox|radio [checked]>`` → ``☑`` / ``☐`` (state kept);
      * ``<br>`` and block boundaries (``</p>``, ``</div>``, ``<li>``, ``</tr>``)
        → newlines so the layout survives;
      * every other tag is dropped but its text is kept;
      * HTML entities are unescaped (``convert_charrefs=True``).
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
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag.lower() in self._BLOCK_TAGS:
            self._buf.append("\n")

    def handle_data(self, data):
        self._buf.append(data)

    def get_text(self) -> str:
        raw = "".join(self._buf)
        lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in raw.split("\n")]
        out: List[str] = []
        for ln in lines:
            if ln or (out and out[-1]):
                out.append(ln)
        return "\n".join(out).strip()


def html_fragment_to_text(fragment: str) -> str:
    """Plain-text rendering of a non-table HTML ``fragment`` (checkbox glyphs,
    line breaks preserved). Returns ``""`` when it has no visible text."""
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
    block(s), in source order, flattened via `html_fragment_to_text`. Empty for
    a bare table so the common case stays a pure table render."""
    if not text or "<table" not in text.lower():
        return ""
    outside = _TABLE_BLOCK_RE.sub("\n", text)
    return html_fragment_to_text(outside)


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
                content = []
                for c in cells:
                    t, _cs, rs, imgs = _cell_parts(c)
                    if (t or "").strip():
                        content.append((t, rs, imgs))
                # Re-span content cells across the free segments, one per
                # segment. Only when the counts line up exactly — otherwise we
                # can't unambiguously map cells to segments, so leave as-is.
                if content and len(content) == len(free_segs):
                    rebuilt = []
                    for (t, rs, imgs), si in zip(content, free_segs):
                        b0, b1 = bounds[si], bounds[si + 1]
                        rebuilt.append((t, max(1, b1 - b0), rs, imgs))
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


def parse_table_grid(
    rows: List[Tuple[List[Tuple[str, int, int, int]], bool]],
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

    # Repair header/body column-granularity mismatch from OCR under-segmentation
    # before placing cells, so the correction propagates to rowspan placement.
    rows = _normalize_column_granularity(rows)

    n_rows = len(rows)

    def _row_width(row_cells):
        return sum(max(1, _cell_parts(c)[1]) for c in row_cells)

    widths = [_row_width(r[0]) for r in rows]
    raw_max = min(max(widths, default=1) or 1, MAX_TABLE_COLS)

    def _place(target_cols: int):
        """Place cells into a target-width grid; report last non-empty col."""
        anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]] = {}
        img_map: Dict[Tuple[int, int], int] = {}
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
                for rr in range(row_idx, row_idx + rs):
                    for cc in range(col, col + cs):
                        occ[rr][cc] = True
                if (cell_text or "").strip():
                    last_nonempty = max(last_nonempty, col + cs)
                col += cs
        return anchors, occ, last_nonempty, img_map

    # First pass at the widest observed width to learn where real content ends.
    _a0, _o0, content_extent, _i0 = _place(raw_max)
    # Modal width, but never below the real-content extent, never above the cap.
    modal = Counter(widths).most_common(1)[0][0] if widths else 1
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


    cell_anchors, occupied, _, img_by_anchor = _place(max_cols)
    _merge_duplicate_rowspan_labels(cell_anchors, occupied, max_cols, n_rows)
    return max_cols, n_rows, cell_anchors, occupied, img_by_anchor


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
    left-to-right by its (pre-reflow) bbox centroid."""
    pb = pic.get("_orig_bbox") or pic.get("bbox") or []
    if len(pb) != 4:
        return (0.0, 0.0)
    return ((pb[1] + pb[3]) / 2.0, (pb[0] + pb[2]) / 2.0)


def _assign_pictures_by_img_structure(
    pictures: List[Dict],
    bbox: List[int],
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    img_by_anchor: Dict[Tuple[int, int], int],
    n_rows: int,
) -> Optional[Dict[Tuple[int, int], List[Dict]]]:
    """Assign recovered pictures to the `<img>` cells parsed from the table HTML,
    driven by each picture's LAYOUT BBOX Y-POSITION (not by a count match).

    `img_by_anchor` maps a cell anchor to how many `<img>` placeholders that cell
    held, which authoritatively marks WHICH cells hold pictures. Each image cell
    owns a vertical band = the fraction of the table height its row-span covers
    (a rowspan-3 image cell is a 3-row-tall band, correctly taller than a 1-row
    cell). Every recovered picture is dropped into the image cell whose band
    contains its (pre-reflow) bbox y-centroid (nearest band centre on a miss/tie).

    This is robust to a picture/placeholder COUNT MISMATCH — the old code fell
    back to equal-height-band geometry the moment `len(pics) != total_slots`,
    which mis-placed the ROW (it only voted on the column). Placing by bbox keeps
    each picture in the cell its position indicates: e.g. one picture in the row-3
    image cell and two in the row-4 image cell when that is where their bboxes sit.
    It survives translation, which never touches the empty <img> cells.

    The `<img>` count per cell is honoured as a soft CAPACITY: while a cell is
    already at capacity and another image cell is still empty, a picture prefers
    the empty cell if it is at least as close in y.

    Returns None only when there is no usable `<img>` structure at all, so the
    caller falls back to the geometric scorer.
    """
    if not img_by_anchor or n_rows <= 0:
        return None
    tbl_y1 = float(bbox[1])
    tbl_h = max(1.0, float(bbox[3]) - float(bbox[1]))

    cells: List[Dict] = []
    for rc, n in img_by_anchor.items():
        if n <= 0 or rc not in cell_anchors:
            continue
        _t, _cs, rs, _h = cell_anchors[rc]
        r0 = rc[0]
        r1 = min(n_rows, r0 + max(1, rs))
        lo = tbl_y1 + tbl_h * (r0 / n_rows)
        hi = tbl_y1 + tbl_h * (r1 / n_rows)
        cells.append({
            "anchor": rc, "cap": n,
            "lo": lo, "hi": hi, "mid": (lo + hi) / 2.0,
            "count": 0,
        })
    if not cells:
        return None

    def _pic_cy(pic: Dict) -> float:
        pb = pic.get("_orig_bbox") or pic.get("bbox") or []
        if len(pb) != 4:
            return tbl_y1
        return (float(pb[1]) + float(pb[3])) / 2.0

    out: Dict[Tuple[int, int], List[Dict]] = {}
    for pic in sorted(pictures, key=_pic_sort_key):
        cy = _pic_cy(pic)
        containing = [c for c in cells if c["lo"] <= cy <= c["hi"]]
        with_cap = [c for c in containing if c["count"] < c["cap"]]
        pool = with_cap or [c for c in cells if c["count"] < c["cap"]] or cells
        best = min(pool, key=lambda c: abs(cy - c["mid"]))
        best["count"] += 1
        out.setdefault(best["anchor"], []).append(pic)
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

    Translation breaks the simple "short text = image cell" signal: a 3-CJK
    label like 横截面 expands to "Corte transversal" / "Cross section",
    which then outscores neighbouring numeric cells like `15.5` under any
    length-based tier. Two refinements keep this working:

    1. Per-row scoring picks a candidate (row, col) for each picture using
       emptiness + horizontal proximity, just like before — this still
       resolves the picture-in-its-own-row case (e.g. the `Âncora II`
       rowspan cell that is literally empty).
    2. Pictures whose bboxes share an x-range (same column visually) are
       grouped, and the GROUP's column index is a majority vote across its
       per-picture candidates. So even when one picture's row has only
       numeric neighbours, it inherits the column from siblings whose row
       contained the empty cell.

    The vote is the key fix: a single empty cell anywhere in the picture
    cluster's row range is enough to anchor the whole cluster to that
    column, regardless of what translation did to label text.
    """
    out: Dict[Tuple[int, int], List[Dict]] = {}
    if not pictures or n_rows <= 0 or max_cols <= 0:
        return out

    # Authoritative <img>-structure assignment first (by picture bbox y-position);
    # geometry is the fallback only when there are no <img> cells at all.
    structural = _assign_pictures_by_img_structure(
        pictures, bbox, cell_anchors, img_by_anchor or {}, n_rows,
    )
    if structural is not None:
        return structural

    tbl_x1, tbl_y1, tbl_x2, tbl_y2 = bbox
    tbl_w_px = max(1.0, tbl_x2 - tbl_x1)
    tbl_h_px = max(1.0, tbl_y2 - tbl_y1)
    row_band_h = tbl_h_px / n_rows
    weight_total = sum(col_weight) or max_cols
    col_edges_px = [tbl_x1]
    for cw in col_weight:
        col_edges_px.append(col_edges_px[-1] + tbl_w_px * (cw / weight_total))

    # For each row, list every cell anchor whose VERTICAL SPAN covers that
    # row — anchors at earlier rows that continue via rowspan are included.
    cells_by_row: Dict[int, List[Tuple[int, int, int, str]]] = {}
    for (r, c), (txt, cs, rs, _h) in cell_anchors.items():
        for rr in range(r, min(n_rows, r + max(1, rs))):
            cells_by_row.setdefault(rr, []).append((r, c, cs, txt or ""))

    def _emptiness_tier(text: str) -> int:
        """0 = empty, 1 = very short label, 2 = anything longer."""
        s = (text or "").strip()
        if not s:
            return 0
        if _display_width(s) <= 6:
            return 1
        return 2

    def _score_picture(pic: Dict) -> Tuple[Optional[Tuple[int, int]],
                                            List[Tuple[int, int, int, float]]]:
        """Return (best (row,col), all candidates) for a single picture.

        The 2nd element lets the cluster vote inspect ALL candidates per
        picture, not just the winner — important when one picture's
        empty-cell signal should propagate to siblings whose own row has
        only long label cells after translation.
        """
        pb = pic.get("_orig_bbox") or pic.get("bbox") or []
        if len(pb) != 4:
            return None, []
        cx_px = (pb[0] + pb[2]) / 2.0
        cy_px = (pb[1] + pb[3]) / 2.0
        row_idx = max(0, min(n_rows - 1, int((cy_px - tbl_y1) / row_band_h)))

        candidates: List[Tuple[int, int, int, float]] = []
        for dr in range(n_rows):
            offsets = (0,) if dr == 0 else (dr, -dr)
            for off in offsets:
                r = row_idx + off
                if r < 0 or r >= n_rows:
                    continue
                for (anchor_r, c, cs, txt) in cells_by_row.get(r) or []:
                    tier = _emptiness_tier(txt)
                    mid = (col_edges_px[c] + col_edges_px[min(max_cols, c + cs)]) / 2.0
                    dist = abs(cx_px - mid)
                    candidates.append((anchor_r, c, tier, dist))
                if candidates:
                    break
            if candidates:
                break
        if not candidates:
            return None, []
        candidates.sort(key=lambda t: (t[2], t[3]))
        return (candidates[0][0], candidates[0][1]), candidates

    # Per-picture preferred anchors AND full candidate lists.
    individual: List[Tuple[Dict, Optional[Tuple[int, int]],
                           List[Tuple[int, int, int, float]]]] = []
    for p in pictures:
        best, cands = _score_picture(p)
        individual.append((p, best, cands))

    # Cluster pictures by x-overlap: two pictures are in the same cluster
    # when their bbox x-spans overlap by ≥ 50% of the smaller span. This
    # catches vertically-stacked images of similar width (the common
    # "all images sit in one column" pattern) without flagging unrelated
    # diagrams elsewhere on the page.
    def _x_overlap(a: Dict, b: Dict) -> float:
        ab = a.get("_orig_bbox") or a.get("bbox") or []
        bb = b.get("_orig_bbox") or b.get("bbox") or []
        if len(ab) != 4 or len(bb) != 4:
            return 0.0
        ax1, _, ax2, _ = ab
        bx1, _, bx2, _ = bb
        inter = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        smaller = max(1.0, min(ax2 - ax1, bx2 - bx1))
        return inter / smaller

    n = len(pictures)
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _x_overlap(pictures[i], pictures[j]) >= 0.5:
                _union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(_find(i), []).append(i)

    # For each cluster, decide the consensus column. Strategy:
    #
    # 1. Tier-weighted voting from EVERY picture's full candidate list (not
    #    just its winner). An empty cell (tier 0) anywhere in any sibling's
    #    candidate set is worth more than many tier-1 numeric-cell votes —
    #    a single empty cell in the cluster's row range is the cleanest
    #    "this is the picture column" signal, and survives translation.
    # 2. Among the picture's own candidates, restrict to that consensus
    #    column. If the picture has no candidate in that column (rare —
    #    only when the cluster spans multiple distinct image columns),
    #    fall back to the picture's individual winner.
    TIER_WEIGHT = {0: 100, 1: 1, 2: 0.01}

    for members in clusters.values():
        col_score: Dict[int, float] = {}
        for idx in members:
            for (_ar, c, tier, dist) in individual[idx][2]:
                # Closer cells contribute more even within the same tier,
                # so adjacent-column ties resolve toward the picture's
                # actual centroid.
                proximity = 1.0 / (1.0 + dist / max(1.0, tbl_w_px / max_cols))
                col_score[c] = col_score.get(c, 0.0) + TIER_WEIGHT.get(tier, 0.01) * proximity
        if not col_score:
            continue
        consensus_col = max(col_score.keys(), key=lambda c: (col_score[c], -c))

        for idx in members:
            anchor = individual[idx][1]
            if anchor is None:
                continue
            anchor_r, _orig_c = anchor
            if (anchor_r, consensus_col) in cell_anchors:
                final = (anchor_r, consensus_col)
            else:
                # Find a row anchor in the consensus column whose vertical
                # span covers this picture's row.
                final = None
                for (ar, ac), (_txt, _cs, rs, _h) in cell_anchors.items():
                    if ac != consensus_col:
                        continue
                    if ar <= anchor_r < ar + max(1, rs):
                        final = (ar, ac)
                        break
                if final is None:
                    final = anchor
            out.setdefault(final, []).append(pictures[idx])
    return out


def compute_col_weights(
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    max_cols: int,
) -> List[int]:
    """Per-column weights inferred from cell text.

    A column's soft weight is a HIGH PERCENTILE (75th) of its cells' width
    contributions — not the max, and not the median. The max let a single
    outlier cell (one long ``H1, H2, … H15`` list among empty cells) blow a
    column out. The median under-serves the common real case where a column is
    short in MOST rows yet holds long paragraphs in several rows (e.g. a
    '特征判定' answer column: many '否'/'固定。' plus a few 60-180 char answers) —
    its median is ~one glyph, so it was squeezed to a 1-glyph-per-line vertical
    strip and its long cells clipped. The 75th percentile gives such a column
    enough width for its long cells while ignoring a lone outlier and empty
    placeholder cells. A hard floor from the longest unbreakable token AND a
    content-aware label minimum are still applied per column (via max) so no
    column is squeezed below what it needs.
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


def _inherit_header_colspans(
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    occupied: List[List[bool]],
    max_cols: int,
    n_rows: int,
) -> None:
    """Make body rows inherit the table's column grouping.

    Many source tables group several columns under one spanning header (e.g. a
    'Torque de falha' header with colspan=2 over 'Valor' + 'Modo' sub-columns).
    Summary rows ('Total', 'Average', 'Média', ...) then carry a single value
    under that whole group and leave the other grouped columns blank. OCR
    commonly emits those as a value cell plus a SEPARATE empty cell rather than
    one `colspan` cell, so the value lands in one sub-column instead of
    spanning the group as it does in the source.

    This pass detects each colspan group and, for every other row in which
    exactly ONE column of the group carries text and the rest are empty,
    replaces those cells with a single value cell spanning the whole group.
    Rows that fill more than one grouped column (real data rows) are left
    untouched. Mutates `cell_anchors`/`occupied` in place.

    General by construction — it keys only on:
      * the presence of a colspan>1 cell (the group), and
      * per-row emptiness within that group,
    with no table-, column-, document-, or language-specific assumptions. It
    also tolerates headers emitted as plain <td> rows (no <thead>) and values
    that sit in any sub-column of the group, not just the leftmost.
    """
    # 1) Column groups = ranges spanned by a colspan>1 cell. Prefer header
    #    cells; if the table marked no header, fall back to the first row,
    #    which is where a grouping header almost always sits.
    spanning = [
        (c, c + cs)
        for (r, c), (txt, cs, rs, is_h) in cell_anchors.items()
        if is_h and cs > 1
    ]
    if not spanning:
        spanning = [
            (c, c + cs)
            for (r, c), (txt, cs, rs, is_h) in cell_anchors.items()
            if r == 0 and cs > 1
        ]
    if not spanning:
        return
    # Dedup + drop nested/overlapping (keep the widest, leftmost-first).
    groups: List[Tuple[int, int]] = []
    for a, b in sorted(set(spanning), key=lambda g: (g[0], -(g[1] - g[0]))):
        if groups and a < groups[-1][1]:
            continue
        groups.append((a, b))

    # A colspan is only a genuine collapsible SUB-COLUMN GROUP if some row fills
    # the ENTIRE group together — every grouped column non-empty in one row. That
    # is the positive evidence the sub-columns form ONE unit (e.g. a
    # `Torque de falha`→`Valor|Modo` group whose data rows fill both), so a
    # single summary value under the group (a `Média`/`Total` row) truly spans it.
    #
    # A weaker "≥2 of the group filled somewhere" test over-collapses two
    # different real tables:
    #   * a wide title/logo/banner `colspan` (never fills ≥2) — the 7→6 bug, and
    #   * a RISK MATRIX where `Severidade (S)` spans 5 independent severity
    #     columns (`1 Desprezível … 5 Desastres`): a row filling 2 of the 5
    #     (`H1…` under 2 Leve + `H3,H11` under 3 Grave) would wrongly mark the
    #     group "real", then a row with a single positional value (`H9,H12` under
    #     3 Grave only) got merged across all 5 columns, losing which severity it
    #     maps to. Requiring the FULL width to be filled at least once keeps such
    #     positional matrices split (no row ever fills all 5), while genuine
    #     tight groups (all sub-columns filled together) still collapse. The
    #     gate only ever REJECTS more groups than the old test, so it cannot
    #     reintroduce the 7→6 over-merge.
    def _is_real_group(a: int, b: int) -> bool:
        for rr in range(n_rows):
            filled = 0
            for c in range(a, b):
                anc = cell_anchors.get((rr, c))
                # Only BODY cells count: the group's own SUB-HEADER row fills every
                # column with a label (e.g. `1 Desprezível … 5 Desastres`), which
                # would spuriously look like "the whole group filled together". A
                # header row is evidence of labelled sub-columns, NOT of them being
                # filled as one data unit, so skip header cells.
                if anc is not None and not anc[3] and (anc[0] or "").strip():
                    filled += 1
            if filled == (b - a):
                return True
        return False

    groups = [g for g in groups if _is_real_group(*g)]
    if not groups:
        return

    for r in range(n_rows):
        for (a, b) in groups:
            present = [
                (c, cell_anchors[(r, c)])
                for c in range(a, b)
                if (r, c) in cell_anchors
            ]
            # Need every grouped column present as its own cell (cs==1) — skip
            # when a rowspan from above already occupies part of the group, or
            # when the group itself is the spanning header cell.
            if len(present) != (b - a) or any(v[1] != 1 for _c, v in present):
                continue
            nonempty = [(c, v) for (c, v) in present if (v[0] or "").strip()]
            if len(nonempty) != 1:
                continue  # 0 = leave blank; 2+ = real data row, leave split
            _val_c, val_v = nonempty[0]
            t, _cs, rs, is_h = val_v
            # Collapse the group into one value cell at the group start,
            # spanning the full width. Covered columns stay `occupied`, so the
            # render loop emits them as gridSpan continuation (skips them).
            for c, _v in present:
                del cell_anchors[(r, c)]
            cell_anchors[(r, a)] = (t, b - a, rs, is_h)
            for c in range(a, b):
                occupied[r][c] = True


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
# one when the translated text genuinely can't fit the original table size. 3.5pt
# gives dense many-row tables the extra headroom to fit every cell inside the
# source bbox (rows stay hRule="exact") before the grow-backstop is ever needed.
_HARD_MIN_FONT_PT = 3.5


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

    # Precompute each text cell's needed height once (font is fixed for this
    # call). Map each cell to the row it is CHARGED to: a single-row cell to its
    # own row; a rowspan cell to the LAST row of its span, so growing that row
    # (the balancer's job) relieves the whole merged cell. Rowspan cells were
    # previously skipped entirely, leaving tall merged labels clipped.
    _charged: List[Tuple[int, float, int, int]] = []  # (charge_row, need_h, r0, rs)
    for (r0, c), (txt, cs, rs, is_h) in cell_anchors.items():
        if not (txt or "").strip():
            continue
        cell_w_pt = sum(col_w_emus[c:c + max(1, cs)]) / EMU_PER_PT
        bold = bool(is_h or r0 == 0)
        need_h = _needed_lines(txt, cell_w_pt, size_px, bold) \
            * _line_h_pt(size_px, bold, has_cjk(txt)) \
            + reserve_pt_by_cell.get((r0, c), 0.0)
        charge_row = min(n_rows - 1, r0 + max(1, rs) - 1)
        _charged.append((charge_row, need_h, r0, max(1, rs)))

    def _row_deficit(heights: List[float], r: int) -> float:
        """Max (needed_h − avail_h) in pt charged to row r (>0 = clip). A rowspan
        cell's available height is the sum over the rows it covers."""
        worst = 0.0
        for (charge_row, need_h, r0, rs) in _charged:
            if charge_row != r:
                continue
            avail = sum(heights[r0:r0 + rs]) if rs > 1 else heights[r]
            worst = max(worst, need_h - avail)
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
    """Wrap `text` to `box_w_pt` at `size_pt`; return (wrapped_lines,
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
    """A recovered picture's source height (pre-reflow bbox height in pixels;
    falls back to the raster's own pixel height). Used only to split a shared
    rowspan between stacked pictures in proportion to their sizes — units
    cancel, so the raw pixel height is fine."""
    pb = pic.get("_orig_bbox") or pic.get("bbox") or []
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
    orig_bbox: Optional[List[int]] = None,
) -> None:
    text = (entry.get("text") or "").strip()
    if not text:
        return
    bbox = entry.get("bbox")
    if not bbox or len(bbox) != 4:
        return
    # Reflow may have shifted/grown this table's bbox downward while the
    # contained pictures stayed frozen at their original coordinates. Picture
    # → cell assignment must therefore be computed in the PRE-reflow frame
    # (the frame the picture coords still live in); using the post-reflow
    # bbox would map the first picture into the header row. Cell sizing
    # (column widths, row heights) still uses the live `bbox`.
    assign_bbox = orig_bbox if (orig_bbox and len(orig_bbox) == 4) else bbox
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
        trailing_h_emu = min(trailing_h_emu, int(h * 0.6))
        h = max(1, h - trailing_h_emu)

    if not rows:
        # Fall back to text rendering inside a positioned box. Pin the bbox
        # (`body_auto_fit=False`) so an oversize translation can't grow past
        # the source rectangle; `fit_multiline` will shrink-or-truncate so the
        # rendered content always fits.
        x1, y1, x2, y2 = bbox
        box_w_pt = max(1.0, (x2 - x1) / ctx.zoom)
        box_h_pt = max(1.0, (y2 - y1) / ctx.zoom)
        t_style = dict(style)
        base_size_pt = float(t_style.get("size") or 11.0)
        fitted, lines = fit_multiline(
            text, box_w_pt, box_h_pt, max_size_pt=base_size_pt,
        )
        line_pt = None
        render_text = text
        # `lines is None` → doesn't fit even at the floor. Wrap the full text at
        # the floor and let the box grow (spAutoFit) instead of clipping.
        fallback_auto_fit = lines is None
        if lines is None:
            t_style["size"] = min(base_size_pt, 6.0)
            _f = get_font(max(1, int(round(6.0))))
            render_text = (
                "\n".join(wrap_to_width(text, _f, box_w_pt * 0.93, int(round(6.0))))
                if _f is not None else text
            )
        else:
            if fitted < base_size_pt:
                t_style["size"] = fitted
            if lines:
                render_text = "\n".join(lines)
                if len(lines) >= 2:
                    line_pt = box_h_pt / len(lines)
        runs_xml = build_run_xml(render_text, t_style)
        para_xml = build_paragraph_xml(runs_xml, line_pt=line_pt)
        ctx.xml_chunks.append(
            build_anchored_textbox_xml(
                x, y, w, h, para_xml, ctx._next_id(),
                body_auto_fit=fallback_auto_fit,
                rotation=table_rotation,
                # `x/y/w/h` came from rotated_shape_geometry when turned, which
                # is the transposed extent the shape-rotating path expects.
                rotate_shape=True,
            )
        )
        return

    max_cols, n_rows, cell_anchors, occupied, img_by_anchor = parse_table_grid(rows)

    # Make summary/total rows inherit the header's column grouping so a single
    # value under a multi-column header spans that group (as in the source)
    # instead of sitting in one sub-column with blank cells beside it.
    _inherit_header_colspans(cell_anchors, occupied, max_cols, n_rows)

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
            pictures_for_table, assign_bbox, col_weight, cell_anchors,
            max_cols, n_rows, img_by_anchor,
        )
        # A picture column's need is a MINIMUM WIDTH (the picture's true physical
        # width), not a proportional weight. Recording it as a weight (the old
        # `pic_w_px / 4.4`) let the image column's huge pixel count dominate the
        # proportional split and squeeze the numeric columns to a few points —
        # which is what clipped the '直径/D' etc. headers. Fold the physical
        # width into the per-column floor below instead.
        zoom = max(1e-6, float(getattr(ctx, "zoom", 1.0) or 1.0))
        for (_r, c), pics in pic_assignments.items():
            for pic in pics:
                pb = pic.get("_orig_bbox") or pic.get("bbox") or []
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
    # little padding), so the column is guaranteed wide enough to show the image
    # at full size without upscaling the whole column out of proportion. Cap a
    # single picture column at 55% of the table so the data columns keep a
    # usable share.
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
    _bal_size_pt = min(declared_size_pt, max(7.0, _bal_narrow_pt / 5.0), 12.0)
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
    tbl_pr_xml = (
        '<w:tblPr>'
        f'<w:tblW w:w="{w}" w:type="dxa"/>'
        '<w:tblLayout w:type="fixed"/>'
        f'{borders_xml}'
        f'{cell_mar_xml}'
        '</w:tblPr>'
    )

    # Per-cell font cap based on the narrowest column to keep cells readable.
    narrowest_col_pt = min(col_w_emus) / EMU_PER_PT
    size_ceiling_pt = min(
        declared_size_pt, max(7.0, narrowest_col_pt / 5.0), 12.0,
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
    # past the cell border (rows use hRule="exact", a hard clip) — which is what
    # clipped the anchor illustrations. We compute each picture's height when
    # scaled to the cell's column width, add a small per-picture pad, and if the
    # anchor's rowspan rows don't already provide that much height, grow those
    # rows evenly to cover the deficit. Text-only cells / picture-free tables
    # are untouched.
    row_min_pts = [line_h_pt * rl for rl in row_lines_arr]
    _pic_reserve_emu_by_cell: Dict[Tuple[int, int], int] = {}
    if pic_assignments:
        _zoom = max(1e-6, float(getattr(ctx, "zoom", 1.0) or 1.0))
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
                pb = pic.get("_orig_bbox") or pic.get("bbox") or []
                if len(pb) != 4 or pb[2] <= pb[0] or pb[3] <= pb[1]:
                    continue
                iw_pt = (pb[2] - pb[0]) / _zoom
                ih_pt = (pb[3] - pb[1]) / _zoom
                # height once the picture is scaled down to fit the column width
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
                    row_min_pts[r] = max(row_min_pts[r], row_h_pts[r])

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
    # Per-row picture-reserve floor: a picture cell in ANY column (not just
    # column 0) must keep enough height for its image. A rowspan picture cell's
    # reserve is spread evenly across the rows it covers so each of those rows is
    # floored by its share — otherwise `_normalise_to` could reclaim the reserved
    # height from a non-first-column picture row and clip the image.
    _row_reserve_floor = [0.0] * n_rows
    for (a_r, a_c), resv_pt in reserve_pt_by_cell.items():
        anc = cell_anchors.get((a_r, a_c))
        rs = max(1, anc[2]) if anc else 1
        span = list(range(a_r, min(n_rows, a_r + rs)))
        if not span:
            continue
        share = resv_pt / len(span)
        for r in span:
            _row_reserve_floor[r] = max(_row_reserve_floor[r], share)

    def _min_rows_for(size_px: int) -> List[float]:
        lh = _line_h_pt(size_px, False, False)
        return [
            max(lh, _row_reserve_floor[r])
            for r in range(n_rows)
        ]

    # When True, even the hard-min font can't fit the content in the source bbox
    # height; the grow-backstop below lets the table extend downward so text is
    # NEVER clipped (rows switch to hRule="atLeast"). Effectively unreachable for
    # real tables at the 3.5pt floor — it exists only to honour "no clip".
    _grow_to_fit = False
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
        _normalise_to(bbox_h_pt)   # rows sum to bbox height (never grow)
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
            # Even the hard-min font can't fit in the fixed bbox. Rather than
            # clip, signal the grow-backstop so the table extends downward.
            _grow_to_fit = True
            break
        fit_size_px = max(_HARD_MIN_FONT_PT, fit_size_px - 0.5)

    # Lock the per-cell emission font to the fitted size and keep totals exact.
    cell_size_pt = float(fit_size_px)
    cell_size_px = max(1, int(round(fit_size_px)))
    row_min_pts = _min_rows_for(cell_size_px)

    if not _grow_to_fit:
        _normalise_to(bbox_h_pt)
        # FINAL clip verification at the exact size/geometry we will emit. The
        # fit loop breaks as soon as `_any_cell_overflows` is clean, but the
        # subsequent `_normalise_to` (rounding, floor reclaim) or a rowspan cell
        # the row-balancer can only partly relieve could leave a residual
        # overflow. Rather than emit a clipped cell, escalate to the grow-backstop
        # so the row(s) grow and the box extends downward (hRule="atLeast").
        if _any_cell_overflows(
            cell_anchors, col_w_emus, row_h_pts, max_cols, n_rows,
            cell_size_px, reserve_pt_by_cell,
        ):
            _grow_to_fit = True

    if _grow_to_fit:
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
                # Per-cell fit. Translation expansion is non-uniform — one
                # cell may become 8 chars while its neighbour becomes 80 —
                # so each cell is re-fitted against (cell_w, row_h). The
                # base size is `cell_size_pt` (the table-wide cap derived
                # from the narrowest column); `fit_multiline` shrinks
                # further inside that budget and truncates with `…` when
                # even the floor doesn't fit, so cell content never
                # overflows the fixed row height (`hRule="exact"`).
                cell_h_pt_for_fit = sum(
                    row_h_pts[row_idx + dr] for dr in range(max(1, rs))
                )
                cell_w_pt_for_fit = cell_w_emu / EMU_PER_PT
                # When this cell also holds pictures, fit the text into the
                # space left ABOVE them (subtract the reserved picture height)
                # so the label and the image don't collide / clip each other.
                _reserve_emu = _pic_reserve_emu_by_cell.get((row_idx, col_idx), 0)
                if _reserve_emu > 0:
                    cell_h_pt_for_fit = max(
                        natural_h_px * 1.06,
                        cell_h_pt_for_fit - _reserve_emu / EMU_PER_PT,
                    )
                # The coupled redistribution + font fallback above already sized
                # the columns, rows and `cell_size_px` so that EVERY cell's full
                # text fits its cell at that size — so we render at exactly the
                # fitted size and wrap the FULL text (never an ellipsis, never a
                # smaller-than-fitted 6pt path that would clip). This is what
                # guarantees no clipped / hidden / scrollable cells.
                cell_style = dict(style)
                cell_style["size"] = cell_size_pt if cell_text else cell_size_pt
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
                pics_here = pic_inline_by_cell.get((row_idx, col_idx)) or []
                pic_paragraphs = []
                cell_h_emu = int(round(row_h_pts[row_idx] * EMU_PER_PT))
                if rs > 1:
                    cell_h_emu = sum(
                        int(round(row_h_pts[row_idx + dr] * EMU_PER_PT))
                        for dr in range(rs)
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
                tc_pr_parts = [f'<w:tcW w:w="{cell_w_emu}" w:type="dxa"/>']
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
                tc_pr_parts.append('<w:vAlign w:val="center"/>')
                cells_xml.append(
                    "<w:tc>"
                    f'<w:tcPr>{"".join(tc_pr_parts)}</w:tcPr>'
                    f"{para_xml}{''.join(pic_paragraphs)}"
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
                        f'<w:tcPr><w:tcW w:w="{cont_w_emu}" w:type="dxa"/>'
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
                    f'<w:tcPr><w:tcW w:w="{col_w_emus[col_idx]}" w:type="dxa"/>'
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
        tr_pr = f'<w:trPr><w:trHeight w:val="{row_h_twips}" w:hRule="{_hrule}"/></w:trPr>'
        rows_xml_parts.append("<w:tr>" + tr_pr + "".join(cells_xml) + "</w:tr>")

    # Build the grid from the FINAL, fully-balanced column widths (see note at
    # the pre-emission balance call above).
    grid_xml = "<w:tblGrid>" + ("".join(
        f'<w:gridCol w:w="{cw}"/>' for cw in col_w_emus
    )) + "</w:tblGrid>"
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