"""Hybrid layout adaptation engine for translated document reconstruction.

Implements width-first fitting + cascading push-down: when a text entry can't
hold its translated content, we FIRST try to widen it to the right into free
space (obstacle-aware, never overlapping a neighbour or the page edge), which
reduces line-wrapping and often removes the need to grow taller at all. Only
the height still unaccounted for triggers vertical growth, and that growth
cascades downward to push every entry below it.

Pictures cascade-move with every other entry; containment detection in
json_to_docx.py and in-table cell assignment in table.py are computed against
pre-reflow snapshots, so they stay correct regardless of movement. Tables and
other non-text entries shift with the cascade. Page overflow is resolved by
uniform vertical-only scaling, never by creating new pages.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

_TEXT_CATEGORIES = {
    "Text", "List-item", "Section-header", "Paragraph",
    "Caption", "Title", "Page-header", "Page-footer", "Footnote",
}

# Entries that expand themselves AND trigger cascade push on entries below.
_CASCADE_CATEGORIES = {
    "Text", "List-item", "Section-header", "Paragraph", "Caption", "Title",
    "Footnote",
}

# Pinned entries: never expand, never pushed, never scaled.
_FIXED_CATEGORIES = {"Page-header", "Page-footer"}


_PUSH_EXCLUDED: set = set()

# Safety gap (in points, scaled by zoom at use site) kept between a widened
# box and the obstacle to its right, so neighbours never end up touching.
_GAP_PT = 6.0

# Small breathing room (points) added after the text's hugged width so the
# last glyph isn't flush against the box edge.
_PAD_PT = 4.0


_WIDTH_SAFETY = 0.93


def _hugged_x2(entry: Dict, zoom: float, x1: float, x2_cur: float) -> Optional[float]:
    """Right edge that makes the box hug its text: width of the longest
    rendered line + one character of breathing space. Left edge stays fixed.

    Measured at the box's CURRENT width so the wrap (and therefore the line
    count and height) is the one the box already has — hugging only trims the
    empty right margin, it never re-wraps. The caller clamps the result to the
    current width so this can only narrow, never widen (widening for overflow
    is handled separately).
    """
    from .text_fit import (
        fit_multiline, get_font, measure_width_px, _TEXTBOX_EDGE_PAD_PT,
    )

    text = (entry.get("text") or "").strip()
    if not text:
        return None
    cat = entry.get("category", "")
    style = entry.get("style") or {}
    base = float(style.get("size") or 11.0)
    bold = bool(style.get("bold")) or cat in (
        "Title", "Section-header", "Page-header", "Page-footer", "Caption",
    )
    box_w_pt = max(1.0, (x2_cur - x1) / zoom)
    size_pt, lines = fit_multiline(
        text, box_w_pt, 10000.0, max_size_pt=base, min_size_pt=6.0, bold=bold,
        edge_pad_pt=_TEXTBOX_EDGE_PAD_PT,
    )
    if not lines:
        return None
    size_px = max(1, int(round(size_pt)))
    font = get_font(size_px, bold=bold)
    if font is None:
        return None
    longest_pt = max(measure_width_px(ln, font, size_px) for ln in lines)
    one_char_pt = measure_width_px("M", font, size_px)  # one-letter pad
    # Fixed edge pad (zero-inset text box), consistent with the fitter — the old
    # `/ _WIDTH_SAFETY` inflated the hug width ~7.5% and over-widened the box.
    hug_w_pt = longest_pt + _TEXTBOX_EDGE_PAD_PT + one_char_pt
    return x1 + hug_w_pt * zoom


def _needed_height_px(entry: Dict, zoom: float) -> Optional[float]:
    """Return the pixel height that fit_multiline says the translated text
    needs at the entry's current (possibly already grown) width, or None if
    measurement is not possible."""
    from .text_fit import fit_multiline, _TEXTBOX_EDGE_PAD_PT

    text = (entry.get("text") or "").strip()
    if not text:
        return None
    bbox = entry.get("bbox")
    if not bbox or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_w_pt = max(1.0, (x2 - x1) / zoom)
    # Use a very tall height so fit_multiline always picks max_size and returns
    # the natural line count — we only care about the line splits here.
    category = entry.get("category", "")
    style = entry.get("style") or {}
    base_size_pt = float(style.get("size") or 11.0)
    bold = bool(style.get("bold")) or category in (
        "Title", "Section-header", "Page-header", "Page-footer", "Caption",
    )

    _size_pt, lines = fit_multiline(
        text,
        box_w_pt,
        box_h_pt=10000.0,   # unconstrained height — get natural line count
        max_size_pt=base_size_pt,
        min_size_pt=6.0,
        bold=bold,
        edge_pad_pt=_TEXTBOX_EDGE_PAD_PT,
    )
    if not lines:
        return None

    from .text_fit import get_font
    size_px = max(1, int(round(_size_pt * 1.0)))  # dpi=72 so pt==px at 72dpi
    font = get_font(size_px, bold=bold)
    if font is None:
        return None
    asc, desc = font.getmetrics()
    from .text_fit import has_cjk
    natural_h = asc + desc
    if has_cjk(text):
        natural_h = max(natural_h, size_px * 1.2)
    line_h = natural_h * 1.10   # same safety as _wrapped_lines_if_fit
    return math.ceil(line_h * len(lines))


def _vertical_band_overlap(a, b) -> bool:
    """True if bboxes `a` and `b` share any vertical extent. Used to decide
    whether a neighbour actually sits beside this box (same horizontal band)
    versus merely above or below it."""
    return a[1] < b[3] and a[3] > b[1]


# Label categories that title the table they sit above. When OCR places one so
# it vertically overlaps a Table (its title dropped inside the table's top), we
# lift it back above the table.
_OVERLAP_RESOLVE_CATEGORIES = {"Caption", "Title", "Section-header", "Section-Header"}
# Gap (pt, scaled by zoom) kept between a lifted caption's bottom and table top.
_TABLE_GAP_PT = 4.0


def _lift_captions_above_overlapping_tables(valid: List[Dict], zoom: float) -> None:
    """Move any caption/title/section-header that VERTICALLY OVERLAPS a Table to
    sit just ABOVE that table's top edge (captions label the table below them).
    Pure bbox translation — no cascade, no scaling — and a strict no-op when
    nothing overlaps. Keys only on category + geometry.

    A Caption/Title/Section-header is a BLOCK-LEVEL label, never a table cell
    (cell text lives inside the table's own HTML), so a separate label entry that
    overlaps a table is always a mis-placed title, not a legitimate in-table
    label — we do NOT treat geometric containment as a reason to leave it. Only a
    label already entirely above the table is left alone. When a label overlaps
    several tables it is lifted above the topmost overlapping one."""
    tables = [
        e for e in valid
        if e.get("category") == "Table"
        and e.get("bbox") and len(e["bbox"]) == 4
    ]
    if not tables:
        return
    gap_px = _TABLE_GAP_PT * zoom
    for e in valid:
        if e.get("category") not in _OVERLAP_RESOLVE_CATEGORIES:
            continue
        eb = e.get("bbox")
        if not eb or len(eb) != 4:
            continue
        ex1, ey1, ex2, ey2 = [float(v) for v in eb]
        target_top: Optional[float] = None
        for t in tables:
            tx1, ty1, tx2, ty2 = [float(v) for v in t["bbox"]]
            # Already entirely above the table → nothing to fix.
            if ey2 <= ty1:
                continue
            # Must share a horizontal band (sits over the table) AND overlap
            # vertically (dropped into it) to count as an overlap to resolve.
            if not (ex1 < tx2 and ex2 > tx1):
                continue
            if not (ey1 < ty2 and ey2 > ty1):
                continue
            ty1_target = ty1  # lift above this table's top
            if target_top is None or ty1_target < target_top:
                target_top = ty1_target
        if target_top is None:
            continue
        ch = ey2 - ey1
        new_bottom = target_top - gap_px
        new_top = new_bottom - ch
        if new_top < 0.0:            # clamp to page top; still clears the table
            new_top = 0.0
            new_bottom = ch
        e["bbox"] = [ex1, new_top, ex2, new_bottom]


def _min_width_x2_for_height(
    entry: Dict,
    zoom: float,
    x1: float,
    y1: float,
    x2_cur: float,
    x2_max: float,
    target_h_px: float,
) -> Optional[float]:
    """Smallest right edge in [x2_cur, x2_max] at which the translated text
    fits within `target_h_px` (the box's ORIGINAL height). Returns None when
    even the maximum available width can't make the text fit that height —
    in which case the caller falls back to full width + vertical growth.

    This is what makes the box *hug* its text: instead of widening all the way
    to the obstacle, we widen only as far as needed to remove the extra line
    wraps, then stop. Binary search over the right edge; `_needed_height_px`
    re-measures at each trial width. The entry's bbox is restored before
    returning so the caller owns the final mutation.
    """
    orig = list(entry["bbox"])

    def needed_at(x2: float) -> Optional[float]:
        entry["bbox"] = [x1, y1, x2, orig[3]]
        return _needed_height_px(entry, zoom)

    h_max = needed_at(x2_max)
    if h_max is None or h_max > target_h_px + 0.5:
        entry["bbox"] = orig
        return None  # widening alone can't fit the original height

    lo, hi, best = x2_cur, x2_max, x2_max
    for _ in range(24):
        if hi - lo <= 1.0:
            break
        mid = (lo + hi) / 2.0
        h = needed_at(mid)
        if h is not None and h <= target_h_px + 0.5:
            best, hi = mid, mid
        else:
            lo = mid
    entry["bbox"] = orig
    return best


def _right_expansion_limit_px(
    entry: Dict,
    all_entries: List[Dict],
    page_w_px: float,
    gap_px: float,
) -> float:
    """Furthest the entry's right edge may move without hitting the page edge
    or any neighbour that sits in the same vertical band.

    The left edge is held fixed (preserves reading alignment); only the right
    edge grows. Any obstacle that starts to the right of our left edge AND
    overlaps our vertical band caps the expansion at `obstacle.x1 - gap_px`.
    The `ox1 > ex1` test also catches a neighbour straddling our current right
    edge, so we never widen into something already overlapping us.

    Returns a limit that is never to the left of the current right edge, so
    the caller can safely compare against `x2`.
    """
    ex1, ey1, ex2, ey2 = [float(v) for v in entry["bbox"]]
    limit = page_w_px  # margins are zeroed in this pipeline; page edge is the cap
    ebox = (ex1, ey1, ex2, ey2)
    for other in all_entries:
        if other is entry:
            continue
        ob = other.get("bbox")
        if not ob or len(ob) != 4:
            continue
        ox1, oy1, ox2, oy2 = [float(v) for v in ob]
        if ox1 > ex1 and _vertical_band_overlap(ebox, (ox1, oy1, ox2, oy2)):
            limit = min(limit, ox1 - gap_px)
    return max(ex2, limit)  # never propose a limit left of the current edge


def reflow_page_entries(
    entries: List[Dict],
    page_w_pt: float,
    page_h_pt: float,
    zoom: float,
) -> List[Dict]:
    """Expand text-entry bboxes so translated text fits, preferring horizontal
    growth into free space and falling back to cascading vertical growth.

    Single pass, top-to-bottom over cascade-category entries:

      1. Measure the height the translated text needs at the box's current
         width. If it already fits, skip.

      2. WIDTH-FIRST: compute how far the right edge can move without hitting
         the page edge or any neighbour in the same vertical band, widen into
         that space, and re-measure. Fewer line-wraps frequently means the
         text now fits the original height — no vertical growth, no cascade.

      3. HEIGHT FALLBACK: for whatever still overflows, grow the bottom edge
         and push every subsequent cascade entry down by the same delta,
         preserving ordering and preventing vertical overlap. There is no
         ceiling — entries may push past the page bottom; layout_page()
         handles overflow via uniform vertical scaling.

    The left edge is never moved, so reading alignment is preserved and
    left-hand neighbours are never disturbed.

    Pictures cascade-move with the rest of the page (they are no longer
    frozen), so text growing above a picture pushes the picture down instead of
    overlapping it. Containment and in-table assignment use pre-reflow
    snapshots, so they remain correct despite the movement. Tables, Formulas,
    and other passive entries shift with the cascade but do not expand
    themselves.

    Page-header and Page-footer entries are excluded entirely: they stay at
    their original positions and are never pushed or scaled.
    """
    if not entries:
        return []

    valid = [e for e in entries if e.get("bbox") and len(e["bbox"]) == 4]
    page_h_px = page_h_pt * zoom
    page_w_px = page_w_pt * zoom
    gap_px = _GAP_PT * zoom

    # Resolve OCR-placed caption/title overlaps with tables BEFORE sorting, so
    # the corrected top edge participates in the top-to-bottom cascade below.
    _lift_captions_above_overlapping_tables(valid, zoom)

    fixed_entries = [e for e in valid if e.get("category") in _FIXED_CATEGORIES]
    cascade_entries = [e for e in valid if e.get("category") not in _FIXED_CATEGORIES]

    # Sort cascade entries top-to-bottom by their top edge.
    cascade_entries.sort(key=lambda e: float(e["bbox"][1]))

    # ── Width-first fitting, then cascading height expansion ────────────────
    for i, entry in enumerate(cascade_entries):
        if entry.get("category") not in _CASCADE_CATEGORIES:
            continue  # Pictures/Tables/Formulas: passive — they shift but don't expand
        # Rotated entries are passive too: every measurement below
        # (_needed_height_px, the width-first fit, the downward cascade) assumes
        # text flows left-to-right and grows downward. For turned text those
        # axes are swapped, so expanding here would grow the box across its own
        # reading direction.
        if float(entry.get("rotation") or 0.0):
            continue

        x1, y1, x2, y2 = [float(v) for v in entry["bbox"]]
        cur_h = y2 - y1
        needed = _needed_height_px(entry, zoom)

        if needed is not None and needed > cur_h:
            # Overflows its height. Widen to the RIGHT into free space
            # (obstacle-aware) only as far as needed to fit the ORIGINAL
            # height; if even the full available width isn't enough, take the
            # full width (fewest wraps) and grow height, cascading downward.
            right_limit = _right_expansion_limit_px(entry, valid, page_w_px, gap_px)
            grow_height = False
            if right_limit > x2 + 1.0:
                fit_x2 = _min_width_x2_for_height(
                    entry, zoom, x1, y1, x2, right_limit, cur_h,
                )
                if fit_x2 is not None:
                    x2 = min(right_limit, fit_x2 + _PAD_PT * zoom)
                    entry["bbox"] = [x1, y1, x2, y2]
                else:
                    x2 = right_limit
                    entry["bbox"] = [x1, y1, x2, y2]
                    widened = _needed_height_px(entry, zoom)
                    if widened is not None and widened > cur_h:
                        needed, grow_height = widened, True
            else:
                grow_height = True  # no room to widen → grow height

            if grow_height and needed is not None:
                delta = (y1 + needed) - y2
                if delta > 0:
                    y2 = y1 + needed
                    entry["bbox"] = [x1, y1, x2, y2]
                    for j in range(i + 1, len(cascade_entries)):
                        if cascade_entries[j].get("category") in _PUSH_EXCLUDED:
                            continue
                        jx1, jy1, jx2, jy2 = [float(v) for v in cascade_entries[j]["bbox"]]
                        cascade_entries[j]["bbox"] = [jx1, jy1 + delta, jx2, jy2 + delta]

        x2_now = float(entry["bbox"][2])
        hug_x2 = _hugged_x2(entry, zoom, x1, x2_now)
        if hug_x2 is not None:
            hug_x2 = min(x2_now, hug_x2)
            if hug_x2 < x2_now - 1.0:
                entry["bbox"] = [x1, entry["bbox"][1], hug_x2, entry["bbox"][3]]

    return cascade_entries + fixed_entries


def layout_page(
    entries: List[Dict],
    page_w_pt: float,
    page_h_pt: float,
    zoom: float,
) -> List[Dict]:
    """Reflows entries for a single source page and returns exactly one page
    dict, preserving the original page count.

    After cascading reflow, if the cascade entries collectively exceed the
    page height, a uniform vertical-only scale factor is applied to compress
    them back within bounds.  x-coordinates are never modified by scaling.
    Fixed entries (Page-header, Page-footer) are excluded from scaling.

    Always returns a list of exactly one page dict so that
    json_to_docx.py's physical_pages.extend(layout_page(...)) works
    correctly without any changes to the caller.
    """
    reflowed = reflow_page_entries(entries, page_w_pt, page_h_pt, zoom)

    page_h_px = page_h_pt * zoom

    cascade = [e for e in reflowed if e.get("category") not in _FIXED_CATEGORIES]
    max_bottom = max((float(e["bbox"][3]) for e in cascade), default=0.0)

    if max_bottom > page_h_px > 0:
        scale_y = page_h_px / max_bottom
        for entry in cascade:
            x1, y1, x2, y2 = [float(v) for v in entry["bbox"]]
            entry["bbox"] = [x1, y1 * scale_y, x2, y2 * scale_y]

    return [
        {
            "entries": reflowed,
            "page_width_pt": page_w_pt,
            "page_height_pt": page_h_pt,
            "zoom": zoom,
        }
    ]