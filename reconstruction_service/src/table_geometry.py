"""Content-vs-bbox sanity checks for table geometry.

Shared by BOTH `ocr_reconstruction.table` and `translation_reconstruction.table`
— the two near-duplicate renderers — so this particular rule cannot drift between
them the way ~600 other lines already have.

Why this exists
---------------
`render_table` derives its width from the OCR'd bbox exactly once::

    x, y, w, h = bbox_px_to_emu(bbox, ...)

and from then on `w` is an AXIOM: the proportional split, the floor lifting, the
trim loop, the drift fixup and `_balance_col_widths` all conserve it exactly.
There are per-column FLOORS (`min_col_emus`) but no ceiling anywhere, and
`bbox_px_to_emu` clamps only against the PAGE, never against content.

That is correct as long as the bbox is. When the layout model boxes a small table
with its containing block's rectangle — which happens when a mislabelled `Form`
block is rescued by `_looks_like_table`, among other ways — the renderer faithfully
stretches a 2x2 grid across the whole page. Row heights follow: `_normalise_to`
has an explicit under-budget branch that INFLATES rows "so the table fills its
bbox exactly", so the table also grows tall enough to swallow the page footer.

So: measure what the content actually wants, and when the bbox is wildly larger
than that, stop trusting the bbox's extent. Its POSITION is still the model's
honest statement about where the table lives, which is why the clamp re-centres
INSIDE the original bbox rather than on the page.

Deliberately dependency-free
----------------------------
No imports from either package. The two text-measurement helpers this needs
(`_display_width`, `_longest_token_width`) exist in both copies and bottom out in
each package's own `text_fit.is_cjk_char`, so they are INJECTED as callables
rather than imported — importing one package's copy into the other would create a
cross-package dependency that does not exist today. It also makes this module
trivially unit-testable with plain lambdas.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

# EMU per point. Duplicated from each package's `geometry.py` rather than
# imported, to keep this module free of package imports (see module docstring).
EMU_PER_PT = 12700

# How many times taller than its rows actually need a bbox may be before it is
# treated as a CONTAINER rectangle rather than the table's own extent.
#
# WHY HEIGHT AND NOT WIDTH. The obvious test — bbox width vs the content's
# natural width — cannot work, because `bbox_px_to_emu` clamps the bbox to the
# PAGE before the renderer ever sees it. A runaway bbox therefore arrives no
# wider than one page, and measurement across real geometries put the reported
# regression at only 1.24-2.04x natural width, BELOW ordinary healthy tables
# (a normal 3-column table measures 2.84x, a CJK 2-column one 2.92x). No width
# threshold separates the two.
#
# Height is not page-clamped the same way, and "how many rows of text would fit
# in this box" is a far sharper question than "how wide could this text be":
#
#     the reported regression, zoom 2.0 -> 4.8                 -> 8.0-22x
#     normal tables                                            -> ~2.0-2.3x
#     sparse forms / signature blocks (worst honest cases)      -> ~5.0-5.4x
#
# 7.0 sits above every legitimate sparse layout measured, with room below the
# regression's FLOOR — which is 8.0 exactly, at the highest zoom these documents
# use (4.8x). A threshold of 8.0 would sit precisely on that floor and miss the
# bug on the most heavily zoomed scans.
#
# Lowering this further is not a free "stricter" knob: the next thing it reaches
# is spacious forms and signature blocks, which are correct as they are.
BBOX_OVERSIZE_RATIO = 7.0

# Per-cell horizontal padding, matching `_usable_w_pt` in both table.py copies
# (which subtracts this from each side of a cell before fitting text).
CELL_MARGIN_PT = 5.4

# Extra display units per column so a label is not flush against the border.
# Mirrors the `+ 2` in the `content_min_disp` floor code in both copies.
_COL_PAD_DISPLAY = 2


def natural_table_width_emu(
    cell_anchors: Dict[Tuple[int, int], Tuple],
    max_cols: int,
    col_weight: List[int],
    declared_size_pt: float,
    display_width_fn: Callable[[str], int],
    longest_token_fn: Callable[[str], int],
) -> int:
    """Width in EMU this table's content would take if nothing constrained it.

    A COMFORTABLE width, not a minimum: the point of comparison is "does the
    bbox make sense for this content", so a hard minimum would make almost any
    bbox look oversized and the trigger would fire constantly.

    Two per-column sources, maxed:

    1. ``col_weight[c]`` — already a 75th-percentile display width per column
       from `compute_col_weights`, with unbreakable-token and content-min floors
       folded in. Converted to points with the SAME ``pt_per_display`` factor the
       existing per-column floor code uses, so this ceiling and those floors are
       measured on one stick and can never contradict each other.
    2. The longest unbreakable token in any cell of the column — a column must
       not be declared narrower than a word it is required to hold. Colspan
       cells share their demand across the columns they cover, matching how
       `compute_col_weights` and the `content_min_disp` loop both do it.

    Returns 0 when there is nothing to measure, which callers treat as
    "no opinion" and skip the clamp.

    NOTE: deliberately does NOT use `measure_width_px`/`get_font`. Those need a
    settled font size, and the font size is an OUTPUT of the fitting loop that
    consumes this value — measuring in real pixels here would be circular. The
    display-unit path is what the surrounding floor code already trusts and is
    font-independent.
    """
    if max_cols <= 0 or not cell_anchors:
        return 0

    # Same conversion factor as the `content_min_disp` floor code: roughly half
    # a glyph advance per display unit (one CJK glyph = 2 display units = ~1 em).
    pt_per_display = max(1.0, float(declared_size_pt) * 0.5)

    token_disp = [0] * max_cols
    for (_r, c), anchor in cell_anchors.items():
        if c >= max_cols:
            continue
        txt = anchor[0]
        cs = max(1, min(int(anchor[1]), max_cols - c))
        share = max(1, longest_token_fn(txt) // cs) if cs > 1 else longest_token_fn(txt)
        for cc in range(c, min(max_cols, c + cs)):
            token_disp[cc] = max(token_disp[cc], share)

    total_pt = 0.0
    for c in range(max_cols):
        weight_disp = col_weight[c] if c < len(col_weight) else 0
        disp = max(int(weight_disp or 0), token_disp[c])
        if disp <= 0:
            continue
        # Padding + both cell margins, so the result is directly comparable to
        # the grid width `w` (which likewise spans border-to-border).
        total_pt += (disp + _COL_PAD_DISPLAY) * pt_per_display + 2 * CELL_MARGIN_PT

    if total_pt <= 0:
        return 0
    return int(round(total_pt * EMU_PER_PT))


def bbox_is_container(bbox_h_pt: float, total_lines: int,
                      line_h_pt: float) -> bool:
    """True when a bbox is far too tall for the rows it holds.

    The signal that the rectangle belongs to something ENCLOSING the table — a
    page frame, a form panel, a mislabelled ``Form`` block — rather than to the
    table itself. ``total_lines`` is the summed WRAPPED line count across rows
    (not the row count), so a short table of tall multi-line cells is correctly
    read as needing that height and is never flagged.

    Cheap and total: no geometry beyond the numbers `render_table` has already
    computed by this point.
    """
    if bbox_h_pt <= 0 or total_lines <= 0 or line_h_pt <= 0:
        return False
    return bbox_h_pt > total_lines * line_h_pt * BBOX_OVERSIZE_RATIO


def clamp_to_natural_width(
    x: int, w: int, natural_w: int,
) -> Tuple[int, int, bool]:
    """Narrow a table to its natural width and re-centre it in the given box.

    Applied only once `bbox_is_container` has established that the bbox is not
    the table's own rectangle. At that point the width is not the table's either,
    so it is replaced outright by what the content wants rather than being scaled
    by any ratio.

    Returns ``(x, w, clamped)``; when there is nothing to do — no natural width,
    or the box is already no wider than natural — the inputs come back UNCHANGED
    with ``False``, leaving the caller's pipeline bit-for-bit unaffected.

    Centring is within the ORIGINAL bbox, not the page: the container's position
    still tells you where the table lives, and centring on the page would throw a
    table out of its column in a multi-column layout.
    """
    if w <= 0 or natural_w <= 0 or natural_w >= w:
        return x, w, False
    return x + (w - natural_w) // 2, natural_w, True


# ── Obstacle-aware growth limits ─────────────────────────────────────────────
# A table may need to grow beyond its source bbox: translated text expands, and
# `render_table`'s grow-backstop would rather enlarge the table than clip a cell.
# The only bound that existed for that growth was the PAGE EDGE, on the height
# axis alone, so a grown table would run straight over whatever sat below it —
# the page number being the visible symptom.
#
# The general rule these two implement: a table grows from its FIXED TOP-LEFT
# origin until it meets either the page edge or the nearest neighbouring entry
# that overlaps it on the perpendicular axis, minus a small gap. Neither axis may
# grow into other layout.
#
# They are the axis-symmetric pair of the obstacle scans the TEXT path already
# trusts — `text_entry._downward_text_room_pt` (vertical) and
# `reflow._right_expansion_limit_px` (horizontal) — restated here in the shared,
# import-free module so both table.py copies get identical behaviour and it
# cannot drift from the text path.
#
# Both take the entry list as a plain sequence of dicts (not a ctx), keeping this
# module free of package imports and trivially unit-testable. Both return None
# for "unobstructed", which callers MUST treat as "no opinion" rather than zero —
# that is what makes an absent/empty entry list a strict no-op.

# Gap left between a grown table and its neighbour, in PIXELS at the page's zoom.
# Matches `_GAP_PX` in `text_entry._downward_text_room_pt`.
_OBSTACLE_GAP_PX = 4.0


def _own_bbox(entry) -> Optional[Tuple[float, float, float, float]]:
    """The growing entry's own rectangle, used to exclude its contained content."""
    try:
        ob = entry.get("bbox") if entry is not None else None
    except AttributeError:
        return None
    if not ob or len(ob) != 4:
        return None
    try:
        return (float(ob[0]), float(ob[1]), float(ob[2]), float(ob[3]))
    except (TypeError, ValueError):
        return None


def _bboxes(entries: Sequence, skip,
            own: Optional[Tuple[float, float, float, float]] = None,
            ) -> List[Tuple[float, float, float, float]]:
    """Valid 4-number bboxes from `entries`, excluding the entry `skip` itself
    and anything CLAIMED AS `skip`'s OWN cell content.

    The exclusion rule is essential, not a refinement. A table's own cell
    pictures are ordinary `Image` entries on the same page whose bboxes sit
    INSIDE the table's rectangle. Treated as neighbours they are the nearest
    thing to the right of the table's left edge, so the growth cap collapses the
    table to the gap before its own first picture — measured on a real document:
    a 710pt table capped to 58pt. Content a box contains cannot be an obstacle
    to that box.

    "Claimed" is `_table_owner_id == id(skip)` when the caller has set that
    marker (`json_to_docx.py` sets it once, when it decides which pictures a
    table's own HTML actually places via `<img>` cells) — NOT bare geometric
    containment. A picture can sit inside a table's outer bbox (its margin, a
    de-rotated frame, ...) without belonging to any of its cells; such a
    picture is a real obstacle and must still bound the table's width. Callers
    that never set the marker (or pass entries without it) fall back to pure
    containment, so this stays a drop-in for any caller unaware of ownership.

    Tolerant by design: entries with a missing/short/non-numeric bbox are simply
    not obstacles, because a malformed neighbour must never be able to shrink a
    table (the failure mode would be an unexplained squeeze, far worse than the
    overlap this guards against).
    """
    skip_id = id(skip) if skip is not None else None
    out: List[Tuple[float, float, float, float]] = []
    for other in entries or ():
        if other is skip:
            continue
        try:
            ob = other.get("bbox")
        except AttributeError:
            continue
        if not ob or len(ob) != 4:
            continue
        try:
            box = (float(ob[0]), float(ob[1]), float(ob[2]), float(ob[3]))
        except (TypeError, ValueError):
            continue
        if own is not None and (
            box[0] >= own[0] and box[2] <= own[2]
            and box[1] >= own[1] and box[3] <= own[3]
        ):
            has_owner_key = False
            try:
                has_owner_key = "_table_owner_id" in other
            except TypeError:
                pass
            if has_owner_key:
                # Caller resolved ownership explicitly: exclude only when
                # THIS table claimed it. `None` means "confirmed unowned"
                # (e.g. a standalone picture) and must NOT be excluded even
                # though it is geometrically inside `own`.
                if other.get("_table_owner_id") == skip_id:
                    continue
            else:
                continue                  # no ownership info: fall back to containment
        out.append(box)
    return out


def downward_room_pt(
    entries: Sequence, entry, x1: float, x2: float, y1: float, zoom: float,
) -> Optional[float]:
    """Vertical room (pt) from `y1` down to the nearest entry BELOW that
    horizontally overlaps the span ``[x1, x2]``, less a small gap.

    Horizontal overlap is required so an entry in another column is never an
    obstacle. Returns None when nothing is below — growth is then bounded only by
    the page, exactly as before.

    All inputs are in PIXELS (the frame the layout JSON uses); the result is in
    POINTS, ready to bound `_avail_pt`.
    """
    if zoom <= 0:
        return None
    gap = _OBSTACLE_GAP_PX * zoom
    nearest = None
    own = _own_bbox(entry)
    for ox1, oy1, ox2, oy2 in _bboxes(entries, entry, own):
        if oy1 <= y1:                      # not below our top edge
            continue
        if not (ox1 < x2 and ox2 > x1):    # no horizontal overlap
            continue
        if nearest is None or oy1 < nearest:
            nearest = oy1
    if nearest is None:
        return None
    return max(0.0, (nearest - gap - y1) / zoom)


def rightward_room_pt(
    entries: Sequence, entry, x1: float, y1: float, y2: float, zoom: float,
) -> Optional[float]:
    """Horizontal room (pt) from `x1` right to the nearest entry that starts to
    our right and shares our vertical band ``[y1, y2]``, less a small gap.

    Mirror of `downward_room_pt` on the other axis. The ``ox1 > x1`` test (rather
    than ``ox1 >= x2``) also catches a neighbour already straddling our right
    edge, so a table can never be widened further into something it is already
    touching.

    Returns None when nothing is to the right.
    """
    if zoom <= 0:
        return None
    gap = _OBSTACLE_GAP_PX * zoom
    nearest = None
    own = _own_bbox(entry)
    for ox1, oy1, ox2, oy2 in _bboxes(entries, entry, own):
        if ox1 <= x1:                      # not to our right
            continue
        if not (oy1 < y2 and oy2 > y1):    # no vertical band overlap
            continue
        if nearest is None or ox1 < nearest:
            nearest = ox1
    if nearest is None:
        return None
    return max(0.0, (nearest - gap - x1) / zoom)


# ── Dropped-column detection from picture geometry ───────────────────────────
# A picture must sit at least this fraction of a column width away from the
# column its `<img>` was parsed into before that counts as a real disagreement.
# Comfortably above sub-column jitter (a picture is inset inside its cell) and
# comfortably below one whole column.
_COL_SHIFT_TOLERANCE = 0.55


def dropped_column_index(
    pictures: Sequence,
    bbox: Sequence,
    img_cols: Sequence,
    max_cols: int,
    col_weight: Optional[Sequence] = None,
) -> Optional[int]:
    """Index of a column the OCR dropped, inferred from where pictures actually are.

    THE PROBLEM. A layout model can omit a table column entirely — typically a
    blank label column at the left edge, which carries no text for the model to
    notice. Every emitted row is then uniformly one cell too narrow, so
    `parse_table_grid`'s modal-width logic has nothing to disagree with and
    faithfully builds a grid one column short, with EVERY cell shifted.

    THE SIGNAL. Text inside a table gets no page entry of its own, so there is no
    text geometry to check against — but pictures do. Each `<img>` placeholder's
    parsed column says where the HTML thinks the picture is; the matching
    `Image` entry's bbox says where it physically IS. When the grid is missing a
    column, every picture sits a whole column-width to the RIGHT of where its
    `<img>` was placed. Measured on a real page: four diagrams parsed into
    column 0 while their bboxes sat squarely in column 1's band.

    `pictures` and `img_cols` are parallel: `img_cols[i]` is the grid column that
    `pictures[i]`'s placeholder was parsed into. Returns the index at which to
    INSERT one blank column, or None to leave the grid alone.

    Deliberately conservative — it reshapes a table only when the evidence is
    unanimous, so it is inert on every well-formed document:

    * needs at least one picture paired with an `<img>` cell (a table with no
      pictures has no signal, and is never touched);
    * EVERY picture must disagree, and by the SAME whole-column shift — one
      stray picture, or an inconsistent set, never reshapes a grid;
    * the shift must exceed `_COL_SHIFT_TOLERANCE` of a column, so a picture
      merely inset within its own cell does not count;
    * only ever reports a shift to the RIGHT (a missing column), because the
      caller may only INSERT — it never removes or re-orders columns.
    """
    if max_cols <= 0 or not pictures or not img_cols:
        return None
    if len(pictures) != len(img_cols):
        return None
    if len(bbox) != 4:
        return None
    x0 = float(bbox[0])
    width = float(bbox[2]) - x0
    if width <= 0:
        return None

    # Proportional column edges — the same space the picture assigner uses.
    if col_weight and len(col_weight) == max_cols and sum(col_weight) > 0:
        total = float(sum(col_weight))
        edges = [0.0]
        for cw in col_weight:
            edges.append(edges[-1] + float(cw) / total)
    else:
        edges = [i / max_cols for i in range(max_cols + 1)]

    shifts = []
    for pic, parsed_col in zip(pictures, img_cols):
        pb = (pic.get("_orig_bbox") or pic.get("bbox") or []) \
            if hasattr(pic, "get") else []
        if len(pb) != 4:
            return None
        if not (0 <= int(parsed_col) < max_cols):
            return None
        # Fractional position of the picture's centre across the table.
        frac = ((float(pb[0]) + float(pb[2])) / 2.0 - x0) / width
        # Which column does it physically land in?
        actual = None
        for c in range(max_cols):
            if edges[c] <= frac < edges[c + 1]:
                actual = c
                break
        if actual is None:                       # outside the table's own span
            return None
        # Measure the disagreement in COLUMN WIDTHS, so a picture merely inset
        # inside its cell reads as ~0 rather than as a shift.
        col_w = edges[int(parsed_col) + 1] - edges[int(parsed_col)]
        if col_w <= 0:
            return None
        centre_of_parsed = (edges[int(parsed_col)] + edges[int(parsed_col) + 1]) / 2.0
        if (frac - centre_of_parsed) / col_w < _COL_SHIFT_TOLERANCE:
            return None                          # this picture does NOT disagree
        shifts.append(actual - int(parsed_col))

    if not shifts:
        return None
    # Unanimous, rightward, and exactly one column: anything else is ambiguous
    # and the grid is left untouched.
    if len(set(shifts)) != 1 or shifts[0] != 1:
        return None
    # Insert before the leftmost column any picture was parsed into — that is
    # the boundary the whole grid is shifted across.
    return min(int(c) for c in img_cols)
