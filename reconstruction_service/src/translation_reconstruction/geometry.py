"""Geometry plumbing: EMU/pt constants, bbox conversion, page normalisation,
and per-page section setup. Pure helpers — no OOXML strings here."""
from __future__ import annotations

from typing import Dict

from docx import Document
from docx.shared import Emu


EMU_PER_PT = 12700        # OOXML uses English Metric Units: 1 pt = 12700 EMU
EMU_PER_INCH = 914400
DEFAULT_PAGE_WIDTH_PT = 612.0   # US Letter
DEFAULT_PAGE_HEIGHT_PT = 792.0


def normalize_page(page_or_entries) -> Dict:
    """Accept the new {entries, page_width_pt, ...} shape OR a raw list of
    entries (legacy / image fallback). Returns a normalized dict."""
    if isinstance(page_or_entries, dict):
        entries = page_or_entries.get("entries")
        if entries is None and "layout_result" in page_or_entries:
            entries = page_or_entries.get("layout_result")
        return {
            "entries": entries or [],
            "page_width_pt": page_or_entries.get("page_width_pt") or DEFAULT_PAGE_WIDTH_PT,
            "page_height_pt": page_or_entries.get("page_height_pt") or DEFAULT_PAGE_HEIGHT_PT,
            "zoom": page_or_entries.get("zoom") or 1.0,
            "page_index": page_or_entries.get("page_index", 0),
            # carry image_obj through if upstream attached it (Picture render)
            "_raw": page_or_entries,
        }
    if isinstance(page_or_entries, list):
        return {
            "entries": page_or_entries,
            "page_width_pt": DEFAULT_PAGE_WIDTH_PT,
            "page_height_pt": DEFAULT_PAGE_HEIGHT_PT,
            "zoom": 1.0,
            "page_index": 0,
            "_raw": {},
        }
    return {
        "entries": [],
        "page_width_pt": DEFAULT_PAGE_WIDTH_PT,
        "page_height_pt": DEFAULT_PAGE_HEIGHT_PT,
        "zoom": 1.0,
        "page_index": 0,
        "_raw": {},
    }


def add_section_for_page(doc: Document, width_pt: float, height_pt: float,
                         first: bool):
    """Add (or configure) a section sized to the original page with zero
    margins. The first page reuses the default section; subsequent pages
    add a new one preceded by a page break."""
    if first:
        section = doc.sections[0]
    else:
        # New section starts on a new page automatically.
        section = doc.add_section()
    section.page_width = Emu(int(round(width_pt * EMU_PER_PT)))
    section.page_height = Emu(int(round(height_pt * EMU_PER_PT)))
    section.left_margin = Emu(0)
    section.right_margin = Emu(0)
    section.top_margin = Emu(0)
    section.bottom_margin = Emu(0)
    section.header_distance = Emu(0)
    section.footer_distance = Emu(0)
    section.gutter = Emu(0)
    # Each section gets its own header/footer so per-page text routed there
    # doesn't bleed into other pages.
    section.different_first_page_header_footer = False
    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False
    return section


def alignment_for_bbox(bbox, page_w_pt: float, zoom: float) -> str:
    """Pick left/center/right based on where the bbox sits horizontally."""
    if not bbox or len(bbox) != 4:
        return "left"
    x1, _y1, x2, _y2 = bbox
    cx_pt = ((x1 + x2) / 2.0) / max(1e-6, zoom)
    third = page_w_pt / 3.0
    if cx_pt < third:
        return "left"
    if cx_pt > 2 * third:
        return "right"
    return "center"


# ── Rotation ────────────────────────────────────────────────────────────────
# Layout entries may carry an optional `rotation`: FLOAT DEGREES,
# COUNTER-CLOCKWISE-POSITIVE, where absent/0 means upright. The CJK side-labels
# that motivated this (glyphs turned 90° clockwise, reading bottom-to-top) are
# `rotation = 90.0`. Every reader must go through `normalize_rotation` so a
# missing key, None, a string, and 0 all collapse to the same upright no-op.

# Detections land near-but-not-exactly on a quarter turn (measured: -0.73°,
# and real scans give 89.2/90.4). Snapping inside this tolerance is what lets
# the common case use the cheap `bodyPr vert=` renderer — which preserves the
# box geometry — instead of the shape-rotating `a:xfrm rot=` path.
ROTATION_SNAP_TOLERANCE_DEG = 5.0


def normalize_rotation(value) -> float:
    """Coerce `value` to degrees in [0, 360), snapping near-quarter-turns.

    Returns 0.0 for None/""/non-numeric, so `entry.get("rotation")` on an
    upright (or pre-rotation-feature) entry is always safe.
    """
    try:
        deg = float(value)
    except (TypeError, ValueError):
        return 0.0
    if deg != deg or deg in (float("inf"), float("-inf")):  # NaN / inf
        return 0.0
    deg %= 360.0
    for quarter in (0.0, 90.0, 180.0, 270.0, 360.0):
        if abs(deg - quarter) <= ROTATION_SNAP_TOLERANCE_DEG:
            return quarter % 360.0
    return deg


def is_quarter_turn(rotation) -> bool:
    """True for exactly 90°/270° — the orientations `bodyPr vert=` can express.

    180° is deliberately excluded: there is no `vert` value for it, so it takes
    the `a:xfrm rot=` path along with arbitrary angles.
    """
    return normalize_rotation(rotation) in (90.0, 270.0)


def is_upright(rotation) -> bool:
    """True when no rotation handling is needed at all (the default path)."""
    return normalize_rotation(rotation) == 0.0


def rotated_shape_geometry(bbox_px, rotation, zoom: float,
                           page_w_pt: float, page_h_pt: float):
    """EMU position+extent for a shape rotated by `rotation` about its centre.

    `a:xfrm rot=` spins the shape around its own centre, and Word/LibreOffice
    read `wp:extent` as the UNROTATED extent. So to make a rotated shape cover
    the source bbox we emit the transposed extent and shift the anchor by half
    the difference — otherwise a tall-narrow label lands rotated but offset by
    (w-h)/2 in both axes.

    Upright and 180° rotations need no transposition and fall through to plain
    `bbox_px_to_emu`. Clamping happens against the shape's real on-page
    footprint, which for a quarter turn is the transposed rectangle.
    """
    rot = normalize_rotation(rotation)
    if not is_quarter_turn(rot):
        return bbox_px_to_emu(bbox_px, zoom, page_w_pt, page_h_pt)

    x1, y1, x2, y2 = bbox_px
    x_pt = max(0.0, x1 / zoom)
    y_pt = max(0.0, y1 / zoom)
    w_pt = max(1.0, (x2 - x1) / zoom)
    h_pt = max(1.0, (y2 - y1) / zoom)

    # Centre is rotation-invariant; rebuild the unrotated (transposed) box around it.
    cx_pt = x_pt + w_pt / 2.0
    cy_pt = y_pt + h_pt / 2.0
    rw_pt, rh_pt = h_pt, w_pt
    rx_pt = cx_pt - rw_pt / 2.0
    ry_pt = cy_pt - rh_pt / 2.0

    # Clamp the FOOTPRINT (the original w/h box), then re-derive the offset so
    # the shape stays centred on the same point.
    if x_pt + w_pt > page_w_pt:
        cx_pt = min(cx_pt, max(w_pt / 2.0, page_w_pt - w_pt / 2.0))
    if y_pt + h_pt > page_h_pt:
        cy_pt = min(cy_pt, max(h_pt / 2.0, page_h_pt - h_pt / 2.0))
    rx_pt = cx_pt - rw_pt / 2.0
    ry_pt = cy_pt - rh_pt / 2.0

    return (
        int(round(rx_pt * EMU_PER_PT)),
        int(round(ry_pt * EMU_PER_PT)),
        int(round(rw_pt * EMU_PER_PT)),
        int(round(rh_pt * EMU_PER_PT)),
    )


def bbox_px_to_emu(bbox_px, zoom: float, page_w_pt: float, page_h_pt: float):
    """Convert a 2x-pixel bbox to EMU position+extent, clamped to the page."""
    x1, y1, x2, y2 = bbox_px
    x_pt = max(0.0, x1 / zoom)
    y_pt = max(0.0, y1 / zoom)
    w_pt = max(1.0, (x2 - x1) / zoom)
    h_pt = max(1.0, (y2 - y1) / zoom)
    # Clamp so shapes don't run off the page (Word may render off-page,
    # but viewers handle on-page geometry better).
    if x_pt > page_w_pt:
        x_pt = max(0.0, page_w_pt - 1.0)
    if y_pt > page_h_pt:
        y_pt = max(0.0, page_h_pt - 1.0)
    if x_pt + w_pt > page_w_pt:
        w_pt = max(1.0, page_w_pt - x_pt)
    if y_pt + h_pt > page_h_pt:
        h_pt = max(1.0, page_h_pt - y_pt)
    return (
        int(round(x_pt * EMU_PER_PT)),
        int(round(y_pt * EMU_PER_PT)),
        int(round(w_pt * EMU_PER_PT)),
        int(round(h_pt * EMU_PER_PT)),
    )
