"""Rotation detection + rotated-table reconstruction tests.

Covers the three defects that let a rotated page render upright:

  1. `ooxml.build_anchored_textbox_xml` routed every quarter turn to
     `bodyPr vert=`, which reflows text inside cells but leaves a `<w:tbl>`
     grid upright — so a rotated table got a TRANSPOSED extent on a container
     that never actually turned. `rotate_shape=True` selects `a:xfrm rot=`.
  2. Nothing produced `rotation` outside diagram labels, so no entry or table
     was ever marked rotated in the first place.
  3. `_link_table_continuations` compared a POINT threshold against PIXEL
     coordinates, so on a zoomed scan no table ever counted as "at top of
     page" and multi-page tables never shared column widths.

Run:  python -m pytest ocr_service/tests/test_rotation.py -v
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
RECON_SRC = Path(__file__).resolve().parents[2] / "reconstruction_service" / "src"
for p in (str(SRC), str(RECON_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("docx", reason="python-docx required")

from ocr_reconstruction.ooxml import build_anchored_textbox_xml  # noqa: E402
from ocr_reconstruction.geometry import (  # noqa: E402
    normalize_rotation, rotated_shape_geometry,
)
import rotation_measure  # noqa: E402
from rotation_detect import detect_entry_rotations  # noqa: E402


def _box(rotation=0.0, rotate_shape=False):
    return build_anchored_textbox_xml(
        100, 200, 300, 400, "<w:p/>", 7,
        rotation=rotation, rotate_shape=rotate_shape,
    )


def _vert(xml):
    m = re.search(r'vert="([^"]+)"', xml)
    return m.group(1) if m else None


def _rot(xml):
    m = re.search(r'<a:xfrm rot="(\d+)"', xml)
    return int(m.group(1)) if m else None


# ── 1. The two mechanisms are mutually exclusive, and the caller picks ───────

def test_upright_emits_no_rotation_markup():
    """The load-bearing invariant: rotation=0 must be byte-identical to the
    pre-rotation output, whatever `rotate_shape` says."""
    base = build_anchored_textbox_xml(100, 200, 300, 400, "<w:p/>", 7)
    assert base == _box(0.0, False)
    assert base == _box(0.0, True)
    assert _vert(base) is None and _rot(base) is None


@pytest.mark.parametrize("deg,expected_vert", [(90.0, "vert270"), (270.0, "vert")])
def test_text_quarter_turn_uses_vert(deg, expected_vert):
    """Default path: a rotated LABEL turns its text and keeps box geometry."""
    xml = _box(deg, rotate_shape=False)
    assert _vert(xml) == expected_vert
    assert _rot(xml) is None


@pytest.mark.parametrize("deg,expected_rot", [(90.0, 16200000), (270.0, 5400000)])
def test_table_quarter_turn_uses_xfrm_rot(deg, expected_rot):
    """`rotate_shape=True` spins the SHAPE — the only way to turn a <w:tbl>.
    OOXML `rot` is clockwise-positive 60000ths, our schema is CCW-positive."""
    xml = _box(deg, rotate_shape=True)
    assert _rot(xml) == expected_rot
    assert _vert(xml) is None, "vert= and rot= must never both be emitted"


def test_non_quarter_angles_always_rotate_the_shape():
    """180° and arbitrary angles have no `vert` spelling, so both modes agree."""
    for deg in (180.0, 45.0):
        assert _rot(_box(deg, False)) == _rot(_box(deg, True))
        assert _vert(_box(deg, False)) is None


def test_near_quarter_turn_snaps_and_tiny_angle_is_upright():
    """Real detections land at 89.2/90.4, and 358° is noise around upright."""
    assert _vert(_box(89.2, False)) == "vert270"
    assert _rot(_box(89.2, True)) == 16200000
    assert _vert(_box(358.0, False)) is None and _rot(_box(358.0, False)) is None


def test_rotated_geometry_transposes_extent():
    """A quarter turn must hand back the transposed extent, centred on the same
    point — otherwise the shape lands offset by (w-h)/2 in both axes."""
    bbox = (0, 0, 400, 100)          # 4:1 landscape box, zoom 1
    x, y, w, h = rotated_shape_geometry(bbox, 90.0, 1.0, 600.0, 800.0)
    ux, uy, uw, uh = rotated_shape_geometry(bbox, 0.0, 1.0, 600.0, 800.0)
    assert (w, h) == (uh, uw), "extent should be transposed"
    assert x + w // 2 == ux + uw // 2, "centre must be preserved in x"
    assert y + h // 2 == uy + uh // 2, "centre must be preserved in y"


# ── 2. Detection ────────────────────────────────────────────────────────────

def _entry(cat, bbox, text):
    return {"category": cat, "bbox": list(bbox), "text": text}


def test_tall_narrow_entry_is_detected_as_quarter_turn():
    """The real measurements from the document that motivated this: a sidebar
    header and two operator captions, all recorded as if upright."""
    entries = [
        _entry("Section-Header", (2118, 1479, 2177, 2078), "GF3Z3001 带线锚钉主体-4.5"),
        _entry("Text", (236, 421, 310, 1069), "操作员: 张帅 2019.8.22"),
        _entry("Text", (174, 1472, 298, 2191), "检验员: 罗晓晨 2019.8.22"),
    ]
    assert detect_entry_rotations(entries) == 3
    assert all(e["rotation"] == 90.0 for e in entries)
    assert all(e["rotation_source"] == "bbox-aspect" for e in entries)


def test_upright_entries_are_left_untouched():
    """No `rotation` key at all on upright text — the JSON must stay
    byte-identical to pre-feature output for unrotated documents."""
    entries = [
        _entry("Text", (501, 2927, 779, 3035), "操作员: 张帅"),
        _entry("Caption", (888, 603, 1562, 662), "带线锚钉主体-4.5 外观检测报表"),
        _entry("Text", (642, 3130, 1220, 3249), "李明云 2019.8.22"),
    ]
    assert detect_entry_rotations(entries) == 0
    assert all("rotation" not in e for e in entries)


def test_single_glyph_is_never_rotated():
    """A one-character box is naturally square-ish; its aspect means nothing."""
    entries = [_entry("Text", (0, 0, 20, 200), "A")]
    assert detect_entry_rotations(entries) == 0
    assert "rotation" not in entries[0]


def test_tables_are_excluded_from_the_bbox_test():
    """A landscape table on a portrait page is WIDE, so the aspect test would
    both miss it and misfire on tall upright tables. Tables go through the
    OCR-majority path instead."""
    entries = [_entry("Table", (203, 217, 300, 3319), "<table><tr><td>x</td></tr></table>")]
    assert detect_entry_rotations(entries) == 0
    assert "rotation" not in entries[0]


def test_existing_rotation_is_not_overwritten():
    """Diagram labels arrive already stamped by `diagram_text_recovery`."""
    e = _entry("Text", (0, 0, 30, 300), "已经旋转")
    e["rotation"] = 270.0
    assert detect_entry_rotations([e]) == 0
    assert e["rotation"] == 270.0


def test_kill_switch_disables_detection(monkeypatch):
    monkeypatch.setattr(rotation_measure, "ROT_ENABLE", False)
    entries = [_entry("Text", (236, 421, 310, 1069), "操作员: 张帅 2019.8.22")]
    assert detect_entry_rotations(entries) == 0
    assert "rotation" not in entries[0]


def test_line_rotation_matches_documented_measurements():
    """The populations quoted in `rotation_measure`: rotated labels measured
    aspect 3.2-6.2, horizontal ones 0.12-0.13."""
    assert rotation_measure.line_rotation((0, 0, 24, 148), "生产和生产后信息") == 90.0
    assert rotation_measure.line_rotation((0, 0, 163, 19), "预期用途、识别特征") == 0.0
    # A genuinely skewed quad reports its own tilt, not a quarter turn.
    skew = rotation_measure.line_rotation(
        (0, 0, 100, 30), "skewed", [(0, 0), (100, -20), (100, 10), (0, 30)],
    )
    assert 5.0 <= skew <= 60.0


def test_modal_rotation_picks_majority_and_breaks_ties_low():
    assert rotation_measure.modal_rotation([90.0, 90.0, 0.0]) == 90.0
    assert rotation_measure.modal_rotation([]) == 0.0
    assert rotation_measure.modal_rotation([0.0, 90.0]) == 0.0


# ── 3. Rotated table renders a rotated container, end to end ────────────────

_TBL = (
    "<table><thead><tr><th>序号</th><th>尺寸</th></tr></thead>"
    "<tbody><tr><td>13</td><td>15.51</td></tr>"
    "<tr><td>14</td><td>15.52</td></tr></tbody></table>"
)


def _render_page(rotation=None):
    from docx import Document
    from ocr_reconstruction.shape_context import ShapeContext
    from ocr_reconstruction.table import render_table

    entry = {"category": "Table", "bbox": [310, 175, 2220, 3312], "text": _TBL}
    if rotation is not None:
        entry["rotation"] = rotation
    ctx = ShapeContext(
        Document(), page_w_pt=595.0, page_h_pt=841.0,
        zoom=4.1666667, shape_id_start=1000,
    )
    render_table(ctx, entry)
    return "".join(ctx.xml_chunks)


def test_rotated_table_can_emit_shape_rotation():
    """`rotate_shape` does select the shape-rotating OOXML. Kept as a unit
    guarantee even though the PAGE-LEVEL de-rotation below is what actually
    ships — renderers ignore `a:xfrm rot=` on a shape holding a `<w:tbl>`
    (measured: 255 table cells came back upright), so this path alone cannot
    deliver a rotated table."""
    xml = _render_page(rotation=90.0)
    assert _rot(xml) == 16200000
    assert 'vert="' not in xml
    assert "<w:tbl>" in xml


def test_upright_table_emits_no_rotation():
    xml = _render_page(rotation=None)
    assert _rot(xml) is None and 'vert="' not in xml
    assert "<w:tbl>" in xml


# ── 3b. Page-level de-rotation: what actually makes rotation render ─────────

def _rot_page(entries, w=595.0, h=841.0, zoom=4.1666667):
    return {
        "page_index": 0, "page_width_pt": w, "page_height_pt": h,
        "zoom": zoom, "entries": entries,
    }


def test_rotated_page_becomes_landscape_and_content_upright():
    """A page whose content is mostly rotated is turned into an upright
    LANDSCAPE page: axes swap, bboxes map into the new frame, and no entry
    keeps a `rotation` (which would double-rotate it)."""
    from ocr_reconstruction.json_to_docx import _derotate_rotated_pages

    tbl = {"category": "Table", "bbox": [310, 175, 2220, 3312],
           "text": _TBL, "rotation": 90.0}
    page = _rot_page([tbl])
    _derotate_rotated_pages([page])

    assert page["page_width_pt"] == 841.0, "width/height must swap"
    assert page["page_height_pt"] == 595.0
    assert page["page_rotation_applied"] == 90.0
    assert "rotation" not in tbl, "page carries the turn now"
    assert tbl["derotated_with_page"] is True
    # The table was 1910x3137 px (portrait-tall); after mapping it must be
    # wider than tall, i.e. it fits the landscape page.
    x1, y1, x2, y2 = tbl["bbox"]
    assert (x2 - x1) > (y2 - y1)


def test_upright_page_is_untouched_by_derotation():
    from ocr_reconstruction.json_to_docx import _derotate_rotated_pages

    e = {"category": "Table", "bbox": [273, 508, 2113, 2646], "text": _TBL}
    page = _rot_page([e])
    before = list(e["bbox"])
    _derotate_rotated_pages([page])
    assert page["page_width_pt"] == 595.0 and page["page_height_pt"] == 841.0
    assert e["bbox"] == before
    assert "page_rotation_applied" not in page


def test_few_rotated_labels_do_not_flip_the_page():
    """A couple of turned side-labels on an otherwise upright page must keep
    their per-entry `vert=` rendering, not flip the whole section."""
    from ocr_reconstruction.json_to_docx import _derotate_rotated_pages

    big = {"category": "Table", "bbox": [200, 200, 2200, 3200], "text": _TBL}
    label = {"category": "Section-Header", "bbox": [2118, 1479, 2177, 2078],
             "text": "GF3Z3001", "rotation": 90.0}
    page = _rot_page([big, label])
    _derotate_rotated_pages([page])
    assert page["page_width_pt"] == 595.0, "page must stay portrait"
    assert label["rotation"] == 90.0, "label keeps its own rotation"


def test_derotation_maps_bbox_within_the_new_page_bounds():
    """Every mapped bbox must land inside the swapped page extent."""
    from ocr_reconstruction.json_to_docx import _derotate_rotated_pages

    zoom = 4.1666667
    entries = [
        {"category": "Table", "bbox": [310, 175, 2220, 3312], "text": _TBL,
         "rotation": 90.0},
        {"category": "Text", "bbox": [236, 421, 310, 1069], "text": "操作员",
         "rotation": 90.0},
    ]
    page = _rot_page(entries, zoom=zoom)
    _derotate_rotated_pages([page])
    pw = page["page_width_pt"] * zoom
    ph = page["page_height_pt"] * zoom
    for e in entries:
        x1, y1, x2, y2 = e["bbox"]
        assert -1 <= x1 < x2 <= pw + 1, f"x out of range: {e['bbox']} vs {pw}"
        assert -1 <= y1 < y2 <= ph + 1, f"y out of range: {e['bbox']} vs {ph}"


def test_translation_copy_derotates_identically():
    from translation_reconstruction.json_to_docx import _derotate_rotated_pages

    tbl = {"category": "Table", "bbox": [310, 175, 2220, 3312],
           "text": _TBL, "rotation": 90.0}
    page = _rot_page([tbl])
    _derotate_rotated_pages([page])
    assert (page["page_width_pt"], page["page_height_pt"]) == (841.0, 595.0)
    assert "rotation" not in tbl


# ── 4. Table continuation chain links across pages (unit bug) ───────────────

def _table_page(top, has_thead, zoom=4.1666667):
    head = "<thead><tr><th>a</th><th>b</th></tr></thead>" if has_thead else ""
    body = "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
    return {
        "page_index": 0, "page_width_pt": 595.0, "page_height_pt": 841.0,
        "zoom": zoom,
        "entries": [{
            "category": "Table",
            "bbox": [200, top, 2100, top + 3000],
            "text": f"<table>{head}{body}</table>",
        }],
    }


def test_continuation_links_when_table_is_at_top_of_zoomed_page():
    """A table 42pt from the top measures 175px at zoom 4.17. Comparing that
    against a bare `60` made every page fail, so no chain ever formed."""
    from ocr_reconstruction.json_to_docx import _link_table_continuations

    pages = [_table_page(217, True), _table_page(175, False)]
    _link_table_continuations(pages)

    head = pages[0]["entries"][0]
    cont = pages[1]["entries"][0]
    assert len(head.get("_table_chain") or []) == 2, "pages should form one chain"
    assert head.get("_shared_col_weights"), "head must carry shared weights"
    assert head["_shared_col_weights"] == cont["_shared_col_weights"], (
        "continuation must reuse the head's column widths"
    )


def test_table_far_down_the_page_is_not_a_continuation():
    """The threshold must still reject a table that genuinely starts mid-page."""
    from ocr_reconstruction.json_to_docx import _link_table_continuations

    pages = [_table_page(217, True), _table_page(1800, False)]
    _link_table_continuations(pages)
    assert len(pages[0]["entries"][0].get("_table_chain") or []) == 1


def test_thead_starts_a_new_logical_table():
    from ocr_reconstruction.json_to_docx import _link_table_continuations

    pages = [_table_page(217, True), _table_page(175, True)]
    _link_table_continuations(pages)
    assert len(pages[0]["entries"][0].get("_table_chain") or []) == 1


def test_translation_copy_links_identically():
    """Both reconstruction packages carry the fix — they are independent copies."""
    from translation_reconstruction.json_to_docx import _link_table_continuations

    pages = [_table_page(217, True), _table_page(175, False)]
    _link_table_continuations(pages)
    assert len(pages[0]["entries"][0].get("_table_chain") or []) == 2


def test_normalize_rotation_is_robust_to_junk():
    for junk in (None, "", "abc", float("nan"), float("inf")):
        assert normalize_rotation(junk) == 0.0
    assert normalize_rotation("90") == 90.0
    assert normalize_rotation(450.0) == 90.0
