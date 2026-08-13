"""De-rotated table re-OCR: angle mapping, acceptance guard, no-op safety.

WHY THIS PASS EXISTS. A landscape table scanned onto a portrait sheet reaches the
model sideways, its 25 columns squeezed across the page's short axis (~64px each
for 5-character handwriting). The model drops cells; the reconstruction grid then
fills left-to-right so the survivors shift one column left and land under the
WRONG header, with the final column empty. Measured on the live model, same crop:
0/12 body rows at full width as-scanned vs 12/12 de-rotated.

WHAT THESE TESTS PROTECT. The acceptance guard is the load-bearing part: a re-OCR
must be able to ADD a dropped value but must NEVER silently rewrite a measurement
(`15.53` -> `15.58` on an inspection report is worse than the hole it fixes), lose
a row, or narrow the grid. And the transpose direction must come from the detected
angle — rotating the wrong way was measured to make the output worse (18-wide rows).

Run:  python -m pytest ocr_service/tests/test_table_derotate.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
RECON_SRC = Path(__file__).resolve().parents[2] / "reconstruction_service" / "src"
for p in (str(SRC), str(RECON_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("PIL", reason="Pillow required")
from PIL import Image  # noqa: E402

# `table_derotate_reocr` imports chandra_ocr (vLLM client) at module scope; skip
# cleanly where that dependency is unavailable rather than failing the suite.
tdr = pytest.importorskip(
    "table_derotate_reocr", reason="chandra_ocr/vLLM deps required")


def _table(rows, header=None, cols=None):
    """Build table HTML. `header` is a list of (text, colspan) pairs."""
    html = ["<table border='1'>"]
    if header:
        html.append("<thead><tr>")
        for text, cs in header:
            span = f" colspan='{cs}'" if cs > 1 else ""
            html.append(f"<th{span}>{text}</th>")
        html.append("</tr></thead>")
    html.append("<tbody>")
    for r in rows:
        html.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    html.append("</tbody></table>")
    return "".join(html)


# ── Angle -> transpose. Direction is derived, never hardcoded. ──────────────

def test_quarter_turns_map_to_a_transpose():
    assert tdr.upright_transpose(90.0) == Image.ROTATE_90
    assert tdr.upright_transpose(270.0) == Image.ROTATE_270


def test_near_quarter_turns_snap():
    """Real detections land at 89.2 / 90.4, not exactly 90."""
    assert tdr.upright_transpose(89.2) == Image.ROTATE_90
    assert tdr.upright_transpose(90.4) == Image.ROTATE_90
    assert tdr.upright_transpose(271.0) == Image.ROTATE_270


def test_upright_and_unmeasurable_angles_do_nothing():
    for value in (None, 0, 0.0, "", "abc", float("nan"), 358.0, 2.0):
        assert tdr.upright_transpose(value) is None, value


def test_180_is_excluded_on_purpose():
    """Upside-down text leaves COLUMN geometry unchanged, so a re-OCR gains
    nothing on the axis this module exists to fix."""
    assert tdr.upright_transpose(180.0) is None


def test_transpose_actually_makes_a_landscape_crop_upright():
    """Sanity check on the geometry, not just the constant: a tall-narrow crop
    (a landscape table stored sideways) must come back wide."""
    sideways = Image.new("RGB", (100, 300))
    upright = sideways.transpose(tdr.upright_transpose(90.0))
    assert upright.size == (300, 100)


# ── Grid shape measurement ──────────────────────────────────────────────────

def test_grid_shape_counts_full_width_rows():
    html = _table([["1", "2", "3"], ["4", "5", "6"]])
    cols, rows, full = tdr._grid_shape(html)
    assert (cols, rows, full) == (3, 2, 2)


def test_grid_shape_sees_a_short_row_as_not_full():
    html = _table([["1", "2", "3"], ["4", "5"]])
    cols, rows, full = tdr._grid_shape(html)
    assert cols == 3 and rows == 2
    assert full == 1, "the 2-cell row does not cover the 3-column grid"


def test_grid_shape_is_safe_on_junk():
    assert tdr._grid_shape("") == (0, 0, 0)
    assert tdr._grid_shape("not a table") == (0, 0, 0)


# ── Content preservation: the guard that matters most ───────────────────────

def test_signatures_ignore_formatting_but_keep_values():
    a = tdr._cell_signatures(_table([["15.53", "ok"]]))
    b = tdr._cell_signatures(_table([[" 15.53 ", "OK"]]))
    assert a == b, "whitespace/case must not change the signature"


def test_adding_a_missing_cell_is_accepted():
    old = tdr._cell_signatures(_table([["15.53", "15.51", "15.54"]]))
    new = tdr._cell_signatures(_table([["15.53", "15.51", "15.52", "15.54"]]))
    assert tdr._is_superset(old, new), "a recovered value must be allowed"


def test_rewriting_a_measurement_is_rejected():
    """THE critical guard: 15.53 -> 15.58 is data corruption, not a repair."""
    old = tdr._cell_signatures(_table([["15.53", "15.51"]]))
    new = tdr._cell_signatures(_table([["15.58", "15.51"]]))
    assert not tdr._is_superset(old, new)


def test_dropping_a_duplicate_is_rejected():
    """Multiset, not set — for the raw containment helper."""
    old = tdr._cell_signatures(_table([["ok", "ok", "ok"]]))
    new = tdr._cell_signatures(_table([["ok", "ok"]]))
    assert not tdr._is_superset(old, new)


# ── _content_preserved: the guard the pass actually uses ────────────────────
#
# A strict superset test was measured to be WRONG here. When the sideways read
# drops a cell, the left-shift DUPLICATES a neighbouring group's values into the
# gap; the de-rotated read fixes the shift, so those spurious copies correctly
# disappear. Real counts on the affected page: φ4.51 16->15, φ4.52 15->14,
# φ4.53 19->17, φ4.54 14->13 — the 尺寸2 group falling from 64 to its correct 60
# while 尺寸1 rose 60->64 and ok rose 60->72 (all 12 结论 values recovered).

def test_removing_misplaced_duplicates_is_accepted():
    """The real repair: a few copies of a repeated value go away because they
    were never supposed to be there."""
    old = _table([["φ4.51", "φ4.51", "φ4.51", "15.53"]])
    new = _table([["φ4.51", "φ4.51", "15.53", "15.52"]])
    assert tdr._content_preserved(old, new)


def test_a_value_vanishing_entirely_is_rejected():
    """A rewritten measurement makes its old value disappear outright — the
    failure this guard exists to stop."""
    old = _table([["15.53", "15.51"]])
    new = _table([["15.58", "15.51"]])
    assert not tdr._content_preserved(old, new)


def test_a_large_multiplicity_drop_is_rejected():
    """Losing many copies is data loss, not de-duplication."""
    old = _table([["ok"] * 12])
    new = _table([["ok", "ok"]])
    assert not tdr._content_preserved(old, new)


def test_reordering_alone_is_tolerated():
    """Signatures are a sorted multiset, so a pure reorder preserves content —
    the column-count checks in the guard catch structural change instead."""
    old = tdr._cell_signatures(_table([["a1", "b2"]]))
    new = tdr._cell_signatures(_table([["b2", "a1"]]))
    assert tdr._is_superset(old, new)


# ── The pass is a strict no-op without rotation or a raster ─────────────────

def _run(pages, enabled=True):
    """Drive the pass. `enabled` forces the gate on because the SHIPPING default
    is off (`OCR_ROTATION=false`) — these tests exercise the pass itself, so they
    must not depend on the deployment default either way. The switch's own
    behaviour is covered by `test_master_switch_*`."""
    import asyncio
    prev = tdr.REOCR_ROTATED_TABLES
    tdr.REOCR_ROTATED_TABLES = enabled
    try:
        return asyncio.new_event_loop().run_until_complete(
            tdr.reocr_rotated_tables(pages))
    finally:
        tdr.REOCR_ROTATED_TABLES = prev


def test_upright_tables_are_never_touched(monkeypatch):
    """No rotation stamped -> no candidates -> no model call at all."""
    called = []

    async def _boom(img):
        called.append(img)
        raise AssertionError("must not OCR an upright table")

    monkeypatch.setattr(tdr, "ocr_image_async", _boom)
    html = _table([["1", "2"]])
    page = {
        "page_index": 0,
        "original_image": Image.new("RGB", (200, 200)),
        "layout_result": [{"category": "Table", "bbox": [0, 0, 100, 100],
                           "text": html}],
    }
    _run([page])
    assert not called
    assert page["layout_result"][0]["text"] == html


def test_rotated_table_without_a_raster_is_skipped(monkeypatch):
    """The raster is freed later in the pipeline; a missing one must not crash."""
    async def _boom(img):
        raise AssertionError("must not OCR without a raster")

    monkeypatch.setattr(tdr, "ocr_image_async", _boom)
    page = {
        "page_index": 0,
        "layout_result": [{"category": "Table", "bbox": [0, 0, 100, 100],
                           "text": _table([["1", "2"]]), "rotation": 90.0}],
    }
    _run([page])          # must not raise


def test_kill_switch_disables_the_pass(monkeypatch):
    async def _boom(img):
        raise AssertionError("kill switch must prevent any OCR")

    monkeypatch.setattr(tdr, "ocr_image_async", _boom)
    page = {
        "page_index": 0,
        "original_image": Image.new("RGB", (200, 200)),
        "layout_result": [{"category": "Table", "bbox": [0, 0, 100, 100],
                           "text": _table([["1", "2"]]), "rotation": 90.0}],
    }
    # enabled=False is the gate under test — `_run` would otherwise force it on.
    _run([page], enabled=False)


# ── End-to-end guard behaviour with a stubbed model ─────────────────────────

def _rotated_page(html):
    return {
        "page_index": 3,
        "original_image": Image.new("RGB", (400, 800)),
        "layout_result": [{"category": "Table", "bbox": [10, 10, 300, 700],
                           "text": html, "rotation": 90.0}],
    }


def _stub(monkeypatch, returned_html):
    async def _fake(img):
        return [{"category": "Table", "text": returned_html}]
    monkeypatch.setattr(tdr, "ocr_image_async", _fake)


def test_a_wider_grid_is_accepted(monkeypatch):
    """The page-2 case: every row was internally consistent but the whole grid
    resolved one column too narrow (24 instead of 25)."""
    old = _table([["a1", "b2", "c3"], ["d4", "e5", "f6"]])
    new = _table([["a1", "b2", "c3", "z9"], ["d4", "e5", "f6", "y8"]])
    _stub(monkeypatch, new)
    page = _rotated_page(old)
    _run([page])
    entry = page["layout_result"][0]
    assert entry["text"] == new
    assert entry["source"] == "derotate-reocr"


def test_more_full_width_rows_is_accepted(monkeypatch):
    """The page-1 case: grid width already right, but rows were short."""
    old = _table([["a1", "b2", "c3"], ["d4", "e5"]])
    new = _table([["a1", "b2", "c3"], ["d4", "e5", "f6"]])
    _stub(monkeypatch, new)
    page = _rotated_page(old)
    _run([page])
    assert page["layout_result"][0]["text"] == new


def test_a_narrower_grid_is_rejected(monkeypatch):
    old = _table([["a1", "b2", "c3"], ["d4", "e5", "f6"]])
    new = _table([["a1", "b2"], ["d4", "e5"]])
    _stub(monkeypatch, new)
    page = _rotated_page(old)
    _run([page])
    assert page["layout_result"][0]["text"] == old
    assert "source" not in page["layout_result"][0]


def test_lost_rows_are_rejected(monkeypatch):
    old = _table([["a1", "b2"], ["c3", "d4"], ["e5", "f6"]])
    new = _table([["a1", "b2"], ["c3", "d4"]])
    _stub(monkeypatch, new)
    page = _rotated_page(old)
    _run([page])
    assert page["layout_result"][0]["text"] == old


def test_a_rewritten_value_is_rejected_even_if_wider(monkeypatch):
    """Structural improvement must NOT buy permission to change a measurement."""
    old = _table([["15.53", "15.51", "15.54"]])
    new = _table([["15.58", "15.51", "15.52", "15.54"]])   # wider, but 53 -> 58
    _stub(monkeypatch, new)
    page = _rotated_page(old)
    _run([page])
    assert page["layout_result"][0]["text"] == old, (
        "a re-OCR that rewrites a value must be refused")


def test_the_real_world_repair_shape_is_accepted(monkeypatch):
    """End-to-end shape of the measured fix: the grid gains full-width rows AND
    a few misplaced duplicates disappear. This must be ACCEPTED — an earlier
    strict-superset guard refused it, leaving the table broken."""
    old = _table([["15.53", "15.51", "φ4.51", "φ4.51"],
                  ["15.52", "15.54", "φ4.53"]])          # row 2 short
    new = _table([["15.53", "15.51", "15.52", "φ4.51"],
                  ["15.52", "15.54", "15.53", "φ4.53"]])  # complete, de-shifted
    _stub(monkeypatch, new)
    page = _rotated_page(old)
    _run([page])
    entry = page["layout_result"][0]
    assert entry["text"] == new, "the real repair shape must be accepted"
    assert entry["source"] == "derotate-reocr"


def test_no_improvement_keeps_the_original(monkeypatch):
    same = _table([["a1", "b2"], ["c3", "d4"]])
    _stub(monkeypatch, same)
    page = _rotated_page(same)
    _run([page])
    assert page["layout_result"][0]["text"] == same
    assert "source" not in page["layout_result"][0]


def test_ocr_failure_is_swallowed(monkeypatch):
    """One bad table must never fail the page."""
    async def _raise(img):
        raise RuntimeError("vLLM exploded")
    monkeypatch.setattr(tdr, "ocr_image_async", _raise)
    old = _table([["a1", "b2"]])
    page = _rotated_page(old)
    _run([page])
    assert page["layout_result"][0]["text"] == old


def test_reocr_returning_no_table_keeps_the_original(monkeypatch):
    async def _no_table(img):
        return [{"category": "Text", "text": "just prose"}]
    monkeypatch.setattr(tdr, "ocr_image_async", _no_table)
    old = _table([["a1", "b2"]])
    page = _rotated_page(old)
    _run([page])
    assert page["layout_result"][0]["text"] == old


# ── Collapsed baseline (the `cols 40->25, rows 1->15` rejections) ────────────

def test_collapsed_baseline_may_be_narrowed(monkeypatch):
    """When the sideways read finds NO row structure, every cell runs together
    into one row that saturates MAX_TABLE_COLS. Requiring `new_cols >= old_cols`
    then let the BROKEN parse win purely because 40 > 25. Observed on two real
    pages as `cols 40->25, rows 1->15`."""
    # The re-read finds the SAME values, only arranged into real rows — that is
    # what a genuine repair looks like, and it keeps `_content_preserved` happy.
    # Rows must also be DISTINCT: `parse_table_grid` collapses byte-identical
    # rows, which would make the fixture measure 1 row and mask the behaviour.
    values = [f"v{i}" for i in range(tdr._COLLAPSED_COLS)]   # saturates the cap
    wide_single_row = _table([values])
    structured = _table([values[r * 5:(r + 1) * 5] for r in range(8)])
    _stub(monkeypatch, structured)
    page = _rotated_page(wide_single_row)
    _run([page])
    entry = page["layout_result"][0]
    assert entry["text"] == structured
    assert entry["source"] == "derotate-reocr"


def test_a_normal_baseline_still_may_not_be_narrowed(monkeypatch):
    """The collapsed-baseline exception must NOT become a general licence to
    narrow: with real rows present, fewer columns is a worse reading."""
    old = _table([["a1", "b2", "c3", "d4"], ["e5", "f6", "g7", "h8"]])
    narrower = _table([["a1", "b2"], ["e5", "f6"]])
    _stub(monkeypatch, narrower)
    page = _rotated_page(old)
    _run([page])
    assert page["layout_result"][0]["text"] == old
    assert "source" not in page["layout_result"][0]


def test_collapsed_exception_still_requires_content_preserved(monkeypatch):
    """A collapsed baseline relaxes the GEOMETRY checks only. A re-read that
    rewrites a measurement is still refused."""
    wide_single_row = _table([[f"1{i}.5{i}" for i in range(tdr._COLLAPSED_COLS)]])
    rewritten = _table([["99.99", "88.88"], ["77.77", "66.66"]])
    _stub(monkeypatch, rewritten)
    page = _rotated_page(wide_single_row)
    _run([page])
    assert page["layout_result"][0]["text"] == wide_single_row


# ── Proportional drop tolerance (the 30-column `full 1->15` rejections) ──────

def test_drop_tolerance_scales_with_multiplicity():
    """A flat cap of 3 was calibrated on 25-column tables and then rejected
    wider ones whose de-shift legitimately removes more duplicates."""
    assert tdr._max_count_drop(3) == 3          # floor governs small counts
    assert tdr._max_count_drop(12) == 3
    assert tdr._max_count_drop(24) == 6         # proportional takes over
    assert tdr._max_count_drop(40) == 10


def test_tolerance_never_drops_below_the_flat_floor():
    """Whatever the fraction, a value may always shed at least _MAX_COUNT_DROP —
    the new rule must not be STRICTER than the one it replaces."""
    for had in range(1, 60):
        assert tdr._max_count_drop(had) >= tdr._MAX_COUNT_DROP


def test_a_high_multiplicity_duplicate_purge_is_accepted():
    """A value appearing 24x losing 6 copies is the wide-table de-shift shape."""
    # 24 copies of x9 spread over distinct rows (identical rows would collapse).
    old = _table([["x9", "x9", "x9", "x9", "x9", "x9", f"tag{r}"]
                  for r in range(4)])
    new = _table([["x9", "x9", "x9", "x9", f"tag{r}", "", ""]
                  for r in range(4)])
    # x9 falls 24 -> 16, a drop of 8; the proportional rule allows 6 of 24...
    assert tdr._content_preserved(old, new) is False
    # ...while a drop within tolerance (24 -> 19) is accepted.
    ok = _table([["x9", "x9", "x9", "x9", "x9", f"tag{r}", ""]
                 for r in range(4)] + [["x9", "", "", "", "", "", ""]])
    assert tdr._content_preserved(old, ok) is True


def test_a_vanished_value_is_still_rejected_at_any_multiplicity():
    """Proportional tolerance must never let a value disappear entirely."""
    old = _table([["15.53", "15.51"], ["15.52", "15.54"]])
    new = _table([["15.53", "15.51"], ["15.52", "99.99"]])
    assert tdr._content_preserved(old, new) is False


# ── The OCR_ROTATION master switch ──────────────────────────────────────────

def _reload_gates(env):
    """Reload both rotation modules under `env` and return their gate values.

    The gates are module-level constants read at import time, so the switch can
    only be tested by reloading — the same idiom `test_rotation.py` uses.
    """
    import importlib
    import os
    saved = {k: os.environ.get(k) for k in (
        "OCR_ROTATION", "OCR_DETECT_ROTATION", "OCR_DETECT_ROTATION_TABLES",
        "OCR_REOCR_ROTATED_TABLES")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        os.environ.update(env)
        import rotation_detect
        importlib.reload(rotation_detect)
        importlib.reload(tdr)
        return (rotation_detect.DETECT_ENABLE, rotation_detect.DETECT_TABLES,
                tdr.REOCR_ROTATED_TABLES)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import rotation_detect
        importlib.reload(rotation_detect)
        importlib.reload(tdr)


def test_master_switch_defaults_off():
    """The SHIPPING default: no rotation work unless asked for."""
    detect, tables, reocr = _reload_gates({})
    assert (detect, tables, reocr) == (False, False, False)


def test_master_switch_true_enables_the_whole_chain():
    """One value must configure every stage — a half-configured pipeline flips
    pages to landscape without recovering their dropped cells."""
    detect, tables, reocr = _reload_gates({"OCR_ROTATION": "true"})
    assert (detect, tables, reocr) == (True, True, True)


def test_master_switch_accepts_the_usual_truthy_spellings():
    for value in ("1", "true", "TRUE", "yes", "on", " true "):
        assert _reload_gates({"OCR_ROTATION": value}) == (True, True, True), value


def test_specific_gate_overrides_a_true_master():
    """Fine-grained vars stay usable for debugging ONE stage."""
    detect, tables, reocr = _reload_gates({
        "OCR_ROTATION": "true", "OCR_DETECT_ROTATION_TABLES": "false"})
    assert detect is True
    assert tables is False          # the expensive tier-2 probe, off
    assert reocr is True


def test_specific_gate_can_enable_against_a_false_master():
    detect, tables, reocr = _reload_gates({
        "OCR_ROTATION": "false", "OCR_REOCR_ROTATED_TABLES": "true"})
    assert (detect, tables) == (False, False)
    assert reocr is True


# ── the widened collapsed-baseline guard (page 53) ───────────────────────────

def test_widest_row_reads_the_raw_html_not_the_capped_grid():
    """`parse_table_grid` caps at MAX_TABLE_COLS, so it CANNOT report a run-on
    row. The guard needs what the model actually emitted."""
    import table_derotate_reocr as td
    from ocr_reconstruction.table import MAX_TABLE_COLS
    html = ("<table><tbody><tr>"
            + "".join(f"<td>c{i}</td>" for i in range(700))
            + "</tr></tbody></table>")
    assert td._widest_row_cells(html) == 700 > MAX_TABLE_COLS


def test_widest_row_is_safe_on_junk():
    import table_derotate_reocr as td
    for value in ("", "no table here", "<table></table>"):
        assert td._widest_row_cells(value) == 0


def test_widest_row_picks_the_largest_of_several_rows():
    import table_derotate_reocr as td
    html = ("<table><tbody>"
            "<tr><td>a</td><td>b</td></tr>"
            "<tr>" + "".join(f"<td>{i}</td>" for i in range(90)) + "</tr>"
            "<tr><td>c</td></tr></tbody></table>")
    assert td._widest_row_cells(html) == 90


def test_the_measured_page_53_shape_counts_as_collapsed():
    """The real rotated page parses to `rows 4, cols 40` — a HEADER that is fine
    plus one 697-cell run-on row. The original `old_rows <= 1` test missed it and
    declined the repair (1 of 7 accepted); the raw-HTML test catches it."""
    import table_derotate_reocr as td
    from ocr_reconstruction.table import MAX_TABLE_COLS
    html = ("<table><tbody>"
            "<tr><td>序号</td><td>注塑参数</td><td>尺寸</td><td>外观</td><td>结论</td></tr>"
            "<tr>" + "".join(f"<td>h{i}</td>" for i in range(11)) + "</tr>"
            "<tr>" + "".join(f"<td>g{i}</td>" for i in range(15)) + "</tr>"
            "<tr>" + "".join(f"<td>v{i}</td>" for i in range(697)) + "</tr>"
            "</tbody></table>")
    _cols, rows, _full = td._grid_shape(html)
    assert rows > 1, "grid sees several rows, so `old_rows <= 1` cannot fire"
    assert td._widest_row_cells(html) > MAX_TABLE_COLS, "but the run-on IS visible"


def test_a_well_formed_wide_table_is_not_a_collapsed_baseline():
    """A genuinely wide table at the cap must NOT be treated as structureless, or
    the guard would wave through a re-read that narrows a legitimate grid."""
    import table_derotate_reocr as td
    from ocr_reconstruction.table import MAX_TABLE_COLS
    html = ("<table><tbody>"
            + "".join("<tr>" + "".join(f"<td>r{r}c{c}</td>"
                                       for c in range(MAX_TABLE_COLS)) + "</tr>"
                      for r in range(8))
            + "</tbody></table>")
    assert td._widest_row_cells(html) == MAX_TABLE_COLS
    assert not td._widest_row_cells(html) > MAX_TABLE_COLS
