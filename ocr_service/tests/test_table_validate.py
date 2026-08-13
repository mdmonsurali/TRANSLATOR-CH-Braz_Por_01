"""One table validator: health measurement, comparative scoring, repair safety.

WHY THIS REPLACES `test_table_collapse.py`. The service had one re-OCR pass per
observed failure shape, each with its own trigger and guard. `table_validate`
measures table HEALTH instead, so every previously-special-cased shape is now one
signal among several, and repair keeps whichever read SCORES better rather than
demanding the re-read win on every axis.

The measurements these tests encode, all from stored runs of one 168-page document:

    collapsed   12 data rows run together into a 701-cell <tr> -> the renderer's
                40-col grid places 67 of 728 cells; the well-formed version of the
                SAME page places 326 of 326. So raw <td> counts are a MISLEADING
                quality metric — the output with more cells rendered worse.
    degenerate  678 of 697 cells were the literal "ok" (a repeat loop)
    declined    on the rotated subset, all 8 repairs were rejected by the old
                guard, including `cols 26->25, rows 15->15, full 12->15` — three
                empty rows filled, refused for having one fewer column. The
                comparative score must ACCEPT that trade.

Run:  python -m pytest ocr_service/tests/test_table_validate.py -v
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

tv = pytest.importorskip("table_validate", reason="chandra_ocr/vLLM deps required")

from ocr_reconstruction.table import MAX_TABLE_COLS  # noqa: E402


def _rows_table(rows):
    """Well-formed: one <tr> per row."""
    html = ["<table border='1'><tbody>"]
    for r in rows:
        html.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    html.append("</tbody></table>")
    return "".join(html)


def _collapsed_table(rows):
    """The failure shape: every cell of every row inside ONE <tr>."""
    flat = [c for r in rows for c in r]
    return ("<table border='1'><tbody><tr>"
            + "".join(f"<td>{c}</td>" for c in flat)
            + "</tr></tbody></table>")


def _grid(n_rows=12, n_cols=25):
    # Distinct values per row: `parse_html_table_rows` COLLAPSES byte-identical
    # consecutive rows, which would make a fixture measure 1 row and hide the
    # behaviour under test.
    return [[f"r{r}c{c}" for c in range(n_cols)] for r in range(n_rows)]


# ── health measurement ──────────────────────────────────────────────────────

def test_a_well_formed_table_is_healthy():
    h = tv.table_health(_rows_table(_grid(15, 25)))
    assert h.healthy is True
    assert h.placed == h.emitted == 375
    assert h.cols == 25 and h.rows == 15


def test_the_collapsed_shape_is_unhealthy_and_says_why():
    """The measured page: a row too wide for the grid cannot be placed."""
    h = tv.table_health(_collapsed_table(_grid(12, 25)))
    assert h.unplaceable is True
    assert h.healthy is False
    assert h.placed < h.emitted
    assert any("exceeds" in r for r in h.reasons())


def test_placed_fraction_exposes_the_silent_loss():
    """The core measurement: 300 emitted cells, far fewer placeable."""
    h = tv.table_health(_collapsed_table(_grid(12, 25)))
    assert h.emitted == 300
    assert h.placed_frac < 0.5


def test_the_ok_loop_is_degenerate():
    """678 of 697 cells being one value carries no data."""
    cells = ["1", "495", "295"] + ["ok"] * 700
    html = ("<table><tbody><tr>"
            + "".join(f"<td>{c}</td>" for c in cells) + "</tr></tbody></table>")
    assert tv.table_health(html).degenerate is True


def test_a_legitimate_verdict_column_is_not_degenerate():
    """Real tables DO hold many "ok" cells — 5 of 9 columns is not a loop."""
    rows = [[f"{r}", f"7.6{r}", f"1.9{r}", f"0.8{r}"] + ["ok"] * 5
            for r in range(12)]
    assert tv.table_health(_rows_table(rows)).degenerate is False


def test_truncation_makes_a_table_unhealthy():
    """A generation that hit the output cap lost its tail, however good the
    HTML it did emit looks."""
    html = _rows_table(_grid(15, 25))
    assert tv.table_health(html, truncated=False).healthy is True
    assert tv.table_health(html, truncated=True).healthy is False


def test_health_is_safe_on_junk():
    for value in ("", "not a table", "<table></table>", None):
        h = tv.table_health(value or "")
        assert h.emitted == 0 and h.placed == 0


def test_the_boundary_is_max_table_cols():
    at_cap = _rows_table([[f"c{i}" for i in range(MAX_TABLE_COLS)]])
    over = _rows_table([[f"c{i}" for i in range(MAX_TABLE_COLS + 1)]])
    assert tv.table_health(at_cap).unplaceable is False
    assert tv.table_health(over).unplaceable is True


# ── comparative scoring: the substantive change ─────────────────────────────

def test_a_de_collapsed_read_scores_better():
    rows = _grid(12, 25)
    bad = tv.table_health(_collapsed_table(rows))
    good = tv.table_health(_rows_table(rows))
    assert good.score() > bad.score()


def test_filling_empty_rows_wins_even_while_LOSING_a_column():
    """The exact case the old guard declined 8 times: `cols 26->25, full 12->15`.
    A spurious 26th column is worth less than three complete rows."""
    baseline = _rows_table([[f"r{r}c{c}" for c in range(25)] + [""]
                            for r in range(12)]
                           + [[""] * 26 for _ in range(3)])
    repaired = _rows_table(_grid(15, 25))
    old = tv.table_health(baseline)
    new = tv.table_health(repaired)
    assert new.full_rows > old.full_rows
    assert new.score() > old.score(), "comparative score must accept this trade"


def test_a_degenerate_read_never_beats_a_real_one():
    """Degeneracy is the first score component, so an `ok` loop cannot win on
    cell count — which is exactly how a raw-count metric would be fooled."""
    real = tv.table_health(_rows_table(_grid(12, 25)))
    loop_cells = ["ok"] * 900
    loop = tv.table_health(
        "<table><tbody><tr>"
        + "".join(f"<td>{c}</td>" for c in loop_cells) + "</tr></tbody></table>")
    assert loop.emitted > real.emitted, "the loop has MORE cells"
    assert real.score() > loop.score(), "yet the real table must win"


# ── the repair driver ───────────────────────────────────────────────────────

def _page(html, rotation=None):
    entry = {"category": "Table", "bbox": [100, 100, 2300, 3300], "text": html}
    if rotation is not None:
        entry["rotation"] = rotation
    return {
        "page_index": 55,
        "original_image": Image.new("RGB", (2480, 3505), "white"),
        "layout_result": [entry],
    }


def _run(pages, enabled=True):
    import asyncio
    prev = tv.VALIDATE_TABLES
    tv.VALIDATE_TABLES = enabled
    try:
        return asyncio.new_event_loop().run_until_complete(
            tv.validate_tables(pages))
    finally:
        tv.VALIDATE_TABLES = prev


def _stub(monkeypatch, html):
    async def _fake(img):
        return [{"category": "Table", "text": html}]
    monkeypatch.setattr(tv, "ocr_image_async", _fake)


def test_healthy_tables_cost_zero_model_calls(monkeypatch):
    called = []

    async def _boom(img):
        called.append(1)
        raise AssertionError("must not re-OCR a healthy table")
    monkeypatch.setattr(tv, "ocr_image_async", _boom)
    html = _rows_table(_grid(15, 25))
    page = _page(html)
    _run([page])
    assert not called
    assert page["layout_result"][0]["text"] == html
    assert "source" not in page["layout_result"][0]


def test_an_unhealthy_table_is_repaired(monkeypatch):
    rows = _grid(12, 25)
    _stub(monkeypatch, _rows_table(rows))
    page = _page(_collapsed_table(rows))
    _run([page])
    entry = page["layout_result"][0]
    assert entry["text"] == _rows_table(rows)
    assert entry["source"] == "validate-reocr"


def test_a_worse_reread_is_rejected(monkeypatch):
    """A re-read that places a SMALLER fraction of its cells loses.

    Note the fixture cannot simply be "a smaller collapsed table": a 6-row
    collapse places the same 40 cells out of fewer emitted, so its placed
    FRACTION is genuinely higher and preferring it is correct. Real degradation
    means fewer cells actually reaching the grid.
    """
    good = _rows_table(_grid(12, 25))          # healthy baseline, 300/300
    _stub(monkeypatch, _collapsed_table(_grid(12, 25)))   # 40/300
    page = _page(good)
    # Force the repair path even though the baseline is healthy, so the guard
    # itself is what rejects the worse read rather than the trigger.
    page["layout_result"][0]["truncated"] = True
    _run([page])
    assert page["layout_result"][0]["text"] == good
    assert "source" not in page["layout_result"][0]


def test_a_rewritten_measurement_is_rejected(monkeypatch):
    """Shape alone is not enough — inventing values is worse than ugly HTML."""
    old = _collapsed_table([[f"15.{i:02d}" for i in range(60)]])
    _stub(monkeypatch, _rows_table([["99.99", "88.88"], ["77.77", "66.66"]]))
    page = _page(old)
    _run([page])
    assert page["layout_result"][0]["text"] == old


def test_the_kill_switch_prevents_all_work(monkeypatch):
    async def _boom(img):
        raise AssertionError("kill switch must prevent any OCR")
    monkeypatch.setattr(tv, "ocr_image_async", _boom)
    old = _collapsed_table(_grid(12, 25))
    page = _page(old)
    _run([page], enabled=False)
    assert page["layout_result"][0]["text"] == old


def test_ocr_failure_is_swallowed(monkeypatch):
    async def _raise(img):
        raise RuntimeError("vLLM exploded")
    monkeypatch.setattr(tv, "ocr_image_async", _raise)
    old = _collapsed_table(_grid(12, 25))
    page = _page(old)
    _run([page])
    assert page["layout_result"][0]["text"] == old


def test_reocr_returning_no_table_keeps_the_original(monkeypatch):
    async def _no_table(img):
        return [{"category": "Text", "text": "just prose"}]
    monkeypatch.setattr(tv, "ocr_image_async", _no_table)
    old = _collapsed_table(_grid(12, 25))
    page = _page(old)
    _run([page])
    assert page["layout_result"][0]["text"] == old


def test_a_page_without_a_raster_is_skipped(monkeypatch):
    async def _boom(img):
        raise AssertionError("no raster -> no crop -> no OCR")
    monkeypatch.setattr(tv, "ocr_image_async", _boom)
    old = _collapsed_table(_grid(12, 25))
    page = {"page_index": 1, "layout_result": [
        {"category": "Table", "bbox": [0, 0, 100, 100], "text": old}]}
    _run([page])
    assert page["layout_result"][0]["text"] == old


# ── rotation is now an input to repair, not a separate pass ─────────────────

def test_a_rotated_crop_is_turned_upright_before_reocr(monkeypatch):
    """The de-rotate behaviour must survive the merge: a table stamped 90 deg
    must reach the model UPRIGHT, otherwise its columns stay crushed onto the
    page's short axis and the re-read fails the same way as the original."""
    seen = {}

    async def _capture(img):
        seen["size"] = img.size
        return [{"category": "Table", "text": _rows_table(_grid(12, 25))}]
    monkeypatch.setattr(tv, "ocr_image_async", _capture)

    page = _page(_collapsed_table(_grid(12, 25)), rotation=90.0)
    # bbox 100,100 -> 2300,3300 is TALL (2200 x 3200 + margins)
    _run([page])
    w, h = seen["size"]
    assert w > h, "a 90-degree table must be transposed to landscape before OCR"


def test_an_upright_table_is_not_transposed(monkeypatch):
    seen = {}

    async def _capture(img):
        seen["size"] = img.size
        return [{"category": "Table", "text": _rows_table(_grid(12, 25))}]
    monkeypatch.setattr(tv, "ocr_image_async", _capture)

    page = _page(_collapsed_table(_grid(12, 25)))     # no rotation key
    _run([page])
    w, h = seen["size"]
    assert h > w, "an upright crop must reach the model unrotated"


# ── the measured real-world case ────────────────────────────────────────────

def test_the_real_world_page_55_shape(monkeypatch):
    """Reproduces the measured page and asserts on PLACED cells, not raw counts:
    the collapsed version emits MORE <td> (728 vs 326) yet renders far worse."""
    header = [["序号", "注塑参数", "尺寸", "外观", "结论"]]
    data = _grid(12, 25)
    old = _collapsed_table(header + data)
    new = _rows_table(header + data)

    before = tv.table_health(old)
    assert before.unplaceable and not before.healthy

    _stub(monkeypatch, new)
    page = _page(old)
    _run([page])
    entry = page["layout_result"][0]
    assert entry["source"] == "validate-reocr"

    after = tv.table_health(entry["text"])
    assert after.healthy is True
    assert after.placed > before.placed
    assert after.placed == after.emitted, "every cell placeable after repair"


# ── the ground-truth signal: ruled-line column count ────────────────────────
#
# WHY THIS EXISTS. Every other signal reads the model's OWN output, so a read that
# is internally consistent but missing whole columns is invisible. The measured
# case (`problem.png`): a 7-column injection-parameters form emitted as 4 columns,
# placing 43 of 43 cells — healthy by every HTML-only metric — while three columns
# of real data (周期, 位置, 背压) were never transcribed. Live `/detect_grid` on
# that crop reports 9 column boundaries against the 4 emitted.

def _health(html, ruled):
    return tv.table_health(html, ruled_cols=ruled)


def test_missing_columns_is_detected_from_the_ruling():
    """The exact problem.png shape: internally perfect, three columns absent."""
    html = _rows_table(_grid(13, 4))
    assert _health(html, 0).healthy is True, "no ruling measured -> HTML-only"
    h = _health(html, 9)
    assert h.missing_columns is True
    assert h.healthy is False
    assert any("ruling shows 9 columns" in r for r in h.reasons())


def test_a_matching_ruling_leaves_a_table_alone():
    """The healthy 8-column header table on the same page must not be flagged."""
    html = _rows_table(_grid(4, 8))
    assert _health(html, 7).missing_columns is False
    assert _health(html, 8).missing_columns is False
    assert _health(html, 9).missing_columns is False   # 9 < 8*1.5


def test_the_ratio_threshold_is_where_measurement_put_it():
    """Measured on 13 tables: healthy scored 0.29-1.29, every table at 1.5+ was
    genuinely defective. The boundary must sit in that gap."""
    html = _rows_table(_grid(6, 4))
    assert _health(html, 5).missing_columns is False   # 1.25x — within noise
    assert _health(html, 6).missing_columns is True    # 1.50x — flagged


def test_an_unmeasured_ruling_never_flags():
    """0 means "not measured" — probe off, unreachable, or borderless table. It
    must never be read as "zero columns", or every table would be flagged."""
    html = _rows_table(_grid(13, 4))
    assert _health(html, 0).missing_columns is False
    assert _health(html, 0).healthy is True


def test_sparse_ruling_is_not_trusted():
    """Below the minimum the ruling is too sparse to mean anything (a borderless
    or part-ruled table), so it must abstain rather than guess."""
    html = _rows_table(_grid(13, 1))
    assert _health(html, 2).missing_columns is False


def test_a_wider_reread_beats_a_narrow_one_even_at_a_lower_placed_fraction():
    """Ranking `missing_columns` above `placed_frac` is deliberate: a 7-column
    read that places 90% of its cells must beat a 4-column read that places
    100% of far fewer, because the missing columns are real data."""
    narrow = _health(_rows_table(_grid(13, 4)), 9)
    wide_rows = _grid(13, 9)
    wide_rows[-1] = wide_rows[-1][:7] + ["", ""]      # slightly ragged tail
    wide = _health(_rows_table(wide_rows), 9)
    assert narrow.placed_frac >= wide.placed_frac
    assert wide.score() > narrow.score()


def test_the_reread_is_judged_against_the_same_ruling(monkeypatch):
    """A re-read must carry the SAME `ruled_cols`, so `missing_columns` is
    comparable on both sides.

    Asserted on the scores directly rather than through the driver: a re-read
    with MORE rows legitimately wins the content tie-break even while equally
    column-deficient, so a driver-level assertion would be testing the tie-break,
    not the ruling. What matters here is that the defect is visible on both sides
    instead of only on the baseline.
    """
    narrow_old = tv.table_health(_rows_table(_grid(13, 4)), ruled_cols=9)
    narrow_new = tv.table_health(_rows_table(_grid(13, 4)), ruled_cols=9)
    assert narrow_old.missing_columns and narrow_new.missing_columns
    assert narrow_new.score() == narrow_old.score()

    # If the re-read were judged WITHOUT the ruling its defect would vanish and
    # it would outrank the baseline purely by measuring less.
    unjudged = tv.table_health(_rows_table(_grid(13, 4)), ruled_cols=0)
    assert unjudged.missing_columns is False
    assert unjudged.score() > narrow_old.score(), (
        "this is exactly the false win the shared ruling prevents")


def test_a_recovered_wide_reread_is_accepted(monkeypatch):
    """The repair path for problem.png: 4 columns -> 9, matching the ruling."""
    async def _wide(img):
        return [{"category": "Table", "text": _rows_table(_grid(13, 9))}]
    monkeypatch.setattr(tv, "ocr_image_async", _wide)

    async def _ruled(crop):
        return 9
    monkeypatch.setattr(tv, "ruled_columns", _ruled)

    page = _page(_rows_table(_grid(13, 4)))
    _run([page])
    entry = page["layout_result"][0]
    assert entry["source"] == "validate-reocr"
    assert tv.table_health(entry["text"], ruled_cols=9).healthy is True


def test_a_probe_failure_falls_back_to_html_only(monkeypatch):
    """paddle_service being down must never fail a page or block repair of the
    defects the HTML alone can prove."""
    async def _boom(crop):
        raise RuntimeError("paddle_service unreachable")
    monkeypatch.setattr(tv, "ruled_columns", _boom)

    async def _fixed(img):
        return [{"category": "Table", "text": _rows_table(_grid(12, 25))}]
    monkeypatch.setattr(tv, "ocr_image_async", _fixed)

    page = _page(_collapsed_table(_grid(12, 25)))
    _run([page])          # must not raise
    assert page["layout_result"][0]["text"] is not None


def test_the_grid_probe_can_be_disabled(monkeypatch):
    """`OCR_VALIDATE_GRID=false` must make the probe a strict no-op."""
    import asyncio
    monkeypatch.setattr(tv, "GRID_CHECK", False)

    async def _boom(*a, **k):
        raise AssertionError("probe must not run when disabled")
    monkeypatch.setattr(tv.httpx, "AsyncClient", _boom)
    got = asyncio.new_event_loop().run_until_complete(
        tv.ruled_columns(Image.new("RGB", (80, 80), "white")))
    assert got == 0
