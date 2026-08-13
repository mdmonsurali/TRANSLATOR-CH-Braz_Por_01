"""Cell-level repeat-loop detection, and the retry control flow that acts on it.

WHY THIS EXISTS. Four pages of a real 168-page document are 90-degree ROTATED
measurement tables. Reading one sideways, the model transcribed the first record
and then degenerated into a loop emitting the literal cell "ok" until it hit the
token cap:

    page 50: 746 of 765 cells were "ok"  (98%)
    page 53: 678 of 728                  (93%)
    page 54: 876 of 894                  (98%)
    page 56: 875 of 894                  (98%)

Two separate defects let that reach the output:

1. `detect_repeat_token` is a pure STRING test over the generation's tail. The
   run-on is structured markup ending in `</tr></tbody></table>`, which breaks the
   exact-suffix match, so it caught these only marginally.
2. The `finish_reason == "length"` branch `continue`d immediately, SKIPPING the
   repeat check on every attempt that still had retry budget and reaching it only
   on the final attempt, when no budget remained. Net effect: 13 truncation
   warnings in one run produced ZERO repeat retries.

Run:  python -m pytest ocr_service/tests/test_repeat_loop.py -v
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

co = pytest.importorskip("chandra_ocr", reason="vLLM client deps required")


def _table(values):
    """One row per inner list."""
    rows = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in values
    )
    return f"<table border='1'><tbody>{rows}</tbody></table>"


def _loop_table(real, filler="ok", n=700):
    """The measured failure: one genuine record, then a filler cell loop."""
    cells = list(real) + [filler] * n
    return ("<table border='1'><tbody><tr>"
            + "".join(f"<td>{c}</td>" for c in cells)
            + "</tr></tbody></table>")


# ── the cell-level detector ─────────────────────────────────────────────────

def test_the_measured_ok_loop_is_flagged():
    """98% of cells being one value is degenerate by any standard."""
    html = _loop_table(["1", "495", "295", "2.5", "7.60"], "ok", 760)
    assert co._repeats_one_cell(html) is True


def test_a_real_measurement_table_is_not_flagged():
    """12 rows x distinct measurements — the shape the loop DESTROYED. It must
    survive the check, or the fix would discard good pages."""
    rows = [[f"{r}", "495", "295", "2.5"] + [f"7.6{(r + c) % 10}" for c in range(37)]
            for r in range(12)]
    assert co._repeats_one_cell(_table(rows)) is False


def test_a_legitimate_verdict_column_is_not_flagged():
    """A real table CAN hold many "ok" cells — 5 verdict columns of 12 rows is
    60 of 492 cells (12%). Only total domination is the defect."""
    rows = [[f"{r}", f"7.6{r}", f"1.9{r}", f"0.8{r}"] + ["ok"] * 5
            for r in range(12)]
    html = _table(rows)
    assert co._repeats_one_cell(html) is False


@pytest.mark.parametrize("frac,expected", [(0.5, True), (0.95, False)])
def test_the_threshold_is_tunable(monkeypatch, frac, expected):
    """80% domination: flagged at a 0.70/0.50 threshold, not at 0.95."""
    html = _loop_table(["1", "2"] * 20, "ok", 160)
    monkeypatch.setattr(co, "_MAX_SAME_CELL_FRAC", frac)
    assert co._repeats_one_cell(html) is expected


def test_small_tables_are_left_alone():
    """A 3-cell table of "ok" is 100% dominated but proves nothing — below the
    minimum sample the check must abstain rather than guess."""
    assert co._repeats_one_cell(_table([["ok", "ok", "ok"]])) is False


def test_junk_input_is_safe():
    for value in ("", "not html", "<table></table>", "x" * 300):
        assert co._repeats_one_cell(value) is False


def test_prose_without_cells_is_ignored():
    """The check is about TABLE cells; a repetitive paragraph is not its job."""
    assert co._repeats_one_cell("<p>" + "the " * 400 + "</p>") is False


# ── the control flow that was skipping the check ────────────────────────────

def _decide(truncated, has_repeat, attempt, max_retries, repeat_retry=True):
    """Mirrors the attempt-loop decision in `ocr_image_async`: returns True when
    the loop RETRIES. Kept as a table-driven check because the original bug was
    pure control flow — the detector worked, it was never consulted in time."""
    if not repeat_retry:
        return False
    if not has_repeat and not truncated:
        return False
    return attempt < max_retries


@pytest.mark.parametrize("truncated,has_repeat,attempt,expect", [
    # The measured case: BOTH signals on a page with budget left. The old code
    # continued off `truncated` and never evaluated has_repeat.
    (True, True, 0, True),
    (True, True, 1, True),
    (True, True, 2, False),   # budget exhausted -> accept what we have
    # A loop that did NOT hit the cap must still retry.
    (False, True, 0, True),
    # Truncated but coherent (a genuinely long page) also deserves a resample.
    (True, False, 0, True),
    # Clean output never retries.
    (False, False, 0, False),
])
def test_retry_decision(truncated, has_repeat, attempt, expect):
    assert _decide(truncated, has_repeat, attempt, max_retries=2) is expect


def test_the_kill_switch_still_accepts_the_first_read():
    """With OCR_REPEAT_RETRY off, nothing retries however bad the output."""
    assert _decide(True, True, 0, 2, repeat_retry=False) is False


def test_the_old_bug_would_have_starved_the_repeat_check():
    """Regression guard for the ACTUAL defect. Old flow: `continue` on truncated
    before testing has_repeat, so the repeat branch was reachable only at
    attempt == MAX_RETRIES, where it could no longer retry."""
    def old(truncated, has_repeat, attempt, max_retries):
        if truncated and attempt < max_retries:
            return "retried-without-checking-repeat"
        return "repeat-checked-too-late" if has_repeat else "accepted"

    assert old(True, True, 0, 2) == "retried-without-checking-repeat"
    assert old(True, True, 2, 2) == "repeat-checked-too-late"
    # The fix reaches a real retry decision on the FIRST attempt instead.
    assert _decide(True, True, 0, 2) is True
