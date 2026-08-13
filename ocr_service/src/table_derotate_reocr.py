"""Recover table cells Chandra drops when a table is scanned SIDEWAYS, by
re-OCRing the table crop turned upright.

THE PROBLEM. A landscape inspection table scanned onto a portrait sheet reaches
the model rotated 90°: its columns are stacked across the page's SHORT axis and
every value is turned on its side. Measured on a real 25-column form, after
``scale_to_fit`` shrinks the 2480x3505 raster into Chandra's 3072x2048 envelope,
each column gets ~64 px of width — for 5-character handwritten readings like
``15.53``. The model then silently drops cells, and because the reconstruction
grid fills left-to-right the survivors shift one column left and land under the
WRONG header: a ``尺寸2`` value filed under ``尺寸1``, and the final ``结论``
column empty. That is corrupted measurement data, not a layout glitch.

Reconstruction cannot repair it — given 24 values for 25 slots there is no way to
know which slot was skipped, and the header carries no signal either (all five
``尺寸1`` leaves share one nominal). The values only exist in the pixels, so the
fix belongs here.

THE FIX, measured on the live model (same crop, same model, two orientations):

    as-scanned   page 1: 0/12 body rows at full width; page 2: grid resolves 24 wide
    de-rotated   page 1: 12/12 at full width;          page 2: resolves 25 wide

and the recovered values are CORRECT, not merely the right count — row 1's
``尺寸1`` came back ``15.53 15.51 15.52 15.52 15.54`` with ``结论 = ok``, matching
the source scan including the duplicated ``15.52`` that was being dropped.

DIRECTION MATTERS. Rotating the wrong way measurably degrades the result (an
attempt with the opposite transpose returned 18-wide rows). The transpose is
therefore always derived from the detected ``rotation``, never hardcoded.

STATUS. The repair LOOP moved to `table_validate`, which measures table health
generally instead of triggering on rotation alone, de-rotates the crop using
`upright_transpose` below, and scores both reads comparatively. This module is now
primarily the home of the shared primitives that repair depends on:

    upright_transpose    rotation angle -> the PIL transpose that rights a crop
    _grid_shape          (cols, rows, rows_at_full_width) on the renderer's grid
    _widest_row_cells    raw-HTML row width, independent of the capped grid
    _content_preserved   multiset value guard: tolerates the duplicates a
                         de-shift removes, never a value disappearing
    _extract_table_html  pick the table from a re-OCR's entries

`reocr_rotated_tables` is retained as the rotation-only entry point (still
exercised by `test_table_derotate.py`), but `main` no longer calls it — a rotated
table now reaches the same repair path as any other unhealthy one.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import List, Optional, Tuple

from PIL import Image

from chandra_ocr import ocr_image_async
# The same grid parser the renderer uses, so "how many columns" here means
# exactly what reconstruction will build.
from ocr_reconstruction.table import (
    MAX_TABLE_COLS, parse_html_table_rows, parse_table_grid,
)

log = logging.getLogger("ocr_service")

# Kill switch, defaulting to the rotation MASTER switch (`OCR_ROTATION`, default
# off) so one value configures the whole chain — see `rotation_detect` for why the
# default is off and when to turn it on. Set this explicitly only to debug this
# stage in isolation.
#
# Each accepted repair costs one extra model call on a crop that is nearly a full
# page of pixels (measured 3136x1904 vs a page's 2100x2968), so this is genuinely
# expensive even though it only fires on rotated tables. It is also a strict
# no-op without detection: nothing stamps `rotation`, so there are no candidates.
REOCR_ROTATED_TABLES = os.getenv(
    "OCR_REOCR_ROTATED_TABLES",
    os.getenv("OCR_ROTATION", "false")).strip().lower() in {
    "1", "true", "yes", "on"}

# Margin (px) around the table bbox when cropping — these tables sit flush
# against an outer ruling the model reads better with a little air. Matches
# `table_reocr._CROP_MARGIN_PX`.
_CROP_MARGIN_PX = 12

# How many rotated-table re-OCR calls may be in flight at once. A de-rotated
# table crop is nearly a full page of pixels (measured: 3136x1904 vs a page's
# 2100x2968), so these cost about what a page costs and the same batching logic
# applies — see `main._ocr_pages_concurrently`. Serialising them made a 5-table
# document pay 5x a single call's latency for no reason.
_REOCR_CONCURRENCY = max(1, int(os.getenv("OCR_REOCR_CONCURRENCY", "4")))

_TR_RE = re.compile(r"<tr\b", re.IGNORECASE)
_TR_SPLIT_RE = re.compile(r"<tr\b[^>]*>", re.IGNORECASE)
_TD_COUNT_RE = re.compile(r"<t[dh]\b", re.IGNORECASE)


def _widest_row_cells(table_html: str) -> int:
    """Cell count of the widest ``<tr>`` in the RAW html (0 when there are none).

    Deliberately independent of `parse_table_grid`, which collapses and caps: this
    reports what the model actually emitted, so a run-on row is visible even when
    the grid parser has already truncated it to MAX_TABLE_COLS.
    """
    if not table_html:
        return 0
    return max(
        (len(_TD_COUNT_RE.findall(chunk))
         for chunk in _TR_SPLIT_RE.split(table_html)[1:]),
        default=0,
    )

# A baseline this wide with <=1 row has no row structure at all — the sideways
# read ran every cell together into one row, which `parse_table_grid` then caps
# at MAX_TABLE_COLS. Tied to the renderer's own cap so the two cannot drift.
_COLLAPSED_COLS = MAX_TABLE_COLS

# Rotation (degrees CCW-positive, the schema `rotation_measure` produces) mapped
# to the PIL transpose that returns the crop to upright. Derived, never guessed:
# the opposite transpose was measured to make the output worse.
_UPRIGHT_TRANSPOSE = {
    90.0: Image.ROTATE_90,
    270.0: Image.ROTATE_270,
}


def upright_transpose(rotation) -> Optional[int]:
    """PIL transpose constant that makes a crop rotated by `rotation` upright,
    or None when there is nothing to do (upright, 180°, or unmeasurable).

    180° is excluded deliberately: the text is upside-down but the COLUMN
    geometry is unchanged, so a re-OCR gains nothing on the axis this module
    cares about.
    """
    try:
        deg = float(rotation)
    except (TypeError, ValueError):
        return None
    if deg != deg:                      # NaN
        return None
    deg %= 360.0
    for quarter, transpose in _UPRIGHT_TRANSPOSE.items():
        if abs(deg - quarter) <= 5.0:   # detections land at 89.2 / 90.4
            return transpose
    return None


def _cell_signatures(table_html: str) -> List[str]:
    """Normalised non-empty cell contents, as a sorted multiset.

    Letters/digits/CJK only, lowercased — the same normalisation idea
    `checkbox_reocr._alnum_cjk` uses to prove a re-OCR kept the original
    wording. Used to guarantee a re-read may only ADD a value, never rewrite
    one: `15.53` becoming `15.58` must be rejected, not accepted as a "repair".
    """
    rows = parse_html_table_rows(table_html)
    if not rows:
        return []
    _mc, _nr, anchors, _occ, _img = parse_table_grid(rows)
    sigs = []
    for (_r, _c), value in anchors.items():
        text = re.sub(r"[^0-9a-z㐀-䶿一-鿿]", "",
                      (value[0] or "").lower())
        if text:
            sigs.append(text)
    return sorted(sigs)


def _is_superset(old_sigs: List[str], new_sigs: List[str]) -> bool:
    """True when `new_sigs` contains every signature in `old_sigs` (with at
    least the same multiplicity)."""
    from collections import Counter
    new_counts = Counter(new_sigs)
    for sig, need in Counter(old_sigs).items():
        if new_counts.get(sig, 0) < need:
            return False
    return True


# How far a single value's multiplicity may fall before it counts as real data
# loss rather than the removal of a MISPLACED duplicate. See
# `_content_preserved` for why any tolerance is needed at all.
#
# A FLOOR, not the whole rule: the number of misplaced duplicates a de-shift
# removes scales with how many cells were dropped, which scales with the table's
# width. A flat 3 was calibrated on 25-column tables and then wrongly rejected
# wider ones — measured on 30-column pages that improved from 12-15 full-width
# rows to 14-15 (`full 1->15` on one page) but tripped the flat cap. See
# `_max_count_drop`.
_MAX_COUNT_DROP = int(os.getenv("OCR_REOCR_MAX_COUNT_DROP", "3"))

# Fraction of a value's own multiplicity that may be shed on top of the floor.
# A value appearing 19 times can lose 5 (19 * 0.25 rounded down, vs the floor's
# 3); a value appearing 3 times is still governed by the floor and cannot vanish.
_MAX_COUNT_DROP_FRAC = float(os.getenv("OCR_REOCR_MAX_COUNT_DROP_FRAC", "0.25"))


def _max_count_drop(had: int) -> int:
    """How many occurrences of a value may legitimately disappear.

    Proportional so the rule holds across table widths, floored so it is never
    stricter than the flat tolerance it replaces.
    """
    return max(_MAX_COUNT_DROP, int(had * _MAX_COUNT_DROP_FRAC))


def _content_preserved(old_html: str, new_html: str) -> bool:
    """Did the re-OCR keep the original content, allowing for the duplicates a
    correct repair necessarily removes?

    A strict superset test is WRONG here, and this was measured: when the
    sideways read drops a cell, the left-shift duplicates a neighbouring group's
    values into the gap. The de-rotated read then fixes the shift, so those
    spurious copies legitimately disappear. On a real page the counts moved

        φ4.51 16->15   φ4.52 15->14   φ4.53 19->17   φ4.54 14->13

    i.e. the ``尺寸2`` group fell from 64 occurrences to its correct 60, while
    ``尺寸1`` rose 60->64 and ``ok`` rose 60->72 (recovering all 12 dropped
    ``结论`` values). A superset guard reads that as "lost 6 values" and refuses
    the very repair it should accept.

    So instead of demanding containment, require that no single value LOSES more
    than `_MAX_COUNT_DROP` occurrences and that no value vanishes entirely.
    That still blocks the failure this guard exists for — a rewritten
    measurement (``15.53`` -> ``15.58``) makes a value disappear outright, which
    is rejected — while tolerating the small multiplicity shifts a de-shift
    produces.
    """
    from collections import Counter
    old_counts = Counter(_cell_signatures(old_html))
    new_counts = Counter(_cell_signatures(new_html))
    for sig, had in old_counts.items():
        now = new_counts.get(sig, 0)
        if now == 0:
            # The value is gone entirely: either rewritten or dropped. Never OK.
            return False
        if had - now > _max_count_drop(had):
            return False
    return True


def _grid_shape(table_html: str) -> Tuple[int, int, int]:
    """(max_cols, n_rows, rows_at_full_width) for a table's HTML on the grid the
    renderer will build. `rows_at_full_width` counts rows every column of which
    is covered — by an own cell or an inherited rowspan."""
    rows = parse_html_table_rows(table_html)
    if not rows:
        return 0, 0, 0
    max_cols, n_rows, _anchors, occupied, _img = parse_table_grid(rows)
    full = 0
    for r in range(n_rows):
        if r < len(occupied) and all(
            occupied[r][c] for c in range(min(max_cols, len(occupied[r])))
        ):
            full += 1
    return max_cols, n_rows, full


def _extract_table_html(entries: List[dict]) -> Optional[str]:
    """The `text` of the single Table entry a table-crop re-OCR should return.
    Prefers the largest table by row count when the model emits more than one."""
    tables = [e for e in entries if e.get("category") == "Table" and e.get("text")]
    if not tables:
        return None
    return max(tables, key=lambda e: len(_TR_RE.findall(e["text"]))).get("text")


async def _reocr_one_rotated_table(
    table_entry: dict, page_img: Image.Image, page_index: int,
) -> bool:
    """Re-OCR one rotated table's crop upright and replace `text` when strictly
    better. Returns True when replaced. Never raises on a single bad table."""
    transpose = upright_transpose(table_entry.get("rotation"))
    if transpose is None:
        return False

    old_html = table_entry.get("text") or ""
    if "<table" not in old_html.lower():
        return False

    bbox = table_entry.get("bbox")
    if not bbox or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = (int(float(v)) for v in bbox)
    except (TypeError, ValueError):
        return False
    if x2 <= x1 or y2 <= y1:
        return False

    cx0 = max(0, x1 - _CROP_MARGIN_PX)
    cy0 = max(0, y1 - _CROP_MARGIN_PX)
    cx1 = min(page_img.width, x2 + _CROP_MARGIN_PX)
    cy1 = min(page_img.height, y2 + _CROP_MARGIN_PX)
    if cx1 <= cx0 or cy1 <= cy0:
        return False

    old_cols, old_rows, old_full = _grid_shape(old_html)
    if old_rows <= 0:
        return False

    try:
        crop = page_img.crop((cx0, cy0, cx1, cy1)).convert("RGB")
        upright = crop.transpose(transpose)
        entries = await ocr_image_async(upright)
    except Exception as exc:  # noqa: BLE001 — one bad table must not fail the page
        log.warning("[table-derotate] page %d: re-OCR failed: %s",
                    page_index, exc)
        return False

    new_html = _extract_table_html(entries or [])
    if not new_html:
        log.info("[table-derotate] page %d: re-OCR returned no table; keeping "
                 "original", page_index)
        return False

    new_cols, new_rows, new_full = _grid_shape(new_html)

    # Acceptance guard — replace ONLY when strictly better AND non-destructive:
    #   * a real table (>=1 row) that did not LOSE rows,
    #   * the grid did not get NARROWER (a re-read that finds fewer columns is a
    #     worse reading, not a repair),
    #   * strictly more complete columns overall — either a wider grid (the
    #     page-2 case: 24 -> 25) or more rows filling the width (the page-1
    #     case: 0/12 -> 12/12),
    #   * and no value was rewritten or wholesale lost (`_content_preserved`
    #     tolerates the misplaced DUPLICATES a de-shift correctly removes, but
    #     never a value disappearing).
    #
    # EXCEPTION — a COLLAPSED baseline. When the sideways read fails to find any
    # row structure at all, the whole table parses as one enormous row that
    # saturates `MAX_TABLE_COLS` (observed: `cols 40->25, rows 1->15`). Requiring
    # `new_cols >= old_cols` then lets the BROKEN parse win purely because 40 > 25,
    # which is backwards: a 1-row 40-column grid is not a table. Treat such a
    # baseline as having no usable structure, so a re-read that actually finds
    # rows may narrow the grid.
    # Measured on a real rotated page (53): the sideways read emits a HEADER that
    # parses fine plus ONE run-on data row of 697 cells, giving `rows 4, cols 40`.
    # `old_rows <= 1` misses that shape and declined the repair (1 of 7 accepted),
    # so test the RAW HTML instead: a <tr> wider than the renderer's cap cannot be
    # placed whatever the row count, which is the objective defect.
    collapsed_baseline = (
        (old_rows <= 1 and old_cols >= _COLLAPSED_COLS)
        or _widest_row_cells(old_html) > MAX_TABLE_COLS
    )
    wider = new_cols > old_cols
    more_complete = new_full > old_full
    preserved = _content_preserved(old_html, new_html)
    accept = (
        new_rows >= 1
        and new_rows >= old_rows
        and (new_cols >= old_cols or collapsed_baseline)
        and (wider or more_complete or collapsed_baseline)
        and preserved
    )

    if accept:
        table_entry["text"] = new_html
        table_entry["source"] = "derotate-reocr"
        log.info(
            "[table-derotate] page %d: replaced table (cols %d->%d, rows %d->%d, "
            "full-width rows %d->%d)",
            page_index, old_cols, new_cols, old_rows, new_rows, old_full, new_full,
        )
        return True

    log.info(
        "[table-derotate] page %d: re-OCR not an improvement (cols %d->%d, "
        "rows %d->%d, full %d->%d, content-preserved=%s); keeping original",
        page_index, old_cols, new_cols, old_rows, new_rows, old_full, new_full,
        preserved,
    )
    return False


async def reocr_rotated_tables(pages: List[dict]) -> List[dict]:
    """Re-OCR every rotated Table entry from an upright crop, replacing its HTML
    when the result is strictly better. Mutates `pages` in place.

    Must run AFTER `rotation_detect.detect_rotation` (which supplies the angle)
    and BEFORE `_release_page_rasters` (which frees the raster cropped here).
    """
    if not REOCR_ROTATED_TABLES:
        return pages

    # Collect every candidate across the whole document first, so the semaphore
    # bounds concurrency globally. Gathering per page would serialise a document
    # whose rotated tables are spread one-per-page — the common case here.
    jobs: List[Tuple[dict, Image.Image, int]] = []
    for page in pages:
        entries: List[dict] = page.get("layout_result") or []
        if not entries:
            continue
        candidates = [
            e for e in entries
            if e.get("category") == "Table"
            and upright_transpose(e.get("rotation")) is not None
        ]
        if not candidates:
            continue
        page_img: Optional[Image.Image] = (
            page.get("original_image") or page.get("image")
        )
        if page_img is None:
            continue
        page_index = int(page.get("page_index", 0))
        jobs.extend((entry, page_img, page_index) for entry in candidates)

    if not jobs:
        return pages

    sem = asyncio.Semaphore(_REOCR_CONCURRENCY)

    async def _guarded(entry: dict, page_img: Image.Image, page_index: int) -> bool:
        async with sem:
            return await _reocr_one_rotated_table(entry, page_img, page_index)

    # Each job mutates a distinct entry dict, so concurrent writes cannot race.
    # return_exceptions keeps one failed table from aborting the rest.
    results = await asyncio.gather(
        *(_guarded(*job) for job in jobs), return_exceptions=True
    )

    replaced = 0
    for job, outcome in zip(jobs, results):
        if isinstance(outcome, BaseException):
            log.warning("[table-derotate] page %d: re-OCR failed (%s); "
                        "keeping original", job[2], type(outcome).__name__)
        elif outcome:
            replaced += 1

    if replaced:
        log.info("[table-derotate] repaired %d rotated table(s)", replaced)
    return pages
