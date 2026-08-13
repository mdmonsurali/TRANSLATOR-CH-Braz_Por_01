"""One table-correctness validator: measure health, repair once, keep the better.

WHY THIS REPLACES FOUR PASSES. The service accumulated one re-OCR pass per
observed failure shape — rotated tables, collapsed rows, dropped image rows — each
with its own trigger, threshold, acceptance guard and tests. Three of them cropped
the SAME table bbox and could each spend a model call on it, and the collapse
pass's trigger was already contained in the de-rotate pass's guard. Every new
failure mode wanted a fifth pass.

The generalisation: stop asking "which known way did this break?" and ask "is this
table healthy?". Every failure observed so far violates one measurable property of
the emitted HTML, and all of the measurements already existed somewhere:

    unplaceable  a <tr> wider than MAX_TABLE_COLS cannot be placed at all — the
                 renderer's grid is capped, so the overflow is silently dropped
                 (measured: 701-cell row -> 67 of 728 cells placed)
    placement    placed/emitted cells on the grid the renderer will build
    completeness rows every column of which is covered (`rows_at_full_width`)
    degeneracy   one value dominating the cells — the `ok`-loop shape (measured:
                 678 of 697 cells were the literal "ok")
    truncation   the generation hit the output cap mid-table

WHY A COMPARATIVE SCORE. The old guards demanded a re-read beat the baseline on
EVERY axis while implicitly trusting that baseline. On a table known to be
sideways that is backwards, and it showed: on the 24-page rotated subset all 8
repairs were declined, including `cols 26->25, rows 15->15, full 12->15` — a
re-read that filled three empty rows, rejected because it had one fewer column.
Here both reads are scored the same way and the better one wins, so a repair that
trades a spurious column for three complete rows is accepted.

`_content_preserved` still has the final say: a re-read that rewrote or lost a
value is refused however good its shape, because a plausible-looking table with
invented measurements is worse than an ugly one with real ones.

Public surface
--------------
table_health(html, truncated=False) -> TableHealth   pure, no model call
validate_tables(pages)                               async; repair in place
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import httpx
from PIL import Image

from chandra_ocr import _repeats_one_cell, ocr_image_async
# Reuse the proven primitives rather than reimplementing them: the same grid
# parser the renderer uses, the multiset content guard, the raw-HTML row width,
# and the rotation->transpose mapping.
from table_derotate_reocr import (
    _content_preserved,
    _extract_table_html,
    _grid_shape,
    _widest_row_cells,
    upright_transpose,
)
from ocr_reconstruction.table import (
    MAX_TABLE_COLS,
    parse_html_table_rows,
    parse_table_grid,
)

log = logging.getLogger("ocr_service")

VALIDATE_TABLES = os.getenv("OCR_VALIDATE_TABLES", "true").strip().lower() in {
    "1", "true", "yes", "on"}

# Compare the emitted column count against the RULED LINES in the crop. This is
# the only ground-truth signal available: every other check reads the model's own
# output, so a table that is internally consistent but missing whole columns looks
# perfectly healthy. Measured case: a 7-column form emitted as 4 columns, placing
# 43 of 43 cells — healthy by every HTML-only metric, yet three columns of real
# data (周期, 位置, 背压) were never transcribed.
GRID_CHECK = os.getenv("OCR_VALIDATE_GRID", "true").strip().lower() in {
    "1", "true", "yes", "on"}
PADDLE_SERVICE_URL = os.getenv("PADDLE_SERVICE_URL", "http://paddle_service:8005")
_GRID_TIMEOUT_SEC = float(os.getenv("OCR_VALIDATE_GRID_TIMEOUT_SEC", "60"))

# Flag when the ruling implies at least this many times more columns than the
# model emitted. Measured on a 4-page document, 13 tables: healthy tables scored
# 0.29-1.29 and every table at 1.5+ was genuinely defective (missing columns or
# ragged rows missing their trailing cells). 1.5 sits in the gap, and being a
# RATIO it adapts to table width instead of assuming a fixed column count.
_GRID_COL_RATIO = float(os.getenv("OCR_VALIDATE_GRID_RATIO", "1.5"))
# Below this the ruling is too sparse to trust (a borderless or part-ruled table).
_GRID_MIN_COLS = int(os.getenv("OCR_VALIDATE_GRID_MIN_COLS", "3"))

# Margin (px) around the table bbox when cropping. Matches the sibling passes.
_CROP_MARGIN_PX = 12

# Concurrent re-OCR calls. A table crop costs about what a page costs, so this is
# bounded the way page OCR is bounded in `main._ocr_pages_concurrently`.
_CONCURRENCY = max(1, int(os.getenv("OCR_VALIDATE_CONCURRENCY", "4")))

# A grid this sparse is missing real content. 0.85 sits below every healthy table
# measured in four stored runs (a legitimately ragged table still places ~0.9 of
# its cells) and well above the collapsed shape (67/728 = 0.09).
_MIN_PLACED_FRAC = float(os.getenv("OCR_VALIDATE_MIN_PLACED", "0.85"))


@dataclass(frozen=True)
class TableHealth:
    """Everything known about one table's shape, with no model call."""
    cols: int
    rows: int
    full_rows: int
    emitted: int          # cells the model actually wrote
    placed: int           # cells the renderer can actually place
    widest_row: int
    degenerate: bool
    truncated: bool
    # Columns implied by the crop's ruled lines. 0 means "not measured" (the probe
    # is off, the crop had no usable ruling, or paddle_service was unreachable),
    # which must never be read as "no columns".
    ruled_cols: int = 0

    @property
    def missing_columns(self) -> bool:
        """True when the ruling implies materially more columns than were
        emitted. The only check backed by the image rather than the model."""
        if self.ruled_cols < _GRID_MIN_COLS or self.cols < 1:
            return False
        return self.ruled_cols >= self.cols * _GRID_COL_RATIO

    @property
    def placed_frac(self) -> float:
        return (self.placed / self.emitted) if self.emitted else 0.0

    @property
    def full_frac(self) -> float:
        return (self.full_rows / self.rows) if self.rows else 0.0

    @property
    def unplaceable(self) -> bool:
        """A row wider than the renderer's cap: cells CANNOT all be placed. An
        objective defect, not a judgement about what a table should look like."""
        return self.widest_row > MAX_TABLE_COLS

    @property
    def healthy(self) -> bool:
        return not (self.unplaceable or self.degenerate or self.truncated
                    or self.missing_columns
                    or self.placed_frac < _MIN_PLACED_FRAC)

    def reasons(self) -> List[str]:
        out = []
        if self.unplaceable:
            out.append("row of %d cells exceeds the %d-col grid"
                       % (self.widest_row, MAX_TABLE_COLS))
        if self.degenerate:
            out.append("one value dominates the cells")
        if self.truncated:
            out.append("generation hit the output cap")
        if self.missing_columns:
            out.append("ruling shows %d columns but %d were emitted"
                       % (self.ruled_cols, self.cols))
        if self.placed_frac < _MIN_PLACED_FRAC:
            out.append("only %d of %d cells placeable" % (self.placed, self.emitted))
        return out

    def score(self) -> Tuple[int, int, float, float, int]:
        """Comparative quality, higher is better. Ordered by how badly each
        property corrupts the rendered table:

        1. not degenerate — a table of one repeated value carries no data;
        2. not missing columns — whole columns absent is the worst SURVIVABLE
           defect, because the cells that are present look correct; ranking it
           above placed_frac lets a wider read win even if it places a slightly
           smaller fraction of its (now larger) cell count;
        3. placed fraction — cells the reader will actually see;
        4. full-row fraction — completeness of the rows that do exist;
        5. placed count — tie-break toward the read that carries more content.
        """
        return (0 if self.degenerate else 1,
                0 if self.missing_columns else 1,
                round(self.placed_frac, 3),
                round(self.full_frac, 3),
                self.placed)


def table_health(table_html: str, truncated: bool = False,
                 ruled_cols: int = 0) -> TableHealth:
    """Measure one table. Pure and cheap — safe to run on every table.

    `ruled_cols` is the column count observed in the image (see `ruled_columns`);
    pass 0 when it was not measured, which disables the ground-truth check rather
    than reporting a table as empty.
    """
    if not table_html or "<table" not in table_html.lower():
        return TableHealth(0, 0, 0, 0, 0, 0, False, truncated, ruled_cols)

    rows = parse_html_table_rows(table_html)
    if not rows:
        return TableHealth(0, 0, 0, 0, 0, _widest_row_cells(table_html),
                           False, truncated, ruled_cols)

    max_cols, n_rows, anchors, _occupied, _img = parse_table_grid(rows)
    cols, grid_rows, full = _grid_shape(table_html)
    # `emitted` counts what the model wrote; `placed` counts what survives onto
    # the renderer's grid. The gap is the silent loss.
    emitted = sum(len(cells) for cells, _hdr in rows)
    return TableHealth(
        cols=cols or max_cols,
        rows=grid_rows or n_rows,
        full_rows=full,
        emitted=emitted,
        placed=len(anchors),
        widest_row=_widest_row_cells(table_html),
        degenerate=_repeats_one_cell(table_html),
        truncated=truncated,
        ruled_cols=ruled_cols,
    )


async def ruled_columns(crop: Image.Image) -> int:
    """Columns implied by the ruled lines of `crop`, or 0 when unmeasurable.

    Returns 0 — never raises and never guesses — when the probe is disabled,
    paddle_service is unreachable, or the ruling is too sparse to read. A table is
    then judged on the HTML-only signals exactly as before, so this can only add
    detections, never remove them.
    """
    if not GRID_CHECK:
        return 0
    try:
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        async with httpx.AsyncClient(timeout=_GRID_TIMEOUT_SEC) as client:
            resp = await client.post(
                PADDLE_SERVICE_URL.rstrip("/") + "/detect_grid",
                content=buf.getvalue(),
                headers={"Content-Type": "application/octet-stream"},
            )
        resp.raise_for_status()
        payload = resp.json()
        # `ruled` false means the crop has no grid the probe can read (an unruled
        # block, a table of contents). Its column count is noise, so report 0 —
        # "unmeasurable" — rather than a number the health check would act on.
        if not payload.get("ok") or not payload.get("ruled"):
            return 0
        return int(payload.get("cols") or 0)
    except Exception as exc:  # noqa: BLE001 — a probe failure must not fail OCR
        log.warning("[table-validate] grid probe unavailable: %s: %s",
                    type(exc).__name__, exc)
        return 0


def _crop_for(entry: dict, page_img: Image.Image) -> Optional[Image.Image]:
    """The table's crop, turned upright when rotation detection stamped an angle.

    De-rotating first is why a rotated table can be repaired at all: read
    sideways, its columns land on the page's SHORT axis (~64px per column), so
    the model drops cells and shifts the survivors under the wrong header.
    """
    bbox = entry.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    except (TypeError, ValueError):
        return None

    cx0 = max(0, x1 - _CROP_MARGIN_PX)
    cy0 = max(0, y1 - _CROP_MARGIN_PX)
    cx1 = min(page_img.width, x2 + _CROP_MARGIN_PX)
    cy1 = min(page_img.height, y2 + _CROP_MARGIN_PX)
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None

    crop = page_img.crop((cx0, cy0, cx1, cy1)).convert("RGB")
    transpose = upright_transpose(entry.get("rotation"))
    if transpose is not None:
        crop = crop.transpose(transpose)
    return crop


async def _repair_one(entry: dict, page_img: Image.Image, page_index: int,
                      health: TableHealth) -> bool:
    """Re-OCR one unhealthy table and keep whichever read scores better."""
    old_html = entry.get("text") or ""
    crop = _crop_for(entry, page_img)
    if crop is None:
        return False

    try:
        entries = await ocr_image_async(crop)
    except Exception as exc:  # noqa: BLE001 — one bad table must not fail the page
        log.warning("[table-validate] page %d: re-OCR failed: %s", page_index, exc)
        return False

    new_html = _extract_table_html(entries or [])
    if not new_html:
        log.info("[table-validate] page %d: re-OCR returned no table; keeping "
                 "original", page_index)
        return False

    # Judge the re-read against the SAME ruling, so `missing_columns` is
    # comparable on both sides — otherwise a re-read that is still missing its
    # columns would score as if that defect did not exist and could win.
    new = table_health(new_html, ruled_cols=health.ruled_cols)
    better = new.score() > health.score()
    preserved = _content_preserved(old_html, new_html)

    if better and preserved:
        entry["text"] = new_html
        entry["source"] = "validate-reocr"
        log.info(
            "[table-validate] page %d: replaced table (placed %d/%d->%d/%d, "
            "grid %dx%d->%dx%d, full rows %d->%d)",
            page_index, health.placed, health.emitted, new.placed, new.emitted,
            health.cols, health.rows, new.cols, new.rows,
            health.full_rows, new.full_rows,
        )
        return True

    log.info(
        "[table-validate] page %d: keeping original (re-OCR placed %d/%d vs "
        "%d/%d, full rows %d vs %d, better=%s, content-preserved=%s)",
        page_index, new.placed, new.emitted, health.placed, health.emitted,
        new.full_rows, health.full_rows, better, preserved,
    )
    return False


async def validate_tables(pages: List[dict]) -> List[dict]:
    """Measure every table, re-OCR the unhealthy ones once, keep the better read.

    Mutates `pages`. Must run BEFORE `_release_page_rasters`, which frees the
    raster cropped here. Independent of `OCR_ROTATION`: rotation only supplies
    the angle used to turn a crop upright, and unhealthy upright tables are
    repaired either way.
    """
    if not VALIDATE_TABLES:
        return pages

    # Pass 1 — measure. Candidates are every table with pixels available; the
    # ruled-line probe runs on all of them because a table that looks healthy in
    # HTML is exactly the case it exists to catch (43 of 43 cells placed, three
    # columns missing). The probe is CPU-only morphology, no model, so this is
    # cheap relative to a single OCR call.
    candidates: List[Tuple[dict, Image.Image, int]] = []
    checked = 0
    for page in pages:
        entries: List[dict] = page.get("layout_result") or []
        if not entries:
            continue
        page_img: Optional[Image.Image] = (
            page.get("original_image") or page.get("image")
        )
        page_index = int(page.get("page_index", 0))
        for entry in entries:
            if entry.get("category") != "Table" or not entry.get("text"):
                continue
            checked += 1
            if page_img is None:
                continue                  # no pixels -> nothing to measure or fix
            candidates.append((entry, page_img, page_index))

    probe_sem = asyncio.Semaphore(_CONCURRENCY)

    async def _measure(entry: dict, img: Image.Image, idx: int) -> Optional[
            Tuple[dict, Image.Image, int, TableHealth]]:
        crop = _crop_for(entry, img)
        async with probe_sem:
            ruled = await ruled_columns(crop) if crop is not None else 0
        health = table_health(entry["text"], bool(entry.get("truncated")), ruled)
        if health.healthy:
            return None
        log.info("[table-validate] page %d: unhealthy table (%s) — re-OCRing",
                 idx, "; ".join(health.reasons()))
        return (entry, img, idx, health)

    measured = await asyncio.gather(
        *(_measure(*c) for c in candidates), return_exceptions=True
    )
    jobs: List[Tuple[dict, Image.Image, int, TableHealth]] = []
    for outcome in measured:
        if isinstance(outcome, BaseException):
            log.warning("[table-validate] health check raised: %s", outcome)
        elif outcome is not None:
            jobs.append(outcome)

    if not jobs:
        if checked:
            log.info("[table-validate] %d table(s) checked, all healthy", checked)
        return pages

    sem = asyncio.Semaphore(_CONCURRENCY)

    async def _guarded(entry: dict, img: Image.Image, idx: int,
                       health: TableHealth) -> bool:
        async with sem:
            return await _repair_one(entry, img, idx, health)

    # Each job mutates a distinct entry dict, so concurrent writes cannot race.
    # return_exceptions keeps one failed table from aborting the rest.
    results = await asyncio.gather(
        *(_guarded(*job) for job in jobs), return_exceptions=True
    )

    replaced = 0
    for job, outcome in zip(jobs, results):
        if isinstance(outcome, BaseException):
            log.warning("[table-validate] page %d: repair raised: %s",
                        job[2], outcome)
        elif outcome:
            replaced += 1
    log.info("[table-validate] %d table(s) checked, %d unhealthy, %d replaced",
             checked, len(jobs), replaced)
    return pages
