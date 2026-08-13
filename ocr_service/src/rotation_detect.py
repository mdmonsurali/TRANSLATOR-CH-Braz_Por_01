"""Detect rotated layout entries and rotated tables, and stamp `rotation`.

WHY THIS EXISTS. Chandra reports a bbox and a label for every block but says
nothing about orientation, and PaddleOCR's own orientation classifiers are
disabled in `paddle_service`. So a scan of a LANDSCAPE table on a PORTRAIT
sheet — very common for inspection/QC forms — used to arrive at reconstruction
looking upright, and rendered upright: cell text crushed into columns sized for
the wrong axis. The reconstruction side has understood `rotation` on an entry all
along (`ocr_reconstruction/geometry.py`, `table.py` Case B); nothing was
producing it outside diagrams. This module is that missing producer.

WHAT IT STAMPS. `rotation`, in the schema the renderer consumes: FLOAT DEGREES,
COUNTER-CLOCKWISE-POSITIVE, absent/0 = upright. A quarter-turned CJK label
reading bottom-to-top is `90.0`.

TWO TIERS, cheapest first, so a clean upright document costs nothing:

  1. Bbox-only (FREE, no OCR call). A tall-narrow block holding at least two
     characters is quarter-turned. This alone catches rotated side-labels — the
     sidebar `Section-Header` (59x599 px) and operator captions (74x648,
     124x719) on the document that motivated this.
  2. Region OCR (ONE Paddle pass, only for tables). A table's bbox is a poor
     rotation signal: a landscape table on a portrait page is WIDE, not tall, so
     tier 1 can never see it. Instead we OCR the table crop and take the modal
     rotation of the text lines inside it. When a clear majority of the lines
     read as quarter-turned, the whole table is turned.

ACCEPTANCE GUARD. Tier 2 stamps only on a decisive majority
(`_TABLE_ROT_MIN_RATIO`) over at least `_TABLE_ROT_MIN_LINES` lines, so a noisy
crop, a handful of stray detections, or a table with a few rotated column
headers can never rotate an upright table. That last case is real and different:
rotated headers inside an upright grid are `cell_rotations`, not a turned table.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from PIL import Image

from rotation_measure import ROT_ENABLE, line_rotation, modal_rotation

log = logging.getLogger("ocr.rotation")

# MASTER SWITCH for the whole rotation chain — detection here plus the
# de-rotated table re-OCR in `table_derotate_reocr`. DEFAULT OFF.
#
# Why off by default: tier 2 costs one PaddleOCR pass per candidate table, and
# most tables are not rotated. Measured over a 168-page document, 151 probes ran
# and only 32 found rotation — the other 119 spent that work confirming a table
# was already upright.
#
# With this false, nothing stamps `rotation`, so the reconstruction side (which
# is data-driven — see `json_to_docx._derotate_rotated_pages`) renders every page
# upright, exactly as it did before rotation support existed. Set it true for
# documents known to contain rotated tables: landscape scans of QC/inspection
# forms, where an upright render squashes the columns and can clip the table.
#
# The two gates below fall back to it, so ONE value configures the whole chain.
# Setting them explicitly overrides the master and is intended only for debugging
# a single stage — note that detection ON with re-OCR OFF is a half-configured
# state: pages get flipped to landscape but their dropped cells are not
# recovered.
ROTATION_MASTER = os.getenv("OCR_ROTATION", "false")

# Kill switch for this pass alone (the shared `OCR_ROT_ENABLE` disables rotation
# measurement everywhere, including diagram labels).
DETECT_ENABLE = os.getenv(
    "OCR_DETECT_ROTATION", ROTATION_MASTER).strip().lower() in {
    "1", "true", "yes", "on",
}

# Tier 2 runs Paddle on a table crop, so it is gated separately: it costs one OCR
# call per candidate table.
DETECT_TABLES = os.getenv(
    "OCR_DETECT_ROTATION_TABLES", ROTATION_MASTER).strip().lower() in {
    "1", "true", "yes", "on",
}

# Categories eligible for the free bbox-only test. Tables are excluded on
# purpose — see the module docstring (a rotated table is wide, not tall).
_TIER1_CATEGORIES = {
    "Text", "Section-header", "Section-Header", "Caption", "Title",
    "Footnote", "List-item", "List-Group", "Page-Header", "Page-header",
    "Page-Footer", "Page-footer",
}

# Fraction of a table's detections that must be TALLER THAN WIDE before the
# whole table is declared rotated. Measured populations are far apart —
# rotated table pages 0.97-1.00, an upright one 0.27 — so 0.80 sits in the gap
# with room for a few stray detections either way.
_TABLE_ROT_MIN_RATIO = float(os.getenv("OCR_ROT_TABLE_MIN_RATIO", "0.8"))
# Minimum usable text lines in the crop before the ratio means anything.
_TABLE_ROT_MIN_LINES = int(os.getenv("OCR_ROT_TABLE_MIN_LINES", "4"))
# Margin (px) around a table bbox when cropping — a little air helps detect
# lines flush against the outer ruling.
_CROP_MARGIN_PX = 8


def _stamp(entry: dict, rot: float, source: str) -> bool:
    """Record `rot` on `entry`. Only ever stamps a real rotation, so upright
    entries serialize byte-identically to the pre-feature output."""
    if not rot:
        return False
    entry["rotation"] = float(rot)
    entry["rotation_source"] = source
    return True


def detect_entry_rotations(entries: List[dict]) -> int:
    """Tier 1: bbox-only quarter-turn detection. Returns the number stamped.

    Free — no OCR call. Never overwrites a rotation another pass already found
    (diagram labels arrive here already stamped).
    """
    n = 0
    for entry in entries:
        if entry.get("rotation"):
            continue
        if entry.get("category") not in _TIER1_CATEGORIES:
            continue
        bbox = entry.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        text = (entry.get("text") or "").strip()
        if len(text) < 2:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        # Reuse the shared per-line rule: a tall-narrow box with real text is a
        # quarter turn. Passing no `poly` skips the skew branch, which is
        # meaningless for a layout bbox (it is axis-aligned by construction).
        rot = line_rotation((x1, y1, x2, y2), text)
        if rot and _stamp(entry, rot, "bbox-aspect"):
            n += 1
            log.info(
                "[rotation] entry %s %.0fx%.0f -> %.0f° (%s)",
                entry.get("category"), w, h, rot, (text[:24] or "").replace("\n", " "),
            )
    return n


async def _detect_table_rotation(entry: dict, page_img: Image.Image,
                                 detect) -> bool:
    """Tier 2: OCR one table's crop and stamp `rotation` when a decisive
    majority of its text lines read as quarter-turned. Returns True if stamped.
    """
    bbox = entry.get("bbox")
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

    try:
        crop = page_img.crop((cx0, cy0, cx1, cy1)).convert("RGB")
        detections = await detect(crop)
    except Exception as exc:  # noqa: BLE001 — one bad table must not fail OCR
        log.warning("[rotation] table crop OCR failed at %s: %s", bbox, exc)
        return False
    if not detections:
        return False

    # A table is judged by the AGGREGATE shape of its detections, not by
    # per-line quarter-turn classification. `line_rotation`'s aspect threshold
    # (h/w >= 2.0) is calibrated for multi-character CJK LABELS, which measure
    # 3.2-6.2; a table's cells are mostly SHORT numerics ("22", "105", "50")
    # whose boxes only reach ~1.3-2.1 even when the page is turned, so most of
    # them fall below it. Measured on a rotated inspection table, that rule
    # classified only 10-51% of lines as turned — under any sane majority.
    #
    # What DOES separate cleanly is simply how many detections are taller than
    # they are wide. Measured over the same crops (61 and 60 detections):
    #     rotated table pages   frac(h>w) = 0.97 - 1.00   median h/w 1.63-1.75
    #     upright table page    frac(h>w) = 0.27          median h/w 0.63
    # Rotated text runs down the page, so nearly EVERY box is tall — no
    # threshold on the individual aspect needed, just the direction.
    tall = 0
    total = 0
    for det in detections:
        text = (det.get("text") or "").strip()
        if not text:
            continue
        db = det.get("bbox")
        if not db or len(db) != 4:
            continue
        w = max(1.0, float(db[2]) - float(db[0]))
        h = max(1.0, float(db[3]) - float(db[1]))
        total += 1
        if h > w:
            tall += 1

    if total < _TABLE_ROT_MIN_LINES:
        return False
    ratio = tall / total
    if ratio < _TABLE_ROT_MIN_RATIO:
        log.info(
            "[rotation] table at %s: %d/%d detections tall (%.0f%% < %.0f%%) — "
            "left upright", bbox, tall, total, ratio * 100,
            _TABLE_ROT_MIN_RATIO * 100,
        )
        return False

    # Direction: a quarter-turned CJK/numeric table reads bottom-to-top, which
    # is 90° in our CCW-positive schema. Where enough lines DO clear the
    # per-line rule, let them vote on 90 vs 270; otherwise default to 90.
    votes = [
        line_rotation(det["bbox"], (det.get("text") or "").strip(), det.get("poly"))
        for det in detections
        if det.get("bbox") and len(det["bbox"]) == 4
        and len((det.get("text") or "").strip()) >= 2
    ]
    turned = [v for v in votes if v in (90.0, 270.0)]
    rot = modal_rotation(turned) if turned else 90.0

    if not _stamp(entry, rot, "table-ocr-majority"):
        return False
    log.info(
        "[rotation] table at %s -> %.0f° (%d/%d detections tall)",
        bbox, rot, tall, total,
    )
    return True


async def detect_rotation(pages: List[dict]) -> List[dict]:
    """Stamp `rotation` on rotated entries and rotated tables. Mutates `pages`
    in place and returns it.

    Strict no-op for upright documents: tier 1 is pure geometry, and tier 2 only
    OCRs tables and only stamps on a decisive majority. Must run BEFORE
    `_release_page_rasters` — tier 2 needs the page raster.
    """
    if not (DETECT_ENABLE and ROT_ENABLE):
        return pages

    # Imported lazily so the module is usable (and unit-testable) without the
    # PaddleOCR worker: tier 1 needs no OCR at all.
    detect = None
    if DETECT_TABLES:
        try:
            from diagram_text_recovery import _paddle_detect as detect
        except Exception as exc:  # noqa: BLE001
            log.warning("[rotation] PaddleOCR unavailable, tier 2 off: %s", exc)
            detect = None

    n_entries = 0
    n_tables = 0
    for page in pages:
        entries: List[dict] = page.get("layout_result") or []
        if not entries:
            continue
        n_entries += detect_entry_rotations(entries)

        if detect is None:
            continue
        page_img: Optional[Image.Image] = (
            page.get("original_image") or page.get("image")
        )
        if page_img is None:
            continue
        for entry in entries:
            if entry.get("category") != "Table" or entry.get("rotation"):
                continue
            if await _detect_table_rotation(entry, page_img, detect):
                n_tables += 1

    if n_entries or n_tables:
        log.info("[rotation] stamped %d entr%s and %d table(s)",
                 n_entries, "y" if n_entries == 1 else "ies", n_tables)
    return pages
