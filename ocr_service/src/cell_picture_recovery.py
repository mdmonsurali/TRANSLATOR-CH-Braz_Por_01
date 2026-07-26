"""Recover table-cell photos Chandra emitted as bare ``<img alt>`` placeholders.

Chandra's layout prompt tells the model to describe pictures inside a table as
``<img alt="...">`` tags in the ``Table`` block's HTML — alt text only, **no
pixels and no per-cell bbox**. The reconstruction table parser drops ``<img>``
entirely, so those cells render blank. This is distinct from standalone
``Image``/``Figure`` blocks, which carry their own bbox and crop/place fine.

``picture_recovery`` can't help here: it recovers *embedded PDF rasters* inside a
table bbox and is a no-op on scanned pages (no embedded raster) and image inputs.
The photos on a scan live only as ink on the page raster.

Public surface
--------------
recover_table_cell_pictures(pages) — mutates each page dict in place, adding a
synthesized ``Image`` entry (with an ``image_obj`` PIL crop) for every table cell
whose OCR content was an ``<img>`` placeholder. The crop is located by
**content-blob detection** down the image column, because the reconstructed table
grid flexes rows by text length and does NOT match the scan's physical rows, and
scan rulings are typically too faint to detect. Works for any page that carries a
rendered ``original_image`` (scanned PDFs and image uploads alike).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from bs4 import BeautifulSoup
from PIL import Image

# The reconstruction grid functions — reused so our column indexing / widths
# match exactly what the renderer will draw, keeping the synthesized picture's
# bbox in the column the renderer expects.
from ocr_reconstruction.table import (
    compute_col_weights,
    parse_html_table_rows,
    parse_table_grid,
)

log = logging.getLogger("ocr_service")


# Blob detection tuning. A pixel is "ink" below this 8-bit gray level.
INK_LEVEL = 180
# A row is "occupied" when at least ROW_INK_FRAC of the column strip is ink.
# This must be LOW: a photo band fills most of a row (high ink), but a technical
# LINE DRAWING (a screw / diagram) is mostly white — its outline touches only a
# few percent of any given row, so a high threshold (the old 0.12) collapsed
# such bands into stray slivers or dropped them entirely. A low per-row gate
# catches line-art; band-level total-ink gating (below) rejects the false
# positives (blank cell padding, a single thin ruling) that a low gate lets in.
ROW_INK_FRAC = 0.02
# A detected band is kept only if its interior carries at least this fraction of
# ink overall — enough to distinguish a real drawing/photo from empty padding or
# a lone horizontal rule. Line drawings clear this comfortably; blank space does
# not.
BAND_INK_FRAC = 0.012
MIN_BAND_PX = 20
# Merge photo bands separated by a gap smaller than this (a single photo/drawing
# can have a light horizontal seam that briefly drops below ROW_INK_FRAC — e.g.
# the gap between a screw's head and its shaft hatching).
MERGE_GAP_PX = 24
# Inset the crop a couple px so we don't grab the cell border line.
CROP_INSET_PX = 2


def _image_columns(
    table_html: str,
) -> Tuple[int, Dict[int, List[str]]]:
    """Return (max_cols, {col_index: [alt, ...]}) for the ``<img>`` cells in
    `table_html`, indexed on the SAME grid the renderer builds. The alt list is
    in top-to-bottom reading order within the column."""
    rows = parse_html_table_rows(table_html)
    if not rows:
        return 0, {}
    max_cols, n_rows, _cell_anchors, _occ, _img = parse_table_grid(rows)

    # Re-walk the HTML rows placing cells into the same occupancy grid
    # parse_table_grid uses, so an <img> cell's (row, col) matches the anchors.
    soup = BeautifulSoup(table_html, "html.parser")
    tr_list = soup.find_all("tr")
    occ = [[False] * max_cols for _ in range(n_rows)]
    img_cols: Dict[int, List[str]] = {}
    for r_idx, tr in enumerate(tr_list):
        if r_idx >= n_rows:
            break
        cells = tr.find_all(["td", "th"], recursive=False)
        col = 0
        for cell in cells:
            while col < max_cols and occ[r_idx][col]:
                col += 1
            if col >= max_cols:
                break
            try:
                cs = int(cell.get("colspan", 1) or 1)
                rs = int(cell.get("rowspan", 1) or 1)
            except (TypeError, ValueError):
                cs = rs = 1
            cs = min(max(1, cs), max_cols - col)
            rs = min(max(1, rs), n_rows - r_idx)
            img = cell.find("img")
            # Only recover a photo for a cell that is a PURE image placeholder —
            # an <img> and nothing else meaningful. A cell that mixes an <img>
            # with text (e.g. a header logo cell "WEGO ORTHO 威高骨科") is a text
            # cell: cropping it would grab logo+text and float it over the
            # layout. Such cells keep today's behaviour (render the text; the
            # <img> alt is dropped by the table parser), so nothing overlaps.
            if img is not None and not cell.get_text(strip=True):
                img_cols.setdefault(col, []).append(img.get("alt", "") or "")
            for rr in range(r_idx, r_idx + rs):
                for cc in range(col, col + cs):
                    occ[rr][cc] = True
            col += cs
    return max_cols, img_cols


def _column_x_edges(
    table_html: str, tbl_x1: int, tbl_x2: int, max_cols: int,
) -> List[float]:
    """Pixel x-edges of every column, proportional to `compute_col_weights`
    (horizontally accurate against the source, unlike row heights)."""
    rows = parse_html_table_rows(table_html)
    _mc, _nr, cell_anchors, _occ, _img = parse_table_grid(rows)
    weights = compute_col_weights(cell_anchors, max_cols)
    wsum = sum(weights) or max_cols
    edges = [float(tbl_x1)]
    span = tbl_x2 - tbl_x1
    for w in weights:
        edges.append(edges[-1] + span * (w / wsum))
    return edges


def _detect_photo_bands(strip: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous (y0, y1) bands of picture content down a grayscale column strip.

    Works for both dense photos AND sparse line drawings (technical screw /
    diagram art). A band is a run of rows whose ink fraction exceeds the low
    ROW_INK_FRAC gate; runs are merged across gaps < MERGE_GAP_PX, runs shorter
    than MIN_BAND_PX dropped, and — critically — a surviving band is kept only if
    its interior ink fraction clears BAND_INK_FRAC. The low per-row gate lets
    line-art through (its outline touches few pixels per row); the band-level
    gate then throws out blank padding and lone thin rulings that the low gate
    would otherwise admit.
    """
    if strip.ndim != 2 or strip.shape[0] == 0 or strip.shape[1] == 0:
        return []
    ink = strip < INK_LEVEL
    ink_frac = ink.mean(axis=1)
    mask = ink_frac > ROW_INK_FRAC

    bands: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for y, on in enumerate(mask):
        if on and start is None:
            start = y
        elif not on and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(mask)))

    # Merge bands separated by a small gap (a light seam within one photo, or the
    # break between a screw head and its shaft).
    merged: List[Tuple[int, int]] = []
    for b in bands:
        if merged and b[0] - merged[-1][1] < MERGE_GAP_PX:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)

    kept: List[Tuple[int, int]] = []
    for (a, z) in merged:
        if (z - a) < MIN_BAND_PX:
            continue
        # Reject bands that are mostly blank (padding) or a single hairline rule:
        # a genuine drawing/photo fills at least BAND_INK_FRAC of its area.
        if ink[a:z, :].mean() < BAND_INK_FRAC:
            continue
        kept.append((a, z))
    return kept


def _equal_bands(height: int, n: int) -> List[Tuple[int, int]]:
    """Fallback: split `height` into `n` equal bands (used when blob detection
    doesn't return exactly one band per image cell)."""
    if n <= 0:
        return []
    step = height / n
    return [(int(round(i * step)), int(round((i + 1) * step))) for i in range(n)]


def _x_overlaps(a0: float, a1: float, b0: float, b1: float) -> float:
    """Fraction of [a0,a1] covered by [b0,b1] (0..1)."""
    span = max(1.0, a1 - a0)
    return max(0.0, min(a1, b1) - max(a0, b0)) / span


def _bbox_center_inside(
    inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int],
) -> bool:
    """True when `inner`'s centre lies within `outer` — a lenient containment
    test (a table picture's bbox may poke a few px past the table border)."""
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _picture_bboxes(entries: List[Dict]) -> List[Tuple[int, int, int, int]]:
    """Bboxes of every existing picture entry (Image/Figure/Picture) — the ones
    recover_missing_pictures injected plus any the VLM emitted directly. Our own
    cell-recovered entries are excluded (they're added as we go)."""
    out: List[Tuple[int, int, int, int]] = []
    for e in entries:
        if e.get("category") not in ("Image", "Figure", "Picture"):
            continue
        if e.get("source") == "cell-recovered":
            continue
        bb = e.get("bbox")
        if bb and len(bb) == 4:
            out.append(tuple(int(v) for v in bb))
    return out


def _recover_one_table(
    table_entry: Dict, page_img: Image.Image,
    existing_pic_bboxes: List[Tuple[int, int, int, int]],
) -> List[Dict]:
    """Synthesize an ``Image`` entry per ``<img>`` cell in `table_entry`.

    Skips any image COLUMN that an existing picture already covers — on a native
    PDF, ``recover_missing_pictures`` extracts the real embedded diagram rasters
    (correct bbox) *before* this step runs, so re-cropping the same column by
    blob detection would duplicate them (and our blob crop is often wrong on a
    dense multi-diagram column). Existing pictures are the source of truth.

    Returns a list of new entries (possibly empty). Never raises on a single bad
    table — logs and returns what it could recover.
    """
    html = table_entry.get("text") or ""
    bbox = table_entry.get("bbox")
    if not html or not bbox or len(bbox) != 4:
        return []
    tbl_x1, tbl_y1, tbl_x2, tbl_y2 = (int(v) for v in bbox)
    if tbl_x2 <= tbl_x1 or tbl_y2 <= tbl_y1:
        return []

    max_cols, img_cols = _image_columns(html)
    if not img_cols:
        return []

    # Pictures already living inside this table (from recover_missing_pictures
    # or standalone Image/Figure blocks the VLM emitted).
    pics_in_table = [
        p for p in existing_pic_bboxes
        if _bbox_center_inside(p, (tbl_x1, tbl_y1, tbl_x2, tbl_y2))
    ]

    col_edges = _column_x_edges(html, tbl_x1, tbl_x2, max_cols)
    gray_page = np.asarray(page_img.convert("L")).astype(np.float32)
    page_h, page_w = gray_page.shape

    new_entries: List[Dict] = []
    for col, alts in sorted(img_cols.items()):
        if col + 1 >= len(col_edges):
            continue
        col_x0, col_x1 = col_edges[col], col_edges[col + 1]
        # Skip this column if an existing picture already covers most of it —
        # the real image is already placed, so re-cropping would duplicate it.
        if any(
            _x_overlaps(col_x0, col_x1, p[0], p[2]) >= 0.5 for p in pics_in_table
        ):
            log.info(
                "[cell-recover] col %d already covered by an existing "
                "picture; skipping to avoid a duplicate crop", col,
            )
            continue
        cx0 = max(0, min(page_w - 1, int(col_x0)))
        cx1 = max(cx0 + 1, min(page_w, int(col_x1)))
        cy0 = max(0, min(page_h - 1, tbl_y1))
        cy1 = max(cy0 + 1, min(page_h, tbl_y2))

        strip = gray_page[cy0:cy1, cx0:cx1]
        bands = _detect_photo_bands(strip)
        n_want = len(alts)
        # Map blobs to the image cells in reading order. Detection can return a
        # few spurious short bands (a header ruling, a stray text line) on top of
        # the real photos, so when we have MORE bands than cells, keep the
        # `n_want` TALLEST (photos are substantially taller than incidental
        # rulings/text) and restore top-to-bottom order. Only when we have FEWER
        # bands than cells (photos merged or undetectable) do we fall back to an
        # equal split.
        if len(bands) > n_want:
            bands = sorted(
                sorted(bands, key=lambda b: b[1] - b[0], reverse=True)[:n_want]
            )
        elif len(bands) < n_want:
            log.info(
                "[cell-recover] col %d: %d blob(s) < %d img cell(s); "
                "using equal-band fallback", col, len(bands), n_want,
            )
            bands = _equal_bands(cy1 - cy0, n_want)

        for (a, z), alt in zip(bands, alts):
            y0 = max(cy0, cy0 + a + CROP_INSET_PX)
            y1 = min(cy1, cy0 + z - CROP_INSET_PX)
            x0 = min(page_w - 1, cx0 + CROP_INSET_PX)
            x1 = max(x0 + 1, min(page_w, cx1 - CROP_INSET_PX))
            if y1 <= y0 or x1 <= x0:
                continue
            crop = page_img.crop((x0, y0, x1, y1))
            new_entries.append({
                "bbox": [x0, y0, x1, y1],
                "category": "Image",
                "text": alt,
                "image_obj": crop,
                "source": "cell-recovered",
            })
    return new_entries


def _insert_in_reading_order(entries: List[Dict], new_entry: Dict) -> None:
    """Insert `new_entry` before the first entry whose bbox top sits below it,
    keeping the layout list in rough reading order."""
    new_top = (new_entry.get("bbox") or [0, 0, 0, 0])[1]
    for i, e in enumerate(entries):
        bb = e.get("bbox") or [0, 0, 0, 0]
        if len(bb) >= 2 and bb[1] > new_top:
            entries.insert(i, new_entry)
            return
    entries.append(new_entry)


def recover_table_cell_pictures(pages: List[Dict]) -> List[Dict]:
    """For every page with a rendered raster, crop table cells whose OCR content
    was an ``<img>`` placeholder and inject a synthesized ``Image`` entry per
    photo. Returns the same `pages` list, mutated in place.

    Each synthesized entry carries bbox (page-image pixels, inside the owning
    table's bbox and centred in the image column), category ``Image``, text
    (the alt string), an ``image_obj`` PIL crop ready for ``process_pictures``,
    and ``source = "cell-recovered"``.
    """
    for page in pages:
        page_img: Optional[Image.Image] = (
            page.get("original_image") or page.get("image")
        )
        if page_img is None:
            continue
        entries: List[Dict] = page.get("layout_result") or []
        tables = [e for e in entries if e.get("category") == "Table"]
        if not tables:
            continue

        # Snapshot pictures that already exist (native-PDF embedded rasters
        # recovered upstream, or standalone VLM Image/Figure blocks) so we can
        # skip any table column they already cover and never duplicate them.
        existing_pic_bboxes = _picture_bboxes(entries)

        recovered = 0
        for table in tables:
            for new_entry in _recover_one_table(table, page_img, existing_pic_bboxes):
                _insert_in_reading_order(entries, new_entry)
                recovered += 1

        if recovered:
            log.info(
                "[cell-recover] page %s: recovered %d table-cell picture(s)",
                page.get("page_index", "?"), recovered,
            )
        page["layout_result"] = entries
    return pages
