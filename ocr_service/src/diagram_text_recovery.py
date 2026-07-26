"""Recover the TEXT LABELS inside a recovered diagram so they can be translated
and overlaid in place.

Chandra converts a ``Diagram`` block to mermaid *text* and loses the drawing, so
``chandra_ocr.parse_layout_html`` keeps the original crop as an ``Image`` stamped
``source="diagram-recovered"`` (the mermaid text survives only as a fallback).
That means the boxes/diamonds/edge labels inside the flowchart (开始, 风险是否需要
降低?, 是/否, …) render as the original language — the reconstruction side places
the raster untouched.

Detection uses **PaddleOCR**, NOT Chandra: re-OCRing a diagram crop with Chandra
just collapses the connected flowchart back into one mermaid block with no
coordinates. PaddleOCR returns tight per-line boxes + text (incl. edge labels).
Because PaddleOCR's deps (``numpy<2.6``/``opencv<4.13``) conflict with vLLM's, it
runs in an ISOLATED venv (``/opt/paddle-venv``) invoked as a subprocess — see
``paddle_ocr_worker.py``.

Each detected line becomes a ``{bbox (page px), text, style}`` dict; the regions
are attached as ``entry["diagram_regions"]``. The translator then translates each
region's ``text`` and the reconstruction side paints a white-filled textbox over
the original text and drops the translation on top — the drawing stays identical.

Public surface
--------------
recover_diagram_text(pages) — async; for every ``diagram-recovered`` Image entry,
attach ``diagram_regions``. Mutates ``pages`` in place and returns it. A no-op for
pages with no such entry (and no PaddleOCR subprocess is spawned).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from typing import List, Optional

from PIL import Image

log = logging.getLogger("ocr_service")

# Font-size detection for diagram labels — the same box-HEIGHT heuristic
# chandra_style uses for scanned text (size ≈ box_h_pt * 0.7), but tuned for a
# PaddleOCR label box (single line, legitimately small, no reset-to-default).
_SIZE_HEIGHT_FACTOR = 0.7   # matches chandra_style._heuristic_style
_MIN_SIZE_PT = 4.0          # diagram edge labels (是/否) are small
_MAX_SIZE_PT = 28.0

# Line→node clustering. PaddleOCR's detector segments text LINE-by-line, but a
# flowchart node (box/diamond) is one logical label that can span several lines.
# Two lines belong to the same node when they overlap horizontally AND sit close
# vertically. Thresholds are RELATIVE to the lines' own size (never fixed pixels)
# so the rule is document-independent:
#   * horizontal: the shorter line's x-span must be ≥ this fraction covered by
#     the other (stacked node lines share a centred column).
#   * vertical: the gap between one line's bottom and the next's top must be
#     ≤ this multiple of the local line height (node line-spacing is tight; the
#     gap to a DIFFERENT node — across an arrow / whitespace — is far larger),
#     AND ≤ a cap tied to the diagram's TYPICAL (median) line height. The cap
#     matters for tall rotated side-labels: their boxes are hundreds of px high,
#     so a height-relative gap alone would bridge the whitespace between two
#     separate vertical labels. The median-based cap keeps the rule to normal
#     line-spacing regardless of any one box's height.
_CLUSTER_X_OVERLAP_FRAC = 0.30
_CLUSTER_Y_GAP_FACTOR = 0.8
_CLUSTER_Y_GAP_CAP_FACTOR = 1.2   # × median line height (absolute-ish ceiling)

# Isolated PaddleOCR venv + worker (baked into the image; see ocr_service
# Dockerfile). Overridable for local dev / tests.
PADDLE_PYTHON = os.getenv("PADDLE_VENV_PYTHON", "/opt/paddle-venv/bin/python")
PADDLE_WORKER = os.getenv(
    "PADDLE_WORKER",
    os.path.join(os.path.dirname(__file__), "paddle_ocr_worker.py"),
)
PADDLE_LANG = os.getenv("DIAGRAM_OCR_LANG", "ch")
PADDLE_TIMEOUT_SEC = float(os.getenv("DIAGRAM_OCR_TIMEOUT_SEC", "300"))
# Drop low-confidence detections (PaddleOCR emits stray marks like "+"/"一" on
# arrowheads). Kept low (0.45) so faint 是/否 edge labels — recovered by the
# worker's contrast preprocessing but still lower-confidence — survive; the
# alpha-only filter below removes the punctuation noise the low floor lets in.
MIN_CONF = float(os.getenv("DIAGRAM_OCR_MIN_CONF", "0.45"))

# Margin (px) around the diagram bbox when cropping — a little air helps detect
# labels that sit flush against a box ruling.
_CROP_MARGIN_PX = 8


async def _paddle_detect(crop: Image.Image) -> Optional[List[dict]]:
    """Run the isolated PaddleOCR worker on `crop`. Returns a list of
    ``{bbox (crop-local px), text, conf}`` or None on failure."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
            crop.save(f, format="PNG")

        proc = await asyncio.create_subprocess_exec(
            PADDLE_PYTHON, PADDLE_WORKER, tmp_path, PADDLE_LANG,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(), timeout=PADDLE_TIMEOUT_SEC
            )
        except asyncio.TimeoutError:
            proc.kill()
            log.warning("[diagram-text] PaddleOCR worker timed out")
            return None

        if proc.returncode != 0:
            log.warning("[diagram-text] PaddleOCR worker failed (rc=%s): %s",
                        proc.returncode, (err or b"").decode("utf-8", "replace")[-500:])
            return None
        return json.loads((out or b"").decode("utf-8") or "[]")
    except Exception as exc:  # noqa: BLE001 — one bad diagram must not fail OCR
        log.warning("[diagram-text] PaddleOCR invocation error: %s", exc)
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _x_overlap_frac(a: List[float], b: List[float]) -> float:
    """Fraction of the NARROWER x-span [x0,x2] that overlaps the other's."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    narrow = max(1.0, min(a[2] - a[0], b[2] - b[0]))
    return inter / narrow


def _cluster_lines(lines: List[dict]) -> List[List[dict]]:
    """Group PaddleOCR line boxes into per-node clusters using geometry only.

    Two lines join the same node when they overlap horizontally and are
    vertically adjacent (gap ≤ _CLUSTER_Y_GAP_FACTOR × the local line height).
    Union-find over those pairwise links yields the clusters, so a 3-line node
    chains together transitively. Fully document-independent — every threshold is
    relative to the lines' own measured heights.

    `lines` are dicts with a crop-local ``bbox`` [x0,y0,x1,y1]. Returns a list of
    clusters, each a list of the member line dicts.
    """
    n = len(lines)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    # Diagram's typical line height → an absolute ceiling on the mergeable gap,
    # so a single very tall box can't bridge the whitespace between two nodes.
    heights = sorted(max(1.0, ln["bbox"][3] - ln["bbox"][1]) for ln in lines)
    median_h = heights[len(heights) // 2] if heights else 1.0
    gap_cap = _CLUSTER_Y_GAP_CAP_FACTOR * median_h

    for i in range(n):
        bi = lines[i]["bbox"]
        hi = max(1.0, bi[3] - bi[1])
        for j in range(i + 1, n):
            bj = lines[j]["bbox"]
            hj = max(1.0, bj[3] - bj[1])
            if _x_overlap_frac(bi, bj) < _CLUSTER_X_OVERLAP_FRAC:
                continue
            # Vertical gap between the two boxes (0 if they touch/overlap in y).
            gap = max(bi[1], bj[1]) - min(bi[3], bj[3])
            gap_allowed = min(_CLUSTER_Y_GAP_FACTOR * min(hi, hj), gap_cap)
            if gap <= gap_allowed:
                union(i, j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(lines[i])
    return list(groups.values())


async def _recover_one_diagram(entry: dict, page_img: Image.Image,
                               zoom: float) -> List[dict]:
    """Detect text lines inside one diagram entry's crop. Returns a list of
    top-level layout entries (``category="Text"``, ``source="diagram-label"``)
    — one per detected label, in page-pixel space — ready to append to the
    page's ``layout_result``. Empty list when nothing usable is recovered."""
    bbox = entry.get("bbox")
    if not bbox or len(bbox) != 4:
        return []
    x1, y1, x2, y2 = (int(v) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return []

    cx0 = max(0, x1 - _CROP_MARGIN_PX)
    cy0 = max(0, y1 - _CROP_MARGIN_PX)
    cx1 = min(page_img.width, x2 + _CROP_MARGIN_PX)
    cy1 = min(page_img.height, y2 + _CROP_MARGIN_PX)
    if cx1 <= cx0 or cy1 <= cy0:
        return []
    crop = page_img.crop((cx0, cy0, cx1, cy1)).convert("RGB")

    detections = await _paddle_detect(crop)
    if not detections:
        log.info("[diagram-text] no text regions recovered for diagram at %s", bbox)
        return []

    # 1) Keep clean text LINES (crop-local bbox). PaddleOCR detects line-by-line.
    lines: List[dict] = []
    for det in detections:
        text = (det.get("text") or "").strip()
        if not text:
            continue
        if float(det.get("conf") or 0.0) < MIN_CONF:
            continue
        # Skip detections with no actual letters/ideographs — PaddleOCR emits
        # stray marks ("+", "一", "1") on arrowheads/junctions. A label always
        # carries at least one alphabetic/CJK character.
        if not any(ch.isalpha() for ch in text):
            continue
        db = det.get("bbox")
        if not db or len(db) != 4:
            continue
        lines.append({"bbox": [float(v) for v in db], "text": text})

    if not lines:
        log.info("[diagram-text] all detections below conf floor for diagram at %s",
                 bbox)
        return []

    # 2) Cluster lines into per-node labels (a flowchart node's multi-line text
    #    arrives as several line boxes; merge them into ONE label so it reads and
    #    translates as a unit and overlays as one box).
    labels: List[dict] = []
    for cluster in _cluster_lines(lines):
        # Reading order: top-to-bottom, then left-to-right.
        cluster.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
        text = " ".join(ln["text"] for ln in cluster).strip()
        if not text:
            continue
        # Union bbox over the whole node, offset crop-local → page pixels.
        ux0 = min(ln["bbox"][0] for ln in cluster)
        uy0 = min(ln["bbox"][1] for ln in cluster)
        ux1 = max(ln["bbox"][2] for ln in cluster)
        uy1 = max(ln["bbox"][3] for ln in cluster)
        page_bbox = [
            int(round(cx0 + ux0)), int(round(cy0 + uy0)),
            int(round(cx0 + ux1)), int(round(cy0 + uy1)),
        ]
        # Font size from the PER-LINE height (median across the node's lines), NOT
        # the merged multi-line height — otherwise a 2/3-line node would get a
        # 2–3× font. Same box-height heuristic chandra_style uses for scanned text.
        line_heights = sorted(max(1.0, ln["bbox"][3] - ln["bbox"][1]) for ln in cluster)
        median_h = line_heights[len(line_heights) // 2]
        size_pt = round((median_h / max(zoom, 1e-6)) * _SIZE_HEIGHT_FACTOR, 1)
        size_pt = max(_MIN_SIZE_PT, min(size_pt, _MAX_SIZE_PT))
        labels.append({
            # A normal Text layout entry: surfaces in the OCR JSON + markdown and
            # is translated like any other prose. `source="diagram-label"` marks
            # it so reflow pins it to its parent diagram and reconstruction renders
            # it as a white-filled overlay (hiding the original glyphs).
            "bbox": page_bbox,
            "category": "Text",
            "text": text,
            "style": {"font": "Calibri", "size": size_pt, "bold": False,
                      "color": [0, 0, 0], "source": "diagram-heuristic"},
            "source": "diagram-label",
        })

    if not labels:
        return []

    log.info("[diagram-text] recovered %d node label(s) from %d line(s) for "
             "diagram at %s", len(labels), len(lines), bbox)
    return labels


async def recover_diagram_text(pages: List[dict]) -> List[dict]:
    """For every ``diagram-recovered`` Image entry, detect its text labels with
    PaddleOCR and APPEND them to the page's ``layout_result`` as normal
    ``Text`` entries stamped ``source="diagram-label"``. Mutates ``pages`` in
    place.

    Emitting the labels as top-level entries (rather than a nested field) means
    they appear in the OCR JSON + markdown and flow through translation with no
    special-casing; the ``source`` marker drives white-fill overlay rendering.

    A no-op (no PaddleOCR subprocess spawned) for pages without a
    diagram-recovered entry, so documents with no diagrams are untouched.
    """
    for page in pages:
        page_img: Optional[Image.Image] = (
            page.get("original_image") or page.get("image")
        )
        if page_img is None:
            continue
        entries: List[dict] = page.get("layout_result") or []
        diagrams = [
            e for e in entries
            if e.get("source") == "diagram-recovered" and e.get("category") in (
                "Image", "Figure"
            )
        ]
        zoom = float(page.get("zoom") or 1.0)
        new_labels: List[dict] = []
        for diagram in diagrams:
            new_labels.extend(await _recover_one_diagram(diagram, page_img, zoom))
        if new_labels:
            entries.extend(new_labels)
            page["layout_result"] = entries

    return pages
