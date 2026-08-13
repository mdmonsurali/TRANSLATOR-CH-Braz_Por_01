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
Because PaddleOCR brings its own paddlepaddle/paddlex stack (which floats
numpy/opencv against vLLM's pins), it runs as its OWN COMPOSE SERVICE —
``paddle_service`` on port 8005 — which holds the model resident across
diagrams. Crops are posted as raw PNG bytes, so no temp file or path is shared
between processes.

Each detected line becomes a ``{bbox (page px), text, style}`` dict; the regions
are attached as ``entry["diagram_regions"]``. The translator then translates each
region's ``text`` and the reconstruction side paints a white-filled textbox over
the original text and drops the translation on top — the drawing stays identical.

Public surface
--------------
recover_diagram_text(pages) — async; for every ``diagram-recovered`` Image entry,
attach ``diagram_regions``. Mutates ``pages`` in place and returns it. A no-op for
pages with no such entry (and no request to the PaddleOCR service is made).
"""

from __future__ import annotations

import io
import logging
import os
from typing import List, Optional

import httpx
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

# PaddleOCR runs as its own compose service (`paddle_service`), not as an
# in-image venv subprocess: its paddlepaddle/paddlex stack floats numpy/opencv
# against vLLM's pins, and a separate image makes a collision impossible. The
# crop travels as raw bytes in the request body, so there is no temp file and no
# path handed between processes.
PADDLE_SERVICE_URL = os.getenv(
    "PADDLE_SERVICE_URL", "http://paddle_service:8005",
).rstrip("/")
PADDLE_LANG = os.getenv("DIAGRAM_OCR_LANG", "ch")
PADDLE_TIMEOUT_SEC = float(os.getenv("DIAGRAM_OCR_TIMEOUT_SEC", "300"))
# Drop low-confidence detections (PaddleOCR emits stray marks like "+"/"一" on
# arrowheads). Kept low (0.45) so faint 是/否 edge labels — recovered by the
# worker's contrast preprocessing but still lower-confidence — survive; the
# alpha-only filter below removes the punctuation noise the low floor lets in.
MIN_CONF = float(os.getenv("DIAGRAM_OCR_MIN_CONF", "0.45"))

# Rotation detection lives in `rotation_measure` so this module and
# `rotation_detect` share ONE implementation of "is this line rotated". The
# kill switch and thresholds (and their env vars) are documented there.
from rotation_measure import (  # noqa: E402 — grouped with the other tunables
    ROT_ASPECT_MIN, ROT_ENABLE, ROT_SKEW_MIN_DEG,
    line_rotation, modal_rotation, poly_skew_deg,
)

# Margin (px) around the diagram bbox when cropping — a little air helps detect
# labels that sit flush against a box ruling.
_CROP_MARGIN_PX = 8


# One shared client for the whole run: connection reuse matters because a
# document can issue hundreds of diagram crops. The service holds the model
# resident and serialises predicts on its own side, so this side needs no lock.
_client: Optional["httpx.AsyncClient"] = None


def _paddle_client() -> "httpx.AsyncClient":
    """Return the shared HTTP client, creating it on first use."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=PADDLE_SERVICE_URL,
            timeout=httpx.Timeout(PADDLE_TIMEOUT_SEC, connect=10.0),
        )
    return _client


async def shutdown_paddle_worker() -> None:
    """Close the shared client (idempotent).

    Kept under the original name so `main`'s shutdown hook is unchanged; there
    is no longer a subprocess to reap, only a connection pool to release.
    """
    global _client
    client = _client
    _client = None
    if client is not None and not client.is_closed:
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001 — best-effort teardown
            pass


async def _paddle_detect(crop: Image.Image) -> Optional[List[dict]]:
    """Detect text lines in `crop` via the PaddleOCR service. Returns a list of
    ``{bbox (crop-local px), poly, text, conf}`` or None on failure.

    The PNG is posted as the request body. Nothing is written to disk, which is
    what retired the old ``FileNotFoundError``/desync failures: there is no path
    whose lifetime has to outlive a subprocess's read-ahead buffer.
    """
    try:
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        resp = await _paddle_client().post(
            "/detect",
            content=buf.getvalue(),
            params={"lang": PADDLE_LANG},
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("ok"):
            log.warning("[diagram-text] PaddleOCR service error: %s",
                        payload.get("error"))
            return None
        return payload.get("lines") or []
    except Exception as exc:  # noqa: BLE001 — one bad diagram must not fail OCR
        log.warning("[diagram-text] PaddleOCR request failed: %s: %s",
                    type(exc).__name__, exc)
        return None


def _x_overlap_frac(a: List[float], b: List[float]) -> float:
    """Fraction of the NARROWER x-span [x0,x2] that overlaps the other's."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    narrow = max(1.0, min(a[2] - a[0], b[2] - b[0]))
    return inter / narrow


def _y_overlap_frac(a: List[float], b: List[float]) -> float:
    """Fraction of the SHORTER y-span [y0,y1] that overlaps the other's.

    The transpose of `_x_overlap_frac`, used to cluster quarter-turned lines
    (which sit side by side rather than stacked).
    """
    inter = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    short = max(1.0, min(a[3] - a[1], b[3] - b[1]))
    return inter / short


def _cluster_lines(lines: List[dict]) -> List[List[dict]]:
    """Group PaddleOCR line boxes into per-node clusters using geometry only.

    Lines are PARTITIONED BY ROTATION first and clustered within each partition,
    so an upright node and a rotated side-label sitting next to it can never
    merge into one entry. Upright lines take the original code path unchanged.

    Within a partition, two lines join the same node when they overlap along the
    text's cross-axis and are adjacent along its stacking axis (gap ≤
    _CLUSTER_Y_GAP_FACTOR × the local line thickness). For upright text that is
    "overlap in x, adjacent in y"; for quarter-turned text the axes swap, which
    is why a rotated multi-line label never merged before. Union-find over those
    pairwise links yields the clusters, so a 3-line node chains transitively.
    Fully document-independent — every threshold is relative to the lines' own
    measured thickness.

    `lines` are dicts with a crop-local ``bbox`` [x0,y0,x1,y1]. Returns a list of
    clusters, each a list of the member line dicts.
    """
    partitions: dict = {}
    for ln in lines:
        # 90 and 270 stack along the same axis; group them together and let the
        # transposed predicates handle both.
        rot = float(ln.get("rotation") or 0.0)
        partitions.setdefault(1 if rot in (90.0, 270.0) else 0, []).append(ln)

    out: List[List[dict]] = []
    for sideways, group in partitions.items():
        out.extend(_cluster_one_orientation(group, bool(sideways)))
    return out


def _cluster_one_orientation(lines: List[dict],
                             sideways: bool) -> List[List[dict]]:
    """Union-find clustering for one rotation partition.

    `sideways=False` reproduces the original upright behaviour exactly.
    `sideways=True` transposes both predicates: overlap is measured in y and
    adjacency in x, because quarter-turned lines stack horizontally.
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

    def thickness(b) -> float:
        # Cross-axis extent = the line's height for upright text, its width
        # once turned. This is what the gap thresholds are relative to.
        return max(1.0, (b[2] - b[0]) if sideways else (b[3] - b[1]))

    # Diagram's typical line thickness → an absolute ceiling on the mergeable
    # gap, so a single very long box can't bridge the whitespace between nodes.
    thicknesses = sorted(thickness(ln["bbox"]) for ln in lines)
    median_h = thicknesses[len(thicknesses) // 2] if thicknesses else 1.0
    gap_cap = _CLUSTER_Y_GAP_CAP_FACTOR * median_h

    for i in range(n):
        bi = lines[i]["bbox"]
        hi = thickness(bi)
        for j in range(i + 1, n):
            bj = lines[j]["bbox"]
            hj = thickness(bj)
            overlap = (_y_overlap_frac(bi, bj) if sideways
                       else _x_overlap_frac(bi, bj))
            if overlap < _CLUSTER_X_OVERLAP_FRAC:
                continue
            # Gap between the two boxes along the stacking axis (0 if touching).
            if sideways:
                gap = max(bi[0], bj[0]) - min(bi[2], bj[2])
            else:
                gap = max(bi[1], bj[1]) - min(bi[3], bj[3])
            gap_allowed = min(_CLUSTER_Y_GAP_FACTOR * min(hi, hj), gap_cap)
            if gap <= gap_allowed:
                union(i, j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(lines[i])
    return list(groups.values())


def _assign_line_rotations(lines: List[dict]) -> None:
    """Stamp `rotation` (degrees, CCW-positive) on each detected line in place.

    The measurement itself lives in `rotation_measure.line_rotation` — see that
    module for how rotation is detected and why. Kept as a thin wrapper because
    the clustering code below reads `ln["rotation"]` directly.

    A rotated CJK label reads bottom-to-top (glyphs turned 90° clockwise), which
    is `rotation = 90` in our CCW-positive schema.
    """
    if not ROT_ENABLE:
        return
    for ln in lines:
        ln["rotation"] = line_rotation(
            ln["bbox"], ln.get("text") or "", ln.get("poly"),
        )


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
        lines.append({
            "bbox": [float(v) for v in db],
            "text": text,
            "conf": float(det.get("conf") or 0.0),
            "poly": det.get("poly"),
        })

    if not lines:
        log.info("[diagram-text] all detections below conf floor for diagram at %s",
                 bbox)
        return []

    # 1b) Stamp each line's rotation (0 unless it reads as a rotated label).
    _assign_line_rotations(lines)

    # 2) Cluster lines into per-node labels (a flowchart node's multi-line text
    #    arrives as several line boxes; merge them into ONE label so it reads and
    #    translates as a unit and overlays as one box).
    labels: List[dict] = []
    for cluster in _cluster_lines(lines):
        # The cluster's rotation is the modal rotation of its members. They
        # should agree — _cluster_lines never merges across orientations — so
        # this is really just "pick the one value present".
        cluster_rot = modal_rotation(
            [float(ln.get("rotation") or 0.0) for ln in cluster]
        )

        # Reading order follows the text direction: top-to-bottom then
        # left-to-right when upright, but bottom-to-top for a 90° label (which
        # reads upward) and top-to-bottom with columns right-to-left at 270°.
        if cluster_rot == 90.0:
            cluster.sort(key=lambda ln: (-ln["bbox"][3], ln["bbox"][0]))
        elif cluster_rot == 270.0:
            cluster.sort(key=lambda ln: (ln["bbox"][1], -ln["bbox"][0]))
        else:
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
        # Font size from the PER-LINE CROSS-AXIS extent (median across the node's
        # lines), NOT the merged multi-line extent — otherwise a 2/3-line node
        # would get a 2–3× font. Same box-height heuristic chandra_style uses for
        # scanned text.
        #
        # For a quarter-turned line the glyph height is the box WIDTH: its height
        # is the text's LENGTH. Measuring height there gave 58pt and 103pt on the
        # real labels, both silently swallowed by the _MAX_SIZE_PT clamp.
        sideways = cluster_rot in (90.0, 270.0)
        extents = sorted(
            max(1.0, (ln["bbox"][2] - ln["bbox"][0]) if sideways
                else (ln["bbox"][3] - ln["bbox"][1]))
            for ln in cluster
        )
        median_h = extents[len(extents) // 2]
        size_pt = round((median_h / max(zoom, 1e-6)) * _SIZE_HEIGHT_FACTOR, 1)
        size_pt = max(_MIN_SIZE_PT, min(size_pt, _MAX_SIZE_PT))
        label = {
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
        }
        # Only stamp `rotation` when there is one, so upright labels serialize
        # byte-identically to before this feature existed.
        if cluster_rot:
            label["rotation"] = cluster_rot
        labels.append(label)

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
