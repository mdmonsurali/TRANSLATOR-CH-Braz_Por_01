"""PaddleOCR text-line detection as a standalone HTTP service.

Replaces the in-image ``/opt/paddle-venv`` + stdin/stdout subprocess worker that
``ocr_service`` used to drive (``paddle_ocr_worker.py`` +
``diagram_text_recovery._paddle_detect``).

WHY A SEPARATE SERVICE
----------------------
1. Dependency isolation, structurally. PaddleOCR pulls its own
   paddlepaddle/paddlex stack and floats numpy/opencv/pillow against vLLM's
   pins (measured: numpy 2.3.5 vs 2.2.6, opencv-contrib 4.10 vs
   opencv-python-headless 4.13). A separate image makes a collision impossible
   rather than merely unlikely.
2. It deletes a whole bug class. The old protocol passed a TEMP FILE PATH; the
   worker's ``for raw in sys.stdin`` read ahead into its buffer and processed
   requests after the caller's ``finally`` had already unlinked them, producing
   runs of ``FileNotFoundError: /tmp/tmpXXXX.png`` plus desynced responses
   (``Expecting value: line 1 column 1``) and readline overflows (``Separator is
   not found``). Here the image travels IN the request body: no path, no
   unlink, no shared filesystem, no ordering contract to break.
3. A Paddle crash can no longer take the OCR process's worker state with it, and
   the model stays resident across documents.

The detection code below is a VERBATIM port of the worker's ``_detect`` —
identical preprocessing, identical model flags, identical output shape — so
swapping transports does not change what gets detected.

    POST /detect   body = raw image bytes (PNG/JPG), ?lang=ch
                   -> {"ok": true, "lines": [{bbox, poly, text, conf}, ...]}
    POST /detect_batch  multipart files -> {"ok": true, "results": [[...], ...]}
    GET  /health   -> {"status": "ok", "model_loaded": bool}

``bbox``/``poly`` are in INPUT-IMAGE-LOCAL pixels, exactly as before.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
from typing import List

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("paddle_service")

DEFAULT_LANG = os.getenv("DIAGRAM_OCR_LANG", "ch")

# Longest-side cap (px) on the image actually fed to the detector. Diagram crops
# are normally a few hundred px, but a full-page diagram at 300 DPI arrives at
# ~2000x2800 and PaddleOCR's feature maps scale with area — that is what OOM-
# killed this container at a 4 GiB limit. 1600 keeps text comfortably legible
# for detection (these are labels, not dense body copy) while bounding the peak.
# Coordinates are scaled back to the caller's frame, so this is invisible to it.
_MAX_DETECT_PX = int(os.getenv("PADDLE_MAX_DETECT_PX", "1600"))

# PaddleOCR 3.x on this CPU build crashes inside the oneDNN executor
# ("ConvertPirAttribute2RuntimeAttribute not support ArrayAttribute<DoubleAttribute>")
# on the very first predict. Disabling the oneDNN path is what makes 3.x usable
# here; it does not change the model or its output.
_MKLDNN = False

# Built lazily and cached per language so model construction (~2.5 s) is paid
# once for the life of the container, not per request.
_OCR_CACHE: dict = {}

# PaddlePaddle inference is not reentrant and the pipeline object is shared, so
# requests are serialised. This mirrors the `_PADDLE_LOCK` the caller used to
# hold around the subprocess; keeping it here means the guarantee lives with the
# model instead of depending on every caller remembering it.
_PREDICT_LOCK = asyncio.Lock()

app = FastAPI(title="PaddleOCR diagram text detection")


def _get_ocr(lang: str):
    """Return a cached PaddleOCR pipeline for `lang`, building it on first use.

    The doc-orientation / doc-unwarping sub-models are disabled: diagram crops
    are already upright page regions, and each extra sub-model adds both startup
    and per-call cost for no benefit here. Textline orientation stays off too —
    `rotation_detect` derives rotation from box aspect/skew and relies on this
    detector reporting boxes as-scanned.
    """
    ocr = _OCR_CACHE.get(lang)
    if ocr is None:
        from paddleocr import PaddleOCR

        log.info("building PaddleOCR pipeline (lang=%s)", lang)
        ocr = PaddleOCR(
            lang=lang,
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=_MKLDNN,
        )
        _OCR_CACHE[lang] = ocr
        log.info("PaddleOCR pipeline ready (lang=%s)", lang)
    return ocr


def _detect_bytes(img_bytes: bytes, lang: str) -> list:
    """Detect text lines in an in-memory image. Returns [{bbox, poly, text, conf}].

    Verbatim port of `paddle_ocr_worker._detect`, except the source is bytes
    rather than a path — the old version already converted straight to a numpy
    array, so nothing downstream changes.
    """
    import numpy as np
    from PIL import Image, ImageEnhance

    src = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Bound peak memory. PaddleOCR's detector allocates several float feature
    # maps proportional to input area, so a full-page crop is what pushed this
    # container past a 4 GiB cap and killed it mid-request (measured 3.2+ GiB on
    # 2000x2800). Downscale only the LONGEST side above the cap, preserving
    # aspect ratio; `scale` is carried through so returned coordinates stay in
    # the caller's ORIGINAL crop pixels — the caller maps them into page space
    # and would place labels wrong if we silently changed the frame.
    scale = 1.0
    longest = max(src.size)
    if longest > _MAX_DETECT_PX:
        scale = _MAX_DETECT_PX / float(longest)
        src = src.resize(
            (max(1, int(src.width * scale)), max(1, int(src.height * scale))),
            Image.Resampling.LANCZOS,
        )

    # Contrast preprocessing so FAINT-COLOURED labels (the diagram's 是/否 edge
    # marks are a light red/gray that PaddleOCR's detector otherwise misses):
    #   1. per-pixel MIN over R,G,B — faint red text (high R, low G/B) collapses
    #      to a dark value while the white background stays bright; solid black
    #      text is unaffected.
    #   2. strong contrast stretch so those now-dark strokes clear the detector's
    #      threshold. Verified to recover the light 是/否 arrow labels.
    mn = np.min(np.array(src), axis=2).astype("uint8")
    boosted = ImageEnhance.Contrast(Image.fromarray(mn)).enhance(3.0).convert("RGB")
    arr = np.array(boosted)

    res = _get_ocr(lang).predict(arr)
    if not res:
        return []
    # PaddleOCR 3.x returns a list of per-image result dicts with PARALLEL
    # arrays (rec_texts / rec_scores / rec_polys), not 2.x's nested
    # [[box, (text, conf)], ...] rows.
    r0 = res[0]
    texts = r0.get("rec_texts") or []
    scores = r0.get("rec_scores") or []
    polys = r0.get("rec_polys")
    if polys is None:
        polys = r0.get("dt_polys") or []

    out = []
    for i, txt in enumerate(texts):
        try:
            poly = polys[i]
            conf = float(scores[i]) if i < len(scores) else 0.0
        except (IndexError, TypeError, ValueError):
            continue
        # Undo any downscale so every coordinate is in the caller's original
        # crop frame. inv == 1.0 in the common (un-resized) case, so this is a
        # no-op there and byte-identical to the pre-cap behaviour.
        inv = 1.0 / scale
        xs = [float(p[0]) * inv for p in poly]
        ys = [float(p[1]) * inv for p in poly]
        out.append({
            "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            # The raw quad, kept alongside the axis-aligned hull. min/max is
            # lossy: for SKEWED text the quad's tilt is the only record of the
            # angle, and collapsing it here would destroy that irrecoverably.
            # (Note 90°-rotated CJK comes back as an essentially axis-aligned
            # quad — that case is detected by aspect ratio, not by this angle.)
            "poly": [[float(p[0]) * inv, float(p[1]) * inv] for p in poly],
            "text": txt,
            "conf": conf,
        })
    return out


# ── Ruled-line grid detection (OpenCV only, no model) ───────────────────────
#
# A rule is found as SEGMENTS, not full spans. A form with merged header cells has
# no vertical rule running the whole height — measured on a real 7-column table,
# requiring a 40%-of-height span found only 4 boundaries, while segments found 8.
# So the kernel length is a minimum RUN, and a coordinate counts as a rule when
# enough of the axis is covered by such runs.
#
# CALIBRATION. These two numbers were fitted against two REAL tables with known
# answers, because the obvious loose setting counts CJK glyph strokes as rules:
# at 0.06/0.15 a genuinely 4-column table read as 8 columns, its "rules" evenly
# spaced ~173px apart — i.e. character stems. A rule must therefore span a real
# fraction of the axis, while still not requiring a FULL span (a merged header
# leaves no rule crossing the whole height). Measured grid:
#
# A 5x5 sweep over (run_frac, cover) against four crops with known answers — two
# ordinary bordered tables (4 and 3 columns), one heavily merged form (7), and a
# table of CONTENTS that is not ruled at all — found NO setting that gets all four
# right. The trade is structural, not a tuning failure:
#
#   run_frac/cover   4-col   3-col   merged-7   unruled TOC
#   0.30 / 0.30        4      12         8          38    <- noise everywhere
#   0.70 / 0.60        4       3         0           2    <- chosen
#
# Loose settings count CJK glyph strokes and dotted leader lines as rules (the
# TOC read as 38 columns); strict settings are exact on ordinary tables and
# collapse the TOC to a harmless 2, but report 0 for a form whose merged header
# leaves no full-height divider.
#
# We take the STRICT end deliberately. A 0 means "unmeasurable" and disables the
# check for that table, so the cost is a missed detection — while a false column
# count would send a healthy table for needless re-OCR and could let a worse read
# win. Under-reporting is the safe direction; `ruled` reports which it was.
_GRID_MIN_RUN_FRAC = float(os.getenv("PADDLE_GRID_MIN_RUN_FRAC", "0.70"))
_GRID_COVER_FRAC = float(os.getenv("PADDLE_GRID_COVER_FRAC", "0.60"))
# A thick or slightly skewed rule projects across several adjacent pixels; merge
# coordinates closer than this so one printed line counts once.
_GRID_MERGE_PX = int(os.getenv("PADDLE_GRID_MERGE_PX", "8"))
# Cap on the crop actually analysed. Morphology is cheap but linear in area, and
# a full-page table at zoom 4.17 arrives ~2000x3300.
_GRID_MAX_PX = int(os.getenv("PADDLE_GRID_MAX_PX", "2400"))


def _rule_coords(mask, axis: int, extent: int) -> List[int]:
    """Centre coordinates of the rules visible in `mask` along one axis."""
    projection = mask.sum(axis=axis) / 255.0
    threshold = extent * _GRID_COVER_FRAC
    centres: List[int] = []
    start = None
    for i, value in enumerate(projection):
        if value > threshold and start is None:
            start = i
        elif value <= threshold and start is not None:
            centres.append((start + i - 1) // 2)
            start = None
    if start is not None:
        centres.append((start + len(projection) - 1) // 2)

    merged: List[int] = []
    for c in centres:
        if merged and c - merged[-1] < _GRID_MERGE_PX:
            merged[-1] = c            # same printed rule, keep the later centre
        else:
            merged.append(c)
    return merged


def _detect_grid_bytes(img_bytes: bytes) -> dict:
    """Rows/columns implied by the ruled lines of a table crop.

    Returns `rows`/`cols` (cell counts, i.e. boundaries - 1) plus the raw
    boundary coordinates in the ORIGINAL crop's frame, so a caller can map them
    back onto its own geometry.
    """
    import cv2                      # local: keeps service start-up model-free
    import numpy as np

    from PIL import Image

    with Image.open(io.BytesIO(img_bytes)) as src:
        gray = src.convert("L")
        scale = 1.0
        if max(gray.size) > _GRID_MAX_PX:
            scale = _GRID_MAX_PX / float(max(gray.size))
            gray = gray.resize(
                (max(1, int(gray.width * scale)), max(1, int(gray.height * scale))),
                Image.Resampling.LANCZOS,
            )
        arr = np.asarray(gray)

    height, width = arr.shape
    if height < 16 or width < 16:
        return {"rows": 0, "cols": 0, "h_lines": [], "v_lines": [],
                "width": width, "height": height}

    # Adaptive threshold, ink -> white. Local mean handles the uneven exposure of
    # a scan, where a global Otsu cut drops the faintest rules on one side.
    binary = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -2,
    )
    # Each kernel is sized from the axis it runs ALONG, not from min(w, h): a wide
    # short table would otherwise get a vertical kernel scaled to its height and a
    # horizontal one far too short to reject glyph strokes.
    h_run = max(20, int(width * _GRID_MIN_RUN_FRAC))
    v_run = max(20, int(height * _GRID_MIN_RUN_FRAC))
    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (h_run, 1)),
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_run)),
    )

    h_lines = _rule_coords(horizontal, 1, width)
    v_lines = _rule_coords(vertical, 0, height)

    # A crop needs rules on BOTH axes to be a grid at all. `ruled` tells the
    # caller whether the counts mean anything: on a false, it must fall back to
    # the HTML-only signals rather than treat 0 columns as a finding.
    #
    # (An intersection test — keeping only rules that cross the perpendicular ones
    # — was tried here and did NOT separate the unruled table of contents, whose
    # leader dots do cross the text bands. The strict span thresholds above are
    # what actually discriminate.)
    ruled = len(h_lines) >= 2 and len(v_lines) >= 2
    if not ruled:
        h_lines, v_lines = [], []

    inv = (1.0 / scale) if scale else 1.0
    return {
        "ruled": ruled,
        "rows": max(0, len(h_lines) - 1),
        "cols": max(0, len(v_lines) - 1),
        "h_lines": [int(round(c * inv)) for c in h_lines],
        "v_lines": [int(round(c * inv)) for c in v_lines],
        "width": int(round(width * inv)),
        "height": int(round(height * inv)),
    }


@app.get("/health")
async def health() -> dict:
    """Liveness. Deliberately does NOT build the model: compose gates
    `ocr_service` on this, and a page with no diagrams should never wait on
    Paddle warm-up."""
    return {"status": "ok", "model_loaded": bool(_OCR_CACHE)}


@app.post("/detect")
async def detect(request: Request, lang: str = Query(default=None)) -> JSONResponse:
    """Detect text lines in the raw image bytes posted as the request body."""
    img_bytes = await request.body()
    if not img_bytes:
        return JSONResponse({"ok": False, "error": "empty request body"},
                            status_code=400)
    try:
        async with _PREDICT_LOCK:
            lines = await asyncio.to_thread(
                _detect_bytes, img_bytes, lang or DEFAULT_LANG
            )
        return JSONResponse({"ok": True, "lines": lines})
    except Exception as exc:  # noqa: BLE001 — one bad crop must not 500 the run
        log.warning("detect failed: %s: %s", type(exc).__name__, exc)
        return JSONResponse(
            {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        )


@app.post("/detect_grid")
async def detect_grid(request: Request) -> JSONResponse:
    """Count the RULED LINES of a table crop posted as raw image bytes.

    This is the only ground truth the pipeline has about a table's real shape.
    Every other signal reads the model's own output, so a read that is internally
    consistent but missing whole columns looks perfectly healthy: the measured
    case emitted 4 columns for a 7-column form and placed 43 of 43 cells.

    Deliberately NOT PaddleOCR: this is pure OpenCV morphology on the raw bytes,
    so it needs no model, no warm-up and no `_PREDICT_LOCK` (which serialises all
    Paddle inference globally — a per-table probe behind that lock would contend
    with rotation detection and diagram recovery).
    """
    img_bytes = await request.body()
    if not img_bytes:
        return JSONResponse({"ok": False, "error": "empty request body"},
                            status_code=400)
    try:
        result = await asyncio.to_thread(_detect_grid_bytes, img_bytes)
        return JSONResponse({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001 — a bad crop must not 500 the run
        log.warning("detect_grid failed: %s: %s", type(exc).__name__, exc)
        return JSONResponse(
            {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        )


@app.post("/detect_batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    lang: str = Query(default=None),
) -> JSONResponse:
    """Detect over several images in ONE round-trip.

    Preserves the old worker's batched `{"paths": [...]}` shape, which the
    rotation probe uses to OCR one crop at three orientations. Per-image
    failures degrade to an empty list rather than failing the whole batch — one
    unreadable orientation must not lose the other two.
    """
    results: List[list] = []
    try:
        async with _PREDICT_LOCK:
            for f in files:
                try:
                    data = await f.read()
                    results.append(
                        await asyncio.to_thread(
                            _detect_bytes, data, lang or DEFAULT_LANG
                        )
                    )
                except Exception:  # noqa: BLE001
                    results.append([])
        return JSONResponse({"ok": True, "results": results})
    except Exception as exc:  # noqa: BLE001
        log.warning("detect_batch failed: %s: %s", type(exc).__name__, exc)
        return JSONResponse(
            {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        )
