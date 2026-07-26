"""Isolated PaddleOCR worker — text-line detection for diagram crops.

Runs in its OWN virtualenv (``/opt/paddle-venv``), NEVER in the main app env:
PaddleOCR pins ``numpy<2.6`` / ``opencv<4.13``, which conflict with vLLM's
``numpy>=2.0`` / ``opencv>=4.13`` in the shared container Python. Isolating it in
a venv and invoking it as a subprocess (see ``diagram_text_recovery``) keeps
vLLM's environment untouched.

Contract:
    argv[1] = path to an input image (PNG/JPG)
    argv[2] = language code (default "ch")
    stdout  = JSON list of {"bbox":[x1,y1,x2,y2], "text": str, "conf": float}
              in INPUT-IMAGE-LOCAL pixels. Empty list on no text.

Prints nothing but the JSON on stdout (PaddleOCR chatter goes to stderr).
"""
import sys
import json


def main() -> int:
    if len(sys.argv) < 2:
        print("[]", end="")
        return 0
    img_path = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "ch"

    import numpy as np
    from PIL import Image, ImageEnhance

    from paddleocr import PaddleOCR

    src = Image.open(img_path).convert("RGB")
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
    # Detection + recognition, no angle classifier (diagram text is upright).
    ocr = PaddleOCR(use_angle_cls=False, lang=lang, show_log=False)
    res = ocr.ocr(arr, cls=False)
    lines = res[0] if res else []

    out = []
    for ln in (lines or []):
        try:
            box, (txt, conf) = ln
        except (ValueError, TypeError):
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append({
            "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            "text": txt,
            "conf": float(conf),
        })
    json.dump(out, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
