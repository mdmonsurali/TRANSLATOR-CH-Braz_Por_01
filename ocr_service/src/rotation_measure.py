"""Shared rotation MEASUREMENT primitives.

One implementation of "is this text line rotated, and by how much", used by both
consumers so the two can never drift:

* `diagram_text_recovery` — rotation of individual labels inside a diagram.
* `rotation_detect` — rotation of ordinary layout entries and whole tables.

The schema these produce is the one the reconstruction side consumes (see
`ocr_reconstruction/geometry.py`): FLOAT DEGREES, COUNTER-CLOCKWISE-POSITIVE,
where absent/0 means upright. A CJK label turned a quarter turn clockwise (glyphs
reading bottom-to-top) is `90.0`.

HOW ROTATION IS DETECTED (measured, not assumed): PaddleOCR reads a
quarter-turned CJK label CORRECTLY without help — the same crop scores conf 1.00
upright and 1.00 de-rotated — so confidence cannot discriminate. What *does* flip
is the geometry of the detection it returns:

    vertical label   '生产和生产后信息' conf 1.00  box 24x148  aspect 6.17
    de-rotated same  '生产和生产后信息' conf 1.00  box 145x19  aspect 0.13
    horizontal label '预期用途、识别特征' conf 1.00  box 163x19  aspect 0.12

So a line whose text reads confidently while its own detection box is far taller
than it is wide IS rotated text. That needs no second OCR pass, which keeps this
free — no extra latency, no probe budget.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional

# Kill switch: when off, every caller treats all text as upright and the
# pipeline behaves exactly as it did before the rotation feature.
# `DIAGRAM_ROT_ENABLE` is the historical name and is still honoured.
ROT_ENABLE = os.getenv(
    "OCR_ROT_ENABLE", os.getenv("DIAGRAM_ROT_ENABLE", "1"),
).strip().lower() not in {"0", "false", "no", "off"}

# Detection-box aspect (h/w) at or above which a multi-character line is taken
# to be quarter-turned. Real rotated labels measured 3.2-6.2; horizontal ones
# 0.12-0.13. 2.0 sits in the wide gap between those populations.
ROT_ASPECT_MIN = float(
    os.getenv("OCR_ROT_ASPECT_MIN", os.getenv("DIAGRAM_ROT_ASPECT_MIN", "2.0"))
)
# Minimum |tilt| before a quad's baseline angle counts as real skew rather
# than detector noise (measured noise on axis-aligned text: ~0.7°).
ROT_SKEW_MIN_DEG = float(
    os.getenv("OCR_ROT_SKEW_MIN_DEG", os.getenv("DIAGRAM_ROT_SKEW_MIN_DEG", "5.0"))
)


def poly_skew_deg(poly) -> float:
    """Baseline tilt of a PaddleOCR quad, in degrees CCW-positive.

    Only meaningful for genuinely SKEWED text. Ideographic text turned a full
    quarter-turn comes back as a near-axis-aligned quad (measured: -0.73° on a
    90°-rotated label), so this never reports the quarter turn itself — that is
    `line_rotation`'s job.
    """
    try:
        (x0, y0), (x1, y1) = poly[0], poly[1]
    except (TypeError, IndexError, ValueError):
        return 0.0
    # Image y grows downward, so negate to get the usual CCW-positive angle.
    return -math.degrees(math.atan2(float(y1) - float(y0), float(x1) - float(x0)))


def line_rotation(bbox, text: str, poly=None) -> float:
    """Rotation in degrees CCW-positive for one detected text line.

    `bbox` is (x1, y1, x2, y2) in any consistent pixel space. Returns 0.0 for
    upright text, for a disabled kill switch, and for anything unmeasurable.
    """
    if not ROT_ENABLE:
        return 0.0
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return 0.0
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    # Genuinely skewed text: trust the quad's tilt when it is well past
    # detector noise but not near a quarter turn.
    skew = poly_skew_deg(poly)
    if ROT_SKEW_MIN_DEG <= abs(skew) <= 60.0:
        return skew % 360.0

    # Quarter turn: tall-narrow detection box + enough characters to be a real
    # label. The `len >= 2` guard matters because a single glyph is naturally
    # square-ish and its aspect ratio means nothing.
    if (h / w) >= ROT_ASPECT_MIN and len(text or "") >= 2:
        return 90.0
    return 0.0


def modal_rotation(rotations: List[float]) -> float:
    """The most common rotation in `rotations`, ties going to the smaller angle.

    Used to reduce many per-line measurements to one value for the region that
    contains them.
    """
    if not rotations:
        return 0.0
    counts: dict = {}
    for r in rotations:
        counts[r] = counts.get(r, 0) + 1
    return max(sorted(counts), key=lambda r: counts[r])


def aspect_is_quarter_turn(bbox, text: str = "xx") -> Optional[float]:
    """Bbox-only quarter-turn test — no OCR call at all.

    Returns 90.0 for a tall-narrow box holding at least 2 characters, else None.
    This is the free first pass: it catches rotated side-labels straight from the
    layout geometry the VLM already gave us.
    """
    if not ROT_ENABLE:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    if (h / w) >= ROT_ASPECT_MIN and len((text or "").strip()) >= 2:
        return 90.0
    return None
