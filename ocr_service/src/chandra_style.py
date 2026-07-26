"""Attach per-entry font style to Chandra-OCR layout entries.

Chandra emits inline bold/italic markup (<b>/<strong>/<i>/<em>) in each block's
HTML but no numeric font size, colour, or family. This module fills in the
missing style:

  * bold / italic       — read from the block's inline HTML (stashed as
                          `entry["_html"]` by chandra_ocr.parse_layout_html).
  * size / colour /font — for native PDFs, taken from the PDF text layer via
                          PyMuPDF spans (doc_processing.extract_font_spans),
                          matching each entry to the dominant style of the
                          spans inside its bbox. For scans / images / spanless
                          entries, a category + bbox-height heuristic.

Produces the exact `style` dict the reconstruction run-builder reads
(ooxml.build_run_xml): {font, size, bold, italic, color:[r,g,b]}.
This replaces the Unlimited-OCR `font_attribution.py` path.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup


DEFAULT_FONT = "Calibri"
DEFAULT_SIZE_PT = 11.0

HEURISTIC_BOLD_CATEGORIES = {
    "Title", "Section-Header", "Page-Header", "Page-Footer", "Caption",
}
HEURISTIC_ITALIC_CATEGORIES = {"Caption", "Footnote"}

# Chandra labels that carry a raster crop, not attributable text.
PICTURE_LABELS = {"Image", "Figure"}

# East-Asian wide codepoints advance ~1em; used for line-count estimation.
_WIDE_CH_RE = re.compile(
    r"[ᄀ-ᇿ⺀-⻿　-〿぀-ヿ㄰-㆏"
    r"㐀-䶿一-鿿ꀀ-꓏가-힯豈-﫿"
    r"︰-﹏＀-￯]"
)


def _is_wide_char(ch: str) -> bool:
    return bool(_WIDE_CH_RE.match(ch))


# ── Inline bold / italic from Chandra's block HTML ───────────────────────────

def _inline_emphasis(html_fragment: Optional[str]) -> Tuple[bool, bool]:
    """Return (bold, italic) if the block's HTML is dominated by emphasis tags.

    We treat the block as bold/italic when a bold/italic tag wraps a majority
    of the block's text — this matches how the single-`style`-per-entry
    renderer works (it can't do per-run emphasis, so we pick the dominant)."""
    if not html_fragment:
        return (False, False)
    soup = BeautifulSoup(html_fragment, "html.parser")
    total = len(soup.get_text(strip=True))
    if total == 0:
        return (False, False)

    def _covered(tag_names) -> int:
        return sum(
            len(t.get_text(strip=True))
            for t in soup.find_all(tag_names)
        )

    bold = _covered(["b", "strong"]) >= 0.5 * total
    italic = _covered(["i", "em"]) >= 0.5 * total
    return (bold, italic)


# ── bbox-height heuristic (scans / spanless entries) ─────────────────────────

def _bbox_px_to_pt(bbox_px, zoom: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_px
    return (x1 / zoom, y1 / zoom, x2 / zoom, y2 / zoom)


def _estimate_line_count(entry: Dict, zoom: float) -> int:
    bbox = entry.get("bbox") or [0, 0, 0, 0]
    text = (entry.get("text") or "").strip()
    if not text:
        return 1
    width_px = max(1.0, bbox[2] - bbox[0])
    narrow_advance = 7.0 * max(zoom, 1.0)
    wide_advance = 2.0 * narrow_advance
    estimated_text_width = sum(
        wide_advance if _is_wide_char(ch) else narrow_advance for ch in text
    )
    ratio = estimated_text_width / width_px
    by_width = max(1, math.ceil(ratio - 0.25))
    by_newline = text.count("\n") + 1
    return max(by_newline, min(by_width, 50))


def _heuristic_style(entry: Dict, zoom: float) -> Dict:
    bbox = entry.get("bbox") or [0, 0, 0, 0]
    height_px = max(0.0, bbox[3] - bbox[1])
    lines = _estimate_line_count(entry, zoom)
    per_line_height_px = height_px / max(lines, 1)
    size_pt = (per_line_height_px / max(zoom, 1e-6)) * 0.7
    if size_pt < 6.0 or size_pt > 96.0:
        size_pt = DEFAULT_SIZE_PT

    category = entry.get("category", "")
    inline_bold, inline_italic = _inline_emphasis(entry.get("_html"))
    return {
        "font": DEFAULT_FONT,
        "size": round(size_pt, 1),
        "bold": inline_bold or category in HEURISTIC_BOLD_CATEGORIES,
        "italic": inline_italic or category in HEURISTIC_ITALIC_CATEGORIES,
        "color": [0, 0, 0],
        "source": "heuristic",
    }


# ── PDF-span dominant style (native PDFs) ────────────────────────────────────

def _span_center(span_bbox_pt: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = span_bbox_pt
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _point_in_box(px: float, py: float, box: Tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= px <= x1 and y0 <= py <= y1


def _dominant_style(matched_spans: List[Dict]) -> Optional[Dict]:
    """Pick the style covering the most text by character count."""
    if not matched_spans:
        return None
    weights: Dict[Tuple, int] = defaultdict(int)
    representatives: Dict[Tuple, Dict] = {}
    for span in matched_spans:
        key = (
            span.get("font", "") or DEFAULT_FONT,
            round(float(span.get("size") or DEFAULT_SIZE_PT), 1),
            bool(span.get("bold")),
            bool(span.get("italic")),
            tuple(span.get("color_rgb") or (0, 0, 0)),
        )
        weights[key] += len((span.get("text") or "").strip())
        representatives.setdefault(key, span)
    if not weights:
        return None
    winner_key = max(weights.items(), key=lambda kv: kv[1])[0]
    font, size, bold, italic, color = winner_key
    return {
        "font": font or DEFAULT_FONT,
        "size": size,
        "bold": bold,
        "italic": italic,
        "color": list(color),
        "source": "pdf",
    }


def attribute_page(
    entries: List[Dict],
    spans: List[Dict],
    zoom: float,
) -> List[Dict]:
    """Mutate each entry by adding a `style` dict. Returns the same list.

    `spans` are PDF text spans for the page (empty for scans / images), each:
    {"bbox_pt", "text", "font", "size", "bold", "italic", "color_rgb"}.
    When a span-derived style is found it wins (real size/colour/font); the
    inline HTML bold/italic still overrides emphasis on top of it."""
    for entry in entries:
        category = entry.get("category", "")
        if category in PICTURE_LABELS:
            # Rasters carry no attributable text.
            continue
        if entry.get("source") == "diagram-label":
            # PaddleOCR-recovered diagram labels already carry a deliberate style
            # and sit over a raster (no PDF spans there); keep their style so the
            # bbox-heuristic doesn't inflate the font.
            continue
        bbox_px = entry.get("bbox")
        if not bbox_px or len(bbox_px) != 4:
            entry["style"] = _heuristic_style(entry, zoom)
            continue

        bbox_pt = _bbox_px_to_pt(bbox_px, zoom)
        matched = [
            s for s in spans
            if _point_in_box(*_span_center(s["bbox_pt"]), bbox_pt)
        ]
        style = _dominant_style(matched)
        if style is None:
            entry["style"] = _heuristic_style(entry, zoom)
            continue

        # Layer Chandra's inline emphasis over the PDF-derived style — the
        # model catches bold/italic the PDF flags sometimes miss (and vice
        # versa), so OR them.
        inline_bold, inline_italic = _inline_emphasis(entry.get("_html"))
        style["bold"] = bool(style["bold"]) or inline_bold
        style["italic"] = bool(style["italic"]) or inline_italic
        entry["style"] = style
    return entries
