import os
import subprocess
import tempfile
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image


PDF_RENDER_DPI = float(os.getenv("PDF_RENDER_DPI", "300"))
PDF_RENDER_ZOOM = PDF_RENDER_DPI / 72.0
IMAGE_INPUT_DPI = 96  # treat raw image inputs as if rendered at 96 DPI for page sizing


def page_count_for_pdf(pdf_path: str) -> int:
    """Page count without rendering anything."""
    with fitz.open(pdf_path) as pdf_document:
        return len(pdf_document)


def load_page_range(pdf_path: str, start: int, end: int) -> List[Dict]:
    """Render PDF pages [start, end) only — the windowed form of
    `load_pages_from_pdf`.

    Page dicts are identical in shape to `load_pages_from_pdf`, and
    `page_index` stays ABSOLUTE (not window-relative): picture_recovery and
    table_reocr re-open `pdf_source` and index into it by that number, so
    renumbering per window would silently read the wrong page.
    """
    pages: List[Dict] = []
    pdf_document = fitz.open(pdf_path)
    try:
        total = len(pdf_document)
        start = max(0, start)
        end = min(end, total)
        mat = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
        for page_num in range(start, end):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(matrix=mat)
            image = Image.open(BytesIO(pix.tobytes("ppm"))).convert("RGB")
            pages.append({
                "image": image,
                "page_width_pt": float(page.rect.width),
                "page_height_pt": float(page.rect.height),
                "zoom": PDF_RENDER_ZOOM,
                "page_index": page_num,
                "pdf_source": os.path.abspath(pdf_path),
            })
    finally:
        pdf_document.close()
    return pages


def load_pages_from_pdf(pdf_path: str) -> List[Dict]:
    """Render every PDF page to a PIL image and capture its original geometry.

    Returns one dict per page:
        {
          "image":          PIL.Image (RGB, rendered at PDF_RENDER_ZOOM),
          "page_width_pt":  float,   # original PDF page width in points
          "page_height_pt": float,
          "zoom":           float,   # rasterization zoom (image px = pt * zoom)
          "page_index":     int,
          "pdf_source":     str,     # absolute path, used later for font extraction
        }

    Holds every page in memory at once. The OCR pipeline uses `load_page_range`
    instead; this remains for callers that genuinely want the whole document.
    """
    return load_page_range(pdf_path, 0, page_count_for_pdf(pdf_path))


def docx_to_pdf(docx_path: str) -> Tuple[str, str]:
    """Convert DOCX → PDF via LibreOffice.

    Returns (pdf_path, temp_dir). The PDF must outlive page rendering because
    extract_font_spans / picture_recovery / table_reocr re-open it by path
    mid-pipeline, so it cannot be cleaned up eagerly — the CALLER owns
    `temp_dir` and is responsible for removing it when the document is done.
    """
    out_dir = tempfile.mkdtemp(prefix="ocr_docx_pdf_")
    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to", "pdf",
        "--outdir", out_dir,
        docx_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise Exception(f"LibreOffice conversion failed: {result.stderr}")

    pdf_files = [f for f in os.listdir(out_dir) if f.endswith(".pdf")]
    if not pdf_files:
        raise Exception("LibreOffice produced no PDF output")

    return os.path.join(out_dir, pdf_files[0]), out_dir


def load_pages_from_docx(docx_path: str) -> List[Dict]:
    """Convert DOCX → PDF via LibreOffice, then load pages with geometry.

    Leaks the intermediate PDF's temp dir by design — see `docx_to_pdf` for the
    lifetime constraint. Callers that care should use `docx_to_pdf` directly and
    clean up themselves.
    """
    pdf_path, _out_dir = docx_to_pdf(docx_path)
    return load_pages_from_pdf(pdf_path)


def load_pages_from_image(image_path: str) -> List[Dict]:
    """Single-image input — one synthesized page with no PDF source."""
    img = Image.open(image_path).convert("RGB")
    w_pt = img.width * 72.0 / IMAGE_INPUT_DPI
    h_pt = img.height * 72.0 / IMAGE_INPUT_DPI
    return [{
        "image": img,
        "page_width_pt": w_pt,
        "page_height_pt": h_pt,
        # bbox pixels were captured at the image's native resolution; we
        # model that as "1 px per (72/IMAGE_INPUT_DPI) pt", i.e. zoom that
        # makes bbox_px / zoom = points.
        "zoom": IMAGE_INPUT_DPI / 72.0,
        "page_index": 0,
        "pdf_source": None,
    }]


# ── Backwards-compatible shims ─

def load_images_from_pdf(pdf_path: str) -> List[Image.Image]:
    return [p["image"] for p in load_pages_from_pdf(pdf_path)]


def load_images_from_docx(docx_path: str) -> List[Image.Image]:
    return [p["image"] for p in load_pages_from_docx(docx_path)]


# ── Font / style extraction (PyMuPDF) ───────

def _flag_bold(flags: int, font_name: str) -> bool:
    if flags & (1 << 4):
        return True
    lname = (font_name or "").lower()
    return "bold" in lname or "black" in lname or "heavy" in lname


def _flag_italic(flags: int, font_name: str) -> bool:
    if flags & (1 << 1):
        return True
    lname = (font_name or "").lower()
    return "italic" in lname or "oblique" in lname


def _decode_sRGB(packed: int) -> Tuple[int, int, int]:
    return ((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF)


def extract_font_spans(pdf_source: Optional[str], page_index: int) -> List[Dict]:
    """Return per-span style info for one PDF page.

    Each span:
        {"bbox_pt": (x0,y0,x1,y1), "text": str, "font": str, "size": float,
         "bold": bool, "italic": bool, "color_rgb": (r,g,b)}

    Returns [] for non-PDF inputs or pages with no text layer.
    """
    if not pdf_source:
        return []
    spans: List[Dict] = []
    try:
        with fitz.open(pdf_source) as doc:
            if page_index >= len(doc):
                return []
            page = doc.load_page(page_index)
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if not text or not text.strip():
                            continue
                        font_name = span.get("font", "") or ""
                        flags = int(span.get("flags", 0) or 0)
                        color_packed = int(span.get("color", 0) or 0)
                        spans.append({
                            "bbox_pt": tuple(span.get("bbox", (0, 0, 0, 0))),
                            "text": text,
                            "font": font_name,
                            "size": float(span.get("size", 0.0) or 0.0),
                            "bold": _flag_bold(flags, font_name),
                            "italic": _flag_italic(flags, font_name),
                            "color_rgb": _decode_sRGB(color_packed),
                        })
    except Exception:
        return []
    return spans


# ── Markdown serializer ────────

def _entries_from_page(page_or_entries) -> List[Dict]:
    """Accept either the legacy list-of-entries shape or the new
    {entries: [...]} dict per page."""
    if isinstance(page_or_entries, dict) and "entries" in page_or_entries:
        return page_or_entries.get("entries") or []
    if isinstance(page_or_entries, list):
        return page_or_entries
    return []


def layoutjson2md(layout_data, text_key: str = "text",
                  image: Optional[Image.Image] = None) -> str:
    """Convert layout entries (sorted by reading order) into Markdown.

    Accepts either:
      - legacy: List[Dict] of layout entries
      - new:    Dict with "entries" key (per-page envelope)

    `image` is accepted for backwards compatibility and ignored — the page
    raster is not needed to serialize markdown. Keeping it out of the call path
    lets the pipeline free each page's raster as soon as its crops are taken.
    """
    entries = _entries_from_page(layout_data)
    markdown_lines: List[str] = []
    try:
        sorted_items = sorted(
            entries,
            key=lambda x: (
                x.get("bbox", [0, 0, 0, 0])[1],
                x.get("bbox", [0, 0, 0, 0])[0],
            ),
        )
        for item in sorted_items:
            category = item.get("category", "")
            text = item.get(text_key, "")
            if category in ("Image", "Figure"):
                # Reference the stored crop when available (set by
                # _upload_picture_assets), else a descriptive placeholder.
                url = item.get("image_url") or "Image detected"
                alt = (text or "Image").strip() or "Image"
                markdown_lines.append(f"![{alt}]({url})\n")
            elif not text:
                continue
            elif category in ("Page-Header", "Section-Header"):
                markdown_lines.append(f"## {text}\n")
            elif category == "Title":
                markdown_lines.append(f"# {text}\n")
            elif category in ("Text", "Complex-Block", "Bibliography",
                              "Table-Of-Contents"):
                markdown_lines.append(f"{text}\n")
            elif category == "List-Group":
                markdown_lines.append(f"{text}\n")
            elif category == "Table":
                markdown_lines.append(
                    text if text.strip().startswith("<") else f"**Table:** {text}\n"
                )
            elif category in ("Equation-Block", "Formula"):
                markdown_lines.append(f"$$\n{text}\n$$\n")
            elif category in ("Code-Block", "Chemical-Block", "Form"):
                markdown_lines.append(f"```\n{text}\n```\n")
            elif category == "Diagram":
                # Chandra emits mermaid for diagrams.
                markdown_lines.append(f"```mermaid\n{text}\n```\n")
            elif category == "Caption":
                markdown_lines.append(f"*{text}*\n")
            elif category in ("Page-Footer", "Footnote"):
                markdown_lines.append(f"^{text}^\n")
            else:
                # Unknown / future category: render as plain text rather than
                # silently dropping it so OCR output is never lost.
                markdown_lines.append(f"{text}\n")
        return "\n".join(markdown_lines)
    except Exception as e:
        print("Error converting JSON to Markdown:", e)
        return str(layout_data)
