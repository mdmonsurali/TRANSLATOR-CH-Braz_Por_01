import asyncio
import base64
import logging
import os
import re
import tempfile
from collections import Counter
from typing import Tuple

from bs4 import BeautifulSoup
from openai import AsyncOpenAI, OpenAI
from PIL import Image

log = logging.getLogger("ocr_service")

VLLM_PORT = os.getenv("VLLM_PORT", "8888")
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "3600"))

# Served-model name passed to `vllm serve ... --served-model-name`.
MODEL_NAME = os.getenv("OCR_MODEL_NAME", "chandra")

# Chandra's canvas for normalized bboxes (data-bbox values are 0..BBOX_SCALE).
BBOX_SCALE = int(os.getenv("OCR_BBOX_SCALE", "1000"))

# Chandra default output-token budget (chandra/settings.py MAX_OUTPUT_TOKENS).
MAX_OUTPUT_TOKENS = int(os.getenv("OCR_MAX_TOKENS", "12384"))

# Repeat-token retry budget (replaces Unlimited-OCR's ngram logits processor).
MAX_RETRIES = int(os.getenv("OCR_MAX_RETRIES", "6"))


REPEAT_RETRY = os.getenv("OCR_REPEAT_RETRY", "false").strip().lower() in {
    "1", "true", "yes", "on"}


REQUEST_TIMEOUT_SEC = float(os.getenv("OCR_REQUEST_TIMEOUT_SEC", "300"))

# ── Chandra ocr_layout prompt (ported from chandra/prompts.py) ───────────────

ALLOWED_TAGS = [
    "math", "br", "i", "b", "u", "del", "sup", "sub", "table", "tr", "td",
    "p", "th", "div", "pre", "h1", "h2", "h3", "h4", "h5", "ul", "ol", "li",
    "input", "a", "span", "img", "hr", "tbody", "small", "caption", "strong",
    "thead", "big", "code", "chem",
]
ALLOWED_ATTRIBUTES = [
    "class", "colspan", "rowspan", "display", "checked", "type", "border",
    "value", "style", "href", "alt", "align", "data-bbox", "data-label",
]

_PROMPT_ENDING = f"""
Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Emit EVERY checkbox and radio button as an <input type="checkbox"> (or type="radio") tag, adding the checked attribute only when the box is visibly marked. This includes EMPTY / unchecked boxes — never omit a box because it is blank. Do NOT transcribe a checkbox as its label text alone: an option line like "☐是 ☐否" or "是 否" preceded by boxes must become <input type="checkbox"/>是 <input type="checkbox"/>否, preserving the box for each option.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()

OCR_LAYOUT_PROMPT = f"""
OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure
- Chemical-Block
- Diagram
- Bibliography
- Blank-Page

{_PROMPT_ENDING}
""".strip()


# The Chandra labels that carry a cropped raster (Image/Figure). Charts are
# emitted as <img> Image blocks; diagrams become mermaid *text*, so they are
# NOT cropped unless the model actually put an <img> in them.
PICTURE_LABELS = {"Image", "Figure"}
# Labels the model converts to TEXT (mermaid) but which describe an on-page
# drawing we'd rather keep as the original picture. Unlike table-cell <img>
# placeholders these blocks carry their own data-bbox, so the crop is exact.
DIAGRAM_LABELS = {"Diagram"}
# Labels whose text is HTML the reconstruction table renderer consumes.
TABLE_LABELS = {"Table"}
# Labels whose text is LaTeX for the formula renderer.
FORMULA_LABELS = {"Equation-Block"}
# Labels whose HTML is an ordered/unordered list; markers are re-emitted from
# the <ol>/<ul> structure the model may express them with.
LIST_LABELS = {"List-Group"}


_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY", timeout=VLLM_TIMEOUT)
    return _client


def get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY", timeout=VLLM_TIMEOUT)
    return _async_client


# ── Image sizing (ported from chandra/model/util.py scale_to_fit) ────────────

def scale_to_fit(
    img: Image.Image,
    max_size: Tuple[int, int] = (3072, 2048),
    min_size: Tuple[int, int] = (1792, 28),
    grid_size: int = 28,
) -> Image.Image:
    """Resize `img` into the pixel envelope Chandra was trained on (grid-28,
    max 3072x2048, min 1792x28) while preserving aspect ratio as closely as
    the grid allows. Returns the original if no change is needed."""
    resample_method = Image.Resampling.LANCZOS
    width, height = img.size
    if width <= 0 or height <= 0:
        return img

    original_ar = width / height
    current_pixels = width * height
    max_pixels = max_size[0] * max_size[1]
    min_pixels = min_size[0] * min_size[1]

    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5

    w_blocks = max(1, round((width * scale) / grid_size))
    h_blocks = max(1, round((height * scale) / grid_size))

    while (w_blocks * h_blocks * grid_size * grid_size) > max_pixels:
        if w_blocks == 1 and h_blocks == 1:
            break
        if w_blocks == 1:
            h_blocks -= 1
            continue
        if h_blocks == 1:
            w_blocks -= 1
            continue
        ar_w_loss = abs(((w_blocks - 1) / h_blocks) - original_ar)
        ar_h_loss = abs((w_blocks / (h_blocks - 1)) - original_ar)
        if ar_w_loss < ar_h_loss:
            w_blocks -= 1
        else:
            h_blocks -= 1

    new_width = w_blocks * grid_size
    new_height = h_blocks * grid_size
    if (new_width, new_height) == (width, height):
        return img
    return img.resize((new_width, new_height), resample=resample_method)


_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")

_MAX_SAME_CELL_FRAC = float(os.getenv("OCR_MAX_SAME_CELL_FRAC", "0.70"))
_MIN_CELLS_FOR_CHECK = 60


def _repeats_one_cell(html: str) -> bool:
    """True when one value dominates a table's cells — a cell-level repeat loop.

    `detect_repeat_token` is a pure STRING test over the tail, so it can miss (or
    only marginally catch) this shape: the run-on is structured markup, and a
    trailing `</tr></tbody></table>` breaks the exact-suffix match it needs. This
    counts cell VALUES instead, which is what actually degenerated.
    """
    if not html or len(html) < 200:
        return False
    values = [_TAG_RE.sub("", m).strip().lower() for m in _CELL_RE.findall(html)]
    values = [v for v in values if v]
    if len(values) < _MIN_CELLS_FOR_CHECK:
        return False
    top = Counter(values).most_common(1)[0][1]
    return top / len(values) > _MAX_SAME_CELL_FRAC


def detect_repeat_token(
    predicted_tokens: str,
    base_max_repeats: int = 4,
    window_size: int = 500,
    cut_from_end: int = 0,
    scaling_factor: float = 3.0,
) -> bool:
    """True if the tail of the generation is a short sequence repeated many
    times — the classic VLM degeneration loop. Ported from chandra util.py."""
    if cut_from_end > 0:
        predicted_tokens = predicted_tokens[:-cut_from_end]
    for seq_len in range(1, window_size // 2 + 1):
        candidate_seq = predicted_tokens[-seq_len:]
        max_repeats = int(base_max_repeats * (1 + scaling_factor / seq_len))
        repeat_count = 0
        pos = len(predicted_tokens) - seq_len
        if pos < 0:
            continue
        while pos >= 0:
            if predicted_tokens[pos:pos + seq_len] == candidate_seq:
                repeat_count += 1
                pos -= seq_len
            else:
                break
        if repeat_count > max_repeats:
            return True
    return False


# ── Math normalization (KaTeX <math> / LaTeX → renderer-friendly) ────────────

def _katex_to_latex(html_fragment: str) -> str:
    """Strip <math>...</math> wrappers and return the inner LaTeX for the
    formula renderer (which wants bare LaTeX, no $-delimiters, no <math>)."""
    soup = BeautifulSoup(html_fragment, "html.parser")
    maths = soup.find_all("math")
    if maths:
        parts = [m.decode_contents() for m in maths]
        latex = " ".join(p.strip() for p in parts if p.strip())
    else:
        latex = soup.get_text(" ", strip=True)
    # Drop $ delimiters the model may have emitted; the renderer adds its own.
    latex = latex.replace("$$", "").replace("$", "").strip()
    return latex


_INLINE_MATH_TAG_RE = re.compile(r"<math\b[^>]*>(.*?)</math>", re.DOTALL | re.IGNORECASE)


def _balance_inline_dollars(text: str) -> str:
    """Close a dangling inline-math ``$`` so the reconstruction renderer's
    ``$…$`` path fires instead of leaving raw LaTeX literal.

    The VLM frequently opens inline math with a ``$`` but omits the closing one,
    e.g. ``(AAF) $= Q_{10}^{[(60-24)/10]} = 12.13`` — one ``$`` and no partner,
    so ``build_inline_runs`` (which needs a matched pair) renders the whole tail
    as literal ``Q_{10}^{...}``. When a line has an ODD number of ``$`` we treat
    everything from the last unmatched ``$`` to end-of-line as the math span and
    append a closing ``$``. Balanced text (even count, incl. zero) is returned
    unchanged, and ``$$`` display markers are left alone. Applied per line so an
    unbalanced ``$`` can't swallow the rest of a multi-line block."""
    if "$" not in text:
        return text
    out_lines = []
    for line in text.split("\n"):
        # Ignore display-math markers ($$) when counting inline delimiters.
        inline_count = line.replace("$$", "").count("$")
        if inline_count % 2 == 1:
            line = line + "$"
        out_lines.append(line)
    return "\n".join(out_lines)


def _inline_math_to_dollars(html_fragment: str) -> str:
    """Rewrite inline <math>…</math> to `$…$` so the text renderer's inline
    LaTeX path (`build_inline_runs`) can convert it to Unicode. Also balances a
    dangling inline ``$`` the model may have emitted."""
    def _sub(m: re.Match) -> str:
        body = BeautifulSoup(m.group(1), "html.parser").get_text(" ", strip=True)
        return f"${body}$" if body else ""
    return _balance_inline_dollars(_INLINE_MATH_TAG_RE.sub(_sub, html_fragment))


def _list_marker(list_type: str, idx: int) -> str:
    """Marker text for the `idx`-th (0-based) item of an ordered list of the
    given HTML ``type`` (1/a/A/i/I), or a bullet for an unordered list."""
    if list_type == "a":
        return f"{chr(ord('a') + idx % 26)}) "
    if list_type == "A":
        return f"{chr(ord('A') + idx % 26)}) "
    if list_type in ("i", "I"):
        # Lightweight roman numerals (lists rarely exceed a handful of items).
        romans = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
                  "xi", "xii", "xiii", "xiv", "xv"]
        r = romans[idx] if idx < len(romans) else str(idx + 1)
        return f"{r.upper() if list_type == 'I' else r}) "
    if list_type == "bullet":
        return "• "
    return f"{idx + 1}. "          # default: decimal


def _list_group_to_text(html_fragment: str) -> str:
    """Flatten a List-Group block, RE-EMITTING the ordered/unordered markers the
    model expressed as ``<ol>``/``<ul>`` structure.

    Chandra returns lists two ways: markers baked into the text (``<p>b) …</p>``)
    — which survive plain flattening — or as real ``<ol type="a"><li>…`` where the
    marker is implicit in the list semantics and the naive text flatten drops it
    (the reported ``a) b) c)``-missing bug). Here we walk each ``<ol>``/``<ul>``
    and prepend the correct marker per ``<li>`` (honouring ``type`` and ``start``),
    UNLESS the item text already begins with its own marker (so we never double
    up on the baked-in case). Non-list content in the block is flattened normally."""
    soup = BeautifulSoup(_inline_math_to_dollars(html_fragment), "html.parser")
    for inp in soup.find_all("input"):
        itype = (inp.get("type") or "checkbox").lower().rstrip("/").strip()
        if itype in ("checkbox", "radio"):
            inp.replace_with("☑" if inp.has_attr("checked") else "☐")
        else:
            inp.replace_with("")
    for br in soup.find_all("br"):
        br.replace_with("\n")

    _already_marked = re.compile(r"^\s*([0-9]+|[a-zA-Z]|[ivxIVX]+)\s*[.)、]")

    for lst in soup.find_all(["ol", "ul"]):
        if lst.name == "ul":
            ltype = "bullet"
            start = 0
        else:
            ltype = (lst.get("type") or "1").strip() or "1"
            try:
                start = max(1, int(lst.get("start", 1))) - 1
            except (TypeError, ValueError):
                start = 0
        items = lst.find_all("li", recursive=False) or lst.find_all("li")
        for i, li in enumerate(items):
            body = li.get_text(" ", strip=True)
            if not body:
                continue
            marker = "" if _already_marked.match(body) else _list_marker(ltype, start + i)
            li.string = f"{marker}{body}"
        lst.append("\n")

    for block in soup.find_all(["p", "li", "div"]):
        block.append("\n")
    text = soup.get_text()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _html_block_to_text(html_fragment: str) -> str:
    """Flatten a non-table/non-formula block's inner HTML to text the text
    renderer consumes: inline <math> → $…$, <br> → newline, <p> → newline,
    everything else stripped to its text. Bold/italic detection is handled
    separately in chandra_style by inspecting the same HTML."""
    frag = _inline_math_to_dollars(html_fragment)
    soup = BeautifulSoup(frag, "html.parser")
    for inp in soup.find_all("input"):
        itype = (inp.get("type") or "checkbox").lower().rstrip("/").strip()
        if itype in ("checkbox", "radio"):
            inp.replace_with("☑" if inp.has_attr("checked") else "☐")
        else:
            inp.replace_with("")
    # Convert structural breaks to newlines before extracting text.
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(["p", "li", "tr", "div"]):
        block.append("\n")
    text = soup.get_text()
    # Collapse runs of spaces/tabs but keep newlines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── HTML → layout entries (adapted from chandra/output.py parse_layout) ──────

def _parse_bbox(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    try:
        parts = [int(round(float(v))) for v in raw.split()]
    except (ValueError, TypeError):
        return None
    if len(parts) != 4:
        return None
    return parts


def _rescale_bbox_to_pixels(bbox_1000: list[int], img_w: int, img_h: int) -> list[int]:
    """Map a 0..BBOX_SCALE bbox straight onto the ORIGINAL page image in
    pixels. We rescale against the original size (not the scale_to_fit size)
    so downstream geometry — which assumes bbox_px / zoom = points against the
    rendered page — stays correct without a second remap."""
    x1, y1, x2, y2 = bbox_1000
    sx = img_w / BBOX_SCALE
    sy = img_h / BBOX_SCALE
    return [
        max(0, min(img_w, round(x1 * sx))),
        max(0, min(img_h, round(y1 * sy))),
        max(0, min(img_w, round(x2 * sx))),
        max(0, min(img_h, round(y2 * sy))),
    ]


_TABLE_TAG_RE = re.compile(r"<table\b", re.IGNORECASE)


def _looks_like_table(html_fragment: str) -> bool:
    """True when the block HTML contains a real ``<table>`` — used to rescue a
    table the model mislabelled (typically as ``Form``) so it still reconstructs
    as a grid instead of being flattened to plain text."""
    return bool(_TABLE_TAG_RE.search(html_fragment or ""))


def _prepare_table_html(inner: str) -> str:
    """Normalise a table block's HTML for the reconstruction renderer: drop the
    model's per-tag ``data-bbox`` noise and convert in-cell ``<math>…</math>``
    to ``$…$`` (the table parser strips ``<math>`` tags, leaving raw LaTeX like
    ``Q_{10}`` literal; the cell renderer turns ``$…$`` into proper Unicode +
    sub/superscript runs). Balancing is NOT applied — a per-cell stray ``$``
    would corrupt the surrounding HTML, and in-cell math arrives as well-formed
    ``<math>`` pairs."""
    tsoup = BeautifulSoup(inner, "html.parser")
    for tag in tsoup.find_all(attrs={"data-bbox": True}):
        del tag["data-bbox"]
    for mtag in tsoup.find_all("math"):
        body = mtag.get_text(" ", strip=True)
        mtag.replace_with(f"${body}$" if body else "")
    return str(tsoup)


def parse_layout_html(html: str, orig_img: Image.Image) -> list[dict]:
    """Convert Chandra's layout HTML into layout entries.

    Each entry:
        {"bbox": [x1,y1,x2,y2] (px), "category": <Chandra label>, "text": ...}
    and, for Image/Figure blocks, an `image_obj` cropped from `orig_img`.
    Blocks with no valid bbox or the Blank-Page label are dropped.
    """
    img_w, img_h = orig_img.size
    soup = BeautifulSoup(html, "html.parser")
    top_level = soup.find_all("div", recursive=False)
    if not top_level:

        top_level = [d for d in soup.find_all("div") if d.get("data-label")]

    entries: list[dict] = []
    for div in top_level:
        label = div.get("data-label") or "Text"
        if label == "Blank-Page":
            continue
        bbox_1000 = _parse_bbox(div.get("data-bbox"))
        if bbox_1000 is None:
            log.debug("[ocr] block with no/invalid bbox skipped (label=%s)", label)
            continue
        bbox = _rescale_bbox_to_pixels(bbox_1000, img_w, img_h)
        inner = div.decode_contents()

        entry: dict = {"bbox": bbox, "category": label}

        if label in PICTURE_LABELS:
            try:
                entry["image_obj"] = orig_img.crop(tuple(bbox))
            except (ValueError, SystemError):
                pass
            img_tag = BeautifulSoup(inner, "html.parser").find("img")
            entry["text"] = (img_tag.get("alt") if img_tag else "") or ""
        elif label in DIAGRAM_LABELS:
            entry["text"] = _html_block_to_text(inner)
            try:
                entry["image_obj"] = orig_img.crop(tuple(bbox))
                entry["category"] = "Image"
                entry["source"] = "diagram-recovered"
            except (ValueError, SystemError):
                # Invalid crop geometry — leave it as a Diagram text block so
                # the mermaid content still renders through the text path.
                pass
        elif label in TABLE_LABELS or _looks_like_table(inner):
            entry["category"] = "Table"
            entry["text"] = _prepare_table_html(inner)
        elif label in FORMULA_LABELS:
            entry["text"] = _katex_to_latex(inner)
        elif label in LIST_LABELS:
            entry["text"] = _list_group_to_text(inner)
            entry["_html"] = inner
        else:
            entry["text"] = _html_block_to_text(inner)
            # Stash the raw HTML so chandra_style can read inline bold/italic.
            entry["_html"] = inner

        entries.append(entry)
    return entries


def _image_to_b64(img: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _build_messages(img: Image.Image) -> list:
    b64 = _image_to_b64(img)
    return [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": OCR_LAYOUT_PROMPT},
        ],
    }]


def _open_rgb(image_path: str) -> Image.Image:
    with Image.open(image_path) as im:
        return im.convert("RGB")


async def ocr_image_async(orig_img: Image.Image) -> list:
    """Async layout extraction against Chandra for an in-memory RGB image.

    Returns a list of layout entries with pixel bboxes (relative to `orig_img`).
    Retries with rising temperature when the model degenerates into a repeat
    loop (Chandra's guard, replacing the ngram logits processor Unlimited-OCR
    needed). Shared by the full-page path (`process_image_async`) and the
    targeted table re-OCR (`table_reocr`)."""
    model_img = scale_to_fit(orig_img)
    messages = _build_messages(model_img)

    client = get_async_client()
    raw = ""
    for attempt in range(MAX_RETRIES + 1):
        temperature = min(0.2 * attempt, 0.8)
        top_p = 0.1 if attempt == 0 else 0.95
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=MAX_OUTPUT_TOKENS,
                timeout=REQUEST_TIMEOUT_SEC,
            )
        except Exception as exc:
            log.warning("[ocr] request failed/timed out on attempt %d (%s); "
                        "retrying", attempt + 1, type(exc).__name__)
            raw = ""
            continue
        raw = response.choices[0].message.content or ""

        finish = getattr(response.choices[0], "finish_reason", None)
        truncated = finish == "length"
        if truncated:
            log.warning(
                "[ocr] generation hit the %d-token output cap (finish_reason="
                "length) — the tail of this page is TRUNCATED%s",
                MAX_OUTPUT_TOKENS,
                "; retrying" if (REPEAT_RETRY and attempt < MAX_RETRIES) else "",
            )

        if not REPEAT_RETRY:
            break
        has_repeat = (
            detect_repeat_token(raw)
            or (len(raw) > 50 and detect_repeat_token(raw, cut_from_end=50))
            or _repeats_one_cell(raw)
        )
        if not has_repeat and not truncated:
            break
        if attempt >= MAX_RETRIES:
            break
        if has_repeat:
            log.warning("[ocr] repeat-token loop detected, retrying (attempt %d)",
                        attempt + 1)

    if not raw:
        log.error("[ocr] page produced no output after %d attempt(s); "
                  "returning empty layout", MAX_RETRIES + 1)
    return parse_layout_html(raw, orig_img)


async def process_image_async(image_path: str) -> list:
    """Async layout extraction against Chandra for a file on disk. Thin wrapper
    over `ocr_image_async` that opens + RGB-converts the image first."""
    return await ocr_image_async(_open_rgb(image_path))


def process_image(image_path: str, vllm_url: str | None = None) -> list:
    """Sync layout extraction (kept for parity / scripts). vllm_url ignored."""
    orig_img = _open_rgb(image_path)
    model_img = scale_to_fit(orig_img)
    response = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=_build_messages(model_img),
        temperature=0.0,
        top_p=0.1,
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT_SEC,
    )
    return parse_layout_html(response.choices[0].message.content or "", orig_img)
