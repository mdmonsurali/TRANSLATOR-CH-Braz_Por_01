"""Low-level OOXML string builders shared by every renderer.

Holds the namespace constants and the small helpers that turn (text, style,
geometry) tuples into the XML fragments python-docx expects to find inside a
<w:r>: floating textboxes, floating pictures, inline pictures, run/paragraph
properties, XML escapes, image relationships.
"""
from __future__ import annotations

import os
import re
from io import BytesIO
from typing import Dict, List, Optional

from docx import Document


# Per-script defaults are env-overridable per deployment.
_CJK_DEFAULT_FONTS = {
    "zh": os.environ.get("DOCX_FONT_ZH", "SimSun"),        # Chinese
    "ja": os.environ.get("DOCX_FONT_JA", "MS Mincho"),     # Japanese
    "ko": os.environ.get("DOCX_FONT_KO", "Batang"),        # Korean
}

# Face used to render ballot/form symbols (☑/☐/○/✓…) that Latin text fonts lack.
# Must be a font that (a) is installed in the render container and (b) directly
# contains these glyphs. DejaVu Sans satisfies both (ships in the LibreOffice
# image; covers U+2610-2612/25CB/2713…). Override per deployment if needed.
_SYMBOL_FONT = os.environ.get("DOCX_FONT_SYMBOL", "DejaVu Sans")

# Common Latin-only families that carry no CJK glyphs. When one of these is the
# active font AND the text has CJK, we add a CJK eastAsia fallback. Matched
# case-insensitively; extend via DOCX_LATIN_ONLY_FONTS (comma-separated).
_LATIN_ONLY_FONTS = {
    "calibri", "calibri light", "arial", "arial narrow", "helvetica",
    "times new roman", "times", "cambria", "cambria math", "georgia",
    "verdana", "tahoma", "courier new", "consolas", "segoe ui", "roboto",
    "open sans", "liberation sans", "liberation serif", "dejavu sans",
}
_LATIN_ONLY_FONTS |= {
    f.strip().lower()
    for f in os.environ.get("DOCX_LATIN_ONLY_FONTS", "").split(",")
    if f.strip()
}

_RE_KANA = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f]")   # Hiragana/Katakana
_RE_HANGUL = re.compile(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]")  # Hangul
_RE_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")     # Han ideographs
_RE_ANY_CJK = re.compile(
    r"[\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f"
    r"\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f"
    r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)

# Ballot / geometric-symbol glyphs the reconstruction emits for form controls
# (checkboxes U+2610-2612, radios U+25CB/25CF/25EF, ticks U+2713/2714/2717/2718).
# These live in the Unicode "Miscellaneous Symbols"/"Dingbats" blocks \u2014 NOT in
# Latin text faces like Calibri/Arial \u2014 so a run like "checkbox Normal checkbox
# Abnormal" carries no CJK yet still has no glyph in its Latin font and the box
# silently drops in Word / strict renderers. We route such runs' eastAsia slot
# to a symbol-bearing CJK face (SimSun et al. all include these), the same
# escape valve used for CJK.
_RE_FORM_SYMBOL = re.compile(
    r"[\u2610-\u2612\u25cb\u25cf\u25ef\u2713\u2714\u2717\u2718]"
)


def has_cjk(text: str) -> bool:
    return bool(text) and bool(_RE_ANY_CJK.search(text))


def has_form_symbol(text: str) -> bool:
    """True when `text` contains a ballot/radio/tick glyph absent from Latin
    text fonts (see `_RE_FORM_SYMBOL`)."""
    return bool(text) and bool(_RE_FORM_SYMBOL.search(text))


def detect_cjk_script(text: str) -> Optional[str]:
    """Return 'ja' / 'ko' / 'zh' for East-Asian text, else None.

    Order matters: Japanese mixes kana with Han, Korean mixes Hangul with Han,
    so the presence of kana/Hangul disambiguates from Han-only Chinese.
    """
    if not text:
        return None
    if _RE_KANA.search(text):
        return "ja"
    if _RE_HANGUL.search(text):
        return "ko"
    if _RE_HAN.search(text):
        return "zh"
    return None


def resolve_east_asia_font(font: str, text: str, explicit: Optional[str]) -> str:
    """Pick the eastAsia face for a run without overriding a detected font.

    - An explicit per-run eastAsia font always wins.
    - Otherwise the run's own (detected or fallback) `font` is used, so a font
      the detector chose is respected in both slots.
    - When the text has CJK and `font` is a known Latin-only family we
      substitute a language-appropriate CJK default (the no-detection case).
    - When the text has a ballot/form symbol (☑/☐/○/✓…) that a Latin-only
      `font` cannot render — but no CJK to trigger the branch above — we route
      the eastAsia slot to a symbol-bearing CJK face so the glyph shows. Without
      this, Latin-script forms silently lose their checkboxes.
    """
    if explicit:
        return explicit
    if (font or "").strip().lower() in _LATIN_ONLY_FONTS:
        if has_cjk(text):
            script = detect_cjk_script(text)
            return _CJK_DEFAULT_FONTS.get(script) or font
        if has_form_symbol(text):
            # No CJK to pick a regional face — route to a font that actually
            # contains the ballot/symbol glyphs (Latin faces don't).
            return _SYMBOL_FONT or font
    return font



# OOXML namespace URIs we need beyond what python-docx already registers.
NS_WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_WPS = "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
NS_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"


def attrs_for_anchor() -> str:
    """Inline xmlns attrs used on the root element when we hand-roll XML."""
    return (
        f' xmlns:wp="{NS_WP}"'
        f' xmlns:a="{NS_A}"'
        f' xmlns:pic="{NS_PIC}"'
        f' xmlns:r="{NS_R}"'
        f' xmlns:w="{NS_W}"'
        f' xmlns:wps="{NS_WPS}"'
        f' xmlns:mc="{NS_MC}"'
    )


_XML_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}


def xml_escape(text: str) -> str:
    if not text:
        return ""
    out = []
    for ch in text:
        if ch in _XML_ESCAPES:
            out.append(_XML_ESCAPES[ch])
        elif ord(ch) < 0x20 and ch not in ("\n", "\t"):
            continue
        else:
            out.append(ch)
    return "".join(out)


_WHITE_THRESHOLD = 240


def rgb_hex(color) -> str:
    if not color:
        return "000000"
    try:
        r, g, b = int(color[0]), int(color[1]), int(color[2])
    except (TypeError, ValueError, IndexError):
        return "000000"
    # Source PDFs sometimes carry white / near-white text designed for a
    # dark background. The reconstructed DOCX renders on a white page, so
    # rewrite anything near-white to black to keep it readable.
    if r >= _WHITE_THRESHOLD and g >= _WHITE_THRESHOLD and b >= _WHITE_THRESHOLD:
        return "000000"
    return f"{r:02X}{g:02X}{b:02X}"


def half_points(size_pt) -> int:
    try:
        v = float(size_pt)
    except (TypeError, ValueError):
        v = 11.0
    if v <= 0:
        v = 11.0
    return int(round(v * 2))


def build_run_xml(text: str, style: Optional[Dict]) -> str:
    """One or more ``<w:r>`` for ``text`` with ``style`` applied.

    - Newlines become soft line breaks (``<w:br/>``).
    - The Latin face is whatever the detector supplied (fallback: Calibri) and
      is never overridden. A ``w:eastAsia`` face is added so CJK codepoints
      render; it defaults to the detected font and only substitutes a
      language-appropriate CJK font when the detected/fallback font is a
      Latin-only family that cannot show the CJK present (see
      ``resolve_east_asia_font``).
    - Inline LaTeX math (``$...$`` / ``\\(...\\)``) is converted in place:
      operators/units/Greek to Unicode, ``_{}``/``^{}`` to real sub/superscript
      runs. Display ``$$...$$`` formulas are handled upstream (formula.py) and
      never reach here as body text.
    """
    style = style or {}
    base_font = style.get("font") or "Calibri"
    font = xml_escape(base_font)
    east = xml_escape(
        resolve_east_asia_font(base_font, text or "", style.get("eastasia"))
    )
    # Word/LibreOffice only draw a char from the eastAsia face if they classify
    # it as East-Asian; ballot/tick symbols are ambiguous, so a Latin-only ascii
    # face would still be used and the glyph would drop. When the run carries a
    # form symbol its Latin font can't render (and no CJK already forced the
    # substitution), point the ascii/hAnsi/cs faces at the symbol-bearing font
    # too so the glyph is guaranteed to draw. CJK runs are unaffected.
    ascii_font = font
    if (
        base_font.strip().lower() in _LATIN_ONLY_FONTS
        and not has_cjk(text or "")
        and has_form_symbol(text or "")
    ):
        ascii_font = east
    size_hp = half_points(style.get("size") or 11)
    bold = bool(style.get("bold"))
    italic = bool(style.get("italic"))
    color_hex = rgb_hex(style.get("color"))

    rpr_parts = [
        f'<w:rFonts w:ascii="{ascii_font}" w:hAnsi="{ascii_font}"'
        f' w:eastAsia="{east}" w:cs="{ascii_font}"/>',
        f'<w:sz w:val="{size_hp}"/>',
        f'<w:szCs w:val="{size_hp}"/>',
        f'<w:color w:val="{color_hex}"/>',
    ]
    if bold:
        rpr_parts.append("<w:b/>")
        rpr_parts.append("<w:bCs/>")
    if italic:
        rpr_parts.append("<w:i/>")
        rpr_parts.append("<w:iCs/>")
    rpr_inner = "".join(rpr_parts)

    def _make_run(rpr_extra: str, chunk: str) -> str:
        rpr = "<w:rPr>" + rpr_inner + rpr_extra + "</w:rPr>"
        body_parts: List[str] = []
        for i, line in enumerate((chunk or "").split("\n")):
            if i > 0:
                body_parts.append("<w:br/>")
            body_parts.append(
                f'<w:t xml:space="preserve">{xml_escape(line)}</w:t>'
            )
        return f"<w:r>{rpr}{''.join(body_parts)}</w:r>"

    text = text or ""
    if "$" in text or "\\(" in text:
        from .latex_inline import build_inline_runs
        return build_inline_runs(text, rpr_inner, _make_run)
    return _make_run("", text)


def build_paragraph_xml(
    runs_xml: str,
    alignment: Optional[str] = None,
    line_pt: Optional[float] = None,
) -> str:
    """Build a <w:p>. Always zeroes before/after spacing so Word's default
    Normal spacing (1.15x line, 10pt after) can't inflate the block height
    and push content past a fixed-height text box. When `line_pt` is given,
    pins an EXACT line height (twentieths of a point) so N lines occupy
    exactly the box height — measurement equals render, no vertical clip.
    """
    spacing = '<w:spacing w:before="0" w:after="0"'
    if line_pt and line_pt > 0:
        spacing += f' w:line="{max(1, int(round(line_pt * 20)))}" w:lineRule="exact"'
    spacing += "/>"
    jc = f'<w:jc w:val="{alignment}"/>' if alignment else ""
    return f"<w:p><w:pPr>{spacing}{jc}</w:pPr>{runs_xml}</w:p>"


def build_anchored_textbox_xml(
    x_emu: int,
    y_emu: int,
    w_emu: int,
    h_emu: int,
    inner_body_xml: str,
    docpr_id: int,
    body_auto_fit: bool = False,
    fill_rgb: str | None = None,
) -> str:
    """Floating text box anchored at (x,y) on the page, holding arbitrary
    `<w:p>...` content. Returns an `<mc:AlternateContent>` block to drop
    inside a `<w:r>`.

    When `body_auto_fit=True`, the shape uses `<a:spAutoFit/>` so Word grows
    the shape to fit content that overflows the bbox-derived `cy` (tables
    with multi-line CJK cells, text that didn't fit even at min font size).
    Otherwise `<a:noAutofit/>` is used so the layout-faithful exact-fit path
    (frozen wrap + pinned line height in add_text_entry) gets the precise
    box height it measured.

    `fill_rgb` is an optional 6-hex-digit solid fill (e.g. "FFFFFF"). Default
    (None) keeps the box transparent (`<a:noFill/>`). An opaque fill overlays
    diagram labels on top of a raster, hiding the original-language glyphs.
    """
    name = f"TxtBox{docpr_id}"
    autofit_xml = "<a:spAutoFit/>" if body_auto_fit else "<a:noAutofit/>"
    fill_xml = (
        f'<a:solidFill><a:srgbClr val="{fill_rgb}"/></a:solidFill>'
        if fill_rgb else "<a:noFill/>"
    )
    return (
        f'<mc:AlternateContent{attrs_for_anchor()}>'
        f'<mc:Choice Requires="wps">'
        f'<w:drawing>'
        f'<wp:anchor distT="0" distB="0" distL="0" distR="0"'
        f' simplePos="0" relativeHeight="{docpr_id}" behindDoc="0"'
        f' locked="0" layoutInCell="1" allowOverlap="1">'
        f'<wp:simplePos x="0" y="0"/>'
        f'<wp:positionH relativeFrom="page"><wp:posOffset>{x_emu}</wp:posOffset></wp:positionH>'
        f'<wp:positionV relativeFrom="page"><wp:posOffset>{y_emu}</wp:posOffset></wp:positionV>'
        f'<wp:extent cx="{w_emu}" cy="{h_emu}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:wrapNone/>'
        f'<wp:docPr id="{docpr_id}" name="{name}"/>'
        f'<wp:cNvGraphicFramePr/>'
        f'<a:graphic>'
        f'<a:graphicData uri="{NS_WPS}">'
        f'<wps:wsp>'
        f'<wps:cNvSpPr txBox="1"/>'
        f'<wps:spPr><a:xfrm>'
        f'<a:off x="0" y="0"/><a:ext cx="{w_emu}" cy="{h_emu}"/>'
        f'</a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'{fill_xml}'
        f'<a:ln><a:noFill/></a:ln>'
        f'</wps:spPr>'
        f'<wps:txbx><w:txbxContent>{inner_body_xml}</w:txbxContent></wps:txbx>'
        f'<wps:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0"'
        f' anchor="t" anchorCtr="0">{autofit_xml}</wps:bodyPr>'
        f'</wps:wsp>'
        f'</a:graphicData></a:graphic>'
        f'</wp:anchor>'
        f'</w:drawing>'
        f'</mc:Choice>'
        f'<mc:Fallback><w:pict/></mc:Fallback>'
        f'</mc:AlternateContent>'
    )


def build_anchored_picture_xml(
    x_emu: int,
    y_emu: int,
    w_emu: int,
    h_emu: int,
    rel_id: str,
    docpr_id: int,
    pic_name: str,
) -> str:
    """Floating picture anchored at (x,y) on the page."""
    return (
        f'<mc:AlternateContent{attrs_for_anchor()}>'
        f'<mc:Choice Requires="wps">'
        f'<w:drawing>'
        f'<wp:anchor distT="0" distB="0" distL="0" distR="0"'
        f' simplePos="0" relativeHeight="{docpr_id}" behindDoc="0"'
        f' locked="0" layoutInCell="1" allowOverlap="1">'
        f'<wp:simplePos x="0" y="0"/>'
        f'<wp:positionH relativeFrom="page"><wp:posOffset>{x_emu}</wp:posOffset></wp:positionH>'
        f'<wp:positionV relativeFrom="page"><wp:posOffset>{y_emu}</wp:posOffset></wp:positionV>'
        f'<wp:extent cx="{w_emu}" cy="{h_emu}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:wrapNone/>'
        f'<wp:docPr id="{docpr_id}" name="{xml_escape(pic_name)}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic>'
        f'<a:graphicData uri="{NS_PIC}">'
        f'<pic:pic>'
        f'<pic:nvPicPr>'
        f'<pic:cNvPr id="{docpr_id}" name="{xml_escape(pic_name)}"/>'
        f'<pic:cNvPicPr/>'
        f'</pic:nvPicPr>'
        f'<pic:blipFill>'
        f'<a:blip r:embed="{rel_id}"/>'
        f'<a:stretch><a:fillRect/></a:stretch>'
        f'</pic:blipFill>'
        f'<pic:spPr>'
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{w_emu}" cy="{h_emu}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'</pic:spPr>'
        f'</pic:pic>'
        f'</a:graphicData></a:graphic>'
        f'</wp:anchor>'
        f'</w:drawing>'
        f'</mc:Choice>'
        f'<mc:Fallback><w:pict/></mc:Fallback>'
        f'</mc:AlternateContent>'
    )


def build_inline_picture_xml(
    w_emu: int,
    h_emu: int,
    rel_id: str,
    docpr_id: int,
    pic_name: str,
) -> str:
    """Inline picture (flows with paragraph text) — used inside table cells so
    the image sits in the cell instead of floating beside it. Geometry is the
    picture's intrinsic w/h; the cell sizes itself to host it."""
    return (
        f'<w:r xmlns:w="{NS_W}">'
        f'<w:drawing xmlns:wp="{NS_WP}" xmlns:a="{NS_A}" xmlns:pic="{NS_PIC}" xmlns:r="{NS_R}">'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{w_emu}" cy="{h_emu}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{docpr_id}" name="{xml_escape(pic_name)}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic>'
        f'<a:graphicData uri="{NS_PIC}">'
        f'<pic:pic>'
        f'<pic:nvPicPr>'
        f'<pic:cNvPr id="{docpr_id}" name="{xml_escape(pic_name)}"/>'
        f'<pic:cNvPicPr/>'
        f'</pic:nvPicPr>'
        f'<pic:blipFill>'
        f'<a:blip r:embed="{rel_id}"/>'
        f'<a:stretch><a:fillRect/></a:stretch>'
        f'</pic:blipFill>'
        f'<pic:spPr>'
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{w_emu}" cy="{h_emu}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'</pic:spPr>'
        f'</pic:pic>'
        f'</a:graphicData></a:graphic>'
        f'</wp:inline>'
        f'</w:drawing>'
        f'</w:r>'
    )


def parse_xml_fragment(xml_str: str):
    """Parse a fragment that already declares its namespaces inline."""
    from lxml import etree
    # parse with namespace cleanup; the fragment carries its own xmlns decls.
    return etree.fromstring(xml_str)


def add_image_relationship(doc: Document, img_buffer: BytesIO, ext: str) -> str:
    """Add an image part to the document and return its relationship id."""
    img_buffer.seek(0)
    rel_id, _image = doc.part.get_or_add_image(img_buffer)
    return rel_id