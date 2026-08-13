"""Plain-text entry rendering (Title, Section-header, Text, List-item,
Caption, etc.). Floating text box with frozen wrap + pinned exact line
height so Word renders exactly the lines we measured."""
from __future__ import annotations

import re
from typing import Dict, Optional


def render_text_entry(ctx, entry: Dict) -> None:
    from .geometry import (
        bbox_px_to_emu, EMU_PER_PT, is_quarter_turn, normalize_rotation,
    )
    from .ooxml import (
        build_anchored_textbox_xml, build_paragraph_xml, build_run_xml,
    )
    from .text_fit import fit_multiline, get_font, has_cjk, wrap_to_width

    text = (entry.get("text") or "").strip()
    if not text:
        return
    bbox = entry.get("bbox")
    if not bbox or len(bbox) != 4:
        return
    category = entry.get("category", "")
    x, y, w, h = bbox_px_to_emu(
        bbox, ctx.zoom, ctx.page_w_pt, ctx.page_h_pt,
    )
    style = dict(entry.get("style") or {})
    alignment = "center" if category == "Title" else None
    # Strip leading markdown decorations the VLM may have emitted.
    text = re.sub(r"^\s*#+\s*", "", text)
    text = text.replace("**", "")
    # Normalise inline LaTeX ($...$ / \(...\)) — units, ±, subscripts, Greek —
    # to plain Unicode BEFORE measuring/wrapping, so the frozen wrap matches
    # what actually renders (the fixed-height text box hard-clips overflow, so
    # measurement must equal render). Display $$...$$ formulas are a separate
    # entry category and never reach here.
    from .latex_inline import strip_inline_math_to_plain
    text = strip_inline_math_to_plain(text)

    # Binary-search the largest font size at which `text` word-wraps to fit
    # inside the OCR bbox (both width AND height), and capture the exact
    # wrap. Only shrinks; if the OCR-detected size already fits it is
    # returned as-is. We then FREEZE that wrap into hard line breaks and pin
    # an exact line height so Word renders precisely the lines we measured
    # and they fill exactly the box height — no last-line clip.
    x1, y1, x2, y2 = bbox
    box_w_pt = max(1.0, (x2 - x1) / ctx.zoom)
    box_h_pt = max(1.0, (y2 - y1) / ctx.zoom)
    # A quarter-turned entry reads along the box's OTHER axis, so the fitter
    # must measure against the swapped pair — the shape itself is unrotated
    # (`bodyPr vert=` spins only the text), so its geometry stays as computed.
    # Without this the fitter tries to wrap the text across the narrow side and
    # shrinks it to the readable floor: the 21pt-wide side-labels rendered at
    # 5.5pt with words broken mid-token.
    rotation = normalize_rotation(entry.get("rotation"))
    if is_quarter_turn(rotation):
        box_w_pt, box_h_pt = box_h_pt, box_w_pt
    base_size_pt = float(style.get("size") or 11.0)
    fitted, lines = fit_multiline(text, box_w_pt, box_h_pt, max_size_pt=base_size_pt)

    line_pt: Optional[float] = None
    render_text = text
    # `lines is None` means the text can't fit the OCR bbox even at the readable
    # floor. Rather than clipping it to an ellipsis (which silently drops the
    # content — e.g. long footers/titles the OCR gave a too-short bbox), keep
    # the readable floor size, wrap the FULL text, and GROW the box downward to
    # hold every line (spAutoFit). Readable-and-taller beats a lost line.
    # Growth is DOWNWARD-only, which is meaningless once the text is turned:
    # a rotated box would have to grow sideways, straight into whatever sits
    # beside it (on these pages, the diagram itself). Deliberate limitation —
    # rotated text that overruns shrinks to the readable floor instead.
    body_auto_fit = lines is None and not is_quarter_turn(rotation)
    if lines is None:
        floor_pt = 6.0
        style["size"] = min(base_size_pt, floor_pt)
        font = get_font(max(1, int(round(floor_pt))), bold=bool(style.get("bold")))
        if font is not None:
            wrapped = wrap_to_width(text, font, box_w_pt * 0.93, int(round(floor_pt)))
            render_text = "\n".join(wrapped)
            asc, desc = font.getmetrics()
            natural_h = asc + desc
            if has_cjk(text):
                natural_h = max(natural_h, floor_pt * 1.2)
            line_h_pt = natural_h * 1.10
            needed_h_pt = max(box_h_pt, line_h_pt * len(wrapped))
            # Grow the emitted box height to hold all lines. Skipped when the
            # text is turned: `h` is then the box's SHORT side, and stretching
            # it would fatten the label across the drawing instead of
            # lengthening it along its reading direction.
            if not is_quarter_turn(rotation):
                h = max(h, int(round(needed_h_pt * EMU_PER_PT)))
    else:
        if fitted < base_size_pt:
            style["size"] = fitted
        render_text = "\n".join(lines)        # freeze the measured wrap
        if len(lines) >= 2:
            used_size = float(style.get("size") or 11.0)
            exact = box_h_pt / len(lines)      # N lines exactly fill the bbox
            # Cap the per-line slot at ~1.5x the glyph size. When the OCR bbox
            # is taller than the wrapped text needs, filling it exactly would
            # inflate the apparent line spacing; capping only shrinks the slot
            # (spare height falls to the box bottom) so no line can clip.
            line_pt = min(exact, used_size * 1.5)

    # Keep the (possibly grown) box on the page. A text box anchored at `y`
    # grows DOWNWARD, so an entry near the page bottom — most commonly a
    # Page-Footer the OCR gave a too-short bbox, but also any bottom-anchored
    # text — would overflow the page edge and be visually clipped. When the box
    # bottom would pass the page bottom, lift `y` up by the overflow (never above
    # the page top) so every measured line stays inside the printable area.
    page_h_emu = int(round(ctx.page_h_pt * EMU_PER_PT))
    overflow = (y + h) - page_h_emu
    if overflow > 0:
        y = max(0, y - overflow)

    runs_xml = build_run_xml(render_text, style)
    para_xml = build_paragraph_xml(runs_xml, alignment=alignment, line_pt=line_pt)
    ctx.xml_chunks.append(
        build_anchored_textbox_xml(
            x, y, w, h, para_xml, ctx._next_id(),
            body_auto_fit=body_auto_fit,
            rotation=rotation,
        )
    )