"""Plain-text entry rendering (Title, Section-header, Text, List-item,
Caption, etc.) coordinated with text fitting to prevent overlapping.
"""
from __future__ import annotations

import re
from typing import Dict, Optional


# Short-label categories that should GROW WIDTH (into the free right margin)
# rather than wrap to a new line and grow height. A translated caption/header is
# usually longer than its source bbox (e.g. Portuguese vs Chinese); wrapping it
# grows the box downward and overlaps the content below. These read as a single
# label line, so widening the box to fit is both truer to the source and avoids
# the overlap.
_WIDTH_GROW_CATEGORIES = {
    "Caption", "Section-Header", "Section-header", "Title",
    "Page-Header", "Page-header",
}
# Leave this much of the page as a right-hand margin when growing width (pt).
_RIGHT_MARGIN_PT = 30.0


def _downward_text_room_pt(ctx, entry: Dict, x1: float, x2: float,
                           y1: float) -> Optional[float]:
    """Vertical room (pt) from this box's top edge down to the TOP of the nearest
    entry below that horizontally overlaps this box's [x1, x2] span — i.e. how
    tall this box may become before its text would collide with that neighbour's
    text. Returns None when there is no such neighbour (unbounded growth is fine).

    Uses `ctx.page_entries` (the reflowed page). Horizontal overlap is required so
    a box in a different column is never treated as an obstacle. A small gap is
    kept so the two texts don't touch."""
    entries = getattr(ctx, "page_entries", None) or []
    if not entries:
        return None
    zoom = ctx.zoom
    _GAP_PX = 4.0 * zoom
    nearest_top = None
    for other in entries:
        if other is entry:
            continue
        ob = other.get("bbox")
        if not ob or len(ob) != 4:
            continue
        ox1, oy1, ox2, oy2 = [float(v) for v in ob]
        # Must sit BELOW our top edge and horizontally overlap our x-span.
        if oy1 <= y1:
            continue
        if not (ox1 < x2 and ox2 > x1):
            continue
        if nearest_top is None or oy1 < nearest_top:
            nearest_top = oy1
    if nearest_top is None:
        return None
    return max(0.0, (nearest_top - _GAP_PX - y1) / zoom)


def render_text_entry(ctx, entry: Dict) -> None:
    from .geometry import (
        bbox_px_to_emu, EMU_PER_PT, is_quarter_turn, normalize_rotation,
    )
    from .ooxml import build_anchored_textbox_xml, build_paragraph_xml, build_run_xml
    from .text_fit import (
        get_font, has_cjk, wrap_to_width, fit_multiline, measure_width_px,
        _TEXTBOX_EDGE_PAD_PT,
    )

    text = (entry.get("text") or "").strip()
    if not text:
        return
    bbox = entry.get("bbox")
    if not bbox or len(bbox) != 4:
        return
    category = entry.get("category", "")
    style = dict(entry.get("style") or {})
    # Left-align everything (no centering). With the hugged box width the text
    # reads from the original left margin.
    alignment = None

    # Strip markdown decorations the VLM may have emitted.
    text = re.sub(r"^\s*#+\s*", "", text)
    text = text.replace("**", "")

    x1, y1, x2, y2 = [float(v) for v in bbox]
    zoom = ctx.zoom
    # Carried straight through from OCR (the translator preserves unknown entry
    # keys), so a label detected as rotated stays rotated in the translation.
    rotation = normalize_rotation(entry.get("rotation"))

    # Width-grow for short labels: if the text is a single logical line that
    # doesn't fit the source bbox width at its base size, widen the box to the
    # right (up to the page's right margin) so it fits WITHOUT wrapping — instead
    # of wrapping and growing height (which overlaps the content below). Only for
    # single-line labels; multi-line prose keeps its bbox and wraps as before.
    # Rotated entries are excluded: "widen to the right" grows the box ALONG
    # the reading direction once the text is turned, and the obstacle logic
    # below models neighbours in unrotated page space, so it would happily
    # extend a vertical label across the drawing beside it.
    if (category in _WIDTH_GROW_CATEGORIES and "\n" not in text
            and not is_quarter_turn(rotation)):
        _base_pt = float(style.get("size") or 11.0)
        _want_bold = bool(style.get("bold")) or category in (
            "Title", "Section-header", "Section-Header", "Page-header",
            "Page-Header", "Caption",
        )
        _font = get_font(max(1, int(round(_base_pt))), bold=_want_bold)
        if _font is not None:
            _cur_w_pt = max(1.0, (x2 - x1) / zoom)
            _x1_pt = x1 / zoom
            # One-line width the text needs at base size (+ a little slack).
            _text_w_pt = measure_width_px(
                text, _font, int(round(_base_pt))
            ) + 6.0
            # Room available from the box's left edge to the page right margin.
            _avail_w_pt = max(0.0, (ctx.page_w_pt - _RIGHT_MARGIN_PT) - _x1_pt)
            # OBSTACLE-AWARE cap: never widen the box past a right-hand neighbour
            # that shares this box's vertical band. Growing only to the page
            # margin (as before) let a header run its rendered TEXT straight over
            # the neighbour to its right. Reuse reflow's right-limit computation
            # against the page's reflowed entries; falls back to the page-margin
            # available width when there is no obstacle.
            _page_entries = getattr(ctx, "page_entries", None) or []
            if _page_entries:
                from .reflow import _right_expansion_limit_px
                _GAP_PT = 6.0
                _right_limit_px = _right_expansion_limit_px(
                    entry, _page_entries, ctx.page_w_pt * zoom, _GAP_PT * zoom,
                )
                _obstacle_avail_pt = max(0.0, (_right_limit_px - x1) / zoom)
                _avail_w_pt = min(_avail_w_pt, _obstacle_avail_pt)
            if _text_w_pt > _cur_w_pt and _avail_w_pt > _cur_w_pt:
                # Grow to hold the whole text on one line when it fits the
                # available room; otherwise take ALL the available width so it
                # wraps into as FEW lines as possible (never stay narrow — that
                # is what forced the extra wrapped lines + height overflow).
                # Capped at the neighbour, so the grown box never crosses it;
                # if the label still can't fit one line here it wraps and grows
                # DOWNWARD (below), and if it still overflows the font shrinks.
                _new_w_pt = min(_text_w_pt, _avail_w_pt)
                x2 = x1 + _new_w_pt * zoom

    # Add 6 pt of bottom padding in zoom-space so the OOXML textbox has room
    # for Word's internal line-height rounding that PIL measurement can't capture.
    _PAD_PX = 6.0 * zoom
    padded_y2 = min(y2 + _PAD_PX, ctx.page_h_pt * zoom)
    box_w_pt = max(1.0, (x2 - x1) / zoom)
    box_h_pt = max(1.0, (padded_y2 - y1) / zoom)
    # Quarter-turned text reads along the box's other axis — measure against the
    # swapped pair. The shape stays unrotated (`bodyPr vert=` turns only the
    # text), so its emitted geometry is unaffected.
    if is_quarter_turn(rotation):
        box_w_pt, box_h_pt = box_h_pt, box_w_pt

    base_size_pt = float(style.get("size") or 11.0)
    bold_render = bool(style.get("bold")) or category in (
        "Title", "Section-header", "Page-header", "Page-footer", "Caption",
    )

    # Use the robust fitting engine to calculate font size and line splits
    # maintaining a strict readable minimum floor (6.0 pt) to avoid unreadable text.
    fit_size_pt, lines = fit_multiline(
        text,
        box_w_pt,
        box_h_pt,
        max_size_pt=base_size_pt,
        min_size_pt=6.0,  # Human-readable font floor
        bold=bold_render,
        # Zero-inset text box → wrap at ~full width (fixed pad), so a word Word
        # would keep on line 1 isn't pushed to line 2 by a 7% haircut.
        edge_pad_pt=_TEXTBOX_EDGE_PAD_PT,
    )

    x, y, w, h = bbox_px_to_emu(
        [x1, y1, x2, padded_y2], zoom, ctx.page_w_pt, ctx.page_h_pt,
    )

    # Text-safe vertical room: how far the box may grow DOWN before its text
    # would overlap the rendered text of the nearest neighbour below (in the same
    # horizontal band). Box overlap is acceptable; text overlap is not. When the
    # natural (floor-size, full-wrap) height would exceed this room, we shrink the
    # FONT so the text fits the room instead of overrunning the neighbour's text.
    _avail_h_down_pt = _downward_text_room_pt(ctx, entry, x1, x2, y1)

    # `lines is None` means the text can't fit the OCR bbox even at the 6 pt
    # floor. Instead of clipping it to an ellipsis (silently losing content —
    # e.g. long footers/titles the OCR gave a too-short bbox), keep the floor
    # size, wrap the FULL text, and GROW the box downward (spAutoFit) so every
    # line is visible. Fixed boundaries are still used for text that fits.
    # Downward growth is meaningless for turned text — the box would have to
    # grow sideways, into the drawing beside it. Rotated overruns shrink to the
    # readable floor instead (deliberate limitation).
    body_auto_fit = lines is None and not is_quarter_turn(rotation)
    if lines is None:
        floor_pt = 6.0
        style["size"] = min(base_size_pt, floor_pt)
        font = get_font(max(1, int(round(floor_pt))), bold=bold_render)
        if font is not None:
            wrapped = wrap_to_width(
                text, font, max(1.0, box_w_pt - _TEXTBOX_EDGE_PAD_PT),
                int(round(floor_pt)),
            )
            processed_text = "\n".join(wrapped)
            asc, desc = font.getmetrics()
            natural_h = asc + desc
            if has_cjk(text):
                natural_h = max(natural_h, floor_pt * 1.2)
            line_h_pt = natural_h * 1.10
            needed_h_pt = max(box_h_pt, line_h_pt * len(wrapped))
            # If growing down to `needed_h_pt` would overlap the neighbour below's
            # TEXT, try to fit within the available room by shrinking the font
            # (translator only — this whole renderer is the translator path).
            if is_quarter_turn(rotation):
                # Turned text: never touch `h` (the box's short side). The font
                # already shrank to the floor during fitting; growing here would
                # fatten the label across the drawing.
                pass
            elif (_avail_h_down_pt is not None
                    and needed_h_pt > _avail_h_down_pt > 0):
                _fs2, _lines2 = fit_multiline(
                    text, box_w_pt, _avail_h_down_pt,
                    max_size_pt=base_size_pt, min_size_pt=3.5,
                    bold=bold_render, edge_pad_pt=_TEXTBOX_EDGE_PAD_PT,
                )
                if _lines2 is not None:
                    # Font-shrink succeeded: text now fits the room, no overlap.
                    style["size"] = _fs2
                    processed_text = "\n".join(_lines2)
                    body_auto_fit = False
                    needed_h_pt = min(needed_h_pt, _avail_h_down_pt)
                    h = max(1, int(round(needed_h_pt * EMU_PER_PT)))
                else:
                    # Even 3.5pt can't fit the room — keep full text visible
                    # (grow down); box may overlap but no content is dropped.
                    h = max(h, int(round(needed_h_pt * EMU_PER_PT)))
            else:
                h = max(h, int(round(needed_h_pt * EMU_PER_PT)))
        else:
            processed_text = text
    else:
        style["size"] = fit_size_pt
        processed_text = "\n".join(lines) if lines else text

    # Keep the (possibly grown) box on the page. A text box anchored at `y` grows
    # DOWNWARD, so an entry near the page bottom — most commonly a Page-Footer the
    # OCR gave a too-short bbox, but also any bottom-anchored text — would overflow
    # the page edge and be visually clipped. When the box bottom would pass the
    # page bottom, lift `y` up by the overflow (never above the page top) so every
    # measured line stays inside the printable area.
    page_h_emu = int(round(ctx.page_h_pt * EMU_PER_PT))
    overflow = (y + h) - page_h_emu
    if overflow > 0:
        y = max(0, y - overflow)

    runs_xml = build_run_xml(processed_text, style)
    para_xml = build_paragraph_xml(runs_xml, alignment=alignment, line_pt=None)

    ctx.xml_chunks.append(
        build_anchored_textbox_xml(
            x, y, w, h, para_xml, ctx._next_id(),
            body_auto_fit=body_auto_fit,
            rotation=rotation,
        )
    )