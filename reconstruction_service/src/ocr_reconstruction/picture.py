"""Picture handling: dedup, cropping from source pages, and standalone
floating-picture rendering. In-cell picture placement lives in `table.py`
since it has to interact with the cell grid."""
from __future__ import annotations

import uuid
from io import BytesIO
from typing import Dict


# Chandra layout labels that carry a raster crop.
PICTURE_LABELS = {"Image", "Figure", "Picture"}


def _is_picture(entry) -> bool:
    return entry.get("category") in PICTURE_LABELS


def deduplicate_pictures(layout_result):
    """Remove duplicate Picture entries with identical bboxes."""
    seen_bboxes = set()
    deduped = []
    for entry in layout_result:
        if _is_picture(entry):
            bbox_tuple = tuple(entry.get("bbox") or [])
            if bbox_tuple not in seen_bboxes:
                seen_bboxes.add(bbox_tuple)
                deduped.append(entry)
        else:
            deduped.append(entry)
    return deduped


def process_pictures(full_result):
    """Crop Picture regions from each page's original image and attach the
    crop + a stable id/filename to the entry. MinIO upload happens upstream.
    """
    updated_results = []

    for page_idx, page_result in enumerate(full_result, start=1):
        original_img = page_result["original_image"]
        layout_result = page_result["layout_result"]
        markdown_content = page_result.get("markdown_content", "")

        layout_result = deduplicate_pictures(layout_result)

        new_layout_result = []
        img_replacements = []
        picture_count = 0

        for entry in layout_result:
            if _is_picture(entry):
                picture_count += 1
                bbox = entry["bbox"]
                cropped = original_img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

                short = uuid.uuid4().hex[:6]
                img_id = f"page{page_idx}_pic{picture_count}_{short}"
                image_filename = f"{img_id}.png"

                # Preserve the original Chandra label (Image/Figure/Diagram) so
                # the downstream dispatch and picture-asset upload still match.
                new_entry = {
                    **entry,
                    "image_obj": cropped,
                    "id": img_id,
                    "image_filename": image_filename,
                }
                new_layout_result.append(new_entry)

                img_replacements.append(
                    f"![Picture {img_id}](images/{image_filename})"
                )
            else:
                new_layout_result.append(entry)

        for replacement in img_replacements:
            markdown_content = markdown_content.replace(
                "![Image](Image detected)", replacement, 1
            )

        updated_page_result = {
            **page_result,
            "layout_result": new_layout_result,
            "markdown_content": markdown_content,
        }
        updated_results.append(updated_page_result)

    return updated_results


def render_standalone_picture(ctx, entry: Dict) -> None:
    """Emit a Picture entry as a floating page-anchored shape. Used for any
    picture that ISN'T contained in a table (table-contained ones go inline
    inside the matching cell in table.py)."""
    bbox = entry.get("bbox")
    if not bbox or len(bbox) != 4:
        return
    img_obj = entry.get("image_obj")
    if img_obj is None:
        return
    from .geometry import bbox_px_to_emu
    from .ooxml import (
        build_anchored_picture_xml, add_image_relationship,
    )
    x, y, w, h = bbox_px_to_emu(
        bbox, ctx.zoom, ctx.page_w_pt, ctx.page_h_pt,
    )
    buf = BytesIO()
    img_obj.save(buf, format="PNG")
    buf.seek(0)
    rel_id = add_image_relationship(ctx.doc, buf, "png")
    ctx.xml_chunks.append(
        build_anchored_picture_xml(
            x, y, w, h, rel_id, ctx._next_id(),
            pic_name=entry.get("id") or "Picture",
        )
    )


# The white overlay box stays at the DETECTED bbox (never grows). Longer text is
# made to fit by shrinking the font down to this floor. Matches the translation
# path so the OCR (untranslated) docx reads consistently.
_DIAG_LABEL_MIN_FONT_PT = 4.0


def render_diagram_label_overlays(ctx, diagram_entry: Dict,
                                  label_entries: list) -> None:
    """Overlay diagram labels as white-filled text boxes on top of the diagram
    raster. `label_entries` are top-level `source="diagram-label"` Text entries.
    The OCR path does not reflow, so labels render at their detected bbox."""
    if not label_entries:
        return

    from .geometry import bbox_px_to_emu
    from .ooxml import (
        build_anchored_textbox_xml, build_paragraph_xml, build_run_xml,
    )
    from .text_fit import fit_multiline

    zoom = ctx.zoom
    live = diagram_entry.get("bbox")
    orig = diagram_entry.get("bbox") or live  # OCR path: no move, delta is 0
    dx = float(live[0]) - float(orig[0])
    dy = float(live[1]) - float(orig[1])

    for region in label_entries:
        text = (region.get("text") or "").strip()
        rb = region.get("bbox")
        if not text or not rb or len(rb) != 4:
            continue
        x1 = float(rb[0]) + dx
        y1 = float(rb[1]) + dy
        x2 = float(rb[2]) + dx
        y2 = float(rb[3]) + dy
        if x2 <= x1 or y2 <= y1:
            continue

        # Keep the box at the DETECTED bbox — never grow it; shrink the font to fit.
        box_w_pt = max(1.0, (x2 - x1) / zoom)
        box_h_pt = max(1.0, (y2 - y1) / zoom)
        style = dict(region.get("style") or {})
        base_size_pt = float(style.get("size") or 11.0)
        bold_render = bool(style.get("bold"))

        fit_size_pt, lines = fit_multiline(
            text, box_w_pt, box_h_pt,
            max_size_pt=base_size_pt, min_size_pt=_DIAG_LABEL_MIN_FONT_PT,
            bold=bold_render,
        )
        if lines is None:
            style["size"] = _DIAG_LABEL_MIN_FONT_PT
            processed_text = text
        else:
            style["size"] = fit_size_pt
            processed_text = "\n".join(lines)
        body_auto_fit = False

        x, y, bw, bh = bbox_px_to_emu(
            [x1, y1, x2, y2], zoom, ctx.page_w_pt, ctx.page_h_pt,
        )
        runs_xml = build_run_xml(processed_text, style)
        para_xml = build_paragraph_xml(runs_xml, alignment="center", line_pt=None)
        ctx.xml_chunks.append(
            build_anchored_textbox_xml(
                x, y, bw, bh, para_xml, ctx._next_id(),
                body_auto_fit=body_auto_fit, fill_rgb="FFFFFF",
            )
        )
