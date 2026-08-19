"""Top-level entry: turn the OCR layout JSON into a DOCX whose page count,
page size, and per-entry positions match the source document. Delegates
each entry kind to its dedicated renderer module.
"""
from __future__ import annotations

from io import BytesIO
from typing import Dict, List

from docx import Document
from docx.shared import Pt, RGBColor

from .geometry import (
    add_section_for_page, normalize_page, normalize_rotation,
)
from .picture import render_standalone_picture
from .table import (
    render_table, parse_html_table_rows, parse_markdown_table,
    parse_table_grid, compute_col_weights,
)
from .formula import render_formula
from .text_entry import render_text_entry
from .shape_context import ShapeContext
from .reflow import layout_page


# How far below the page-header band (pt, scaled by zoom) a table may start and
# still count as "at the top of the page" for continuation-chain detection.
_TABLE_AT_TOP_PT = 60.0


# ─── Style normalisation ────────────────────────────────────────────────
#
# After translation the styles in the layout JSON are still the source-PDF's
# styles — CJK CIDFont names that downstream Word will substitute with
# something arbitrary, and per-entry sizes that drift across pages (a
# Page-header rendered at 7pt on one page and 9.8pt on another). The
# translator service can't fix this without re-fitting every box, so we do
# it here right before rendering.
#
# Two goals:
#   1. ALL entries use one universally-available font ("Calibri") so the
#      document renders consistently regardless of where it's opened. The
#      CIDFont substitution behaviour was the main reason text appeared
#      clipped — substituted fonts almost always have different metrics
#      than the original.
#   2. Each category gets ONE base size, so siblings of the same role
#      (every Section-header, every Caption, every Page-footer) read
#      identically. The per-entry text_fit step still shrinks below this
#      base when necessary to avoid clipping the actual rendered text, but
#      it starts from a known sensible value, not a random source size.
#
# The base sizes mirror Word's defaults for an A4 report (Title 14, body
# 10, caption/footer 9). Page-header is intentionally a touch smaller than
# body so a running-head doesn't fight the body for attention.

UNIFIED_FONT = "Calibri"

CATEGORY_BASE_SIZE_PT: Dict[str, float] = {
    "Title": 14.0,
    "Section-header": 12.0,
    "Page-header": 9.0,
    "Page-footer": 9.0,
    "Caption": 10.0,
    "Footnote": 8.5,
    "List-item": 10.5,
    "Text": 10.5,
    # Table cells start at the same readable size as body text. This is only the
    # STARTING size — `table.render_table` still fits each cell — but it is now
    # also honoured as a floor there (`TABLE_MIN_FONT_PT`), so a dense table
    # grows rather than shrinking into illegibility.
    "Table": 10.0,
    "Formula": 11.0,
    "Picture": 11.0,  # unused — pictures have no text
}

# Categories that should ALWAYS render in bold weight, regardless of what

ALWAYS_BOLD_CATEGORIES = {
    "Title", "Section-header", "Page-header", "Page-footer", "Caption",
}

DEFAULT_BASE_SIZE_PT = 10.5


def _normalize_entry_styles(layout_results) -> None:
    """Rewrite every entry's `style` so font, size, and weight are
    predictable.

    Mutates `layout_results` in place. Each entry's style.font becomes
    `UNIFIED_FONT`; each entry's style.size becomes the category's base
    size; headings/headers/footers/captions get style.bold=True. Italic
    and color flags pass through untouched so translated emphasis (and
    deliberately non-black runs) survive.
    """
    for raw_page in layout_results:
        if isinstance(raw_page, dict):
            entries = (
                raw_page.get("entries")
                or raw_page.get("layout_result")
                or []
            )
        elif isinstance(raw_page, list):
            entries = raw_page
        else:
            continue
        for entry in entries:
            cat = entry.get("category") or ""
            style = dict(entry.get("style") or {})
            style["font"] = UNIFIED_FONT
            style["size"] = CATEGORY_BASE_SIZE_PT.get(cat, DEFAULT_BASE_SIZE_PT)
            if cat in ALWAYS_BOLD_CATEGORIES:
                style["bold"] = True
            entry["style"] = style


def _parse_table_entry_rows(entry: Dict):
    """Parse the table HTML on `entry` into rows. None when not a table or
    when parsing yields nothing."""
    text = (entry.get("text") or "").strip()
    if not text:
        return None
    if "<table" in text.lower():
        return parse_html_table_rows(text)
    md_rows = parse_markdown_table(text)
    if md_rows:
        return [([(c, 1, 1) for c in r], False) for r in md_rows]
    return None


def _link_table_continuations(layout_results) -> None:
    """Detect tables that continue from a previous page and propagate the
    head table's column-weight vector onto them so column widths stay
    consistent across the page break.

    Heuristic: a Table entry is a continuation of the most recent prior
    Table entry when it has no `<thead>`, the parsed grid has the same
    `max_cols`, and it sits near the top of its page (just below the page
    header). The chain detector also re-computes the shared weights using
    cells from EVERY page in the chain — so a column that only carries
    long text on a continuation page still gets enough width on the head
    page.
    """
    chain_parent_entry = None
    chain_max_cols = 0
    chain_anchors: Dict = {}
    chain_row_offset = 0

    def _finalize_chain():
        if chain_parent_entry is None or not chain_anchors:
            return
        weights = compute_col_weights(chain_anchors, chain_max_cols)
        for ent in chain_parent_entry.get("_table_chain", [chain_parent_entry]):
            ent["_shared_col_weights"] = list(weights)

    for raw_page in layout_results:
        page_zoom = 1.0
        if isinstance(raw_page, dict):
            entries = (
                raw_page.get("entries")
                or raw_page.get("layout_result")
                or []
            )
            # Bboxes are in page PIXELS; the at-top threshold below is in
            # points, so it has to be scaled by this page's zoom.
            try:
                page_zoom = float(raw_page.get("zoom") or 1.0) or 1.0
            except (TypeError, ValueError):
                page_zoom = 1.0
        elif isinstance(raw_page, list):
            entries = raw_page
        else:
            entries = []
        # The page-header bottom — used to decide whether a table is at the
        # top of the page (i.e. immediately after the header band). When
        # there's no detected page-header band, treat anything in the top
        # ~15% of the page as 'at top'.
        header_bottoms = [
            (e.get("bbox") or [0, 0, 0, 0])[3]
            for e in entries
            if e.get("category") == "Page-header"
        ]
        page_header_y2 = max(header_bottoms) if header_bottoms else 0

        # Find the first table on the page (if any) and remember whether any
        # non-table content precedes it — that's the only place a multi-page
        # continuation can sit. A second table on the same page is always a
        # new logical table.
        page_tables = [e for e in entries if e.get("category") == "Table"]
        first_table_id = id(page_tables[0]) if page_tables else None

        for entry in entries:
            if entry.get("category") != "Table":
                continue
            rows = _parse_table_entry_rows(entry)
            if not rows:
                continue
            text = (entry.get("text") or "").strip()
            has_thead = "<thead" in text.lower()
            max_cols, n_rows, cell_anchors, _occ, _img = parse_table_grid(rows)

            bbox = entry.get("bbox") or [0, 0, 0, 0]
            tbl_top = bbox[1] if len(bbox) >= 2 else 0
            # `tbl_top`/`page_header_y2` are PIXELS, so the point threshold has
            # to be scaled by zoom. Comparing points against pixels made this
            # test fail on every page of a 4x-zoom scan (a table 42pt from the
            # top measures 175px), so no chain ever linked and continuation
            # pages each computed their own column widths.
            at_top_of_page = (
                (tbl_top - page_header_y2) <= _TABLE_AT_TOP_PT * page_zoom
            )

            is_continuation = (
                chain_parent_entry is not None
                and not has_thead
                and max_cols == chain_max_cols
                and at_top_of_page
                and id(entry) == first_table_id
            )
            if is_continuation:
                for (r, c), v in cell_anchors.items():
                    chain_anchors[(chain_row_offset + r, c)] = v
                chain_row_offset += n_rows
                chain = chain_parent_entry.setdefault(
                    "_table_chain", [chain_parent_entry]
                )
                chain.append(entry)
            else:
                _finalize_chain()
                chain_parent_entry = entry
                chain_anchors = dict(cell_anchors)
                chain_max_cols = max_cols
                chain_row_offset = n_rows
                chain_parent_entry["_table_chain"] = [chain_parent_entry]

    _finalize_chain()


# Fraction of a page's rotated area that makes the WHOLE PAGE rotated (a
# landscape sheet scanned into a portrait frame) rather than a few turned
# labels sitting on an upright page.
_PAGE_ROT_AREA_RATIO = 0.5

# Fraction of an untagged entry's own height that must overlap a rotated
# sibling's y-range before that entry is folded into the rotated group (see
# "UNTAGGED NEIGHBOURS" below).
_ROT_NEIGHBOR_OVERLAP_RATIO = 0.6


def _derotate_rotated_pages(layout_results) -> None:
    """Turn a page whose CONTENT is rotated into an upright LANDSCAPE page.

    WHY THIS INSTEAD OF ROTATING THE SHAPES. A quarter-turned table cannot be
    expressed by rotating its container: `bodyPr vert=` only reflows text inside
    the cells and leaves the grid upright, and while `a:xfrm rot=` is the right
    OOXML in principle, renderers ignore it on a shape that holds a `<w:tbl>`.
    Measured on the rendered output of this very document: 255 table cells came
    back at text direction (1,0) — dead upright — while only the 5 standalone
    labels (which use `vert=`) actually turned. So the shape-rotation path
    cannot deliver a rotated table in practice.

    What DOES work is to stop fighting the renderer: swap the section's page
    width/height so the page itself is landscape, map every bbox through the
    inverse rotation, and render all content UPRIGHT. That is plain,
    universally-supported OOXML — no renderer can quietly drop it — and it has
    the side benefit that the result is normal editable/translatable text
    rather than a spun shape.

    A page is de-rotated only when rotated entries dominate its content area
    (`_PAGE_ROT_AREA_RATIO`); a couple of turned side-labels on an otherwise
    upright page keep their per-entry `vert=` rendering instead.

    UNTAGGED NEIGHBOURS. OCR's per-entry rotation vote can miss an entry that
    is visually part of the same turned band as its rotated siblings — e.g. a
    wide table sandwiched between two rotated tables, all sharing one y-range,
    where only the outer two got a `table-ocr-majority` angle. Such an entry
    is folded into the rotated side of the ratio (and inherits `page_rot`
    below) when it y-overlaps a rotated entry by at least
    `_ROT_NEIGHBOR_OVERLAP_RATIO` of its own height — a plain quarter-turn
    table's rotation is a property of the physical page region, not of
    whichever entry OCR happened to tag.
    """
    for raw_page in layout_results:
        if not isinstance(raw_page, dict):
            continue
        entries = raw_page.get("entries") or raw_page.get("layout_result") or []
        if not entries:
            continue

        boxes = []  # (entry, x1, y1, x2, y2, area, rot)
        for e in entries:
            bb = e.get("bbox")
            if not bb or len(bb) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in bb)
            except (TypeError, ValueError):
                continue
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            rot = normalize_rotation(e.get("rotation"))
            boxes.append((e, x1, y1, x2, y2, area, rot))

        total_area = sum(b[5] for b in boxes)
        rotated = [b for b in boxes if b[6] in (90.0, 270.0)]
        if total_area <= 0 or not rotated:
            continue

        rotated_ids = {id(b[0]) for b in rotated}
        rot_area = sum(b[5] for b in rotated)
        angles = [b[6] for b in rotated]

        # Fold in untagged entries that share a rotated sibling's y-band —
        # same physical turned region, just missing an OCR angle.
        for e, x1, y1, x2, y2, area, rot in boxes:
            if id(e) in rotated_ids or rot in (90.0, 270.0) or area <= 0:
                continue
            for _re, rx1, ry1, rx2, ry2, _rarea, rrot in rotated:
                overlap = max(0.0, min(y2, ry2) - max(y1, ry1))
                if overlap / max(1.0, y2 - y1) >= _ROT_NEIGHBOR_OVERLAP_RATIO:
                    rot_area += area
                    angles.append(rrot)
                    rotated_ids.add(id(e))
                    break

        if (rot_area / total_area) < _PAGE_ROT_AREA_RATIO:
            continue

        # One angle for the page: whichever turned orientation covers more area.
        page_rot = max(set(angles), key=angles.count)

        try:
            w_pt = float(raw_page.get("page_width_pt"))
            h_pt = float(raw_page.get("page_height_pt"))
            zoom = float(raw_page.get("zoom") or 1.0) or 1.0
        except (TypeError, ValueError):
            continue

        pw = w_pt * zoom          # page extent in PIXELS (bbox space)
        ph = h_pt * zoom

        def _map(bbox):
            x1, y1, x2, y2 = (float(v) for v in bbox)
            if page_rot == 90.0:
                # Content reads bottom-to-top (glyphs turned 90° CW from
                # upright): straightening means rotating the page frame 90°
                # CW, i.e. (x, y) -> (ph - y, x). Verified empirically against
                # PIL's CW raster rotation and against this document's known
                # top-to-bottom strip order.
                return [ph - y2, x1, ph - y1, x2]
            # 270°: content reads top-to-bottom (glyphs turned 90° CCW from
            # upright): straightening rotates the page frame 90° CCW, i.e.
            # (x, y) -> (y, pw - x).
            return [y1, pw - x2, y2, pw - x1]

        for e in entries:
            bb = e.get("bbox")
            if bb and len(bb) == 4:
                e["bbox"] = _map(bb)
            # The page carries the rotation now, so every entry renders upright.
            # Drop it rather than leaving a stale angle for the renderers.
            if normalize_rotation(e.get("rotation")) in (90.0, 270.0):
                e.pop("rotation", None)
                e["derotated_with_page"] = True
            # Any per-cell turn was relative to the old frame; it no longer
            # applies once the page itself is straightened.
            e.pop("cell_rotations", None)
            # A table/text entry is rebuilt as fresh OOXML (real glyphs/grid),
            # so straightening its BBOX is enough — the renderer always draws
            # it upright regardless of source pixel orientation. A raster
            # Image/Figure/Picture is a PHOTO of the rotated page region: its
            # pixels are still turned exactly as the source was, so it needs
            # the same corrective spin applied to its content, not just its
            # position. Recorded here (once, where `page_rot` is known) and
            # consumed by the picture renderers, which are the only ones that
            # touch raw pixels.
            if e.get("category") in ("Image", "Figure", "Picture"):
                e["_pixel_rotation_deg"] = page_rot

        # The section becomes landscape (or back to portrait for a turned
        # landscape original) — a quarter turn always swaps the axes.
        raw_page["page_width_pt"] = h_pt
        raw_page["page_height_pt"] = w_pt
        raw_page["page_rotation_applied"] = page_rot


def json_to_docx(layout_results, output_path="output.docx"):
    """Build a DOCX whose page count, page size, and per-entry positions
    match the source document.

    `layout_results` is a list of page envelopes (new shape) or a list of
    entry-lists (legacy). When called from the live pipeline it's the same
    `pages` list that `process_pictures` returned, which carries
    `image_obj` on Picture entries — those get embedded.
    """
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.font.color.rgb = RGBColor(0, 0, 0)
    pf = style.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0

    shape_counter = 1000  # docPr ids must be unique and >0

    # Straighten any page whose content is rotated BEFORE anything measures it:
    # the section becomes landscape and every bbox is mapped into that frame, so
    # style normalisation, reflow and the renderers below all see ordinary
    # upright geometry.
    _derotate_rotated_pages(layout_results)

    _normalize_entry_styles(layout_results)


    _link_table_continuations(layout_results)

    physical_pages: List[Dict] = []

    pre_reflow_table_bboxes: Dict[int, List] = {}
    for raw_page in layout_results:
        page = normalize_page(raw_page)
        raw_entries: List[Dict] = []
        if isinstance(raw_page, dict):
            raw_entries = (
                raw_page.get("layout_result")
                or raw_page.get("entries")
                or []
            )
        elif isinstance(raw_page, list):
            raw_entries = raw_page
        entries = page["entries"] or raw_entries
        # Pull diagram labels OUT of the reflow stream: they must stay locked to
        # their parent diagram (rendered by render_diagram_label_overlays), not
        # drift with the text cascade. They still live in the JSON as normal Text
        # entries (so OCR output + translation see them); reconstruction just
        # positions them relative to the diagram rather than reflowing them.
        diagram_labels = [e for e in entries if e.get("source") == "diagram-label"]
        entries = [e for e in entries if e.get("source") != "diagram-label"]
        for e in entries:
            if e.get("category") == "Table" and e.get("bbox"):
                pre_reflow_table_bboxes[id(e)] = list(e["bbox"])

            if e.get("category") in ("Image", "Figure", "Picture") and e.get("bbox"):
                e["_orig_bbox"] = list(e["bbox"])
        laid_out = layout_page(
            entries,
            float(page["page_width_pt"]),
            float(page["page_height_pt"]),
            float(page["zoom"]),
        )
        # layout_page returns exactly one physical page per source page; stash the
        # extracted diagram labels on it so the render loop can overlay them.
        for pp in laid_out:
            pp["_diagram_labels"] = diagram_labels
        physical_pages.extend(laid_out)

    for idx, page in enumerate(physical_pages):
        entries = page["entries"]
        page_w_pt = float(page["page_width_pt"])
        page_h_pt = float(page["page_height_pt"])
        page_zoom = float(page["zoom"])

        add_section_for_page(
            doc,
            page_w_pt,
            page_h_pt,
            first=(idx == 0),
        )

        ctx = ShapeContext(
            doc,
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            zoom=page_zoom,
            shape_id_start=shape_counter,
        )
        # Expose the page's (reflowed) entries so per-entry renderers can be
        # obstacle-aware — e.g. a header/caption width-grow must stop before a
        # right-hand neighbour so their rendered TEXT never overlaps.
        ctx.page_entries = entries

        tables = [e for e in entries if e.get("category") == "Table"]

        def _is_picture(entry) -> bool:
            # ocr_service emits pictures as Image/Figure (recovered table-cell
            # photos and diagram crops are also "Image"); "Picture" is accepted
            # for back-compat. Gating on "Picture" alone silently dropped every
            # picture from the translated DOCX.
            cat = entry.get("category")
            if cat in ("Image", "Figure", "Picture"):
                return True
            return cat == "Diagram" and entry.get("image_obj") is not None

        def _bbox_inside(inner, outer) -> bool:
            if not inner or not outer or len(inner) != 4 or len(outer) != 4:
                return False
            return (inner[0] >= outer[0] and inner[2] <= outer[2]
                    and inner[1] >= outer[1] and inner[3] <= outer[3])

        # A table's outer bbox is its full rectangular extent, which usually
        # includes blank margin beyond the text columns (especially after a
        # de-rotation remap). A picture whose bbox falls in that margin is NOT
        # "inside" the table in any meaningful sense unless the table's own
        # HTML actually places an <img> in one of its cells — cache that check
        # per table (id -> bool) rather than per picture.
        _table_has_img_cache: Dict[int, bool] = {}

        def _table_has_image_cells(t: Dict) -> bool:
            key = id(t)
            if key in _table_has_img_cache:
                return _table_has_img_cache[key]
            has_img = False
            for cells, _is_header in parse_html_table_rows(t.get("text") or ""):
                if any(c[3] for c in cells if len(c) >= 4):
                    has_img = True
                    break
            _table_has_img_cache[key] = has_img
            return has_img

        def _picture_in_any_table(pic_entry) -> bool:
            # Compare in the PRE-REFLOW frame: pictures now cascade-move, so
            # their live bbox is post-move/scale; the snapshot is the frame
            # the table snapshots were taken in.
            pb = pic_entry.get("_orig_bbox") or pic_entry.get("bbox")
            for t in tables:
                if not _table_has_image_cells(t):
                    continue
                orig_tb = pre_reflow_table_bboxes.get(id(t)) or t.get("bbox")
                if _bbox_inside(pb, orig_tb):
                    return True
            return False

        standalone_pics = [
            e for e in entries
            if _is_picture(e) and not _picture_in_any_table(e)
        ]
        # Mark these as ownerless BEFORE the containment loop below: even a
        # standalone picture can sit geometrically inside a table's outer
        # bbox (margin space, a de-rotated frame, ...) without being claimed
        # by any of its cells. `_table_owner_id = None` tells `_bboxes` in
        # `table_geometry.py` this picture is a real obstacle, not content
        # the table contains — otherwise pure bbox-containment would still
        # exclude it and the table could grow straight through it.
        for e in standalone_pics:
            e["_table_owner_id"] = None

        # Bucket contained pictures by their owning table so each table
        # renders its own photos inside the matching cell.
        pics_per_table: Dict[int, List[Dict]] = {}
        contained_no_image = []
        for e in entries:
            if not _is_picture(e):
                continue
            pb = e.get("_orig_bbox") or e.get("bbox")
            for t in tables:
                if not _table_has_image_cells(t):
                    continue
                orig_tb = pre_reflow_table_bboxes.get(id(t)) or t.get("bbox")
                if _bbox_inside(pb, orig_tb):
                    e["_table_owner_id"] = id(t)
                    if e.get("image_obj") is not None:
                        pics_per_table.setdefault(id(t), []).append(e)
                    else:
                        # Fall back to floating anchor on top of the table —
                        # better than disappearing entirely.
                        contained_no_image.append(e)
                    break

        for entry in standalone_pics:
            render_standalone_picture(ctx, entry)
        for entry in entries:
            cat = entry.get("category")
            if _is_picture(entry):
                continue
            if cat == "Table":
                render_table(
                    ctx,
                    entry,
                    pictures_for_table=pics_per_table.get(id(entry)),
                    orig_bbox=pre_reflow_table_bboxes.get(id(entry)),
                )
            elif cat in ("Equation-Block", "Formula"):
                # Chandra emits block math as "Equation-Block"; "Formula" is the
                # legacy label. Both must route to the formula renderer, else the
                # LaTeX (e.g. IMC = \frac{...}) renders as raw text.
                render_formula(ctx, entry)
            else:
                render_text_entry(ctx, entry)
        # Floating fallback for contained pictures with no image_obj.
        for entry in contained_no_image:
            render_standalone_picture(ctx, entry)

        # Overlay the (translated) diagram labels on top of their diagram raster.
        # Each label's bbox is in the diagram's ORIGINAL frame, so match it to the
        # diagram whose pre-reflow bbox contains it, then let the overlay apply the
        # diagram's cascade-move delta.
        diagram_labels = page.get("_diagram_labels") or []
        if diagram_labels:
            from .picture import render_diagram_label_overlays
            diagrams = [
                e for e in entries
                if e.get("source") == "diagram-recovered" and e.get("bbox")
            ]
            for diag in diagrams:
                ob = diag.get("_orig_bbox") or diag.get("bbox")
                mine = [
                    lab for lab in diagram_labels
                    if lab.get("bbox") and _bbox_inside(lab["bbox"], ob)
                ]
                render_diagram_label_overlays(ctx, diag, mine)
            # Any label not inside a matched diagram (rare) — anchor to the first
            # diagram if present so it is never silently dropped.
            if diagrams:
                claimed = {
                    id(lab) for diag in diagrams
                    for lab in diagram_labels
                    if lab.get("bbox") and _bbox_inside(
                        lab["bbox"], diag.get("_orig_bbox") or diag.get("bbox"))
                }
                orphans = [lab for lab in diagram_labels if id(lab) not in claimed]
                if orphans:
                    render_diagram_label_overlays(ctx, diagrams[0], orphans)

        ctx.flush()
        shape_counter = ctx.next_id + 1

    doc.save(output_path)
    if isinstance(output_path, BytesIO):
        return output_path
    print(f"DOCX file saved to: {output_path}")
    return output_path