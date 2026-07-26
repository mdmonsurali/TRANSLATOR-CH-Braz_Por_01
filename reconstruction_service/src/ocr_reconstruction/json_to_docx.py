"""Top-level entry: turn the OCR layout JSON into a DOCX whose page count,
page size, and per-entry positions match the source document. Delegates
each entry kind to its dedicated renderer module."""
from __future__ import annotations

from io import BytesIO
from typing import Dict, List

from docx import Document
from docx.shared import Pt, RGBColor

from .geometry import (
    add_section_for_page, normalize_page,
)
from .picture import render_standalone_picture
from .table import (
    render_table, parse_html_table_rows, parse_markdown_table,
    parse_table_grid, compute_col_weights,
) #deduplicate_entries,
from .formula import render_formula
from .text_entry import render_text_entry
from .shape_context import ShapeContext


# Label categories that title the table they sit above. When OCR places one so
# it vertically overlaps a Table (its title dropped inside the table's top), we
# lift it back above the table. (Ported from the translator reflow path.)
_OVERLAP_RESOLVE_CATEGORIES = {"Caption", "Title", "Section-header", "Section-Header"}
# Gap (pt, scaled by zoom) kept between a lifted caption's bottom and table top.
_TABLE_GAP_PT = 4.0


def _lift_captions_above_overlapping_tables(entries: List[Dict], zoom: float) -> None:
    """Move any caption/title/section-header that VERTICALLY OVERLAPS a Table to
    sit just ABOVE that table's top edge (captions label the table below them).
    Pure bbox translation — no cascade, no scaling — and a strict no-op when
    nothing overlaps. Keys only on category + geometry.

    A Caption/Title/Section-header is a BLOCK-LEVEL label, never a table cell
    (cell text lives inside the table's own HTML), so a separate label entry that
    overlaps a table is always a mis-placed title, not a legitimate in-table
    label. Only a label already entirely above the table is left alone. When a
    label overlaps several tables it is lifted above the topmost overlapping one.
    """
    tables = [
        e for e in entries
        if e.get("category") == "Table"
        and e.get("bbox") and len(e["bbox"]) == 4
    ]
    if not tables:
        return
    gap_px = _TABLE_GAP_PT * zoom
    for e in entries:
        if e.get("category") not in _OVERLAP_RESOLVE_CATEGORIES:
            continue
        eb = e.get("bbox")
        if not eb or len(eb) != 4:
            continue
        ex1, ey1, ex2, ey2 = [float(v) for v in eb]
        target_top = None
        for t in tables:
            tx1, ty1, tx2, ty2 = [float(v) for v in t["bbox"]]
            # Already entirely above the table → nothing to fix.
            if ey2 <= ty1:
                continue
            # Must share a horizontal band (sits over the table) AND overlap
            # vertically (dropped into it) to count as an overlap to resolve.
            if not (ex1 < tx2 and ex2 > tx1):
                continue
            if not (ey1 < ty2 and ey2 > ty1):
                continue
            if target_top is None or ty1 < target_top:
                target_top = ty1  # lift above this table's top
        if target_top is None:
            continue
        ch = ey2 - ey1
        new_bottom = target_top - gap_px
        new_top = new_bottom - ch
        if new_top < 0.0:            # clamp to page top; still clears the table
            new_top = 0.0
            new_bottom = ch
        e["bbox"] = [ex1, new_top, ex2, new_bottom]


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
        if isinstance(raw_page, dict):
            entries = (
                raw_page.get("entries")
                or raw_page.get("layout_result")
                or []
            )
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
            if e.get("category") in ("Page-Header", "Page-header")
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
            at_top_of_page = (tbl_top - page_header_y2) <= 60

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


def json_to_docx(layout_results, output_path="output.docx"):
    """Build a DOCX whose page count, page size, and per-entry positions
    match the source document.

    `layout_results` is a list of page envelopes (new shape) or a list of
    entry-lists (legacy). When called from the live pipeline it's the same
    `pages` list that `process_pictures` returned, which carries
    `image_obj` on Picture entries — those get embedded.
    """
    doc = Document()

    # Keep a sensible Normal style so any inline run we don't override
    # still has a reasonable default. Zero the paragraph spacing too:
    # python-docx's default Normal carries 1.15x line spacing and 10pt
    # space-after, which would otherwise inflate any paragraph we don't
    # explicitly override and reintroduce the vertical-clip bug.
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.font.color.rgb = RGBColor(0, 0, 0)
    pf = style.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0

    shape_counter = 1000  # docPr ids must be unique and >0

    # Annotate Table entries that belong to a multi-page chain with a shared
    # column-weight vector so widths stay consistent across the page break.
    _link_table_continuations(layout_results)

    for idx, raw_page in enumerate(layout_results):
        page = normalize_page(raw_page)
        section = add_section_for_page(
            doc,
            float(page["page_width_pt"]),
            float(page["page_height_pt"]),
            first=(idx == 0),
        )

        # The raw page (when from the live pipeline) carries image_obj on
        # Picture entries via the same dict we normalized. Merge raw entries
        # in case we got the legacy shape.
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


        page_w_pt = float(page["page_width_pt"])
        page_zoom = float(page["zoom"])

        # De-overlap pass: OCR sometimes drops a table's caption/title so its
        # bbox falls inside the table's top edge. Lift it back above the table
        # before any entry is positioned (strict no-op when nothing overlaps).
        _lift_captions_above_overlapping_tables(entries, page_zoom)

        ctx = ShapeContext(
            doc,
            page_w_pt=page_w_pt,
            page_h_pt=float(page["page_height_pt"]),
            zoom=page_zoom,
            shape_id_start=shape_counter,
        )

        # Chandra layout labels. Pictures are Image/Figure (and a Diagram that
        # actually carries a raster crop); tables are Table; block math is
        # Equation-Block. "Formula"/"Picture" are accepted for back-compat.
        def _is_picture(entry) -> bool:
            cat = entry.get("category")
            if cat in ("Image", "Figure", "Picture"):
                return True
            # A Diagram is only a raster when a crop was attached; otherwise
            # it's mermaid text and renders through the text path.
            return cat == "Diagram" and entry.get("image_obj") is not None

        def _is_table(entry) -> bool:
            return entry.get("category") == "Table"

        def _is_formula(entry) -> bool:
            return entry.get("category") in ("Equation-Block", "Formula")

        tables = [e for e in entries if _is_table(e)]

        def _bbox_inside(inner, outer) -> bool:
            if not inner or not outer or len(inner) != 4 or len(outer) != 4:
                return False
            return (inner[0] >= outer[0] and inner[2] <= outer[2]
                    and inner[1] >= outer[1] and inner[3] <= outer[3])

        def _picture_in_any_table(pic_entry) -> bool:
            pb = pic_entry.get("bbox")
            return any(_bbox_inside(pb, t.get("bbox")) for t in tables)

        standalone_pics = [
            e for e in entries
            if _is_picture(e) and not _picture_in_any_table(e)
        ]
        # Bucket contained pictures by their owning table so each table
        # renders its own photos inside the matching cell.
        pics_per_table: Dict[int, List[Dict]] = {}
        contained_no_image = []
        for e in entries:
            if not _is_picture(e):
                continue
            pb = e.get("bbox")
            for t in tables:
                if _bbox_inside(pb, t.get("bbox")):
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
            if _is_picture(entry):
                continue
            if entry.get("source") == "diagram-label":
                # Rendered as a white overlay on the diagram below, not as normal
                # positioned text (which would be transparent and let the original
                # glyphs show through).
                continue
            if _is_table(entry):
                render_table(
                    ctx,
                    entry,
                    pictures_for_table=pics_per_table.get(id(entry)),
                )
            elif _is_formula(entry):
                render_formula(ctx, entry)
            else:
                # Every other Chandra label (Text, Section-Header, Caption,
                # Footnote, Page-Header/Footer, List-Group, Code-Block, Form,
                # Table-Of-Contents, Bibliography, Complex-Block,
                # Chemical-Block, a text-only Diagram, or any future label)
                # renders as a positioned text box — content is never dropped.
                render_text_entry(ctx, entry)
        # Floating fallback for contained pictures with no image_obj.
        for entry in contained_no_image:
            render_standalone_picture(ctx, entry)

        # Overlay diagram labels (source="diagram-label") on top of their diagram
        # raster. Match each to the diagram whose bbox contains it.
        diagram_labels = [e for e in entries if e.get("source") == "diagram-label"]
        if diagram_labels:
            from .picture import render_diagram_label_overlays
            diagrams = [
                e for e in entries
                if e.get("source") == "diagram-recovered" and e.get("bbox")
            ]
            for diag in diagrams:
                db = diag.get("bbox")
                mine = [
                    lab for lab in diagram_labels
                    if lab.get("bbox") and _bbox_inside(lab["bbox"], db)
                ]
                render_diagram_label_overlays(ctx, diag, mine)

        ctx.flush()
        shape_counter = ctx.next_id + 1

    doc.save(output_path)
    if isinstance(output_path, BytesIO):
        return output_path
    print(f"DOCX file saved to: {output_path}")
    return output_path