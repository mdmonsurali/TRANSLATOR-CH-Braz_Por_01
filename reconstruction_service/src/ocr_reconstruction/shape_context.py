"""Per-page accumulator. Collects every emitted XML chunk for a page and
flushes them into one paragraph inside the current section. Word anchors the
floating shapes relative to the page they live on, so the flush step is just
'one run, many drawings as children'."""
from __future__ import annotations

from typing import List


class ShapeContext:
    """Accumulates anchored shapes for one page and adds them as one paragraph
    containing one run with many drawings. python-docx lets us drop raw XML
    inside a run via element insertion."""

    def __init__(self, doc, page_w_pt: float, page_h_pt: float,
                 zoom: float, shape_id_start: int):
        self.doc = doc
        self.page_w_pt = page_w_pt
        self.page_h_pt = page_h_pt
        self.zoom = zoom
        self.next_id = shape_id_start
        self.xml_chunks: List[str] = []
        # Entries for the current page; set by json_to_docx so per-entry
        # renderers can be obstacle-aware (a table must not grow over its
        # neighbours). Empty when not provided, which every consumer treats as
        # "no opinion" — so this stays a no-op for callers that never set it.
        self.page_entries: List = []

    def _next_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def flush(self) -> None:
        """Drop every accumulated shape into one paragraph at the end of the
        current section. Word anchors them relative to the page they live on."""
        from .ooxml import parse_xml_fragment

        if not self.xml_chunks:
            # Add an empty paragraph so the section is non-empty (Word requires
            # at least one block in each section).
            self.doc.add_paragraph()
            return
        # Wrap in a single <w:p><w:r>...</w:r></w:p> tree.
        para = self.doc.add_paragraph()
        run = para.add_run()
        run_elem = run._r
        # Inject each anchored shape as a child of <w:r>.
        for chunk in self.xml_chunks:
            elem = parse_xml_fragment(chunk)
            run_elem.append(elem)
