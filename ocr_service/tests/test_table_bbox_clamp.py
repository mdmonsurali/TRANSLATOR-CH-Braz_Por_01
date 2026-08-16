"""Container-bbox guard for table geometry (`table_geometry`).

WHY THIS EXISTS. A small 2-row table rendered stretched across the full text
column of the page. Its layout entry carried bbox [267, 362, 2841, 2067] —
2574x1705 px, essentially the whole page body — because the layout model boxed
the table with its CONTAINING block's rectangle (a page frame / mislabelled
`Form` block rescued by `_looks_like_table`) rather than the table's own extent.

`render_table` derives `w` from that bbox once and then treats it as an axiom,
and `_normalise_to` inflates ROW HEIGHTS to fill the bbox as well, so the
renderer faithfully stretched a 2-row grid across the container.

WHY THE TEST IS ON HEIGHT, NOT WIDTH. The intuitive check — bbox width vs the
content's natural width — cannot work here, and these tests encode why.
`bbox_px_to_emu` clamps the bbox to the PAGE before the renderer sees it, so a
runaway box arrives no wider than one page. Measured across real geometries:

    the reported regression      1.24 - 2.04x natural width
    an ordinary 3-column table   2.84x
    an ordinary CJK 2-column     2.92x

The bug scores BELOW healthy tables — no width threshold separates them. The
height test ("how many rows of text would fit in this box vs how many it holds")
puts the regression at 12-22x while healthy tables sit at ~2.0-2.3x and the worst
honest cases (spacious forms, signature blocks) reach only ~5.4x.

THE NO-OP TESTS MATTER MOST. A guard that fires on ordinary padding would
reshape documents that render correctly today.

Run:  python -m pytest ocr_service/tests/test_table_bbox_clamp.py -v
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
RECON_SRC = Path(__file__).resolve().parents[2] / "reconstruction_service" / "src"
for p in (str(SRC), str(RECON_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import table_geometry as tg  # noqa: E402
from table_geometry import (  # noqa: E402
    BBOX_OVERSIZE_RATIO, EMU_PER_PT, bbox_is_container,
    clamp_to_natural_width, natural_table_width_emu,
)

# Representative wrapped-line height at the ~10.6pt body size these documents
# use. The renderer computes this from real font metrics; a constant is fine
# here because every assertion below is a RATIO.
_LINE_H_PT = 14.8


# ── Measurement stand-ins ────────────────────────────────────────────────────
# `natural_table_width_emu` takes its two text measures as callables precisely
# so it can be exercised without pulling in either package's `text_fit`.

def _display_width(s: str) -> int:
    # Longest LINE, not total length: `<br/>` becomes "\n" during parsing and a
    # column is only as wide as its widest line.
    return max((len(line) for line in (s or "").split("\n")), default=0)


def _longest_token_width(s: str) -> int:
    if not s:
        return 0
    return max((len(t) for line in s.split("\n") for t in line.split()), default=0)


def _anchors(rows):
    """`cell_anchors`-shaped dict: {(r, c): (text, colspan, rowspan, is_header)}."""
    return {
        (r, c): (txt, 1, 1, r == 0)
        for r, row in enumerate(rows)
        for c, txt in enumerate(row)
    }


def _natural(rows, size_pt=10.6):
    max_cols = max(len(r) for r in rows)
    col_weight = [
        max((_display_width(r[c]) for r in rows if c < len(r)), default=1)
        for c in range(max_cols)
    ]
    return natural_table_width_emu(
        _anchors(rows), max_cols, col_weight, size_pt,
        _display_width, _longest_token_width,
    )


# ── Detection: the height test ───────────────────────────────────────────────

def test_the_reported_container_bbox_is_detected():
    """bbox [267,362,2841,2067] holds a 2-row table. At the pipeline's zoom the
    box is hundreds of points tall for ~3 wrapped lines of content."""
    # 1705px tall at zoom 4.0 -> ~426pt; 3 wrapped lines need ~44pt.
    assert bbox_is_container(426.0, 3, _LINE_H_PT)


@pytest.mark.parametrize("zoom,bbox_h_pt", [
    (2.0, 661.0), (3.0, 568.0), (4.0, 426.0), (4.8, 355.0),
])
def test_the_regression_is_detected_at_every_plausible_zoom(zoom, bbox_h_pt):
    """Zoom varies per document (scans run up to ~4.8x). The signal must not
    depend on guessing it."""
    assert bbox_is_container(bbox_h_pt, 3, _LINE_H_PT), f"missed at zoom {zoom}"


@pytest.mark.parametrize("name,bbox_h_pt,lines", [
    ("snug 3-row table", 100.0, 3),
    ("2x padded 3-row", 100.0, 3),
    ("tall 20-row table", 692.0, 20),
    ("1-row header strip", 30.0, 1),
    ("2-row, generous bbox", 148.0, 2),
    ("spacious form", 224.0, 4),
    ("signature block", 240.0, 3),
    ("short table of tall multiline cells", 148.0, 6),
])
def test_healthy_layouts_are_not_flagged(name, bbox_h_pt, lines):
    """Real shapes that render correctly today, including the sparse ones that
    come closest to the trigger. Any of these firing is a regression."""
    assert not bbox_is_container(bbox_h_pt, lines, _LINE_H_PT), name


def test_multiline_cells_are_measured_as_lines_not_rows():
    """A 2-ROW table whose cells wrap to 8 lines legitimately needs 8 lines of
    height. Counting rows instead of wrapped lines would flag it."""
    tall = 8 * _LINE_H_PT * 1.5
    assert not bbox_is_container(tall, 8, _LINE_H_PT)
    # The same box holding only 1 line of content IS a container.
    assert bbox_is_container(tall, 1, _LINE_H_PT)


def test_trigger_boundary():
    need = 4 * _LINE_H_PT
    assert not bbox_is_container(need * (BBOX_OVERSIZE_RATIO - 0.1), 4, _LINE_H_PT)
    assert bbox_is_container(need * (BBOX_OVERSIZE_RATIO + 0.1), 4, _LINE_H_PT)


@pytest.mark.parametrize("h,lines,lh", [
    (0.0, 3, 14.8), (-5.0, 3, 14.8),      # no height
    (400.0, 0, 14.8), (400.0, -1, 14.8),  # no content
    (400.0, 3, 0.0),                       # no font metrics
])
def test_degenerate_inputs_never_flag(h, lines, lh):
    assert not bbox_is_container(h, lines, lh)


# ── Application: narrow + centre ─────────────────────────────────────────────

def test_clamped_table_is_centred_in_the_original_bbox():
    """Equal gutters are what put the small table under its caption instead of
    pinned to the container's left edge."""
    x0, w0 = 100 * EMU_PER_PT, 500 * EMU_PER_PT
    natural = 200 * EMU_PER_PT
    x, w, clamped = clamp_to_natural_width(x0, w0, natural)
    assert clamped and w == natural
    assert (x - x0) == (x0 + w0) - (x + w), "table is not centred"
    assert x >= x0 and x + w <= x0 + w0


def test_clamp_is_a_no_op_when_the_box_is_not_wider_than_natural():
    x0, w0 = 0, 200 * EMU_PER_PT
    for natural in (200 * EMU_PER_PT, 300 * EMU_PER_PT):
        assert clamp_to_natural_width(x0, w0, natural) == (x0, w0, False)


@pytest.mark.parametrize("w,natural", [(0, 100), (500, 0), (500, -1), (-5, 100)])
def test_clamp_rejects_degenerate_geometry(w, natural):
    assert clamp_to_natural_width(10, w, natural) == (10, w, False)


# ── Natural width sanity ─────────────────────────────────────────────────────

def test_short_numeric_table_is_narrow_and_prose_table_is_wide():
    narrow = _natural([["5.35", "236.85"], ["1.20", "300.10"]])
    wide = _natural([
        ["A reasonably long paragraph of explanatory text in this cell",
         "Another long paragraph sitting beside it in the second column"],
    ])
    assert 0 < narrow < wide
    assert narrow < 300 * EMU_PER_PT


def test_natural_width_grows_with_column_count():
    assert _natural([["ab", "cd", "ef", "gh"]]) > _natural([["ab", "cd"]])


def test_empty_input_yields_no_opinion():
    assert natural_table_width_emu({}, 0, [], 11.0,
                                   _display_width, _longest_token_width) == 0
    assert natural_table_width_emu({}, 3, [1, 1, 1], 11.0,
                                   _display_width, _longest_token_width) == 0


def test_unbreakable_token_widens_a_column_beyond_its_average():
    anchors = {(0, 0): ("x", 1, 1, True), (1, 0): ("A" * 40, 1, 1, False)}
    assert natural_table_width_emu(
        anchors, 1, [1], 11.0, _display_width, _longest_token_width,
    ) > 40 * EMU_PER_PT * 0.4


def test_colspan_demand_is_shared_across_covered_columns():
    """A wide colspan cell must not force every column it covers to full width."""
    w_span = natural_table_width_emu(
        {(0, 0): ("A" * 40, 4, 1, True)}, 4, [1, 1, 1, 1], 11.0,
        _display_width, _longest_token_width,
    )
    w_single = natural_table_width_emu(
        {(0, 0): ("A" * 40, 1, 1, True)}, 1, [1], 11.0,
        _display_width, _longest_token_width,
    )
    assert w_span < w_single * 4


# ── End-to-end through the real renderer ─────────────────────────────────────

_REGRESSION_HTML = (
    '<table border="1"><thead><tr>'
    '<th>Torque de falha/<br/>Relação torque de inserção/torque de falha</th>'
    '<th>Força de extração (N)</th></tr></thead>'
    '<tbody><tr><td>5.35</td><td>236.85</td></tr></tbody></table>'
)
_HEALTHY_HTML = (
    '<table border="1"><thead><tr><th>Parameter</th><th>Value</th>'
    '<th>Unit</th></tr></thead><tbody>'
    '<tr><td>Tensile strength</td><td>186.3</td><td>N</td></tr>'
    '<tr><td>Torque ratio</td><td>5.35</td><td>-</td></tr>'
    '</tbody></table>'
)


class _Ctx:
    def __init__(self):
        self.xml_chunks = []
        self.zoom = 2.0
        self.page_w_pt = 595.0
        self.page_h_pt = 842.0
        self._id = 0

    def _next_id(self):
        self._id += 1
        return self._id


def _render_xml(mod, html, bbox, size=10.6):
    """Raw emitted OOXML for one table — the sibling of `_render`, which returns
    geometry. Used by the font assertions, which need to read `w:sz` values."""
    ctx = _Ctx()
    mod.render_table(ctx, {"text": html, "bbox": bbox, "category": "Table",
                           "style": {"size": size, "font": "SimSun"}})
    return "".join(ctx.xml_chunks)


def _render(mod, html, bbox, size=10.6):
    import re
    ctx = _Ctx()
    mod.render_table(ctx, {"text": html, "bbox": bbox, "category": "Table",
                           "style": {"size": size, "font": "SimSun"}})
    xml = "".join(ctx.xml_chunks)
    pos = re.search(r'<wp:positionH[^>]*><wp:posOffset>(-?\d+)', xml)
    ext = re.search(r'<wp:extent cx="(\d+)" cy="(\d+)"', xml)
    assert pos and ext, "no anchored shape emitted"
    return (int(pos.group(1)) / EMU_PER_PT, int(ext.group(1)) / EMU_PER_PT,
            int(ext.group(2)) / EMU_PER_PT)


def _both_renderers():
    pytest.importorskip("PIL", reason="Pillow required")
    import ocr_reconstruction.table as ocr_table
    import translation_reconstruction.table as tr_table
    return [("ocr", ocr_table), ("translation", tr_table)]


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_the_reported_table_renders_small_and_centred(which):
    """The regression, end to end. The container bbox is 1287pt wide (page
    clamped to ~462pt) and 661pt tall; the table must come out near its natural
    width, a couple of lines tall, and centred in the container."""
    mod = dict(_both_renderers())[which]
    bbox = [267, 362, 2841, 2067]
    x, w, h = _render(mod, _REGRESSION_HTML, bbox)

    assert w < 420, f"still stretched: {w:.0f}pt"
    assert h < 120, f"still inflated: {h:.0f}pt tall"
    # Centred within the page-clamped container box.
    box_x, box_w = bbox[0] / 2.0, min(595.0 - bbox[0] / 2.0,
                                      (bbox[2] - bbox[0]) / 2.0)
    assert abs((x - box_x) - ((box_x + box_w) - (x + w))) <= 1.0


@pytest.mark.parametrize("which", ["ocr", "translation"])
@pytest.mark.parametrize("bbox", [
    [300, 400, 1100, 600],   # snug
    [300, 400, 1900, 600],   # 2x padded — must still be untouched
])
def test_healthy_tables_render_identically_with_the_guard_present(which, bbox):
    """The guard must not perturb a table it does not fire on.

    Compared against what `bbox_px_to_emu` alone would produce — NOT against the
    raw bbox. That function independently clamps a box to the page (the 2x-padded
    case is 800pt wide on a 595pt page and comes back 445pt), and attributing its
    clamping to the guard would make this test assert the wrong thing.
    """
    pytest.importorskip("PIL", reason="Pillow required")
    from ocr_reconstruction.geometry import bbox_px_to_emu

    mod = dict(_both_renderers())[which]
    x, w, h = _render(mod, _HEALTHY_HTML, bbox)
    ex, _ey, ew, eh = bbox_px_to_emu(bbox, 2.0, 595.0, 842.0)
    assert abs(x - ex / EMU_PER_PT) < 0.5, "position shifted"
    assert abs(w - ew / EMU_PER_PT) < 0.5, "width changed"
    assert abs(h - eh / EMU_PER_PT) < 0.5, "height changed"


def test_both_renderer_copies_produce_the_same_geometry():
    """~600 lines have already drifted between the two table.py copies; the
    translated DOCX must lay out its tables exactly like the OCR'd one."""
    mods = dict(_both_renderers())
    for html, bbox in ((_REGRESSION_HTML, [267, 362, 2841, 2067]),
                       (_HEALTHY_HTML, [300, 400, 1100, 600])):
        a = _render(mods["ocr"], html, bbox)
        b = _render(mods["translation"], html, bbox)
        assert a == pytest.approx(b, abs=0.5), f"copies diverge on {bbox}"


# ── Both copies must use the shared rule ─────────────────────────────────────

def test_both_reconstruction_copies_import_the_shared_guard():
    for _name, mod in _both_renderers():
        assert mod.bbox_is_container is tg.bbox_is_container
        assert mod.clamp_to_natural_width is tg.clamp_to_natural_width


def test_the_two_copies_apply_the_guard_identically():
    import inspect
    keys = ("_guard_eligible", "_bbox_clamped", "bbox_is_container",
            "clamp_to_natural_width", "_natural_w", "_natural_h_pt")

    def _guard_lines(mod):
        src = inspect.getsource(mod.render_table).splitlines()
        return [ln.strip() for ln in src
                if ln.strip() and not ln.strip().startswith("#")
                and any(k in ln for k in keys)]

    mods = dict(_both_renderers())
    assert _guard_lines(mods["ocr"]) == _guard_lines(mods["translation"])


# ── Obstacle-bounded growth: a table must not grow over other layout ─────────
# A table whose translated text cannot fit its source bbox is allowed to GROW
# (the grow-backstop prefers a taller table to a clipped cell). The only bound
# that existed for that growth was the PAGE EDGE, on the height axis alone, so a
# grown table ran straight over whatever sat beneath it — in the reported file a
# 1016-char cell grew to 508.9pt, hit the page bottom at 595.3pt, and covered the
# page number 72pt below it. The rule is now symmetric: grow until the page edge
# OR the nearest neighbour on that axis, whichever comes first.

def test_downward_room_stops_at_the_nearest_entry_below():
    """Room is measured to the neighbour's TOP edge, less a small gap."""
    table = {"bbox": [100, 100, 500, 400]}
    footer = {"bbox": [200, 600, 300, 640]}
    room = tg.downward_room_pt([table, footer], table, 100, 500, 100, 2.0)
    # (600 - 4*2 - 100) / 2 = 246
    assert room == pytest.approx(246.0)


def test_rightward_room_stops_at_the_nearest_entry_to_the_right():
    table = {"bbox": [100, 100, 500, 400]}
    side = {"bbox": [700, 150, 900, 350]}
    room = tg.rightward_room_pt([table, side], table, 100, 100, 400, 2.0)
    # (700 - 4*2 - 100) / 2 = 296
    assert room == pytest.approx(296.0)


def test_room_is_none_when_nothing_blocks_that_axis():
    """None means "no opinion", so the page edge stays the only bound. An entry
    that misses our band on the perpendicular axis is never an obstacle."""
    table = {"bbox": [100, 100, 500, 400]}
    # Below, but in a different column: not a vertical obstacle.
    elsewhere = {"bbox": [900, 600, 1000, 640]}
    assert tg.downward_room_pt([table, elsewhere], table, 100, 500, 100, 2.0) is None
    # To the right, but in a different row band: not a horizontal obstacle.
    above = {"bbox": [700, 10, 900, 50]}
    assert tg.rightward_room_pt([table, above], table, 100, 100, 400, 2.0) is None


def test_room_is_none_without_entries_so_the_guard_is_a_no_op():
    """Callers that never populate `page_entries` must be unaffected."""
    for entries in ([], None, ()):
        assert tg.downward_room_pt(entries, None, 0, 10, 0, 2.0) is None
        assert tg.rightward_room_pt(entries, None, 0, 0, 10, 2.0) is None


def test_malformed_neighbours_are_not_obstacles():
    """A bad bbox must never squeeze a table — an unexplained shrink would be a
    worse failure than the overlap this guards against."""
    table = {"bbox": [100, 100, 500, 400]}
    junk = [table, {"bbox": [1, 2]}, {"bbox": None}, {}, {"bbox": ["a", 1, 2, 3]}]
    assert tg.downward_room_pt(junk, table, 100, 500, 100, 2.0) is None
    assert tg.rightward_room_pt(junk, table, 100, 100, 400, 2.0) is None


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_a_grown_table_stops_above_the_entry_below_it(which):
    """End to end: the reported page-0 shape. One cell far too long for its
    bbox, with a page-number entry 27pt below the table's bottom edge. The
    rendered table must not reach that entry's top."""
    mod = dict(_both_renderers())[which]
    bbox = [263, 360, 2841, 2067]              # 619 x 410 pt at zoom 2... (see below)
    footer = {"bbox": [1712, 2181, 1782, 2228], "category": "Page-Footer"}
    entry = {"text": '<table border="1"><tr><td></td><td>'
                     + ("palavra " * 260) + "</td><td></td></tr></table>",
             "bbox": bbox, "category": "Table",
             "style": {"size": 10.6, "font": "SimSun"}}
    ctx = _Ctx()
    ctx.page_entries = [entry, footer]
    mod.render_table(ctx, entry)
    import re
    xml = "".join(ctx.xml_chunks)
    pos = re.search(r'<wp:positionV[^>]*><wp:posOffset>(-?\d+)', xml)
    ext = re.search(r'<wp:extent cx="(\d+)" cy="(\d+)"', xml)
    y = int(pos.group(1)) / EMU_PER_PT
    h = int(ext.group(2)) / EMU_PER_PT
    footer_top_pt = footer["bbox"][1] / ctx.zoom
    assert y + h <= footer_top_pt + 0.5, (
        f"table bottom {y + h:.1f}pt runs into the entry at {footer_top_pt:.1f}pt"
    )


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_the_obstacle_bound_is_a_no_op_without_page_entries(which):
    """Same table, no `page_entries`: unchanged from the pre-guard behaviour."""
    mod = dict(_both_renderers())[which]
    bbox = [300, 400, 1100, 600]
    with_ctx = _render(mod, _HEALTHY_HTML, bbox)
    assert with_ctx[1] > 0 and with_ctx[2] > 0


def test_both_reconstruction_copies_import_the_shared_room_helpers():
    for _name, mod in _both_renderers():
        assert mod.downward_room_pt is tg.downward_room_pt
        assert mod.rightward_room_pt is tg.rightward_room_pt


# ── Rebuilding a sub-table the OCR flattened ────────────────────────────────
# Verified against the source PDF's vector lines: the reported table's outer
# grid has NO interior row dividers on the affected pages (one continuation
# row), and every interior rule sits inside a single outer column. The OCR
# nevertheless emitted the sub-table's rows and columns at TOP level, which is
# what produced both "extra row/column" and "table inside table goes to next".
#
# As ever in this module, THE NO-OP TESTS MATTER MOST: a promoter that fires on
# ordinary tables would reshape documents that render correctly today.

def _rows(mod, html):
    return mod.parse_html_table_rows(html)


def _sub_tables(rows):
    """Every nested ("table", rows) block in a parsed row list."""
    return [v for cells, _h in rows for c in cells
            for k, v in (getattr(c, "blocks", None) or ()) if k == "table"]


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_rows_confined_to_an_inset_span_become_one_nested_table(which):
    """Signal A: consecutive rows whose content is inset from the left are the
    sub-table's own rows promoted to top level."""
    mod = dict(_both_renderers())[which]
    html = ("<table>"
            + "".join("<tr><td></td><td></td><td>n%d</td><td>v%d</td>"
                      "<td></td></tr>" % (i, i) for i in range(4))
            + "</table>")
    rows = _rows(mod, html)
    out = mod._promote_flattened_subtables(rows)
    assert out is not rows, "the flattened sub-table was not rebuilt"
    assert len(out) == 1, "the run should collapse to ONE outer row"
    subs = _sub_tables(out)
    assert len(subs) == 1 and len(subs[0]) == 4, "sub-table rows lost"


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_equal_parallel_paragraph_blocks_are_paired_as_sub_rows(which):
    """Signal B: adjacent cells carrying the same number of short <p> blocks are
    the sub-table's columns emitted side by side."""
    mod = dict(_both_renderers())[which]
    html = ("<table><tr><td></td>"
            "<td><p>a1</p><p>a2</p><p>a3</p></td>"
            "<td><p>b1</p><p>b2</p><p>b3</p></td>"
            "<td></td></tr></table>")
    out = mod._promote_flattened_subtables(_rows(mod, html))
    subs = _sub_tables(out)
    assert len(subs) == 1, "parallel blocks were not paired"
    sub = subs[0]
    assert len(sub) == 3, "one sub-row per parallel block"
    assert [c[0] for c in sub[0][0]] == ["a1", "b1"], "blocks paired out of order"


@pytest.mark.parametrize("which", ["ocr", "translation"])
@pytest.mark.parametrize("html,why", [
    ("<table><tr><td>a</td><td>b</td><td>c</td></tr>"
     "<tr><td>d</td><td>e</td><td>f</td></tr></table>",
     "a full-width table has nothing inset"),
    ("<table><tr><td></td><td>only</td><td></td></tr></table>",
     "one confined row is just a row with blanks"),
    ("<table><tr><td>a</td><td>b</td><td></td></tr>"
     "<tr><td>c</td><td>d</td><td></td></tr></table>",
     "TRAILING blanks are ordinary row padding, not nesting"),
    ("<table><tr><td></td><td><p>x1</p><p>x2</p></td>"
     "<td><p>y1</p><p>y2</p><p>y3</p></td></tr></table>",
     "unequal block counts cannot be paired unambiguously"),
    ("<table><tr><td></td><td><p>only</p></td><td><p>one</p></td></tr></table>",
     "a single block per cell is a plain cell"),
])
def test_ordinary_tables_are_left_untouched(which, html, why):
    """Each of these must return the SAME rows object — a strict no-op."""
    mod = dict(_both_renderers())[which]
    rows = _rows(mod, html)
    assert mod._promote_flattened_subtables(rows) is rows, why


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_long_paragraph_columns_are_prose_not_a_sub_table(which):
    """Two neighbouring cells of running prose can trivially agree on their
    paragraph count. Measured in the reported corpus: a real flattened sub-table's
    longest block is 74 chars, this false-positive shape reaches 476."""
    mod = dict(_both_renderers())[which]
    para = "texto " * 40                       # ~240 chars, clearly prose
    html = ("<table><tr><td></td>"
            f"<td><p>{para}</p><p>{para}</p></td>"
            f"<td><p>{para}</p><p>{para}</p></td>"
            "</tr></table>")
    rows = _rows(mod, html)
    assert mod._promote_flattened_subtables(rows) is rows


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_a_real_nested_table_is_never_re_promoted(which):
    """Structure the OCR got right must pass through untouched."""
    mod = dict(_both_renderers())[which]
    html = ("<table><tr><td></td><td><table><tr><td>n</td><td>v</td></tr>"
            "</table></td><td></td></tr></table>")
    rows = _rows(mod, html)
    assert mod._promote_flattened_subtables(rows) is rows


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_promotion_never_drops_content(which):
    """The run's own caption row spans wider than the sub-table's columns; an
    earlier cut discarded exactly that cell. Compare character multisets so a
    reordering (column-major to row-major) is not mistaken for a loss."""
    import re
    from collections import Counter
    mod = dict(_both_renderers())[which]
    html = ("<table>"
            "<tr><td></td><td>n0</td><td>v0</td><td></td></tr>"
            "<tr><td></td><td colspan='3'>caption spanning wide</td></tr>"
            "<tr><td></td><td>n1</td><td>v1</td><td></td></tr>"
            "</table>")

    def _deep(rows):
        acc = []

        def walk(rs):
            for cells, _h in rs:
                for c in cells:
                    blocks = getattr(c, "blocks", None) or ()
                    if any(k == "table" for k, _ in blocks):
                        for k, v in blocks:
                            walk(v) if k == "table" else acc.append(v)
                    else:
                        acc.append(c[0] or "")
        walk(rows)
        return Counter(re.sub(r"\s+", "", "".join(acc)))

    rows = _rows(mod, html)
    out = mod._promote_flattened_subtables(rows)
    assert out is not rows, "expected this shape to promote"
    assert _deep(rows) == _deep(out), "promotion lost or invented content"


def test_the_two_copies_promote_identically():
    """~600 lines have already drifted between the copies; this rule must not."""
    mods = dict(_both_renderers())
    html = ("<table>"
            + "".join("<tr><td></td><td>n%d</td><td>v%d</td><td></td></tr>"
                      % (i, i) for i in range(3))
            + "</table>")
    a = mods["ocr"]._promote_flattened_subtables(_rows(mods["ocr"], html))
    b = mods["translation"]._promote_flattened_subtables(
        _rows(mods["translation"], html))
    assert [[tuple(c)[:4] for c in cells] for cells, _ in a] == \
           [[tuple(c)[:4] for c in cells] for cells, _ in b]


# ── Three-state column classification ───────────────────────────────────────
# The promoter must distinguish a column the row left BLANK from one merely
# COVERED by a `rowspan` anchored in another row, and must treat an `<img>`-only
# cell as CONTENT. Conflating these made it collapse ordinary flat picture
# tables — real data rows were destroyed on a document that rendered correctly
# before the promoter existed.

@pytest.mark.parametrize("which", ["ocr", "translation"])
@pytest.mark.parametrize("html,should_fire,why", [
    # A row under a rowspan legitimately emits fewer <td>s. Its leading columns
    # are COVERED, not blank, so it is not evidence of an inset sub-table.
    ("<table>"
     "<tr><td rowspan='3'>L</td><td>a1</td><td>b1</td><td rowspan='3'>R</td></tr>"
     "<tr><td>a2</td><td>b2</td></tr>"
     "<tr><td>a3</td><td>b3</td></tr></table>",
     False, "rowspan-covered columns are not blank"),
    # A picture column carrying no text is CONTENT, not a blank left margin.
    ("<table>"
     "<tr><td><img alt='x'/></td><td>a1</td><td></td></tr>"
     "<tr><td><img alt='y'/></td><td>a2</td><td></td></tr></table>",
     False, "an <img>-only cell is content"),
    # A run that is inset on the LEFT but runs out to the grid's RIGHT edge is
    # NOT a sub-table — it is an ordinary row group under a tall `rowspan`
    # label, which is exactly this shape. A nested table is ENCLOSED by the
    # outer grid, so it must leave a column free on both sides.
    #
    # This case previously expected True. That expectation was wrong: measured
    # on a real page (`2.6 Clinical Evaluation ...-77-80`, page 2), a
    # `rowspan="16"` label in column 0 produces exactly this `.BCC…`-to-the-edge
    # shape for six ordinary data rows, and promoting them collapsed all six
    # into one flattened cell — destroying the first row and shifting the rest
    # into the wrong columns. The synthetic 3-column example here is degenerate:
    # a "sub-table" filling all remaining width is indistinguishable from plain
    # data rows beside a label, and the real document settles which reading is
    # correct.
    ("<table>"
     "<tr><td rowspan='3'>Label</td><td></td><td>n0</td><td>v0</td></tr>"
     "<tr><td></td><td>n1</td><td>v1</td></tr>"
     "<tr><td></td><td>n2</td><td>v2</td></tr></table>",
     False, "left-inset but reaching the right edge is a row group, not nesting"),
    # Inset on BOTH sides with no rowspan involved — genuinely enclosed, so this
    # is the plain sub-table case and must still be promoted.
    ("<table>"
     "<tr><td></td><td>n0</td><td>v0</td><td></td></tr>"
     "<tr><td></td><td>n1</td><td>v1</td><td></td></tr></table>",
     True, "left-inset blanks with no rowspan are the plain case"),
    ("<table><tr><td>a</td><td>b</td></tr>"
     "<tr><td>c</td><td>d</td></tr></table>",
     False, "full-width data has no inset"),
    ("<table><tr><td>a</td><td>b</td><td></td></tr>"
     "<tr><td>c</td><td>d</td><td></td></tr></table>",
     False, "trailing padding is not nesting"),
])
def test_promotion_classifies_blank_covered_and_content(
    which, html, should_fire, why,
):
    mod = dict(_both_renderers())[which]
    rows = mod.parse_html_table_rows(html)
    fired = mod._promote_flattened_subtables(rows) is not rows
    assert fired is should_fire, why


# ── A sub-table fills its parent cell without touching the border ───────────

def _sub_rows(mod):
    return mod.parse_html_table_rows(
        "<table><tr><td>a</td><td>b</td></tr>"
        "<tr><td>c</td><td>d</td></tr></table>"
    )


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_a_nested_target_height_is_met_in_both_directions(which):
    """A nested table is as FIXED as its parent: it fills a larger cell and
    shrinks into a smaller one, rather than spilling past the cell border."""
    mod = dict(_both_renderers())[which]
    style = {"size": 9.0, "font": "SimSun"}
    w = int(200 * EMU_PER_PT)
    _xml, natural = mod.render_nested_table_xml(_sub_rows(mod), w, style, 9.0)
    assert natural > 0

    for factor in (3.0, 2.0, 1.0, 0.5, 0.25):
        _xml, got = mod.render_nested_table_xml(
            _sub_rows(mod), w, style, 9.0, target_h_pt=natural * factor)
        assert got == pytest.approx(natural * factor, abs=1.0), (
            f"target x{factor} not met: got {got:.1f}pt"
        )


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_nested_rows_are_pinned_exact_so_word_cannot_grow_them(which):
    """`atLeast` let Word grow a sub-table row past the declared height, pushing
    the parent cell out. A fixed table must pin them."""
    mod = dict(_both_renderers())[which]
    xml, _h = mod.render_nested_table_xml(
        _sub_rows(mod), int(200 * EMU_PER_PT), {"size": 9.0, "font": "SimSun"},
        9.0,
    )
    assert 'w:hRule="exact"' in xml
    assert 'w:hRule="atLeast"' not in xml


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_a_leading_sub_table_gets_a_spacer_off_the_cell_border(which):
    """The parent pins `<w:top w:w="0">` margins and `vAlign=top`, so without a
    leading paragraph the sub-table's first row border lands ON the parent's."""
    mod = dict(_both_renderers())[which]
    xml, _h = mod._cell_blocks_xml(
        (("table", _sub_rows(mod)),), int(200 * EMU_PER_PT),
        {"size": 9.0, "font": "SimSun"}, 9.0,
    )
    assert xml.startswith("<w:p "), "sub-table is flush against the cell border"


@pytest.mark.parametrize("which", ["ocr", "translation"])
@pytest.mark.parametrize("n_tables", [1, 2, 3])
def test_stacked_sub_tables_share_the_cell(which, n_tables):
    """N stacked sub-tables must JOINTLY fill the cell — not each sit at natural
    size with the surplus pooled underneath."""
    mod = dict(_both_renderers())[which]
    style = {"size": 9.0, "font": "SimSun"}
    w = int(200 * EMU_PER_PT)
    blocks = tuple(("table", _sub_rows(mod)) for _ in range(n_tables))
    _xml, natural = mod._cell_blocks_xml(blocks, w, style, 9.0)
    target = natural * 2.5
    _xml, got = mod._cell_blocks_xml(blocks, w, style, 9.0, target_h_pt=target)
    assert got == pytest.approx(target, abs=1.0)


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_only_table_blocks_absorb_the_surplus(which):
    """Text keeps its natural height; the surplus goes to the table(s)."""
    import re
    mod = dict(_both_renderers())[which]
    style = {"size": 9.0, "font": "SimSun"}
    w = int(200 * EMU_PER_PT)
    blocks = (("text", "Caption"), ("table", _sub_rows(mod)))
    nat_xml, natural = mod._cell_blocks_xml(blocks, w, style, 9.0)
    fill_xml, got = mod._cell_blocks_xml(
        blocks, w, style, 9.0, target_h_pt=natural * 3)
    assert got == pytest.approx(natural * 3, abs=1.0)

    def _heights(x):
        return [int(v) for v in re.findall(r'<w:trHeight w:val="(\d+)"', x)]

    assert sum(_heights(fill_xml)) > sum(_heights(nat_xml)), "table did not grow"


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_a_cell_without_a_sub_table_ignores_the_target(which):
    """No table block → no target applies → byte-identical to today."""
    mod = dict(_both_renderers())[which]
    style = {"size": 9.0, "font": "SimSun"}
    w = int(200 * EMU_PER_PT)
    blocks = (("text", "just text"),)
    xml_a, h_a = mod._cell_blocks_xml(blocks, w, style, 9.0)
    xml_b, h_b = mod._cell_blocks_xml(blocks, w, style, 9.0, target_h_pt=h_a * 3)
    assert xml_a == xml_b and h_a == pytest.approx(h_b)


def test_the_spacer_paragraph_puts_its_size_in_rpr():
    """`w:sz` is a RUN property. Emitted loose in `w:pPr` Word ignores it, so the
    '1pt' spacer rendered at body size and inflated every sub-table cell."""
    for _name, mod in _both_renderers():
        assert "<w:rPr>" in mod._EMPTY_P
        assert "<w:pPr><w:sz" not in mod._EMPTY_P


# ── A table's own contained pictures are not obstacles ──────────────────────

def test_contained_entries_are_not_growth_obstacles():
    """A table's cell pictures are Image entries INSIDE its own bbox. Treated as
    neighbours they cap the table at the gap before its first picture — measured
    on a real document: a 710pt table capped to 58pt."""
    table = {"bbox": [263, 461, 3217, 1628]}
    own_picture = {"bbox": [520, 1047, 939, 1259], "category": "Image"}
    entries = [table, own_picture]
    assert tg.rightward_room_pt(entries, table, 263, 461, 1628, 4.1667) is None
    assert tg.downward_room_pt(entries, table, 263, 3217, 461, 4.1667) is None


# ── A table is EXACTLY its bbox — it never grows ────────────────────────────
# The old contract preferred growing a table to clipping a cell. A grown table
# runs over the layout beneath it and off the page: measured on a real document,
# a 349pt table rendered 665pt tall with its bottom at 752pt on a 595pt page.
# The layout is now fixed, and nothing may push it past its own rectangle.

def _picture_entry(bbox):
    """A picture entry with a real raster, as the pipeline supplies them."""
    pytest.importorskip("PIL", reason="Pillow required")
    from PIL import Image
    w = max(1, int(bbox[2] - bbox[0]))
    h = max(1, int(bbox[3] - bbox[1]))
    return {"bbox": list(bbox), "category": "Image", "id": "pic",
            "image_obj": Image.new("RGB", (w, h), "white")}


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_a_table_never_renders_taller_than_its_bbox(which):
    """Even when the content demands far more room than the bbox allows."""
    mod = dict(_both_renderers())[which]
    bbox = [200, 200, 1400, 500]              # 600 x 150 pt at zoom 2
    html = ("<table border='1'>"
            + "".join(
                f"<tr><td>{'palavra ' * 40}</td><td>{'texto ' * 40}</td></tr>"
                for _ in range(6))
            + "</table>")
    _x, _w, h = _render(mod, html, bbox)
    bbox_h_pt = (bbox[3] - bbox[1]) / 2.0
    assert h <= bbox_h_pt + 1.0, (
        f"table grew to {h:.0f}pt from a {bbox_h_pt:.0f}pt bbox"
    )


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_oversized_pictures_shrink_instead_of_growing_the_table(which):
    """Pictures are scaled by `min(1.0, scale_w, scale_h)` against the FINAL row
    heights, so a table full of huge images stays its own size."""
    mod = dict(_both_renderers())[which]
    bbox = [200, 200, 1400, 500]
    html = ("<table border='1'>"
            "<tr><td><img alt='a'/></td><td>one</td></tr>"
            "<tr><td><img alt='b'/></td><td>two</td></tr></table>")
    docx = pytest.importorskip("docx", reason="python-docx required")
    ctx = _Ctx()
    ctx.doc = docx.Document()          # add_image_relationship needs a real doc
    entry = {"text": html, "bbox": bbox, "category": "Table",
             "style": {"size": 10.6, "font": "SimSun"}}
    # Each picture is taller than the whole table.
    pics = [_picture_entry([250, 220, 700, 900]),
            _picture_entry([250, 950, 700, 1600])]
    mod.render_table(ctx, entry, pictures_for_table=pics)
    import re
    xml = "".join(ctx.xml_chunks)
    ext = re.search(r'<wp:extent cx="(\d+)" cy="(\d+)"', xml)
    h = int(ext.group(2)) / EMU_PER_PT
    assert h <= (bbox[3] - bbox[1]) / 2.0 + 1.0, "pictures grew the table"


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_no_row_is_emitted_with_hrule_atleast(which):
    """`atLeast` lets Word grow a row past what we declared — the second,
    independent way a table runs onto the next page."""
    mod = dict(_both_renderers())[which]
    bbox = [200, 200, 1400, 500]
    html = ("<table border='1'>"
            + "".join(f"<tr><td>{'palavra ' * 30}</td></tr>" for _ in range(5))
            + "</table>")
    ctx = _Ctx()
    mod.render_table(ctx, {"text": html, "bbox": bbox, "category": "Table",
                           "style": {"size": 10.6, "font": "SimSun"}})
    xml = "".join(ctx.xml_chunks)
    assert 'w:hRule="atLeast"' not in xml


# ── Pictures land in the column their bbox indicates ────────────────────────

_TWO_IMG_COLS = (
    "<table border='1'>"
    "<tr><td>L</td><td><img alt='left'/></td><td>M</td>"
    "<td><img alt='right'/></td></tr></table>"
)


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_pictures_are_placed_by_x_not_only_y(which):
    """Two <img> cells in the SAME row band are indistinguishable by y alone —
    which is how a right-hand diagram ended up in a left-hand column."""
    mod = dict(_both_renderers())[which]
    bbox = [0, 0, 4000, 400]
    mc, nr, anchors, _occ, img = mod.parse_table_grid(
        mod.parse_html_table_rows(_TWO_IMG_COLS))
    weights = mod.compute_col_weights(anchors, mc)
    left = {"bbox": [900, 100, 1100, 300], "category": "Image"}
    right = {"bbox": [3100, 100, 3300, 300], "category": "Image"}
    out = mod._assign_pictures_to_cells(
        [left, right], bbox, weights, anchors, mc, nr, img)
    placed = {a: [p["bbox"][0] for p in v] for a, v in out.items()}
    assert len(placed) == 2, f"both <img> cells should be used: {placed}"
    for anchor, xs in placed.items():
        col = anchor[1]
        # The left-hand cell must hold the left-hand picture, and vice versa.
        assert (col < mc / 2) == (xs[0] < 2000), f"picture crossed columns: {placed}"


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_surplus_pictures_beyond_img_capacity_are_dropped_smallest_first(which):
    """`<img>` placeholders state how many pictures the table can show. Layout
    models emit extra slivers (measured: 3-27pt strips beside 50-150pt
    diagrams); the real diagrams are the large ones."""
    mod = dict(_both_renderers())[which]
    bbox = [0, 0, 4000, 400]
    mc, nr, anchors, _occ, img = mod.parse_table_grid(
        mod.parse_html_table_rows(_TWO_IMG_COLS))
    weights = mod.compute_col_weights(anchors, mc)
    big_l = {"bbox": [900, 100, 1100, 300], "category": "Image"}
    big_r = {"bbox": [3100, 100, 3300, 300], "category": "Image"}
    sliver = {"bbox": [1000, 150, 1008, 250], "category": "Image"}
    out = mod._assign_pictures_to_cells(
        [sliver, big_l, big_r], bbox, weights, anchors, mc, nr, img)
    kept = [p["bbox"] for v in out.values() for p in v]
    assert len(kept) == 2, f"capacity is 2, got {len(kept)}"
    assert sliver["bbox"] not in kept, "the artifact displaced a real diagram"


# ── A rowspan that stops short of the last row ──────────────────────────────

@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_a_truncated_rowspan_is_extended_to_the_last_row(which):
    """OCR under-counts a rowspan that runs to the bottom, leaving a hole that
    renders as a stray extra cell — the reported 'last column has 2 rows but
    reconstruction built 3'."""
    mod = dict(_both_renderers())[which]
    html = ("<table border='1'>"
            "<tr><td>a1</td><td rowspan='2'>spans to bottom</td></tr>"
            "<tr><td>a2</td></tr>"
            "<tr><td>a3</td></tr></table>")
    mc, nr, anchors, occ, _img = mod.parse_table_grid(
        mod.parse_html_table_rows(html))
    assert nr == 3
    assert occ[nr - 1][1], "the trailing hole was left open"
    assert anchors[(0, 1)][2] == 3, "the rowspan was not extended to the bottom"


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_a_genuinely_blank_trailing_cell_is_left_alone(which):
    """Single-row cells followed by a blank are ordinary — not a truncated span."""
    mod = dict(_both_renderers())[which]
    html = ("<table border='1'>"
            "<tr><td>a1</td><td>v1</td></tr>"
            "<tr><td>a2</td><td>v2</td></tr>"
            "<tr><td>a3</td><td></td></tr></table>")
    mc, nr, anchors, _occ, _img = mod.parse_table_grid(
        mod.parse_html_table_rows(html))
    assert all(anchors[(r, 1)][2] == 1 for r in range(nr) if (r, 1) in anchors)


# ── A column the OCR dropped entirely ──────────────────────────────────────
#
# A layout model can omit a whole table column — typically a blank label column,
# which carries no text for it to notice. Every emitted row is then uniformly one
# cell too narrow, so the modal-width logic has nothing to disagree with and
# every cell is placed one column off. Pictures are the only in-table content
# with page geometry of their own, so they are the one available witness.

def _picture_grid_case(n_cols, img_col, pic_frac, n_pics=4):
    """A `n_cols`-wide grid whose <img> cells sit in `img_col`, with pictures
    physically at `pic_frac` across the table."""
    bbox = [0, 0, 900, 400]
    pics = []
    for i in range(n_pics):
        cx = bbox[2] * pic_frac
        pics.append({"bbox": [cx - 20, 10 + i * 90, cx + 20, 80 + i * 90]})
    cols = [img_col] * n_pics
    return pics, bbox, cols


def test_a_dropped_column_is_detected_when_every_picture_disagrees():
    """The reported 'first empty column is missing': every picture sits a full
    column-width right of the column its <img> was parsed into."""
    # 8 parsed columns; pictures physically in the 2nd of 9 → grid is short one.
    pics, bbox, cols = _picture_grid_case(8, img_col=0, pic_frac=1.5 / 9)
    assert tg.dropped_column_index(pics, bbox, cols, 8) == 0


def test_a_table_whose_pictures_agree_is_left_alone():
    """The no-op that matters most: a correct grid is never reshaped."""
    # <img> in column 1 and the pictures really are in column 1 of 9.
    pics, bbox, cols = _picture_grid_case(9, img_col=1, pic_frac=1.5 / 9)
    assert tg.dropped_column_index(pics, bbox, cols, 9) is None


def test_one_disagreeing_picture_does_not_reshape_the_grid():
    """The disagreement must be UNANIMOUS — a single stray picture is not
    evidence that a whole column is missing."""
    pics, bbox, cols = _picture_grid_case(8, img_col=0, pic_frac=1.5 / 9)
    # Drag one picture back into column 0's band so the vote is split.
    pics[0]["bbox"] = [10, 10, 50, 80]
    assert tg.dropped_column_index(pics, bbox, cols, 8) is None


def test_sub_column_jitter_does_not_reshape_the_grid():
    """A picture merely inset inside its own cell is not a column shift."""
    # Pictures sit slightly right of centre in their own column, not a column over.
    pics, bbox, cols = _picture_grid_case(8, img_col=0, pic_frac=0.09)
    assert tg.dropped_column_index(pics, bbox, cols, 8) is None


def test_a_table_with_no_pictures_is_never_reshaped():
    """No pictures → no signal → no change. This is what keeps the rule inert on
    documents whose tables are pure text (measured: all 17 tables of one
    reference document have no in-table Image entries at all)."""
    assert tg.dropped_column_index([], [0, 0, 900, 400], [], 8) is None


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_inserting_the_blank_column_restores_the_true_grid(which):
    """End-to-end: an 8-wide parse becomes the source's 9-wide grid, with the
    blank column leading and every cell shifted back one to the right."""
    mod = dict(_both_renderers())[which]
    html = ("<table border='1'>"
            "<tr><td><img alt='d'/></td><td>AR-1</td><td>5,5 mm</td></tr>"
            "<tr><td><img alt='d'/></td><td>AR-2</td><td>6,5 mm</td></tr>"
            "</table>")
    rows = mod.parse_html_table_rows(html)
    mc, nr, anchors, _occ, img = mod.parse_table_grid(rows)
    assert (mc, nr) == (3, 2)
    mc2, nr2, anchors2, _occ2, img2 = mod.parse_table_grid(rows, insert_blank_col=0)
    assert (mc2, nr2) == (4, 2), "the blank column was not inserted"
    # Column 0 is blank; the content moved one column right, intact.
    assert not (anchors2[(0, 0)][0] or "").strip()
    assert (0, 1) in img2, "the picture cell did not move to column 1"
    assert anchors2[(0, 2)][0] == "AR-1"
    assert anchors2[(1, 2)][0] == "AR-2"


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_inserting_a_column_does_not_trip_the_subtable_promoter(which):
    """A fresh blank LEADING column is exactly the left-inset shape
    `_promote_flattened_subtables` reads as a nested sub-table. Inserting before
    promotion collapsed a real 4-row table into a single row, so the insert must
    happen after it."""
    mod = dict(_both_renderers())[which]
    html = "<table border='1'>" + "".join(
        f"<tr><td><img alt='d'/></td><td>AR-{i}</td><td>{i},5 mm</td></tr>"
        for i in range(4)
    ) + "</table>"
    rows = mod.parse_html_table_rows(html)
    mc, nr, _a, _o, _i = mod.parse_table_grid(rows, insert_blank_col=0)
    assert nr == 4, f"rows collapsed to {nr} — the promoter ate the table"
    assert mc == 4


# ── A rowspan the source contradicts ───────────────────────────────────────

@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_a_contradicted_rowspan_is_retracted(which):
    """The reported 'second column's last row moved to the third column': an
    over-stated rowspan leaves the final row's own cells nowhere to start, so
    the walker pushes every one of them a column to the right."""
    mod = dict(_both_renderers())[which]
    # Mirrors the measured page: col0 legitimately spans rows 3-4, but col1 is
    # ALSO given rowspan=2 where the source rules divide it. Row 4 then has only
    # columns 2+ free, so its three cells are each pushed one column right and a
    # multi-column hole trails at the end.
    html = ("<table border='1'>"
            "<tr><td>h0</td><td>h1</td><td>h2</td><td>h3</td><td>h4</td>"
            "<td>h5</td><td>h6</td></tr>"
            "<tr><td rowspan='2'>L</td><td rowspan='2'>M</td><td>b2</td>"
            "<td>b3</td><td rowspan='3'>R</td><td rowspan='3'>S</td>"
            "<td rowspan='3'>T</td></tr>"
            "<tr><td>c2</td><td>c3</td></tr>"
            "<tr><td rowspan='2'>L2</td><td rowspan='2'>M2</td><td>d2</td>"
            "<td>d3</td></tr>"
            "<tr><td>IMG</td><td>e2</td><td>e3</td></tr></table>")
    mc, nr, anchors, _occ, _img = mod.parse_table_grid(
        mod.parse_html_table_rows(html))
    # Without the repair the last row starts at column 2 and trails a hole.
    assert (4, 1) in anchors, "row 4 did not get its own column-1 cell"
    assert anchors[(4, 1)][0] == "IMG"
    assert anchors[(4, 2)][0] == "e2"
    assert anchors[(3, 1)][2] == 1, "the contradicted rowspan was not retracted"
    # The genuinely-spanning label beside it is untouched.
    assert anchors[(3, 0)][2] == 2, "a correct rowspan was disturbed"


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_a_rowspan_over_genuinely_empty_rows_is_preserved(which):
    """A span whose covered rows really are empty creates no pressure and must
    survive — and `_close_trailing_rowspan_gaps` must still extend it."""
    mod = dict(_both_renderers())[which]
    html = ("<table border='1'>"
            "<tr><td>a1</td><td rowspan='2'>spans</td></tr>"
            "<tr><td>a2</td></tr>"
            "<tr><td>a3</td></tr></table>")
    mc, nr, anchors, occ, _img = mod.parse_table_grid(
        mod.parse_html_table_rows(html))
    assert anchors[(0, 1)][2] == 3, "the two repairs fought over the same row"
    assert all(occ[r][1] for r in range(nr))


# ── Nested sub-table clears its parent cell's borders ───────────────────────

@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_a_nested_subtable_has_a_gap_at_both_borders(which):
    """The sub-table's first and last row borders must not land on the parent
    cell's own borders. The SAME spacer drives both, so the gaps match."""
    mod = dict(_both_renderers())[which]
    assert mod._SPACER_PT >= 2.0, "the spacer is too small to separate borders"
    # The spacer must carry its size as a RUN property; `w:sz` loose in `w:pPr`
    # is ignored by Word, which is what made a previous "1pt" spacer render at
    # body size.
    assert "<w:rPr>" in mod._EMPTY_P and "w:sz" in mod._EMPTY_P
    blocks = (("table", [([mod.Cell("x", 1, 1, 0)], False)]),)
    xml, h = mod._cell_blocks_xml(blocks, 2_000_000, {"size": 9.0}, 9.0)
    assert xml.startswith("<w:p"), "no leading spacer before the sub-table"
    assert xml.rstrip().endswith("</w:p>"), "a w:tc must end with a w:p"
    # Both spacers are accounted in the height the caller reserves.
    assert h >= 2 * mod._SPACER_PT


# ── Rowspan-shadowed rows keep their true columns ───────────────────────────
# Measured on `2.6 Clinical Evaluation Clinical Evidence-77-80`, page 2. The
# table opens with `<td rowspan="16">Força de extração</td>`; every row beneath
# it emits one fewer <td>. Two independent defects followed from that shape, and
# each is pinned below.

_ROWSPAN_LABEL_HTML = (
    "<table><tbody>"
    "<tr><td rowspan='4'>Label</td><td></td><td>1</td><td>0.23</td></tr>"
    "<tr><td></td><td>2</td><td>0.28</td></tr>"
    "<tr><td></td><td>3</td><td>0.38</td></tr>"
    "<tr><td></td><td>4</td><td>0.33</td></tr>"
    "</tbody></table>"
)


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_rows_under_a_tall_rowspan_are_not_collapsed(which):
    """The promoter must leave a plain data block under a `rowspan` label alone.

    It used to read those rows as an inset sub-table and collapse the whole run
    into ONE flattened cell — the first data row vanished and the rest were
    re-columned. The run reaches the grid's right edge, so it is not enclosed
    and cannot be a nested table.
    """
    mod = dict(_both_renderers())[which]
    rows = mod.parse_html_table_rows(_ROWSPAN_LABEL_HTML)
    assert mod._promote_flattened_subtables(rows) is rows, (
        "a row group under a rowspan label was collapsed into a sub-table"
    )


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_every_row_under_a_rowspan_keeps_its_own_row(which):
    """No row is lost, and the data lands in the same column on every row."""
    mod = dict(_both_renderers())[which]
    rows = mod.parse_html_table_rows(_ROWSPAN_LABEL_HTML)
    max_cols, n_rows, anchors, _occ, _img = mod.parse_table_grid(rows)
    assert n_rows == 4, f"expected 4 rows, got {n_rows}"
    # The label owns column 0; the numbered series must line up beneath itself.
    series_col = {
        r: c
        for (r, c), (txt, *_rest) in anchors.items()
        if (txt or "").strip() in ("1", "2", "3", "4")
    }
    assert len(series_col) == 4, f"lost a data row: {series_col}"
    assert len(set(series_col.values())) == 1, (
        f"data rows landed in different columns: {series_col}"
    )


# ── Blank rowspan spacers must not shrink the grid ──────────────────────────

@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_blank_rowspan_spacers_do_not_narrow_the_grid(which):
    """A table whose outer rules are tall blank spacer cells keeps its width.

    Measured on page 3 of the same document: the anchor row emits 10 cells
    (six of them `<td rowspan="12"></td>` verticals) while every data row emits
    only 4. The modal width was taken from raw <td> counts, so the grid was
    built 8 wide — dropping the anchor row's last cells and shifting every data
    row into the wrong column. Effective occupancy (which counts the rowspans
    reaching down from above) is 10 on nearly every row.
    """
    mod = dict(_both_renderers())[which]
    html = (
        "<table><tbody>"
        "<tr><td rowspan='3'></td><td rowspan='3'></td><td>5</td><td>137.40</td>"
        "<td rowspan='3'></td><td rowspan='3'></td><td>3</td><td>148.45</td></tr>"
        "<tr><td>Media</td><td>162.28</td><td>4</td><td>169.24</td></tr>"
        "<tr><td>Desvio</td><td>21.13</td><td>5</td><td>134.53</td></tr>"
        "</tbody></table>"
    )
    rows = mod.parse_html_table_rows(html)
    max_cols, n_rows, anchors, _occ, _img = mod.parse_table_grid(rows)
    assert max_cols == 8, f"grid narrowed to {max_cols} cols; spacers were lost"
    # Nothing may be dropped: the anchor row's four values must all be present.
    row0 = {c: (t or "").strip()
            for (r, c), (t, *_x) in anchors.items() if r == 0}
    assert "148.45" in row0.values(), f"anchor row lost a cell: {row0}"
    # The two data columns must stay aligned across every row.
    for label, value in (("Media", "162.28"), ("Desvio", "21.13")):
        cols = [c for (r, c), (t, *_x) in anchors.items()
                if (t or "").strip() == label]
        vals = [c for (r, c), (t, *_x) in anchors.items()
                if (t or "").strip() == value]
        assert cols and vals and vals[0] == cols[0] + 1, (
            f"{label}/{value} were split across columns"
        )


# ── Table text stays readable ───────────────────────────────────────────────

@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_table_font_floor_is_readable(which):
    """The uniform-shrink fallback must not drive cells into illegibility.

    The floor used to be 3.5pt and was reached routinely — the shrink loop is
    the first thing that gives when a bbox is tight, so dense tables silently
    landed there.

    Two tiers now: a table STARTS at `TABLE_MIN_FONT_PT` (10pt), may shrink as
    far as `TABLE_SOFT_MIN_FONT_PT` (8pt) while staying inside its source bbox,
    and only grows taller when even that overflows.
    """
    mod = dict(_both_renderers())[which]
    assert mod.TABLE_MIN_FONT_PT >= 10.0, (
        f"table start size is {mod.TABLE_MIN_FONT_PT}pt — below the 10pt target"
    )
    assert mod._HARD_MIN_FONT_PT >= 8.0, (
        f"shrink floor is {mod._HARD_MIN_FONT_PT}pt — too small to read"
    )
    # The soft floor must not exceed the start size, or the loop cannot shrink.
    assert mod.TABLE_SOFT_MIN_FONT_PT <= mod.TABLE_MIN_FONT_PT


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_a_dense_table_renders_at_the_readable_floor(which):
    """End to end: a table too dense for its bbox never renders below the soft
    floor — it shrinks to 8pt and then grows, rather than shrinking on."""
    mod = dict(_both_renderers())[which]
    body = "".join(
        f"<tr><td>Row {i} label</td><td>{i}.00</td><td>{i}.50</td></tr>"
        for i in range(24)
    )
    html = f"<table><tbody>{body}</tbody></table>"
    # A deliberately short bbox: 24 rows cannot fit at the start size.
    xml = _render_xml(mod, html, [0, 0, 1200, 400])
    sizes = {int(m) / 2 for m in re.findall(r'<w:sz w:val="(\d+)"/>', xml)}
    body_sizes = {s for s in sizes if s > 2.0}   # ignore the sub-table spacer
    assert body_sizes, "no text emitted"
    assert min(body_sizes) >= mod.TABLE_SOFT_MIN_FONT_PT, (
        f"cells rendered below the {mod.TABLE_SOFT_MIN_FONT_PT}pt soft floor: "
        f"{sorted(body_sizes)}"
    )


@pytest.mark.parametrize("which", ["ocr", "translation"])
def test_a_roomy_table_keeps_the_full_start_size(which):
    """The soft floor is a fallback, not a default: a table with room to spare
    must still render at the 10pt start size, never pre-shrunk."""
    mod = dict(_both_renderers())[which]
    html = (
        "<table><tbody>"
        "<tr><td>a</td><td>b</td></tr>"
        "<tr><td>c</td><td>d</td></tr>"
        "</tbody></table>"
    )
    xml = _render_xml(mod, html, [0, 0, 1200, 600])
    sizes = {int(m) / 2 for m in re.findall(r'<w:sz w:val="(\d+)"/>', xml)}
    body_sizes = {s for s in sizes if s > 2.0}
    assert body_sizes, "no text emitted"
    assert min(body_sizes) >= mod.TABLE_MIN_FONT_PT, (
        f"a roomy table was shrunk below the start size: {sorted(body_sizes)}"
    )
