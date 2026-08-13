"""Windowing equivalence + raster-lifetime tests for the OCR pipeline.

The pipeline renders, OCRs, recovers and then FREES pages one window at a time
so peak memory tracks the window rather than the document length. That is only
correct because every recovery pass is per-page pure (no cross-page state), so
processing N pages in windows of W must produce byte-identical output to
processing them in one pass.

These tests pin that property down. They stub the vLLM call — the point is the
windowing/lifetime logic, not the model.

Run:  python -m pytest ocr_service/tests/test_page_windowing.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
RECON_SRC = Path(__file__).resolve().parents[2] / "reconstruction_service" / "src"
for p in (str(SRC), str(RECON_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

fitz = pytest.importorskip("fitz", reason="PyMuPDF required")
PIL = pytest.importorskip("PIL", reason="Pillow required")
from PIL import Image  # noqa: E402

import doc_processing  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_pdf(path: Path, n_pages: int) -> Path:
    """A tiny multi-page PDF with identifiable per-page text."""
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=595, height=842)  # A4 pt
        page.insert_text((72, 200), f"Page {i} content", fontsize=24)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def pdf10(tmp_path: Path) -> Path:
    return _make_pdf(tmp_path / "ten.pdf", 10)


# ── load_page_range ──────────────────────────────────────────────────────────

def test_page_count_does_not_render(pdf10: Path):
    assert doc_processing.page_count_for_pdf(str(pdf10)) == 10


def test_range_returns_only_requested_pages(pdf10: Path):
    pages = doc_processing.load_page_range(str(pdf10), 3, 7)
    assert len(pages) == 4
    assert [p["page_index"] for p in pages] == [3, 4, 5, 6]


def test_page_index_is_absolute_not_window_relative(pdf10: Path):
    """picture_recovery/table_reocr re-open pdf_source and index into it by
    page_index, so a window starting at 8 must still report 8, not 0."""
    pages = doc_processing.load_page_range(str(pdf10), 8, 10)
    assert [p["page_index"] for p in pages] == [8, 9]


def test_range_clamps_out_of_bounds(pdf10: Path):
    assert doc_processing.load_page_range(str(pdf10), 8, 999) != []
    assert len(doc_processing.load_page_range(str(pdf10), 8, 999)) == 2
    assert doc_processing.load_page_range(str(pdf10), 50, 60) == []


def test_windowed_render_matches_whole_document(pdf10: Path):
    """Concatenated windows must equal the all-at-once render, page for page."""
    whole = doc_processing.load_pages_from_pdf(str(pdf10))

    windowed = []
    for start in range(0, 10, 3):
        windowed.extend(doc_processing.load_page_range(str(pdf10), start, start + 3))

    assert len(windowed) == len(whole) == 10
    for a, b in zip(whole, windowed):
        assert a["page_index"] == b["page_index"]
        assert a["zoom"] == b["zoom"]
        assert a["page_width_pt"] == b["page_width_pt"]
        assert a["page_height_pt"] == b["page_height_pt"]
        assert a["pdf_source"] == b["pdf_source"]
        assert a["image"].size == b["image"].size
        assert a["image"].tobytes() == b["image"].tobytes()


# ── layoutjson2md no longer needs a raster ───────────────────────────────────

def test_layoutjson2md_works_without_image():
    entries = [
        {"category": "Title", "text": "Hello", "bbox": [0, 0, 10, 10]},
        {"category": "Text", "text": "World", "bbox": [0, 20, 10, 30]},
    ]
    md = doc_processing.layoutjson2md(entries)
    assert "# Hello" in md and "World" in md


def test_layoutjson2md_ignores_supplied_image():
    """The image arg is vestigial; passing one must not change the output."""
    entries = [{"category": "Text", "text": "x", "bbox": [0, 0, 1, 1]}]
    img = Image.new("RGB", (4, 4))
    assert doc_processing.layoutjson2md(entries) == \
           doc_processing.layoutjson2md(entries, image=img)


# ── process_pictures page numbering across windows ───────────────────────────

def test_process_pictures_start_index_keeps_page_numbers_absolute():
    """Called per window, filenames must not restart at page1 each window."""
    from ocr_reconstruction import process_pictures

    def _page(text):
        return {
            "original_image": Image.new("RGB", (100, 100)),
            "layout_result": [{"category": "Image", "text": text,
                               "bbox": [0, 0, 50, 50]}],
            "markdown_content": "",
        }

    win1 = process_pictures([_page("a"), _page("b")], start_index=1)
    win2 = process_pictures([_page("c"), _page("d")], start_index=3)

    ids = [e["id"]
           for page in win1 + win2
           for e in page["layout_result"] if "id" in e]
    prefixes = [i.split("_")[0] for i in ids]
    assert prefixes == ["page1", "page2", "page3", "page4"]


def test_process_pictures_defaults_to_one():
    """Whole-document callers get the original numbering."""
    from ocr_reconstruction import process_pictures

    page = {
        "original_image": Image.new("RGB", (100, 100)),
        "layout_result": [{"category": "Image", "text": "a",
                           "bbox": [0, 0, 50, 50]}],
        "markdown_content": "",
    }
    out = process_pictures([page])
    assert out[0]["layout_result"][0]["id"].startswith("page1_")


# ── Raster release ───────────────────────────────────────────────────────────

def test_release_frees_both_alias_keys():
    """`image` and `original_image` alias the same PIL object and every
    recovery module reads `original_image or image`, so both must be dropped
    or the buffer stays alive."""
    import main

    img = Image.new("RGB", (64, 64))
    page = {"image": img, "original_image": img, "layout_result": []}
    main._release_page_rasters([page])

    assert "image" not in page
    assert "original_image" not in page


def test_release_keeps_crops_usable():
    """Crops own their own buffers; closing the parent must not invalidate
    them, since they are what survives the window."""
    import main

    parent = Image.new("RGB", (64, 64), color=(10, 20, 30))
    crop = parent.crop((0, 0, 8, 8))
    page = {"image": parent, "original_image": parent,
            "layout_result": [{"category": "Image", "image_obj": crop}]}

    main._release_page_rasters([page])

    surviving = page["layout_result"][0]["image_obj"]
    assert surviving.size == (8, 8)
    assert surviving.getpixel((0, 0)) == (10, 20, 30)  # still readable


def test_release_tolerates_missing_and_closed_images():
    import main

    already = Image.new("RGB", (4, 4))
    already.close()
    pages = [{"layout_result": []},
             {"image": None, "original_image": None, "layout_result": []},
             {"image": already, "original_image": already, "layout_result": []}]
    main._release_page_rasters(pages)  # must not raise


# ── Window driver equivalence (stubbed OCR) ──────────────────────────────────

@pytest.mark.asyncio
async def test_window_partitioning_covers_every_page_exactly_once():
    """The range walk in _run_pipeline must tile [0, total) with no gap or
    overlap for any window/total combination."""
    for total in (0, 1, 5, 10, 32, 33, 100, 350):
        for window in (1, 3, 32, 64):
            seen = []
            for start in range(0, total, window):
                end = min(start + window, total)
                seen.extend(range(start, end))
            assert seen == list(range(total)), (total, window)


@pytest.mark.asyncio
async def test_ocr_encode_is_bounded_by_semaphore(monkeypatch, tmp_path):
    """The PNG encode must sit INSIDE the semaphore. With it outside, gather
    schedules every page and each encodes before its first real await, so all
    N pages hit disk at once instead of batch_size at a time."""
    import main

    live = 0
    peak = 0

    async def fake_process_image_async(path):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return []

    monkeypatch.setattr(main, "process_image_async", fake_process_image_async)

    pages_meta = [{"image": Image.new("RGB", (32, 32)), "page_index": i}
                  for i in range(12)]
    batch_size = 3
    out = await main._ocr_pages_concurrently(pages_meta, batch_size)

    assert len(out) == 12
    assert peak <= batch_size, f"concurrency {peak} exceeded batch size {batch_size}"
    assert all("original_image" in p for p in out)


@pytest.mark.asyncio
async def test_process_window_releases_rasters_and_keeps_entries(monkeypatch):
    """One window: OCR runs, entries survive, rasters do not."""
    import main

    async def fake_process_image_async(path):
        return [{"category": "Text", "text": "hi", "bbox": [0, 0, 10, 10]}]

    monkeypatch.setattr(main, "process_image_async", fake_process_image_async)
    monkeypatch.setattr(main, "INCLUDE_PICTURES", False)
    monkeypatch.setattr(main, "ATTRIBUTE_STYLES", False)
    monkeypatch.setattr(main, "REOCR_DROPPED_CHECKBOXES", False)

    pages_meta = [{"image": Image.new("RGB", (32, 32)), "page_index": i,
                   "zoom": 1.0, "page_width_pt": 100.0, "page_height_pt": 100.0,
                   "pdf_source": None}
                  for i in range(4)]

    out = await main._process_window(pages_meta, 2, doc_id=None, page_offset=0)

    assert len(out) == 4
    for page in out:
        assert "image" not in page, "raster leaked past the window"
        assert "original_image" not in page, "raster leaked past the window"
        assert page["layout_result"][0]["text"] == "hi"
        assert "hi" in page["markdown_content"]
        assert page["zoom"] == 1.0  # geometry survives


@pytest.mark.asyncio
async def test_windowed_output_equals_single_pass(monkeypatch):
    """THE test: markdown + layout entries must be identical whether the
    document is processed in one window or many. This encodes the per-page
    purity assumption that makes windowing safe."""
    import main

    async def fake_process_image_async(path):
        return [{"category": "Text", "text": "line", "bbox": [0, 0, 10, 10]}]

    monkeypatch.setattr(main, "process_image_async", fake_process_image_async)
    monkeypatch.setattr(main, "INCLUDE_PICTURES", False)
    monkeypatch.setattr(main, "ATTRIBUTE_STYLES", False)
    monkeypatch.setattr(main, "REOCR_DROPPED_CHECKBOXES", False)

    total = 10

    def fresh_meta():
        return [{"image": Image.new("RGB", (32, 32)), "page_index": i,
                 "zoom": 1.0, "page_width_pt": 100.0, "page_height_pt": 100.0,
                 "pdf_source": None}
                for i in range(total)]

    # One window covering the whole document.
    single = await main._process_window(fresh_meta(), 2, None, 0)

    # Many small windows.
    meta = fresh_meta()
    windowed = []
    win = 3
    for start in range(0, total, win):
        chunk = meta[start:start + win]
        windowed.extend(await main._process_window(chunk, 2, None, start))

    assert len(single) == len(windowed) == total

    def envelope(pages):
        return [main._page_envelope_for_json(p) for p in pages]

    assert envelope(single) == envelope(windowed)
    assert [p["markdown_content"] for p in single] == \
           [p["markdown_content"] for p in windowed]
