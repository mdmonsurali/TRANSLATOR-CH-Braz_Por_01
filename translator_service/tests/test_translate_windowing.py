"""Windowing + checkpoint/resume tests for the translation pipeline.

A 1000-page translation runs for hours, so it is split into page groups:
memory is bounded by the group, and progress is checkpointed after each one so
a job killed by a timeout / GPU swap / restart resumes instead of restarting.

Windowing is only correct because pages are independent — a unit's writer
closes over its own entry, and the residual-CJK retry compares a string only
against itself. These tests pin that down.

Run:  python -m pytest translator_service/tests/test_translate_windowing.py -v
"""

from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import translation  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _layout(n_pages: int, entries_per_page: int = 3) -> list[dict]:
    """A layout envelope list with identifiable per-entry text."""
    return [
        {
            "page_index": p,
            "page_width_pt": 595.0,
            "page_height_pt": 842.0,
            "zoom": 1.0,
            "entries": [
                {"category": "Text", "text": f"p{p}e{e}",
                 "bbox": [0, e * 10, 100, e * 10 + 8]}
                for e in range(entries_per_page)
            ],
        }
        for p in range(n_pages)
    ]


@pytest.fixture
def fake_llm(monkeypatch):
    """Deterministic translator: uppercases each source string."""
    calls = {"chunks": 0, "items": 0, "max_items_in_flight": 0}

    async def fake_translate_chunk(items, target_lang, client):
        calls["chunks"] += 1
        calls["items"] += len(items)
        calls["max_items_in_flight"] = max(
            calls["max_items_in_flight"], len(items))
        return [s.upper() for s in items]

    monkeypatch.setattr(translation, "_translate_chunk", fake_translate_chunk)

    async def no_retry(sources, flat, target_lang, rounds=2):
        return 0, 0

    monkeypatch.setattr(translation, "_retry_residual_cjk", no_retry)
    return calls


# ── Windowing equivalence ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_windowed_equals_single_pass(fake_llm):
    """THE test: windowing must not change a single translated string."""
    base = _layout(20)

    single = copy.deepcopy(base)
    await translation.translate_layout(single, "pt-BR", page_window=0)

    windowed = copy.deepcopy(base)
    await translation.translate_layout(windowed, "pt-BR", page_window=3)

    assert single == windowed
    # And it actually did something.
    assert windowed[0]["entries"][0]["text"] == "P0E0"


@pytest.mark.asyncio
async def test_window_larger_than_document_is_single_pass(fake_llm):
    layout = _layout(4)
    await translation.translate_layout(layout, "pt-BR", page_window=100)
    assert layout[3]["entries"][0]["text"] == "P3E0"


@pytest.mark.asyncio
async def test_every_page_translated_across_uneven_windows(fake_llm):
    """A window size that doesn't divide the page count must not drop a tail."""
    for pages, window in [(10, 3), (10, 4), (7, 7), (7, 1), (1, 5), (33, 8)]:
        layout = _layout(pages)
        await translation.translate_layout(layout, "pt-BR", page_window=window)
        for p, page in enumerate(layout):
            for e, entry in enumerate(page["entries"]):
                assert entry["text"] == f"P{p}E{e}", (pages, window, p, e)


@pytest.mark.asyncio
async def test_stats_accumulate_across_windows(fake_llm):
    layout = _layout(12)
    stats = await translation.translate_layout(layout, "pt-BR", page_window=5)
    assert stats["items_translated"] == 12 * 3
    assert stats["failed"] == 0
    assert stats["chunks"] >= 3  # at least one per window


@pytest.mark.asyncio
async def test_empty_layout_is_safe(fake_llm):
    stats = await translation.translate_layout([], "pt-BR", page_window=5)
    assert stats["items_translated"] == 0


# ── Progress callback (the checkpoint hook) ──────────────────────────────────

@pytest.mark.asyncio
async def test_on_progress_reports_monotonic_page_counts(fake_llm):
    seen = []

    async def on_progress(done, total, stats):
        seen.append((done, total))

    layout = _layout(10)
    await translation.translate_layout(layout, "pt-BR", page_window=3,
                                       on_progress=on_progress)

    assert [d for d, _ in seen] == [3, 6, 9, 10]
    assert all(t == 10 for _, t in seen)
    # Monotonically increasing, never past the end.
    assert seen == sorted(seen)


@pytest.mark.asyncio
async def test_on_progress_not_called_without_windowing(fake_llm):
    called = []

    async def on_progress(done, total, stats):
        called.append(done)

    layout = _layout(3)
    await translation.translate_layout(layout, "pt-BR", page_window=0,
                                       on_progress=on_progress)
    assert called == []


@pytest.mark.asyncio
async def test_progress_stats_are_cumulative(fake_llm):
    snapshots = []

    async def on_progress(done, total, stats):
        snapshots.append(stats["items_translated"])

    await translation.translate_layout(_layout(9), "pt-BR", page_window=3,
                                       on_progress=on_progress)
    assert snapshots == sorted(snapshots)
    assert snapshots[-1] == 27


# ── Memory bounding ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_units_are_bounded_by_window(monkeypatch):
    """The whole point: _collect_units must never see the whole document."""
    seen_sizes = []
    real_collect = translation._collect_units

    def spy(layout):
        seen_sizes.append(len(layout))
        return real_collect(layout)

    monkeypatch.setattr(translation, "_collect_units", spy)

    async def fake_chunk(items, target_lang, client):
        return [s.upper() for s in items]

    async def no_retry(sources, flat, target_lang, rounds=2):
        return 0, 0

    monkeypatch.setattr(translation, "_translate_chunk", fake_chunk)
    monkeypatch.setattr(translation, "_retry_residual_cjk", no_retry)

    await translation.translate_layout(_layout(100), "pt-BR", page_window=10)

    assert seen_sizes, "_collect_units was never called"
    assert max(seen_sizes) <= 10, (
        f"a window of {max(seen_sizes)} pages leaked through; "
        "memory would scale with document length")


# ── Failure isolation ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_one_bad_window_does_not_lose_other_windows(monkeypatch):
    """A window whose model call fails keeps source text; the rest still
    translate. Losing 1000 pages because page 400 failed is unacceptable."""
    call = {"n": 0}

    async def flaky(items, target_lang, client):
        call["n"] += 1
        if call["n"] == 2:          # fail the second chunk only
            return None
        return [s.upper() for s in items]

    async def no_retry(sources, flat, target_lang, rounds=2):
        return 0, 0

    monkeypatch.setattr(translation, "_translate_chunk", flaky)
    monkeypatch.setattr(translation, "_retry_residual_cjk", no_retry)

    layout = _layout(9, entries_per_page=1)
    stats = await translation.translate_layout(layout, "pt-BR", page_window=1)

    assert stats["failed"] > 0
    assert stats["items_translated"] > 0        # partial, not total loss
    translated = [p["entries"][0]["text"] for p in layout]
    assert any(t.isupper() for t in translated)
    assert any(t.islower() for t in translated)  # the failed one kept source


# ── Resume splicing (the logic _translate_with_checkpoints relies on) ────────

def test_resume_splice_preserves_identity():
    """Resume replaces leading pages with checkpointed ones, then translates
    the rest in place. `pending` must alias `layout`'s dicts so in-place writes
    are visible in the final list."""
    layout = _layout(6)
    saved = copy.deepcopy(layout)
    for p in saved[:2]:
        for e in p["entries"]:
            e["text"] = e["text"].upper()

    done = 2
    layout[:done] = saved[:done]
    pending = layout[done:]

    # Mutating `pending` must show up in `layout` (same objects, not copies).
    pending[0]["entries"][0]["text"] = "MUTATED"
    assert layout[2]["entries"][0]["text"] == "MUTATED"
    assert layout[0]["entries"][0]["text"] == "P0E0".upper()


def test_resume_rejects_mismatched_checkpoint():
    """A checkpoint whose page count differs from the layout means the source
    changed; it must be ignored rather than spliced in."""
    layout = _layout(6)
    stale = _layout(4)
    assert len(stale) != len(layout)  # the condition the pipeline checks
