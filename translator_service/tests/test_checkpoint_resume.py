"""Checkpoint / resume behaviour of _translate_with_checkpoints.

Covers the guarantee that matters for a 1000-page job: work already done is
never redone, and a job that dies mid-flight resumes instead of restarting.

Run:  python -m pytest translator_service/tests/test_checkpoint_resume.py -v
"""

from __future__ import annotations

import copy
import sys
import uuid as _uuid
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pipeline  # noqa: E402
import translation  # noqa: E402


def _layout(n_pages: int) -> list[dict]:
    return [
        {"page_index": p, "page_width_pt": 595.0, "page_height_pt": 842.0,
         "zoom": 1.0,
         "entries": [{"category": "Text", "text": f"p{p}", "bbox": [0, 0, 9, 9]}]}
        for p in range(n_pages)
    ]


class FakeStorage:
    """In-memory stand-in for the MinIO + Postgres checkpoint surface."""

    def __init__(self):
        self.row: dict = {}
        self.checkpoint: list[dict] | None = None
        self.checkpoint_writes = 0
        self.deleted = False
        self.fail_next_checkpoint = False

    async def fetch_translation(self, trans_id):
        return dict(self.row) if self.row else None

    async def update_translation(self, trans_id, **fields):
        self.row.update(fields)

    async def put_checkpoint(self, trans_id, layout):
        if self.fail_next_checkpoint:
            self.fail_next_checkpoint = False
            raise RuntimeError("minio down")
        self.checkpoint_writes += 1
        self.checkpoint = copy.deepcopy(layout)
        return f"translations/{trans_id}/checkpoint.json"

    async def fetch_checkpoint(self, trans_id):
        return copy.deepcopy(self.checkpoint) if self.checkpoint else None

    async def delete_checkpoint(self, trans_id):
        self.deleted = True
        self.checkpoint = None


@pytest.fixture
def env(monkeypatch):
    store = FakeStorage()
    monkeypatch.setattr(pipeline, "storage", store)
    monkeypatch.setattr(pipeline, "TRANSLATE_PAGE_WINDOW", 3)

    async def fake_chunk(items, target_lang, client):
        return [s.upper() for s in items]

    async def no_retry(sources, flat, target_lang, rounds=2):
        return 0, 0

    monkeypatch.setattr(translation, "_translate_chunk", fake_chunk)
    monkeypatch.setattr(translation, "_retry_residual_cjk", no_retry)
    return store


@pytest.mark.asyncio
async def test_checkpoints_written_per_window(env):
    layout = _layout(10)
    await pipeline._translate_with_checkpoints(
        layout, trans_id=_uuid.uuid4(), target_lang="pt-BR",
        original_name="doc.pdf")

    # windows of 3 over 10 pages -> 4 checkpoints
    assert env.checkpoint_writes == 4
    assert env.row["pages_done"] == 10
    assert env.row["page_count"] == 10
    assert all(p["entries"][0]["text"].isupper() for p in layout)


@pytest.mark.asyncio
async def test_resume_skips_completed_pages(env):
    """The core guarantee: pages already translated are not re-sent."""
    trans_id = _uuid.uuid4()
    full = _layout(9)

    # Simulate a prior run that finished 6 of 9 pages then died.
    partial = copy.deepcopy(full)
    for p in partial[:6]:
        p["entries"][0]["text"] = p["entries"][0]["text"].upper()
    env.checkpoint = partial
    env.row = {"checkpoint_key": "k", "pages_done": 6}

    sent: list[str] = []
    original = translation._translate_chunk

    async def spy(items, target_lang, client):
        sent.extend(items)
        return await original(items, target_lang, client)

    translation._translate_chunk = spy
    try:
        layout = copy.deepcopy(full)
        stats = await pipeline._translate_with_checkpoints(
            layout, trans_id=trans_id, target_lang="pt-BR",
            original_name="doc.pdf")
    finally:
        translation._translate_chunk = original

    # Only pages 6,7,8 went to the model.
    assert sorted(sent) == ["p6", "p7", "p8"]
    assert stats["resumed_pages"] == 6
    # Every page ends up translated: spliced ones plus newly done ones.
    assert [p["entries"][0]["text"] for p in layout] == \
           [f"P{i}" for i in range(9)]


@pytest.mark.asyncio
async def test_fully_checkpointed_job_does_no_work(env):
    trans_id = _uuid.uuid4()
    done = _layout(5)
    for p in done:
        p["entries"][0]["text"] = p["entries"][0]["text"].upper()
    env.checkpoint = done
    env.row = {"checkpoint_key": "k", "pages_done": 5}

    layout = _layout(5)
    stats = await pipeline._translate_with_checkpoints(
        layout, trans_id=trans_id, target_lang="pt-BR",
        original_name="doc.pdf")

    assert stats["resumed_pages"] == 5
    assert stats["failed"] == 0          # must not trip _guard_translation
    assert all(p["entries"][0]["text"].isupper() for p in layout)


@pytest.mark.asyncio
async def test_mismatched_checkpoint_is_ignored(env):
    """Checkpoint from a different document shape must not corrupt output."""
    env.checkpoint = _layout(4)          # wrong length
    env.row = {"checkpoint_key": "k", "pages_done": 4}

    layout = _layout(9)
    stats = await pipeline._translate_with_checkpoints(
        layout, trans_id=_uuid.uuid4(), target_lang="pt-BR",
        original_name="doc.pdf")

    assert "resumed_pages" not in stats   # started over
    assert all(p["entries"][0]["text"].isupper() for p in layout)


@pytest.mark.asyncio
async def test_unreadable_checkpoint_starts_over(env):
    env.checkpoint = None                 # key recorded but object gone
    env.row = {"checkpoint_key": "k", "pages_done": 5}

    layout = _layout(6)
    await pipeline._translate_with_checkpoints(
        layout, trans_id=_uuid.uuid4(), target_lang="pt-BR",
        original_name="doc.pdf")

    assert all(p["entries"][0]["text"].isupper() for p in layout)


@pytest.mark.asyncio
async def test_checkpoint_write_failure_does_not_kill_job(env):
    """MinIO hiccup at checkpoint time must cost resumability, not the job."""
    env.fail_next_checkpoint = True

    layout = _layout(9)
    stats = await pipeline._translate_with_checkpoints(
        layout, trans_id=_uuid.uuid4(), target_lang="pt-BR",
        original_name="doc.pdf")

    assert stats["failed"] == 0
    assert all(p["entries"][0]["text"].isupper() for p in layout)


@pytest.mark.asyncio
async def test_no_checkpoint_for_short_document(env, monkeypatch):
    """Below one window there is nothing to resume from, so no extra I/O."""
    monkeypatch.setattr(pipeline, "TRANSLATE_PAGE_WINDOW", 25)
    layout = _layout(3)
    await pipeline._translate_with_checkpoints(
        layout, trans_id=_uuid.uuid4(), target_lang="pt-BR",
        original_name="doc.pdf")

    assert env.checkpoint_writes == 0
    assert all(p["entries"][0]["text"].isupper() for p in layout)


@pytest.mark.asyncio
async def test_page_count_recorded_even_without_windowing(env, monkeypatch):
    monkeypatch.setattr(pipeline, "TRANSLATE_PAGE_WINDOW", 25)
    await pipeline._translate_with_checkpoints(
        _layout(3), trans_id=_uuid.uuid4(), target_lang="pt-BR",
        original_name="doc.pdf")
    assert env.row["page_count"] == 3
