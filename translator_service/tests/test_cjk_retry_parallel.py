"""`_retry_residual_cjk` must stay semantically identical after parallelization.

The retry pass used to be fully serial — nested for-loops with an await inside —
while every other call path ran TRANSLATE_CONCURRENCY-wide. On a document with
~80 residual strings that cost 10+ minutes of strictly sequential requests.

Batches now run under asyncio.gather with a semaphore. These tests pin the
behaviour that must NOT change:
  * a retry is accepted only if it actually removed the CJK (never make a
    string worse),
  * a failed/short batch still falls back to one call per item,
  * rounds stay sequential (round 2 sees round 1's results),
  * concurrency is bounded by TRANSLATE_CONCURRENCY.

Run:  python -m pytest translator_service/tests/test_cjk_retry_parallel.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import translation  # noqa: E402


CJK = "医"          # a CJK char, so _contains_cjk() is true


def cjk(s: str) -> str:
    """A source string that still contains CJK (i.e. a retry target)."""
    return f"{CJK}{s}"


def translated(s: str) -> str:
    """What a SUCCESSFUL retry returns: the CJK is gone.

    A stub that merely prefixes the input (e.g. "OK-" + "医a") still contains
    the CJK char, so the code correctly rejects it — strip it here so the stub
    models a real translation.
    """
    return "OK-" + s.replace(CJK, "")


@pytest.fixture
def no_sleep(monkeypatch):
    """Keep TRANSLATE_CONCURRENCY deterministic for these tests."""
    monkeypatch.setattr(translation, "TRANSLATE_CONCURRENCY", 4)


# ── Correctness ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_translates_all_residual_strings(monkeypatch, no_sleep):
    async def force(items, lang, client):
        return [translated(s) for s in items]

    monkeypatch.setattr(translation, "_force_translate", force)

    sources = [cjk(str(i)) for i in range(25)]
    flat = [cjk(str(i)) for i in range(25)]

    retried, still = await translation._retry_residual_cjk(sources, flat, "pt-BR")

    assert retried == 25
    assert still == 0
    assert all(v.startswith("OK-") for v in flat)


@pytest.mark.asyncio
async def test_never_accepts_a_still_cjk_retry(monkeypatch, no_sleep):
    """A retry that still contains CJK must be REJECTED, leaving the original."""
    async def force(items, lang, client):
        return [cjk("still-bad") for _ in items]   # never clears the CJK

    monkeypatch.setattr(translation, "_force_translate", force)

    sources = [cjk("a"), cjk("b")]
    flat = [cjk("orig-a"), cjk("orig-b")]

    retried, still = await translation._retry_residual_cjk(sources, flat, "pt-BR")

    assert retried == 2
    assert still == 2
    assert flat == [cjk("orig-a"), cjk("orig-b")], "must not overwrite with a worse value"


@pytest.mark.asyncio
async def test_never_touches_clean_strings(monkeypatch, no_sleep):
    """Entries without residual CJK are not targets and must be left alone."""
    async def force(items, lang, client):
        return [translated(s) for s in items]

    monkeypatch.setattr(translation, "_force_translate", force)

    sources = [cjk("a"), "already english", cjk("c")]
    flat = [cjk("a"), "already english", "clean result"]

    retried, still = await translation._retry_residual_cjk(sources, flat, "pt-BR")

    assert retried == 1                       # only index 0 qualifies
    assert flat[1] == "already english"
    assert flat[2] == "clean result"


@pytest.mark.asyncio
async def test_empty_targets_is_a_noop(monkeypatch, no_sleep):
    called = []

    async def force(items, lang, client):
        called.append(items)
        return [translated(s) for s in items]

    monkeypatch.setattr(translation, "_force_translate", force)

    flat = ["clean", "also clean"]
    retried, still = await translation._retry_residual_cjk(
        ["clean", "also clean"], flat, "pt-BR")

    assert (retried, still) == (0, 0)
    assert called == [], "must not call the model when nothing has residual CJK"


# ── Per-item fallback ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_failed_batch_falls_back_to_per_item(monkeypatch, no_sleep):
    """A batch returning None must still retry each item individually."""
    seen_sizes = []

    async def force(items, lang, client):
        seen_sizes.append(len(items))
        if len(items) > 1:
            return None                      # batch call fails
        return [translated(items[0])]            # per-item call succeeds

    monkeypatch.setattr(translation, "_force_translate", force)

    sources = [cjk(str(i)) for i in range(6)]
    flat = list(sources)

    retried, still = await translation._retry_residual_cjk(sources, flat, "pt-BR")

    assert still == 0
    assert all(v.startswith("OK-") for v in flat)
    assert 6 in seen_sizes                   # the batch attempt happened
    assert seen_sizes.count(1) == 6          # then one call per item


@pytest.mark.asyncio
async def test_wrong_length_batch_falls_back(monkeypatch, no_sleep):
    async def force(items, lang, client):
        if len(items) > 1:
            return ["only-one"]              # wrong length for the batch
        return [translated(items[0])]

    monkeypatch.setattr(translation, "_force_translate", force)

    sources = [cjk("a"), cjk("b"), cjk("c")]
    flat = list(sources)

    _, still = await translation._retry_residual_cjk(sources, flat, "pt-BR")
    assert still == 0
    assert all(v.startswith("OK-") for v in flat)


# ── Rounds stay sequential ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_second_round_sees_first_round_results(monkeypatch, no_sleep):
    """Rounds must remain ordered: round 2 only retries what round 1 left."""
    rounds_seen = []
    call = {"n": 0}

    async def force(items, lang, client):
        call["n"] += 1
        rounds_seen.append(len(items))
        # Round 1 fails to clear CJK; round 2 succeeds.
        if call["n"] <= 1:
            return [cjk("nope") for _ in items]
        return [translated(s) for s in items]

    monkeypatch.setattr(translation, "_force_translate", force)

    sources = [cjk("a"), cjk("b")]
    flat = list(sources)

    retried, still = await translation._retry_residual_cjk(
        sources, flat, "pt-BR", rounds=2)

    assert still == 0, "round 2 should have cleared what round 1 could not"
    assert all(v.startswith("OK-") for v in flat)


@pytest.mark.asyncio
async def test_stops_early_when_nothing_pending(monkeypatch, no_sleep):
    """If round 1 clears everything, round 2 must not issue any calls."""
    calls = {"n": 0}

    async def force(items, lang, client):
        calls["n"] += 1
        return [translated(s) for s in items]

    monkeypatch.setattr(translation, "_force_translate", force)

    sources = [cjk("a"), cjk("b")]
    flat = list(sources)
    await translation._retry_residual_cjk(sources, flat, "pt-BR", rounds=2)

    assert calls["n"] == 1, "one batch cleared everything; no second round"


# ── The actual speedup ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_batches_run_concurrently(monkeypatch, no_sleep):
    """THE regression test: batches must overlap, not run one at a time.

    With 5 batches of 10 and a serial loop, peak concurrency is 1. Parallelized
    it should reach TRANSLATE_CONCURRENCY.
    """
    live = 0
    peak = 0

    async def force(items, lang, client):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return [translated(s) for s in items]

    monkeypatch.setattr(translation, "_force_translate", force)

    sources = [cjk(str(i)) for i in range(50)]   # 5 batches of 10
    flat = list(sources)

    await translation._retry_residual_cjk(sources, flat, "pt-BR")

    assert peak > 1, "batches ran serially — the parallelization regressed"
    assert peak <= 4, f"concurrency {peak} exceeded TRANSLATE_CONCURRENCY=4"


@pytest.mark.asyncio
async def test_concurrency_is_bounded(monkeypatch):
    """The semaphore must cap in-flight calls even with many batches."""
    monkeypatch.setattr(translation, "TRANSLATE_CONCURRENCY", 2)

    live = 0
    peak = 0

    async def force(items, lang, client):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return [translated(s) for s in items]

    monkeypatch.setattr(translation, "_force_translate", force)

    sources = [cjk(str(i)) for i in range(80)]
    flat = list(sources)
    await translation._retry_residual_cjk(sources, flat, "pt-BR")

    assert peak <= 2, f"concurrency {peak} exceeded TRANSLATE_CONCURRENCY=2"


@pytest.mark.asyncio
async def test_result_matches_serial_reference(monkeypatch, no_sleep):
    """Same stubbed model, same inputs -> same `flat` a serial pass would give."""
    def model(s: str) -> str:
        # Deterministic, order-independent: clears CJK for even indices only,
        # so half the targets are accepted and half stay at their source value.
        return translated(s) if int(s[1:]) % 2 == 0 else cjk("stuck")

    async def force(items, lang, client):
        return [model(s) for s in items]

    monkeypatch.setattr(translation, "_force_translate", force)

    sources = [cjk(str(i)) for i in range(30)]

    # Reference: what a serial implementation would produce.
    expected = [model(s) if model(s) and CJK not in model(s) else s
                for s in sources]

    flat = list(sources)
    await translation._retry_residual_cjk(sources, flat, "pt-BR")

    assert flat == expected
