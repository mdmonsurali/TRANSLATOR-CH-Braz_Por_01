"""A stuck generation must not hang the whole document.

Observed in production: a 1-page scanned PDF pushed the model into a degenerate
state, the chat.completions.create call never returned (no timeout was set),
the retry loop never advanced, and the document sat in `processing` for 65+
minutes. These tests pin the recovery behaviour.

Run:  python -m pytest ocr_service/tests/test_request_timeout.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
RECON = Path(__file__).resolve().parents[2] / "reconstruction_service" / "src"
for p in (str(SRC), str(RECON)):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

import chandra_ocr  # noqa: E402


# parse_layout_html wants top-level <div>s carrying data-label + data-bbox
# (bbox is 0-1000 normalized). Anything else parses to zero entries.
_GOOD_HTML = '<div data-label="Text" data-bbox="0 0 500 500">hello</div>'


class _Msg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class _Resp:
    def __init__(self, content):
        self.choices = [_Msg(content)]


class FakeCompletions:
    """Scripted chat.completions.create."""

    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls = []
        self.timeouts_seen = []

    async def create(self, **kw):
        self.calls.append(kw)
        self.timeouts_seen.append(kw.get("timeout"))
        b = self.behaviours.pop(0) if self.behaviours else "ok"
        if b == "timeout":
            raise TimeoutError("request timed out")
        if b == "hang":
            await asyncio.sleep(3600)      # would hang forever
        if b == "repeat":
            return _Resp("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        return _Resp(_GOOD_HTML)


class FakeClient:
    def __init__(self, behaviours):
        self.chat = type("C", (), {})()
        self.chat.completions = FakeCompletions(behaviours)


@pytest.fixture
def img():
    return Image.new("RGB", (64, 64), color=(255, 255, 255))


# ── The timeout is actually passed ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_timeout_is_sent_on_every_attempt(monkeypatch, img):
    client = FakeClient(["repeat", "repeat", "ok"])
    monkeypatch.setattr(chandra_ocr, "get_async_client", lambda: client)
    monkeypatch.setattr(chandra_ocr, "MAX_RETRIES", 3)
    monkeypatch.setattr(chandra_ocr, "REQUEST_TIMEOUT_SEC", 42.0)

    await chandra_ocr.ocr_image_async(img)

    assert client.chat.completions.timeouts_seen == [42.0, 42.0, 42.0]


# ── Recovery from a timeout ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timeout_advances_to_next_attempt(monkeypatch, img):
    """A timed-out attempt must not end the loop — the next attempt runs."""
    client = FakeClient(["timeout", "ok"])
    monkeypatch.setattr(chandra_ocr, "get_async_client", lambda: client)
    monkeypatch.setattr(chandra_ocr, "MAX_RETRIES", 3)

    entries = await chandra_ocr.ocr_image_async(img)

    assert len(client.chat.completions.calls) == 2
    assert entries, "should have recovered on the retry"


@pytest.mark.asyncio
async def test_all_attempts_timeout_returns_empty_not_hang(monkeypatch, img):
    """Exhausting the budget yields an empty layout so the DOCUMENT survives."""
    client = FakeClient(["timeout", "timeout", "timeout", "timeout"])
    monkeypatch.setattr(chandra_ocr, "get_async_client", lambda: client)
    monkeypatch.setattr(chandra_ocr, "MAX_RETRIES", 3)

    entries = await asyncio.wait_for(chandra_ocr.ocr_image_async(img), timeout=5)

    assert entries == []
    assert len(client.chat.completions.calls) == 4   # MAX_RETRIES + 1


@pytest.mark.asyncio
async def test_connection_error_is_also_survivable(monkeypatch, img):
    """Any exception, not just TimeoutError, must degrade rather than raise."""
    class Boom(FakeCompletions):
        async def create(self, **kw):
            self.calls.append(kw)
            raise ConnectionError("vllm went away")

    client = FakeClient([])
    client.chat.completions = Boom([])
    monkeypatch.setattr(chandra_ocr, "get_async_client", lambda: client)
    monkeypatch.setattr(chandra_ocr, "MAX_RETRIES", 2)

    entries = await chandra_ocr.ocr_image_async(img)
    assert entries == []


@pytest.mark.asyncio
async def test_partial_failure_still_uses_good_attempt(monkeypatch, img):
    """A timeout followed by a good response must return the good content,
    not the empty string left over from the failed attempt."""
    client = FakeClient(["timeout", "repeat", "ok"])
    monkeypatch.setattr(chandra_ocr, "get_async_client", lambda: client)
    monkeypatch.setattr(chandra_ocr, "MAX_RETRIES", 3)

    entries = await chandra_ocr.ocr_image_async(img)
    assert entries, "final good attempt should win"


# ── The regression this prevents ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_page_cannot_hang_forever(monkeypatch, img):
    """THE regression test: with no timeout a hanging call blocked the document
    indefinitely. The real client enforces the deadline, so here we assert the
    timeout value is handed to it on every call — if someone drops the kwarg,
    this fails."""
    client = FakeClient(["ok"])
    monkeypatch.setattr(chandra_ocr, "get_async_client", lambda: client)
    monkeypatch.setattr(chandra_ocr, "REQUEST_TIMEOUT_SEC", 300.0)

    await chandra_ocr.ocr_image_async(img)

    assert all(t is not None and t > 0
               for t in client.chat.completions.timeouts_seen), \
        "every generation call must carry a timeout"


def test_timeout_is_configurable():
    assert isinstance(chandra_ocr.REQUEST_TIMEOUT_SEC, float)
    assert chandra_ocr.REQUEST_TIMEOUT_SEC > 0
