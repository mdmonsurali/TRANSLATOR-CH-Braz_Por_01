"""5-phase upload → OCR → swap GPU → translate → restore pipeline.

The orchestrator does not write to Postgres or MinIO directly. It calls
ocr_service `/ocr` (Phase 1) and translator_service `/translate/from-history`
(Phase 3) over the docker network; those services own their respective
storage layers and write to the existing `documents` and `translations`
tables so the standard Task History UI picks up everything automatically.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid as _uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional

import httpx

import gpu_swap
from identity import Identity, identity_headers

log = logging.getLogger("orchestrator.pipeline")

OCR_SERVICE_URL = os.environ.get("OCR_SERVICE_URL", "http://ocr_service:8001")
TRANSLATOR_SERVICE_URL = os.environ.get(
    "TRANSLATOR_SERVICE_URL", "http://translator_service:8003",
)
TRANSLATE_TARGET_LANG = os.environ.get("TRANSLATE_TARGET_LANG", "pt-BR")
# Whole-document calls: these must cover the ENTIRE job, not one page. A
# 1000-page OCR pass or translation runs for hours, and the old 15/30-minute
# defaults aborted such jobs long before memory was ever the limit. Translation
# checkpoints every TRANSLATE_PAGE_WINDOW pages, so a timeout here now costs at
# most one window of re-work rather than the whole document.
OCR_TIMEOUT_SEC = float(os.environ.get("OCR_TIMEOUT_SEC", "21600"))        # 6h
TRANSLATE_TIMEOUT_SEC = float(os.environ.get("TRANSLATE_TIMEOUT_SEC", "43200"))  # 12h

SUPPORTED_EXTS = {".pdf", ".docx", ".jpg", ".jpeg", ".png"}

EmitFn = Callable[[str, dict], Awaitable[None]]


class PreparedFile:
    """Drained upload payload kept around for both phases."""
    __slots__ = ("index", "original_name", "content", "content_type", "early_error",
                 "document_id", "ocr_elapsed", "page_count")

    def __init__(self, index: int, original_name: str,
                  content: Optional[bytes], content_type: Optional[str],
                  early_error: Optional[str]):
        self.index = index
        self.original_name = original_name
        self.content = content
        self.content_type = content_type
        self.early_error = early_error
        self.document_id: Optional[str] = None
        self.ocr_elapsed: Optional[float] = None
        self.page_count: Optional[int] = None


# ── Phase 1: OCR ───

async def _call_ocr(client: httpx.AsyncClient, p: PreparedFile,
                     identity: Identity) -> dict:
    resp = await client.post(
        f"{OCR_SERVICE_URL}/ocr",
        files={"file": (p.original_name, p.content,
                        p.content_type or "application/octet-stream")},
        headers=identity_headers(identity),
        timeout=OCR_TIMEOUT_SEC,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"ocr_service returned {resp.status_code}: {resp.text[:500]}"
        )
    return resp.json()


async def _phase_ocr(prepared: List[PreparedFile], identity: Identity,
                      emit: EmitFn) -> None:
    """OCR every file with a non-early-error. Emits one `result` event per
    file with stage='ocr'. Updates the PreparedFile rows in place."""
    log.info("[orchestrator] OCR PHASE START: %d file(s)", len(prepared))
    await emit("status", {"phase": "ocr",
                           "message": f"Starting OCR for {len(prepared)} file(s)"})
    async with httpx.AsyncClient(timeout=OCR_TIMEOUT_SEC) as client:
        for p in prepared:
            if p.early_error:
                log.warning("[orchestrator] OCR SKIP %s: %s",
                            p.original_name, p.early_error)
                await emit("result", {
                    "index":         p.index,
                    "stage":         "ocr",
                    "status":        "error",
                    "original_name": p.original_name,
                    "error":         p.early_error,
                    "elapsed_sec":   0,
                })
                continue
            log.info("[orchestrator] OCR START  %s", p.original_name)
            t0 = time.perf_counter()
            try:
                row = await _call_ocr(client, p, identity)
                p.document_id = row.get("document_id")
                p.page_count = row.get("page_count")
                p.ocr_elapsed = row.get("elapsed_sec",
                                          round(time.perf_counter() - t0, 2))
                log.info(
                    "[orchestrator] OCR DONE   %s  doc_id=%s  pages=%s  %.2fs",
                    p.original_name, p.document_id, p.page_count, p.ocr_elapsed,
                )
                await emit("result", {
                    "index":         p.index,
                    "stage":         "ocr",
                    "status":        "ok",
                    "original_name": p.original_name,
                    "document_id":   p.document_id,
                    "page_count":    p.page_count,
                    "elapsed_sec":   p.ocr_elapsed,
                })
            except Exception as exc:
                log.error("[orchestrator] OCR FAIL   %s: %s",
                          p.original_name, exc)
                await emit("result", {
                    "index":         p.index,
                    "stage":         "ocr",
                    "status":        "error",
                    "original_name": p.original_name,
                    "error":         str(exc)[:500],
                    "elapsed_sec":   round(time.perf_counter() - t0, 2),
                })
    ok = sum(1 for p in prepared if p.document_id)
    log.info("[orchestrator] OCR PHASE END: %d/%d ok", ok, len(prepared))


# ── Phase 3: translate ───────

async def _call_translate(client: httpx.AsyncClient, doc_id: str,
                           identity: Identity, target_lang: str) -> dict:
    """Call translator_service `/translate/from-history` for a single document.

    Translator's SSE route streams `start`/`result`/`end`; for a single ID
    that's three frames. We collect the `result` frame and return its
    decoded JSON payload so the orchestrator can emit its own normalized
    result event.
    """
    payload = {"document_ids": [doc_id], "target_lang": target_lang}
    async with client.stream(
        "POST",
        f"{TRANSLATOR_SERVICE_URL}/translate/from-history",
        json=payload,
        headers=identity_headers(identity),
        timeout=TRANSLATE_TIMEOUT_SEC,
    ) as resp:
        if resp.status_code != 200:
            body = await resp.aread()
            raise RuntimeError(
                f"translator returned {resp.status_code}: "
                f"{body.decode('utf-8', 'replace')[:500]}"
            )
        result_payload: Optional[dict] = None
        buffer = ""
        import json as _json
        async for chunk in resp.aiter_text():
            buffer += chunk
            while True:
                sep = buffer.find("\n\n")
                if sep < 0:
                    break
                frame = buffer[:sep]
                buffer = buffer[sep + 2:]
                event_name = "message"
                data_lines: List[str] = []
                for line in frame.splitlines():
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                if not data_lines:
                    continue
                try:
                    obj = _json.loads("\n".join(data_lines))
                except Exception:
                    continue
                if event_name == "result":
                    result_payload = obj
                elif event_name == "error":
                    raise RuntimeError(
                        obj.get("detail") or "translator service error"
                    )
        if result_payload is None:
            raise RuntimeError("translator returned no result event")
        if result_payload.get("status") != "ok":
            raise RuntimeError(
                result_payload.get("error") or "translation failed"
            )
        return result_payload


async def _phase_translate(prepared: List[PreparedFile], identity: Identity,
                            target_lang: str, emit: EmitFn) -> None:
    """Translate every file that OCR'd successfully. Per-file SSE event with
    stage='translate'."""
    targets = [p for p in prepared if p.document_id]
    log.info("[orchestrator] TRANSLATE PHASE START: %d doc(s) -> %s",
             len(targets), target_lang)
    await emit("status", {
        "phase": "translate",
        "message": f"Translating {len(targets)} document(s) to {target_lang}",
    })
    ok = 0
    async with httpx.AsyncClient(timeout=TRANSLATE_TIMEOUT_SEC) as client:
        for p in targets:
            log.info("[orchestrator] TRANSLATE START  %s  doc_id=%s",
                     p.original_name, p.document_id)
            t0 = time.perf_counter()
            try:
                row = await _call_translate(client, p.document_id,
                                              identity, target_lang)
                elapsed = row.get("elapsed_sec",
                                   round(time.perf_counter() - t0, 2))
                log.info(
                    "[orchestrator] TRANSLATE DONE   %s  trans_id=%s  "
                    "items=%s  chunks=%s  failed=%s  %.2fs",
                    p.original_name, row.get("translation_id"),
                    row.get("items_translated"), row.get("chunks"),
                    row.get("failed"), elapsed,
                )
                ok += 1
                await emit("result", {
                    "index":            p.index,
                    "stage":            "translate",
                    "status":           "ok",
                    "original_name":    p.original_name,
                    "document_id":      p.document_id,
                    "translation_id":   row.get("translation_id"),
                    "target_lang":      row.get("target_lang", target_lang),
                    "items_translated": row.get("items_translated"),
                    "chunks":           row.get("chunks"),
                    "failed":           row.get("failed"),
                    "elapsed_sec":      elapsed,
                })
            except Exception as exc:
                log.error("[orchestrator] TRANSLATE FAIL   %s: %s",
                          p.original_name, exc)
                await emit("result", {
                    "index":         p.index,
                    "stage":         "translate",
                    "status":        "error",
                    "original_name": p.original_name,
                    "document_id":   p.document_id,
                    "error":         str(exc)[:500],
                    "elapsed_sec":   round(time.perf_counter() - t0, 2),
                })
    log.info("[orchestrator] TRANSLATE PHASE END: %d/%d ok",
             ok, len(targets))


# ── Top-level orchestration ──────

async def prepare_uploads(files) -> List[PreparedFile]:
    """Drain UploadFile bodies upfront so the SSE generator can keep going
    after the request body is consumed. Mirrors ocr_service's `prepared`
    pattern."""
    prepared: List[PreparedFile] = []
    for i, upload in enumerate(files):
        original = Path(upload.filename or f"upload_{i}").name
        ext = Path(original).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            prepared.append(PreparedFile(
                index=i, original_name=original,
                content=None, content_type=upload.content_type,
                early_error=f"Unsupported file type: {ext}",
            ))
            continue
        content = await upload.read()
        if not content:
            prepared.append(PreparedFile(
                index=i, original_name=original,
                content=None, content_type=upload.content_type,
                early_error="Empty upload",
            ))
            continue
        prepared.append(PreparedFile(
            index=i, original_name=original, content=content,
            content_type=upload.content_type, early_error=None,
        ))
    return prepared


async def run_pipeline(prepared: List[PreparedFile], identity: Identity,
                        target_lang: str, emit: EmitFn,
                        set_phase: Callable[[str], None]) -> dict:
    """Drive the five phases. Returns a final summary dict for the `end`
    event. The caller wraps this in an SSE StreamingResponse."""
    t_total = time.perf_counter()
    target_lang = (target_lang or TRANSLATE_TARGET_LANG).strip() or TRANSLATE_TARGET_LANG

    log.info("=" * 72)
    log.info("[orchestrator] BATCH START: %d file(s), target=%s, user=%s",
             len(prepared), target_lang, identity.id)
    log.info("=" * 72)

    await emit("start", {
        "total":       len(prepared),
        "target_lang": target_lang,
        "items": [
            {"index": p.index, "original_name": p.original_name, "status": "queued"}
            for p in prepared
        ],
    })

    # Phase 0: ensure OCR is the GPU tenant.
    set_phase("swap-to-ocr")
    log.info("[orchestrator] PHASE 0  swap-to-ocr (prep)")
    await gpu_swap.swap_to_ocr(emit)
    log.info("[orchestrator] PHASE 0  OK: ocr_service is the GPU tenant")

    # Phase 1: OCR every file.
    set_phase("ocr")
    log.info("[orchestrator] PHASE 1  ocr")
    await _phase_ocr(prepared, identity, emit)

    have_any_ocr = any(p.document_id for p in prepared)

    if have_any_ocr:
        # Phase 2: swap GPU to translator.
        set_phase("swap-to-translator")
        log.info("[orchestrator] PHASE 2  swap-to-translator "
                 "(stop ocr, start translator)")
        try:
            await gpu_swap.swap_to_translator(emit)
            log.info("[orchestrator] PHASE 2  OK: translator_service is "
                     "the GPU tenant")
        except Exception as exc:
            log.exception("[orchestrator] PHASE 2  FAIL: %s", exc)
            await emit("status", {
                "phase":   "swap-to-translator",
                "status":  "error",
                "message": f"GPU swap to translator failed: {exc}",
            })
            # Skip phase 3 and go straight to recovery.
            have_any_ocr = False

    if have_any_ocr:
        # Phase 3: translate each OCR'd doc one by one.
        set_phase("translate")
        log.info("[orchestrator] PHASE 3  translate")
        await _phase_translate(prepared, identity, target_lang, emit)

    # Phase 4: restore OCR mode for the next batch. Best-effort.
    set_phase("swap-to-ocr")
    log.info("[orchestrator] PHASE 4  swap-back (stop translator, "
             "restart ocr)")
    try:
        await emit("status", {
            "phase":   "swap-to-ocr",
            "message": "Restoring OCR mode for the next batch",
        })
        await gpu_swap.swap_to_ocr(emit)
        log.info("[orchestrator] PHASE 4  OK: ocr_service is back online")
    except Exception as exc:
        log.warning("[orchestrator] PHASE 4  WARN: swap-back failed: %s", exc)
        await emit("status", {
            "phase":   "swap-to-ocr",
            "status":  "warning",
            "message": f"Could not restore OCR mode automatically: {exc}",
        })

    set_phase("idle")

    ok_ocr = sum(1 for p in prepared if p.document_id)
    failed = sum(1 for p in prepared if not p.document_id)
    elapsed = round(time.perf_counter() - t_total, 2)
    log.info("=" * 72)
    log.info("[orchestrator] BATCH END: %d/%d ocr ok, %d failed, %.2fs total",
             ok_ocr, len(prepared), failed, elapsed)
    log.info("=" * 72)
    summary = {
        "total":             len(prepared),
        "ok_ocr":            ok_ocr,
        "failed":            failed,
        "total_elapsed_sec": round(time.perf_counter() - t_total, 2),
    }
    return summary
