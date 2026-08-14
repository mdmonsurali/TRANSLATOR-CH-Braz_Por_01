"""orchestrator_service — FastAPI entry point.

Routes:
    GET  /health                              docker socket + downstream probes
    GET  /status                              current swap state + active phase
    POST /run/batch                           SSE pipeline (upload → OCR → swap → translate → restore)
    GET  /translations/{id}/{kind}            stream translated_json | translated_docx
    GET  /translations/{id}/preview           DOCX → PDF via ocr_service /preview-input
    POST /translate/batch-zip                 ZIP bundle of multiple translations

The artifact reads are owned here (rather than translator_service) so users
can download finished translations even when translator_service has been
stopped between batches.
"""

from __future__ import annotations

import asyncio
import io
import json as _json
import logging
import os
import time
import uuid as _uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

import gpu_swap
import pipeline
import storage
from content_disposition import content_disposition, safe_zip_name
from identity import Identity, identity_headers, require_user

log = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

OCR_SERVICE_URL = os.environ.get("OCR_SERVICE_URL", "http://ocr_service:8001")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await storage.init_pool()
    storage.init_minio()
    log.info("storage layer ready (postgres + minio)")
    # Make OCR the default GPU tenant on boot: stop translator if running,
    # start OCR if not, wait for its healthcheck. Without this, a stale
    # translator container from a prior run can sit on enough VRAM to
    # prevent vLLM from claiming its share.
    try:
        await gpu_swap.swap_to_ocr()
    except Exception:
        log.exception("startup swap_to_ocr failed; continuing so /status "
                      "and /health remain reachable")
    yield
    await storage.close_pool()


app = FastAPI(title="Orchestrator Service", version="1.0", lifespan=lifespan)


class _SilencePathsFilter(logging.Filter):
    """Drop uvicorn access lines for noisy poll endpoints so real events
    aren't drowned. Match the path inside the formatted access message."""
    def __init__(self, paths: tuple[str, ...]) -> None:
        super().__init__()
        self._paths = paths

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(f' {p} ' in msg for p in self._paths)


logging.getLogger("uvicorn.access").addFilter(
    _SilencePathsFilter(("/health", "/status"))
)

# Single-flight: only one batch at a time. The lock is held across the full
# pipeline so two callers can't fight over the GPU.
_batch_lock = asyncio.Lock()
_current_phase = "idle"


def _set_phase(phase: str) -> None:
    global _current_phase
    _current_phase = phase


@app.get("/health")
async def health():
    if not gpu_swap.docker_reachable():
        raise HTTPException(503, "docker socket not reachable")
    return {"status": "ok", "docker": "ok",
            "gpu_swap_enabled": gpu_swap.GPU_SWAP_ENABLED}


@app.get("/status")
async def status(identity: Identity = Depends(require_user)):
    snap = await gpu_swap.snapshot()
    snap["current_phase"] = _current_phase
    snap["batch_in_flight"] = _batch_lock.locked()
    return snap


def _sse(event: str, payload: dict) -> bytes:
    return (
        f"event: {event}\n"
        f"data: {_json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
    ).encode("utf-8")


@app.post("/run/batch")
async def run_batch(files: List[UploadFile] = File(...),
                     target_lang: Optional[str] = Form(default=None),
                     identity: Identity = Depends(require_user)):
    if not files:
        raise HTTPException(400, "No files uploaded")

    if _batch_lock.locked():
        raise HTTPException(
            409,
            "Another batch is already running. Wait for it to finish "
            "(or check /status).",
        )

    # Drain uploads while we still own the request body.
    prepared = await pipeline.prepare_uploads(files)
    if not prepared:
        raise HTTPException(400, "No valid files")

    async def event_stream():
        out_q: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        async def emit(event: str, payload: dict) -> None:
            await out_q.put((event, payload))

        async def run_and_finish():
            try:
                async with _batch_lock:
                    summary = await pipeline.run_pipeline(
                        prepared=prepared,
                        identity=identity,
                        target_lang=target_lang or os.environ.get(
                            "TRANSLATE_TARGET_LANG", "pt-BR",
                        ),
                        emit=emit,
                        set_phase=_set_phase,
                    )
                await emit("end", summary)
            except Exception as exc:
                log.exception("pipeline crashed")
                await emit("error", {"detail": str(exc)[:500]})
            finally:
                _set_phase("idle")
                await out_q.put(SENTINEL)

        worker = asyncio.create_task(run_and_finish())
        try:
            while True:
                item = await out_q.get()
                if item is SENTINEL:
                    return
                event, payload = item
                yield _sse(event, payload)
        finally:
            if not worker.done():
                worker.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Artifact serving ───
# Lets users download translated DOCX / JSON / preview while
# translator_service is stopped between batches.

_ARTIFACT_KINDS = {"translated_json", "translated_docx"}


def _parse_trans_id(raw: str) -> _uuid.UUID:
    try:
        return _uuid.UUID(raw)
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid translation id")


@app.post("/translations-by-source")
async def translations_by_source(payload: dict,
                                  identity: Identity = Depends(require_user)):
    """Body: {"document_ids": ["uuid", ...]}. Returns a compact map
    {source_document_id: {translation_id, target_lang, ...}} for each
    document in the list that has a successful translation owned by the
    caller. Used by Task History to render translation-download buttons.
    """
    raw_ids = payload.get("document_ids") or []
    if not isinstance(raw_ids, list):
        raise HTTPException(400, "document_ids must be a list")
    doc_uuids: list[_uuid.UUID] = []
    for raw in raw_ids:
        try:
            doc_uuids.append(_uuid.UUID(str(raw)))
        except (ValueError, TypeError):
            continue  # silently drop bad ids — partial result is fine
    mapping = await storage.translations_by_source(identity.id, doc_uuids)
    return JSONResponse(content=mapping)


@app.post("/translate/batch-zip")
async def translate_batch_zip(payload: dict = Body(...),
                               identity: Identity = Depends(require_user)):
    """ZIP of N translations. Folder layout matches /ocr/batch-zip but renamed
    so users get a self-describing archive:

        Translate_result/<original_stem>/translate_<original_stem>.docx
        Translate_result/<original_stem>/translate_<original_stem>.json

    Hosted here (not translator_service) because translator_service is
    stopped between batches in GPU-swap mode.
    """
    ids = payload.get("ids") or []
    if not isinstance(ids, list) or not ids:
        raise HTTPException(400, "Body must be {\"ids\": [...]}")

    buf = io.BytesIO()
    written = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for raw in ids:
            try:
                trans_id = _uuid.UUID(str(raw))
            except (ValueError, TypeError):
                continue
            row = await storage.fetch_translation(trans_id, identity.id)
            if not row or row.get("status") != "ok":
                continue
            stem = safe_zip_name(
                Path(row.get("original_name") or str(trans_id)).stem,
                str(trans_id),
            )
            for kind, ext in (("translated_docx", "docx"),
                              ("translated_json", "json")):
                try:
                    _fname, _ctype, body = await storage.get_artifact_bytes(
                        row, kind,
                    )
                except (KeyError, FileNotFoundError):
                    continue
                zf.writestr(
                    f"Translate_result/{stem}/translate_{stem}.{ext}",
                    body,
                )
                written += 1

    if written == 0:
        raise HTTPException(404, "No translated artifacts found for the given ids")

    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition":
                'attachment; filename="Translate_result.zip"',
        },
    )


# NOTE: declaration order matters here. FastAPI matches routes in the order
# they are registered; the literal "/preview" path must be declared BEFORE
# the more general "/{kind}" path or every request to /preview is captured
# by the {kind} route and rejected with 400 "Unknown kind: preview".

@app.get("/translations/{trans_id}/preview")
async def preview_translation(trans_id: str,
                               identity: Identity = Depends(require_user)):
    """DOCX → PDF preview. Fetches the translated DOCX from MinIO ourselves,
    then forwards it to ocr_service /preview-input (which runs LibreOffice).
    Avoids needing translator_service to be up."""
    tid = _parse_trans_id(trans_id)
    row = await storage.fetch_translation(tid, identity.id)
    if row is None:
        raise HTTPException(404, "Translation not found")
    try:
        filename, _ctype, docx_bytes = await storage.get_artifact_bytes(
            row, "translated_docx",
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    async with httpx.AsyncClient(timeout=600) as client:
        try:
            pdf_resp = await client.post(
                f"{OCR_SERVICE_URL}/preview-input",
                files={"file": (
                    filename, docx_bytes,
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document",
                )},
                headers=identity_headers(identity),
            )
        except httpx.ConnectError:
            raise HTTPException(503, "OCR service is unavailable for preview")
    if pdf_resp.status_code != 200:
        raise HTTPException(pdf_resp.status_code, pdf_resp.text)
    return Response(
        content=pdf_resp.content,
        media_type="application/pdf",
    )


@app.get("/translations/{trans_id}/{kind}")
async def get_translation_artifact(trans_id: str, kind: str,
                                    identity: Identity = Depends(require_user)):
    if kind not in _ARTIFACT_KINDS:
        raise HTTPException(400, f"Unknown kind: {kind}")
    tid = _parse_trans_id(trans_id)
    row = await storage.fetch_translation(tid, identity.id)
    if row is None:
        raise HTTPException(404, "Translation not found")
    try:
        filename, content_type, body = await storage.get_artifact_bytes(row, kind)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    disposition = "attachment" if kind == "translated_docx" else "inline"
    return Response(
        content=body,
        media_type=content_type,
        headers={
            "Content-Disposition": content_disposition(
                disposition, filename, str(row.get("id") or trans_id),
            ),
        },
    )
