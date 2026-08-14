"""translator_service — FastAPI entry point.

Routes (all require X-User-Id / X-User-Role identity headers, trusted
because this port is only reachable from the docker network):

    GET    /health                         postgres + minio + llama probe
    POST   /translate/batch                SSE multi-file pipeline
    POST   /translate/batch-zip            ZIP bundle of translated DOCX
    GET    /translations                   list owner-scoped rows
    GET    /translations/{id}              one row metadata
    GET    /translations/{id}/{kind}       stream translated_json|translated_docx
    DELETE /translations/{id}              owner-scoped delete

The /translate/batch SSE event shape mirrors ocr_service/ocr/batch so the
frontend can reuse the same parser.
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

from fastapi import Body, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

import storage
import llama_client
from content_disposition import content_disposition, safe_zip_name
from identity import Identity, require_user
from pipeline import (
    SUPPORTED_EXTS,
    TRANSLATE_TARGET_LANG,
    translate_from_doc_id,
    translate_one,
)

log = logging.getLogger("translator_service")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ARTIFACT_KINDS = {"translated_json", "translated_docx"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await storage.init_pool()
    storage.init_minio()
    await storage.init_schema()
    log.info("storage layer ready (postgres + minio)")
    yield
    await storage.close_pool()


app = FastAPI(title="Translator Service", version="1.0", lifespan=lifespan)


class _SilencePathsFilter(logging.Filter):
    """Drop uvicorn access lines for /health so the actual events show."""
    def __init__(self, paths: tuple[str, ...]) -> None:
        super().__init__()
        self._paths = paths

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(f' {p} ' in msg for p in self._paths)


logging.getLogger("uvicorn.access").addFilter(
    _SilencePathsFilter(("/health",))
)


def _assert_owner_or_404(row: dict, user_id: _uuid.UUID) -> None:
    if row.get("owner_id") != user_id:
        raise HTTPException(404, "Translation not found")


@app.get("/health")
async def health():
    if not await llama_client.health():
        raise HTTPException(503, "llama.cpp not reachable")
    try:
        await storage.health_postgres()
    except Exception as e:
        raise HTTPException(503, f"postgres not reachable: {e}")
    try:
        storage.health_minio()
    except Exception as e:
        raise HTTPException(503, f"minio not reachable: {e}")
    return {"status": "ok", "llama": "ok", "postgres": "ok", "minio": "ok"}


def _sse(event: str, payload: dict) -> str:
    return (f"event: {event}\n"
            f"data: {_json.dumps(payload, ensure_ascii=False, default=str)}\n\n")


@app.post("/translate/batch")
async def translate_batch(files: List[UploadFile] = File(...),
                           target_lang: Optional[str] = None,
                           identity: Identity = Depends(require_user)):
    if not files:
        raise HTTPException(400, "No files uploaded")

    lang = (target_lang or TRANSLATE_TARGET_LANG).strip() or TRANSLATE_TARGET_LANG

    # Drain the request body now — once StreamingResponse starts the upload
    # stream is gone. Mirrors the same pattern in ocr_service.
    prepared: list[dict] = []
    for upload in files:
        original = Path(upload.filename or "upload").name
        ext = Path(original).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            prepared.append({"original_name": original, "ext": ext,
                             "content": None, "content_type": upload.content_type,
                             "early_error": f"Unsupported file type: {ext}"})
            continue
        content = await upload.read()
        if not content:
            prepared.append({"original_name": original, "ext": ext,
                             "content": None, "content_type": upload.content_type,
                             "early_error": "Empty upload"})
            continue
        prepared.append({"original_name": original, "ext": ext,
                         "content": content, "content_type": upload.content_type,
                         "early_error": None})

    async def event_stream():
        t0 = time.perf_counter()
        items = [
            {"index": i, "original_name": p["original_name"], "status": "queued"}
            for i, p in enumerate(prepared)
        ]
        yield _sse("start", {"total": len(prepared), "items": items,
                              "target_lang": lang})

        for i, p in enumerate(prepared):
            if p["early_error"]:
                yield _sse("result", {
                    "index":         i,
                    "original_name": p["original_name"],
                    "status":        "error",
                    "error":         p["early_error"],
                    "elapsed_sec":   0,
                })
                continue
            try:
                row = await translate_one(
                    content=p["content"],
                    original_name=p["original_name"],
                    content_type=p["content_type"],
                    user_id=identity.id,
                    user_role=identity.role,
                    target_lang=lang,
                )
                row["index"] = i
                yield _sse("result", row)
            except HTTPException as exc:
                yield _sse("result", {
                    "index":         i,
                    "original_name": p["original_name"],
                    "status":        "error",
                    "error":         exc.detail,
                    "elapsed_sec":   0,
                })
            except Exception as exc:
                log.exception("[batch] translation failed for %s",
                              p["original_name"])
                yield _sse("result", {
                    "index":         i,
                    "original_name": p["original_name"],
                    "status":        "error",
                    "error":         f"Translation failed: {exc}",
                    "elapsed_sec":   0,
                })

        yield _sse("end", {
            "total":             len(prepared),
            "total_elapsed_sec": round(time.perf_counter() - t0, 2),
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/translator-documents")
async def translator_documents(limit: int = 200, offset: int = 0,
                                identity: Identity = Depends(require_user)):
    """List OCR'd documents the caller can pick for translation.

    Read straight from postgres (not via ocr_service) so this works in
    swap mode where ocr_service is stopped.
    """
    rows = await storage.list_ocr_documents(
        owner_id=identity.id, limit=limit, offset=offset,
    )
    total = await storage.count_ocr_documents(owner_id=identity.id)
    # Stringify UUIDs/timestamps for JSON.
    return JSONResponse(
        content=_json.loads(_json.dumps(rows, default=str)),
        headers={"X-Total-Count": str(total)},
    )


@app.post("/translate/from-history")
async def translate_from_history(payload: dict = Body(...),
                                  identity: Identity = Depends(require_user)):
    """Translate a list of already-OCR'd documents picked from history.
    Body: {"document_ids": ["uuid", ...], "target_lang": "pt-BR"}.

    Designed for swap-mode: ocr_service may be stopped while this runs;
    we read layout JSON + pictures directly from MinIO.
    """
    raw_ids = payload.get("document_ids") or []
    if not isinstance(raw_ids, list) or not raw_ids:
        raise HTTPException(400, "Body must include non-empty document_ids list")
    lang = ((payload.get("target_lang") or TRANSLATE_TARGET_LANG).strip()
            or TRANSLATE_TARGET_LANG)

    # Validate UUIDs upfront. Bad ids become per-item errors, not a 400 that
    # tanks the whole batch.
    prepared: list[dict] = []
    for raw in raw_ids:
        try:
            doc_id = _uuid.UUID(str(raw))
            prepared.append({"doc_id": doc_id, "raw": str(raw),
                             "early_error": None})
        except (ValueError, TypeError):
            prepared.append({"doc_id": None, "raw": str(raw),
                             "early_error": "Invalid document id"})

    async def event_stream():
        t0 = time.perf_counter()
        items = [
            {"index": i, "document_id": p["raw"], "status": "queued"}
            for i, p in enumerate(prepared)
        ]
        yield _sse("start", {"total": len(prepared), "items": items,
                              "target_lang": lang})

        for i, p in enumerate(prepared):
            if p["early_error"]:
                yield _sse("result", {
                    "index":       i,
                    "document_id": p["raw"],
                    "status":      "error",
                    "error":       p["early_error"],
                    "elapsed_sec": 0,
                })
                continue
            try:
                row = await translate_from_doc_id(
                    doc_id=p["doc_id"],
                    user_id=identity.id,
                    target_lang=lang,
                )
                row["index"] = i
                yield _sse("result", row)
            except HTTPException as exc:
                yield _sse("result", {
                    "index":       i,
                    "document_id": p["raw"],
                    "status":      "error",
                    "error":       exc.detail,
                    "elapsed_sec": 0,
                })
            except Exception as exc:
                log.exception("[from-history] failed for %s", p["raw"])
                yield _sse("result", {
                    "index":       i,
                    "document_id": p["raw"],
                    "status":      "error",
                    "error":       f"Translation failed: {exc}",
                    "elapsed_sec": 0,
                })

        yield _sse("end", {
            "total":             len(prepared),
            "total_elapsed_sec": round(time.perf_counter() - t0, 2),
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/translate/batch-zip")
async def translate_batch_zip(payload: dict = Body(...),
                               identity: Identity = Depends(require_user)):
    ids = payload.get("ids") or []
    if not isinstance(ids, list) or not ids:
        raise HTTPException(400, "Body must be {\"ids\": [...]}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for raw in ids:
            try:
                trans_id = _uuid.UUID(str(raw))
            except (ValueError, TypeError):
                continue
            row = await storage.fetch_translation(trans_id)
            if not row or row.get("status") != "ok":
                continue
            if row.get("owner_id") != identity.id:
                continue
            stem = safe_zip_name(Path(row["original_name"]).stem, str(trans_id))
            for kind in ("translated_json", "translated_docx"):
                try:
                    fname, _ctype, body = await storage.get_artifact_bytes(
                        trans_id, kind,
                    )
                except (KeyError, FileNotFoundError):
                    continue
                zf.writestr(f"{stem}/{fname}", body)
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition":
                'attachment; filename="Translations.zip"',
        },
    )


@app.get("/translations")
async def list_translations(limit: int = 50, offset: int = 0,
                            status: Optional[str] = None,
                            identity: Identity = Depends(require_user)):
    rows = await storage.list_translations(
        limit=limit, offset=offset,
        owner_id=identity.id, status=status,
    )
    total = await storage.count_translations(
        owner_id=identity.id, status=status,
    )
    return JSONResponse(
        content=rows,
        headers={"X-Total-Count": str(total)},
    )


@app.get("/translations/{trans_id}")
async def get_translation(trans_id: str,
                          identity: Identity = Depends(require_user)):
    try:
        tid = _uuid.UUID(trans_id)
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid translation id")
    row = await storage.fetch_translation(tid)
    if row is None:
        raise HTTPException(404, "Translation not found")
    _assert_owner_or_404(row, identity.id)
    # asyncpg returns UUID/datetime objects; let JSONResponse stringify them.
    return JSONResponse(content=_json.loads(
        _json.dumps(row, default=str)
    ))


@app.get("/translations/{trans_id}/{kind}")
async def get_translation_artifact(trans_id: str, kind: str,
                                    identity: Identity = Depends(require_user)):
    if kind not in ARTIFACT_KINDS:
        raise HTTPException(400, f"Unknown kind: {kind}")
    try:
        tid = _uuid.UUID(trans_id)
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid translation id")
    row = await storage.fetch_translation(tid)
    if row is None:
        raise HTTPException(404, "Translation not found")
    _assert_owner_or_404(row, identity.id)
    try:
        filename, content_type, body = await storage.get_artifact_bytes(tid, kind)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except KeyError:
        raise HTTPException(404, "Translation not found")

    disposition = ("attachment" if kind == "translated_docx" else "inline")
    return StreamingResponse(
        iter([body]),
        media_type=content_type,
        headers={
            "Content-Disposition": content_disposition(
                disposition, filename, str(tid),
            ),
        },
    )


@app.delete("/translations/{trans_id}")
async def delete_translation(trans_id: str,
                              identity: Identity = Depends(require_user)):
    try:
        tid = _uuid.UUID(trans_id)
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid translation id")
    ok = await storage.delete_translation(tid, identity.id)
    if not ok:
        raise HTTPException(404, "Translation not found")
    return Response(status_code=204)
