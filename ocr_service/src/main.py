"""Chandra-OCR ocr_service — UUID-based persistence via Postgres + MinIO.

Exposes:
    GET    /health                              upstream + Postgres + MinIO probe
    POST   /ocr           (file)                single-file pipeline, returns {document_id}
    POST   /ocr/batch     (files[])             SSE batch pipeline, one event per file
    POST   /ocr/batch-zip ({"ids":[uuid,...]})  ZIP bundle of N documents
    GET    /documents     ?limit&offset&status  list documents (no bytes)
    GET    /documents/{id}                      JSON row (no bytes)
    GET    /documents/{id}/{kind}               stream source|md|json|docx from MinIO
    GET    /preview/{id}                        DOCX -> PDF via LibreOffice
    POST   /preview-input (file)                DOCX upload -> PDF preview

Bytes live in MinIO at documents/{uuid}/{source.ext,output.md,layout.json,output.docx}.
Metadata lives in Postgres `documents` table.
"""

from __future__ import annotations

import asyncio
import io
import json as _json
import logging
import os
import subprocess
import tempfile
import time
import uuid as _uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import httpx
from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from PIL import Image

from doc_processing import (
    extract_font_spans,
    layoutjson2md,
    load_pages_from_docx,
    load_pages_from_image,
    load_pages_from_pdf,
)
from ocr_reconstruction import json_to_docx, process_pictures
from chandra_ocr import process_image_async, PICTURE_LABELS
from chandra_style import attribute_page
from picture_recovery import recover_missing_pictures
from cell_picture_recovery import recover_table_cell_pictures
from table_reocr import recover_dropped_table_rows
from checkbox_reocr import recover_dropped_checkboxes
from diagram_text_recovery import recover_diagram_text

import storage

log = logging.getLogger("ocr_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

VLLM_URL = os.environ.get("VLLM_SERVICE_URL", "http://localhost:8888")
PREVIEW_TIMEOUT_SEC = int(os.environ.get("PREVIEW_TIMEOUT_SEC", "360"))
SUPPORTED_EXTS = {".pdf", ".docx", ".jpg", ".jpeg", ".png"}
PREVIEW_INPUT_EXTS = {".docx", ".doc", ".odt", ".rtf"}

BATCH_SIZE_NATIVE = int(os.getenv("OCR_BATCH_SIZE_NATIVE", "16"))
BATCH_SIZE_SCANNED = int(os.getenv("OCR_BATCH_SIZE_SCANNED", "4"))
BATCH_SIZE_DEFAULT = int(os.getenv("OCR_BATCH_SIZE", "2"))

# Chandra emits bboxes normalized 0-1000; chandra_ocr rescales to page pixels.
INCLUDE_PICTURES = os.getenv("INCLUDE_PICTURES", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
ATTRIBUTE_STYLES = os.getenv("ATTRIBUTE_STYLES", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
# Re-OCR a table's crop when Chandra dropped a row+image on the full-page pass
# (embedded rasters inside the table exceed its <img> cells). Default on; issues
# extra model calls only for the affected tables.
REOCR_DROPPED_ROWS = os.getenv("OCR_REOCR_DROPPED_ROWS", "true").strip().lower() in {
    "1", "true", "yes", "on",
}
# Re-OCR a standalone form block's crop when Chandra transcribed its option
# labels but dropped the checkboxes (empty boxes lost on the downscaled full
# page). Default on; issues extra model calls only for blocks that look like
# they lost a box. Works on scanned pages too (uses only the page raster).
REOCR_DROPPED_CHECKBOXES = os.getenv(
    "OCR_REOCR_DROPPED_CHECKBOXES", "true").strip().lower() in {
    "1", "true", "yes", "on",
}

ARTIFACT_KINDS = {"source", "md", "json", "docx"}


# Identity — trusted from ui_service over the docker network. Drop the
# host port mapping for 8001 in production so this header can't be forged.

class Identity:
    __slots__ = ("id", "role")

    def __init__(self, id: _uuid.UUID, role: str):
        self.id = id
        self.role = role


def require_user(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> Identity:
    if not x_user_id or not x_user_role:
        raise HTTPException(401, "Missing identity headers")
    try:
        uid = _uuid.UUID(x_user_id)
    except (ValueError, TypeError):
        raise HTTPException(401, "Invalid X-User-Id")
    if x_user_role not in {"user", "admin", "master"}:
        raise HTTPException(401, "Invalid X-User-Role")
    return Identity(id=uid, role=x_user_role)


def _assert_owner_or_404(doc: dict, user_id: _uuid.UUID) -> None:
    """Use 404 (not 403) so we don't leak existence of other users' docs."""
    if doc.get("owner_id") != user_id:
        raise HTTPException(404, "Document not found")


# Lifespan: Postgres pool + MinIO client + schema

@asynccontextmanager
async def lifespan(app: FastAPI):
    await storage.init_pool()
    storage.init_minio()
    await storage.init_schema()
    log.info("storage layer ready (postgres + minio)")
    yield
    await storage.close_pool()


app = FastAPI(title="Chandra-OCR Service", version="2.0", lifespan=lifespan)


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


# Pipeline helpers (image extraction + async vLLM fan-out)

def _pages_from_path(in_path: Path) -> List[dict]:
    """One geometry-aware page dict per page for any supported input type.

    Each dict has: image, page_width_pt, page_height_pt, zoom, page_index,
    pdf_source (None for raw image inputs)."""
    ext = in_path.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg"}:
        return load_pages_from_image(str(in_path))
    if ext == ".pdf":
        return load_pages_from_pdf(str(in_path))
    if ext == ".docx":
        return load_pages_from_docx(str(in_path))
    raise HTTPException(415, f"Unsupported file type: {ext}")


def _detect_scan_type(in_path: Path) -> str:
    """Return 'native' if file has extractable text, else 'scanned'."""
    import fitz  # PyMuPDF, local import keeps cold start tight
    ext = in_path.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg"}:
        return "scanned"
    if ext == ".docx":
        return "native"
    if ext == ".pdf":
        try:
            with fitz.open(str(in_path)) as doc:
                for page in doc:
                    if len(page.get_text("text").strip()) >= 32:
                        return "native"
            return "scanned"
        except Exception as e:
            log.warning("scan-type probe failed for %s: %s", in_path.name, e)
            return "scanned"
    return "scanned"


def _batch_size_for(scan_type: str) -> int:
    return {
        "native":  BATCH_SIZE_NATIVE,
        "scanned": BATCH_SIZE_SCANNED,
    }.get(scan_type, BATCH_SIZE_DEFAULT)


async def _ocr_one_page_async(page_meta: dict, sem: asyncio.Semaphore) -> dict:
    """Run OCR for one page and merge the layout result into the geometry dict."""
    img = page_meta["image"]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name, "PNG")
        tmp_path = tmp.name
    try:
        async with sem:
            try:
                layout = await process_image_async(tmp_path)
            except Exception as exc:
                # One bad page must not kill the whole document. Log and
                # return an empty layout so the rest of the pages still
                # contribute to the markdown/JSON/DOCX outputs.
                log.warning("[ocr] page failed, returning empty layout: %s", exc)
                layout = []
    finally:
        os.unlink(tmp_path)
    page = {
        **page_meta,
        "original_image": img,
        "layout_result": layout,
        "markdown_content": "",
    }
    return page


async def _ocr_pages_concurrently(pages_meta: List[dict], batch_size: int) -> List[dict]:
    sem = asyncio.Semaphore(max(1, batch_size))
    tasks = [_ocr_one_page_async(p, sem) for p in pages_meta]
    return await asyncio.gather(*tasks)


def _attribute_styles(pages: List[dict]) -> None:
    """For each page, extract PDF spans (if any) and attach a `style` dict
    to every layout entry. Mutates pages in place."""
    for page in pages:
        entries = page.get("layout_result") or []
        spans = extract_font_spans(page.get("pdf_source"), page.get("page_index", 0))
        attribute_page(entries, spans, float(page.get("zoom") or 1.0))


_NON_JSON_ENTRY_FIELDS = {
    "image_obj",      # PIL.Image — not JSON-serializable
    "_table_chain",   # back-reference into other entries; would loop
    "_shared_col_weights",  # rebuilt on the next reconstruction pass
    "_html",          # raw block HTML kept only for style attribution
}


def _clean_entry_for_json(entry: dict) -> dict:
    """Strip non-serializable / non-JSON-safe fields from a layout entry."""
    return {k: v for k, v in entry.items() if k not in _NON_JSON_ENTRY_FIELDS}


def _page_envelope_for_json(page: dict) -> dict:
    """New per-page JSON shape: page-level geometry + entries with style."""
    return {
        "page_index": page.get("page_index", 0),
        "page_width_pt": page.get("page_width_pt"),
        "page_height_pt": page.get("page_height_pt"),
        "zoom": page.get("zoom"),
        "entries": [_clean_entry_for_json(e) for e in page.get("layout_result") or []],
    }


async def _upload_picture_assets(doc_id: _uuid.UUID, pages: List[dict]) -> List[dict]:
    """PNG-encode each Picture crop, upload to MinIO, and patch the entry
    with image_key + relative image_url. Mutates pages in place."""
    tasks: list = []
    for page in pages:
        for entry in page.get("layout_result", []):
            if entry.get("category") not in PICTURE_LABELS:
                continue
            img = entry.get("image_obj")
            fname = entry.get("image_filename")
            if img is None or not fname:
                continue
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            async def _go(e=entry, f=fname, b=png_bytes):
                key = await storage.put_image(doc_id, f, b)
                e["image_key"] = key
                e["image_url"] = f"images/{f}"

            tasks.append(_go())
    if tasks:
        await asyncio.gather(*tasks)
    return pages


# Core pipeline — runs against an on-disk temp file, persists to MinIO + DB

async def _run_pipeline(in_path: Path, source_bytes: bytes, original_name: str,
                         owner_id: _uuid.UUID) -> dict:
    """Process one document end-to-end:
       1. INSERT row with status=queued, PUT source to MinIO.
       2. Render pages → OCR via vLLM → markdown / layout JSON / DOCX bytes.
       3. PUT three artifacts to MinIO in parallel; UPDATE row to ok.

    On failure the row is left with status=error and the original exception
    is re-raised so the caller can surface it (single mode) or convert it to
    an SSE result event (batch mode).
    """
    ext = in_path.suffix.lower()
    doc_id = await storage.insert_document(original_name, ext, source_bytes, owner_id)

    try:
        await storage.update_document(doc_id, status="processing")
        t0 = time.perf_counter()

        scan_type = _detect_scan_type(in_path)
        batch_size = _batch_size_for(scan_type)
        pages_meta = _pages_from_path(in_path)
        log.info("[ocr] %s id=%s: %d page(s), type=%s, batch_size=%d",
                 original_name, doc_id, len(pages_meta), scan_type, batch_size)

        pages = await _ocr_pages_concurrently(pages_meta, batch_size)

        # Recover checkboxes Chandra dropped on standalone form blocks
        # (transcribed the option labels as bare text, no <input>). Re-OCRs just
        # the block crop where the boxes likely got lost. Independent of
        # pictures and works on scanned pages, so it runs unconditionally here.
        if REOCR_DROPPED_CHECKBOXES:
            await recover_dropped_checkboxes(pages)

        # Recover Picture entries the VLM dropped from inside tables. Only
        # runs for native PDFs (image inputs / scans have no embedded raster
        # info to recover from) and only injects pictures whose PDF bbox
        # sits inside an already-detected Table bbox.
        if INCLUDE_PICTURES:
            # First repair any table where Chandra dropped a row+image on the
            # full-page pass (embedded rasters > <img> cells): re-OCR the table
            # crop and swap in the fuller HTML. Runs BEFORE recovery so the
            # recovered pictures map onto the corrected grid.
            if REOCR_DROPPED_ROWS:
                await recover_dropped_table_rows(pages)
            recover_missing_pictures(pages)
            # Recover photos Chandra emitted as bare <img alt> placeholders in
            # table cells (scanned pages / image inputs have no embedded raster
            # for recover_missing_pictures to find). Crops them from the page
            # raster by content-blob detection down each image column.
            recover_table_cell_pictures(pages)
            # Recover the TEXT LABELS inside each recovered diagram (mermaid loses
            # them) by re-OCRing the crop, so the boxes/diamonds/edge labels can
            # be translated and overlaid in place instead of staying in the source
            # language baked into the raster.
            await recover_diagram_text(pages)

        if not INCLUDE_PICTURES:
            for page in pages:
                kept = []
                for e in page["layout_result"]:
                    if e.get("category") not in PICTURE_LABELS:
                        kept.append(e)
                    elif e.get("source") == "diagram-recovered":
                        # A Diagram we relabelled to Image for cropping: with
                        # pictures off, revert to a Diagram text block so the
                        # mermaid content still renders instead of vanishing.
                        e["category"] = "Diagram"
                        e.pop("image_obj", None)
                        kept.append(e)
                    # else: a real picture — drop it (pictures are off).
                page["layout_result"] = kept

        if ATTRIBUTE_STYLES:
            _attribute_styles(pages)

        # Crop + upload pictures first so each entry carries image_url before
        # the markdown serializer runs — that lets the markdown reference the
        # stored crop (images/{filename}) instead of a static placeholder.
        if INCLUDE_PICTURES:
            pages = process_pictures(pages)
            pages = await _upload_picture_assets(doc_id, pages)

        for page in pages:
            page["markdown_content"] = layoutjson2md(page["original_image"], page["layout_result"])

        md_text = "\n\n".join(p["markdown_content"] for p in pages)
        layout_json = [_page_envelope_for_json(p) for p in pages]
        md_bytes = md_text.encode("utf-8")
        json_bytes = _json.dumps(layout_json, indent=2, ensure_ascii=False).encode("utf-8")

        docx_buf = io.BytesIO()
        json_to_docx(pages, output_path=docx_buf)
        docx_bytes = docx_buf.getvalue()

        md_key, json_key, docx_key = await asyncio.gather(
            storage.put_artifact(doc_id, "md",   md_bytes),
            storage.put_artifact(doc_id, "json", json_bytes),
            storage.put_artifact(doc_id, "docx", docx_bytes),
        )

        elapsed = round(time.perf_counter() - t0, 2)
        await storage.update_document(
            doc_id,
            status="ok",
            page_count=len(pages),
            scan_type=scan_type,
            markdown_key=md_key,
            json_key=json_key,
            docx_key=docx_key,
            layout=layout_json,
            elapsed_sec=elapsed,
            error=None,
        )

        return {
            "document_id":   str(doc_id),
            "original_name": original_name,
            "status":        "ok",
            "scan_type":     scan_type,
            "page_count":    len(pages),
            "elapsed_sec":   elapsed,
        }
    except Exception as exc:
        await storage.update_document(doc_id, status="error", error=str(exc))
        raise


# DOCX → PDF preview helper (LibreOffice)

def _convert_to_pdf(src: Path, outdir: Path, timeout: int) -> Path:
    subprocess.run(["pkill", "-9", "-f", "soffice"], check=False, capture_output=True)
    try:
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf",
             "--outdir", str(outdir), str(src)],
            check=True, capture_output=True, timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        log.error("LibreOffice failed: %s", exc.stderr.decode(errors="replace"))
        raise HTTPException(500, "Document -> PDF conversion failed")
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-f", "soffice"], check=False, capture_output=True)
        raise HTTPException(504, "Document -> PDF conversion timed out")

    pdf_path = outdir / (src.stem + ".pdf")
    if not pdf_path.exists():
        raise HTTPException(500, "Converted PDF not found")
    return pdf_path


# Routes

@app.get("/health")
async def health():
    # vLLM upstream
    try:
        r = httpx.get(f"{VLLM_URL}/health", timeout=2.0)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(503, f"vLLM not reachable: {e}")
    # Postgres
    try:
        await storage.health_postgres()
    except Exception as e:
        raise HTTPException(503, f"postgres not reachable: {e}")
    # MinIO
    try:
        storage.health_minio()
    except Exception as e:
        raise HTTPException(503, f"minio not reachable: {e}")
    return {"status": "ok", "vllm": "ok", "postgres": "ok", "minio": "ok"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), identity: Identity = Depends(require_user)):
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(415, f"Unsupported file type: {ext}")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty upload")

    work = Path(tempfile.mkdtemp(prefix="ocr_in_"))
    in_path = work / name
    in_path.write_bytes(content)

    try:
        return await _run_pipeline(in_path, source_bytes=content, original_name=name,
                                   owner_id=identity.id)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("pipeline failed")
        raise HTTPException(500, f"Pipeline failed: {exc}")
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


# ── SSE batch ──────

def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {_json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


@app.post("/ocr/batch")
async def ocr_batch(files: List[UploadFile] = File(...),
                    identity: Identity = Depends(require_user)):
    if not files:
        raise HTTPException(400, "No files uploaded")

    # Read every upload body up-front (the request body is gone once the
    # StreamingResponse starts).
    prepared: list[dict] = []
    for upload in files:
        original = Path(upload.filename or "upload").name
        ext = Path(original).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            prepared.append({"original_name": original, "ext": ext,
                             "content": None,
                             "early_error": f"Unsupported file type: {ext}"})
            continue
        content = await upload.read()
        if not content:
            prepared.append({"original_name": original, "ext": ext,
                             "content": None, "early_error": "Empty upload"})
            continue
        prepared.append({"original_name": original, "ext": ext,
                         "content": content, "early_error": None})

    async def event_stream():
        import shutil
        work_root = Path(tempfile.mkdtemp(prefix="ocr_batch_"))
        t0 = time.perf_counter()
        try:
            items = [
                {"index": i, "original_name": p["original_name"], "status": "queued"}
                for i, p in enumerate(prepared)
            ]
            yield _sse("start", {"total": len(prepared), "items": items})

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

                sub = work_root / f"item_{i}"
                sub.mkdir(parents=True, exist_ok=True)
                in_path = sub / (p["original_name"] or f"item_{i}{p['ext']}")
                in_path.write_bytes(p["content"])

                try:
                    row = await _run_pipeline(
                        in_path, source_bytes=p["content"],
                        original_name=p["original_name"],
                        owner_id=identity.id,
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
                    log.exception("[batch] pipeline failed for %s", p["original_name"])
                    yield _sse("result", {
                        "index":         i,
                        "original_name": p["original_name"],
                        "status":        "error",
                        "error":         f"Pipeline failed: {exc}",
                        "elapsed_sec":   0,
                    })

            yield _sse("end", {
                "total":             len(prepared),
                "total_elapsed_sec": round(time.perf_counter() - t0, 2),
            })
        finally:
            shutil.rmtree(work_root, ignore_errors=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Bulk download ──────

@app.post("/ocr/batch-zip")
async def ocr_batch_zip(payload: dict = Body(...),
                        identity: Identity = Depends(require_user)):
    ids = payload.get("ids") or []
    if not isinstance(ids, list) or not ids:
        raise HTTPException(400, "Body must be {\"ids\": [...]}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for raw in ids:
            try:
                doc_id = _uuid.UUID(str(raw))
            except (ValueError, TypeError):
                continue
            doc = await storage.fetch_document(doc_id)
            if not doc or doc.get("status") != "ok":
                continue
            if doc.get("owner_id") != identity.id:
                continue
            stem = Path(doc["original_name"]).stem or str(doc_id)
            for kind in ("md", "json", "docx"):
                try:
                    filename, _ctype, body = await storage.get_artifact_bytes(doc_id, kind)
                except (KeyError, FileNotFoundError):
                    continue
                zf.writestr(f"{stem}/ocr_{filename}", body)

    data = buf.getvalue()
    if len(data) < 30:
        raise HTTPException(404, "No output files found for the given ids")

    return StreamingResponse(
        iter([data]),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="OCR_results.zip"'},
    )


# ── Document API ──────

@app.get("/documents")
async def list_documents(limit: int = 50, offset: int = 0, status: str | None = None,
                          identity: Identity = Depends(require_user)):
    rows = await storage.list_documents(limit=limit, offset=offset, status=status,
                                        owner_id=identity.id)
    total = await storage.count_documents(owner_id=identity.id, status=status)
    return JSONResponse(
        content=_json.loads(_json.dumps(rows, default=str)),
        headers={
            "X-Total-Count": str(total),
            "Access-Control-Expose-Headers": "X-Total-Count",
        },
    )


def _parse_uuid(raw: str) -> _uuid.UUID:
    try:
        return _uuid.UUID(raw)
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid document id (must be UUID)")


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str, identity: Identity = Depends(require_user)):
    uid = _parse_uuid(doc_id)
    doc = await storage.fetch_document(uid)
    if doc is None:
        raise HTTPException(404, "Document not found")
    _assert_owner_or_404(doc, identity.id)
    # Drop the bulky layout column from the row view; clients fetch it via /json.
    doc.pop("layout", None)
    return JSONResponse(content=_json.loads(_json.dumps(doc, default=str)))


@app.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str, identity: Identity = Depends(require_user)):
    uid = _parse_uuid(doc_id)
    ok = await storage.delete_document(uid, identity.id)
    if not ok:
        # Either missing or not the caller's. Use 404 to avoid leaking existence.
        raise HTTPException(404, "Document not found")
    return Response(status_code=204)


@app.delete("/internal/owner/{owner_id}/documents")
async def delete_owner_documents(owner_id: str,
                                  identity: Identity = Depends(require_user)):
    """Internal: called by auth_service when a user account is deleted, so the
    user's OCR history is wiped from DB + MinIO. Caller must be that same user
    or an admin (the caller already proved identity via X-User-Id headers
    forwarded by auth_service)."""
    target = _parse_uuid(owner_id)
    if identity.role not in {"admin", "master"} and identity.id != target:
        raise HTTPException(403, "Forbidden")
    removed = await storage.delete_all_documents_for_owner(target)
    return {"deleted": removed}


@app.get("/documents/{doc_id}/images/{filename}")
async def get_document_image(doc_id: str, filename: str,
                              identity: Identity = Depends(require_user)):
    uid = _parse_uuid(doc_id)
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    doc = await storage.fetch_document(uid)
    if doc is None:
        raise HTTPException(404, "Image not found")
    _assert_owner_or_404(doc, identity.id)
    try:
        fname, content_type, body = await storage.get_image_bytes(uid, filename)
    except Exception:
        raise HTTPException(404, "Image not found")
    return Response(
        content=body,
        media_type=content_type,
        headers={"Content-Disposition": f'inline; filename="{fname}"'},
    )


@app.get("/documents/{doc_id}/{kind}")
async def get_document_artifact(doc_id: str, kind: str,
                                 identity: Identity = Depends(require_user)):
    if kind not in ARTIFACT_KINDS:
        raise HTTPException(400, f"Unknown kind: {kind}")
    uid = _parse_uuid(doc_id)
    doc = await storage.fetch_document(uid)
    if doc is None:
        raise HTTPException(404, "Document not found")
    _assert_owner_or_404(doc, identity.id)
    try:
        filename, content_type, body = await storage.get_artifact_bytes(uid, kind)
    except KeyError:
        raise HTTPException(404, "Document not found")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    # JSON/MD viewed inline so the UI can pretty-print without download dance.
    disposition = "inline" if kind in {"md", "json"} else f'attachment; filename="ocr_{filename}"'
    return Response(
        content=body,
        media_type=content_type,
        headers={"Content-Disposition": disposition},
    )


# ── DOCX → PDF preview ──────

@app.get("/preview/{doc_id}")
async def preview_docx(doc_id: str, identity: Identity = Depends(require_user)):
    uid = _parse_uuid(doc_id)
    doc = await storage.fetch_document(uid)
    if doc is None:
        raise HTTPException(404, "Document not found")
    _assert_owner_or_404(doc, identity.id)
    try:
        _filename, _ctype, docx_bytes = await storage.get_artifact_bytes(uid, "docx")
    except KeyError:
        raise HTTPException(404, "Document not found")
    except FileNotFoundError:
        raise HTTPException(404, "DOCX not available for this document")

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"{uid}.docx"
        src.write_bytes(docx_bytes)
        pdf_path = _convert_to_pdf(src, Path(tmp), PREVIEW_TIMEOUT_SEC)
        pdf_bytes = pdf_path.read_bytes()

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{uid}.pdf"'},
    )


@app.post("/preview-input")
async def preview_input(file: UploadFile = File(...),
                         identity: Identity = Depends(require_user)):
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in PREVIEW_INPUT_EXTS:
        raise HTTPException(415, f"Cannot preview {ext}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / name
        src.write_bytes(data)
        pdf_path = _convert_to_pdf(src, Path(tmp), timeout=120)
        pdf_bytes = pdf_path.read_bytes()
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{Path(name).stem}.pdf"'},
    )

