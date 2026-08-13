"""Storage layer for OCR — Postgres metadata + MinIO bytes.

Owns:
- One asyncpg connection pool (POSTGRES_*).
- One MinIO client (MINIO_*).
- The documents table lifecycle (init_schema reads database/schema.sql).

Public surface used by main.py:
    init_pool() / close_pool()
    init_minio()
    init_schema()
    health_postgres() / health_minio()
    insert_document(original_name, ext, source_bytes) -> uuid
    update_document(doc_id, **fields) -> None
    put_artifact(doc_id, kind, body) -> str          # kind in {md, json, docx}
    fetch_document(doc_id) -> dict | None
    list_documents(limit, offset, status=None) -> list[dict]
    get_artifact_bytes(doc_id, kind) -> (filename, content_type, bytes)

Bytes paths inside the MinIO bucket:
    documents/{uuid}/source.{ext}
    documents/{uuid}/output.md
    documents/{uuid}/layout.json
    documents/{uuid}/output.docx
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import uuid as _uuid
from pathlib import Path
from typing import Any, Optional

import asyncpg
from minio import Minio
from minio.deleteobjects import DeleteObject
from minio.error import S3Error

log = logging.getLogger("ocr_service.storage")

# ── Config from env ───────────────────────────────────────────────────────

POSTGRES_DSN = (
    f"postgres://{os.getenv('POSTGRES_USER', 'ocr')}"
    f":{os.getenv('POSTGRES_PASSWORD', 'changeme')}"
    f"@{os.getenv('POSTGRES_HOST', 'postgres')}"
    f":{os.getenv('POSTGRES_PORT', '5432')}"
    f"/{os.getenv('POSTGRES_DB', 'ocr')}"
)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "changeme")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "ocr")
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

UPLOAD_CONCURRENCY = int(os.getenv("MINIO_UPLOAD_CONCURRENCY", "8"))
SCHEMA_FILE = Path(os.getenv("SCHEMA_FILE", "/workspace/database/schema.sql"))

# ── Module state ────────

pool: Optional[asyncpg.Pool] = None
minio: Optional[Minio] = None
_upload_sem: Optional[asyncio.Semaphore] = None


# ── Lifecycle ────

async def init_pool() -> None:
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(dsn=POSTGRES_DSN, min_size=1, max_size=10)
        log.info("postgres pool ready: %s", POSTGRES_DSN.rsplit("@", 1)[-1])


async def close_pool() -> None:
    global pool
    if pool is not None:
        await pool.close()
        pool = None


def init_minio() -> None:
    global minio, _upload_sem
    minio = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
        region=MINIO_REGION,
    )
    _upload_sem = asyncio.Semaphore(UPLOAD_CONCURRENCY)
    # Bucket is created by minio-init compose service; verify it exists.
    if not minio.bucket_exists(MINIO_BUCKET):
        log.warning("minio bucket %s not found, creating", MINIO_BUCKET)
        minio.make_bucket(MINIO_BUCKET)
    log.info("minio ready: bucket=%s endpoint=%s", MINIO_BUCKET, MINIO_ENDPOINT)


async def init_schema() -> None:
    if pool is None:
        raise RuntimeError("init_pool() must be called before init_schema()")
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(sql)
    log.info("schema applied from %s", SCHEMA_FILE)


# ── Health probes ────

async def health_postgres() -> None:
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")


def health_minio() -> None:
    if minio is None:
        raise RuntimeError("minio client not initialised")
    # bucket_exists is cheap (HEAD request) and confirms creds + reachability.
    minio.bucket_exists(MINIO_BUCKET)


# ── Helpers ───

ARTIFACT_KEY = {
    "md":   "output.md",
    "json": "layout.json",
    "docx": "output.docx",
}
ARTIFACT_CONTENT_TYPE = {
    "md":   "text/markdown",
    "json": "application/json",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
SOURCE_CONTENT_TYPE = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _source_object_name(doc_id: _uuid.UUID, ext: str) -> str:
    return f"documents/{doc_id}/source{ext}"


def _artifact_object_name(doc_id: _uuid.UUID, kind: str) -> str:
    return f"documents/{doc_id}/{ARTIFACT_KEY[kind]}"


async def _put_object(name: str, body: bytes, content_type: str) -> None:
    """MinIO PUT in a thread, throttled by the upload semaphore."""
    assert _upload_sem is not None and minio is not None
    async with _upload_sem:
        await asyncio.to_thread(
            minio.put_object,
            MINIO_BUCKET,
            name,
            data=_BytesReader(body),
            length=len(body),
            content_type=content_type,
        )


def _get_object_bytes(name: str) -> bytes:
    """Sync MinIO GET — used inside asyncio.to_thread."""
    assert minio is not None
    resp = minio.get_object(MINIO_BUCKET, name)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


class _BytesReader:
    """Minimal file-like wrapper for bytes that the minio SDK accepts."""
    __slots__ = ("_b", "_pos")

    def __init__(self, b: bytes):
        self._b = b
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self._pos + n > len(self._b):
            chunk = self._b[self._pos:]
            self._pos = len(self._b)
            return chunk
        chunk = self._b[self._pos:self._pos + n]
        self._pos += n
        return chunk


# ── CRUD ─────────

async def insert_document(original_name: str, ext: str, source_bytes: bytes,
                          owner_id: _uuid.UUID) -> _uuid.UUID:
    """Create a 'queued' document row and upload its source bytes to MinIO.

    Returns the new document UUID. Caller drives state transitions via
    update_document() and put_artifact().
    """
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    doc_id = _uuid.uuid4()
    source_key = _source_object_name(doc_id, ext)
    await _put_object(
        source_key,
        source_bytes,
        SOURCE_CONTENT_TYPE.get(ext.lower(), "application/octet-stream"),
    )
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO documents (id, original_name, status, source_key, owner_id)
            VALUES ($1, $2, 'queued', $3, $4)
            """,
            doc_id, original_name, source_key, owner_id,
        )
    return doc_id


# `layout` (JSONB) is still writable but the OCR pipeline no longer populates
# it: it duplicated documents/{uuid}/layout.json in MinIO, cost a full extra
# serialization of the document at peak memory, and had no reader — the
# list/detail routes pop the column straight back off.
_ALLOWED_UPDATE_COLS = {
    "status", "scan_type", "page_count", "markdown_key", "json_key",
    "docx_key", "layout", "elapsed_sec", "error",
}


async def update_document(doc_id: _uuid.UUID, **fields: Any) -> None:
    """Update an existing document row. Caller specifies columns by name."""
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    cols = [(k, v) for k, v in fields.items() if k in _ALLOWED_UPDATE_COLS]
    if not cols:
        return
    set_clauses = []
    values = []
    for i, (col, val) in enumerate(cols, start=1):
        set_clauses.append(f"{col} = ${i}")
        # JSONB column: pass JSON string, asyncpg encodes
        if col == "layout" and not isinstance(val, str):
            val = _json.dumps(val, ensure_ascii=False)
        values.append(val)
    set_clauses.append(f"updated_at = now()")
    sql = f"UPDATE documents SET {', '.join(set_clauses)} WHERE id = ${len(values) + 1}"
    values.append(doc_id)
    async with pool.acquire() as conn:
        await conn.execute(sql, *values)


async def put_artifact(doc_id: _uuid.UUID, kind: str, body: bytes) -> str:
    """Upload a generated artifact (md / json / docx) to MinIO. Returns the key."""
    if kind not in ARTIFACT_KEY:
        raise ValueError(f"unknown artifact kind: {kind}")
    key = _artifact_object_name(doc_id, kind)
    await _put_object(key, body, ARTIFACT_CONTENT_TYPE[kind])
    return key


IMAGE_CONTENT_TYPE = "image/png"


def _image_object_name(doc_id: _uuid.UUID, filename: str) -> str:
    safe = os.path.basename(filename)
    return f"documents/{doc_id}/images/{safe}"


async def put_image(doc_id: _uuid.UUID, filename: str, body: bytes) -> str:
    """Upload a cropped Picture PNG to MinIO. Returns the object key."""
    key = _image_object_name(doc_id, filename)
    await _put_object(key, body, IMAGE_CONTENT_TYPE)
    return key


async def get_image_bytes(doc_id: _uuid.UUID, filename: str) -> tuple[str, str, bytes]:
    """Fetch a stored Picture PNG. Returns (filename, content_type, body)."""
    key = _image_object_name(doc_id, filename)
    body = await asyncio.to_thread(_get_object_bytes, key)
    return os.path.basename(filename), IMAGE_CONTENT_TYPE, body


async def fetch_document(doc_id: _uuid.UUID) -> Optional[dict]:
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM documents WHERE id = $1", doc_id)
    return dict(row) if row else None


async def count_documents(owner_id: Optional[_uuid.UUID] = None,
                          status: Optional[str] = None) -> int:
    """Total rows matching the same filters list_documents uses."""
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    async with pool.acquire() as conn:
        if status and owner_id is not None:
            return await conn.fetchval(
                "SELECT count(*)::int FROM documents WHERE status = $1 AND owner_id = $2",
                status, owner_id,
            )
        if status:
            return await conn.fetchval(
                "SELECT count(*)::int FROM documents WHERE status = $1", status,
            )
        if owner_id is not None:
            return await conn.fetchval(
                "SELECT count(*)::int FROM documents WHERE owner_id = $1", owner_id,
            )
        return await conn.fetchval("SELECT count(*)::int FROM documents")


def _delete_minio_prefix(prefix: str) -> int:
    """Synchronous MinIO recursive prefix delete. Returns object count removed.
    Called inside asyncio.to_thread by the async wrappers below."""
    assert minio is not None
    keys = [DeleteObject(obj.object_name) for obj in
            minio.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True)]
    if not keys:
        return 0
    errors = list(minio.remove_objects(MINIO_BUCKET, keys))
    for err in errors:
        log.warning("minio delete error: %s", err)
    return len(keys) - len(errors)


async def delete_document(doc_id: _uuid.UUID, owner_id: _uuid.UUID) -> bool:
    """Delete one document (DB row + all MinIO objects under documents/{id}/).
    Owner-scoped: returns False if the row doesn't exist or belongs to someone
    else, True if the row was deleted."""
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM documents WHERE id = $1 AND owner_id = $2",
            doc_id, owner_id,
        )
    if not result.endswith(" 1"):
        return False
    try:
        removed = await asyncio.to_thread(_delete_minio_prefix, f"documents/{doc_id}/")
        log.info("deleted document %s (%d MinIO objects)", doc_id, removed)
    except Exception as e:
        # DB row is gone; orphan MinIO bytes are recoverable, don't fail the call.
        log.warning("MinIO cleanup failed for %s: %s", doc_id, e)
    return True


async def delete_all_documents_for_owner(owner_id: _uuid.UUID) -> int:
    """Delete every document owned by `owner_id` (DB + MinIO). Returns the
    number of DB rows removed. Used when a user account is deleted."""
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    async with pool.acquire() as conn:
        ids = [r["id"] for r in await conn.fetch(
            "SELECT id FROM documents WHERE owner_id = $1", owner_id,
        )]
        if not ids:
            return 0
        await conn.execute("DELETE FROM documents WHERE owner_id = $1", owner_id)
    for doc_id in ids:
        try:
            await asyncio.to_thread(_delete_minio_prefix, f"documents/{doc_id}/")
        except Exception as e:
            log.warning("MinIO cleanup failed for %s: %s", doc_id, e)
    log.info("deleted %d documents for owner %s", len(ids), owner_id)
    return len(ids)


async def list_documents(limit: int = 50, offset: int = 0,
                         status: Optional[str] = None,
                         owner_id: Optional[_uuid.UUID] = None) -> list[dict]:
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    cols = ("id, original_name, status, scan_type, page_count, "
            "elapsed_sec, error, created_at, updated_at, owner_id")
    async with pool.acquire() as conn:
        if status and owner_id is not None:
            rows = await conn.fetch(
                f"""
                SELECT {cols}
                FROM documents
                WHERE status = $1 AND owner_id = $2
                ORDER BY created_at DESC
                LIMIT $3 OFFSET $4
                """,
                status, owner_id, limit, offset,
            )
        elif status:
            rows = await conn.fetch(
                f"""
                SELECT {cols}
                FROM documents
                WHERE status = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """,
                status, limit, offset,
            )
        elif owner_id is not None:
            rows = await conn.fetch(
                f"""
                SELECT {cols}
                FROM documents
                WHERE owner_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """,
                owner_id, limit, offset,
            )
        else:
            rows = await conn.fetch(
                f"""
                SELECT {cols}
                FROM documents
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit, offset,
            )
    return [dict(r) for r in rows]


async def get_artifact_bytes(doc_id: _uuid.UUID, kind: str) -> tuple[str, str, bytes]:
    """Resolve a document's artifact and return (suggested_filename, content_type, bytes).

    `kind` is one of {source, md, json, docx}. Raises FileNotFoundError if the
    document row exists but the artifact key column is null, or KeyError if the
    document is unknown.
    """
    doc = await fetch_document(doc_id)
    if doc is None:
        raise KeyError(f"document not found: {doc_id}")

    if kind == "source":
        key = doc["source_key"]
        # filename derived from source_key extension
        ext = os.path.splitext(key)[1] or ".bin"
        content_type = SOURCE_CONTENT_TYPE.get(ext.lower(), "application/octet-stream")
        # User-facing name: original + ext
        filename = doc["original_name"]
    else:
        col = {"md": "markdown_key", "json": "json_key", "docx": "docx_key"}[kind]
        key = doc[col]
        if not key:
            raise FileNotFoundError(f"document {doc_id} has no {kind} artifact yet")
        content_type = ARTIFACT_CONTENT_TYPE[kind]
        stem = os.path.splitext(doc["original_name"])[0] or str(doc_id)
        suffix = {"md": ".md", "json": ".json", "docx": ".docx"}[kind]
        filename = f"{stem}{suffix}"

    body = await asyncio.to_thread(_get_object_bytes, key)
    return filename, content_type, body
