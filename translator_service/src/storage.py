"""Storage layer for translator_service — Postgres metadata + MinIO bytes.

Mirrors the shape of ocr_service/storage.py so behaviour (pool lifecycle,
schema apply, MinIO bucket conventions) stays consistent across the two
services. Owner scoping comes from identity headers, same as ocr_service.

MinIO layout:
    translations/{uuid}/translated.json
    translations/{uuid}/translated.docx
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

log = logging.getLogger("translator_service.storage")

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

pool: Optional[asyncpg.Pool] = None
minio: Optional[Minio] = None
_upload_sem: Optional[asyncio.Semaphore] = None


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


async def health_postgres() -> None:
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")


def health_minio() -> None:
    if minio is None:
        raise RuntimeError("minio client not initialised")
    minio.bucket_exists(MINIO_BUCKET)


ARTIFACT_KEY = {
    "translated_json": "translated.json",
    "translated_docx": "translated.docx",
    # Partially-translated layout, rewritten every page-window so a long job
    # can resume instead of restarting. Deleted once the translation is ok.
    "checkpoint": "checkpoint.json",
}
ARTIFACT_CONTENT_TYPE = {
    "translated_json": "application/json",
    "translated_docx":
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "checkpoint": "application/json",
}


def _artifact_object_name(trans_id: _uuid.UUID, kind: str) -> str:
    return f"translations/{trans_id}/{ARTIFACT_KEY[kind]}"


class _BytesReader:
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


async def _put_object(name: str, body: bytes, content_type: str) -> None:
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
    assert minio is not None
    resp = minio.get_object(MINIO_BUCKET, name)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


async def insert_translation(original_name: str, source_document_id: _uuid.UUID,
                              target_lang: str, owner_id: _uuid.UUID) -> _uuid.UUID:
    """Create a 'queued' translations row and return its UUID."""
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    trans_id = _uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO translations
                (id, source_document_id, original_name, target_lang, status, owner_id)
            VALUES ($1, $2, $3, $4, 'queued', $5)
            """,
            trans_id, source_document_id, original_name, target_lang, owner_id,
        )
    return trans_id


_ALLOWED_UPDATE_COLS = {
    "status", "translated_json_key", "translated_docx_key",
    "elapsed_sec", "error", "source_document_id",
    # Checkpointing for long documents — see schema.sql.
    "pages_done", "page_count", "checkpoint_key",
}


async def update_translation(trans_id: _uuid.UUID, **fields: Any) -> None:
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    cols = [(k, v) for k, v in fields.items() if k in _ALLOWED_UPDATE_COLS]
    if not cols:
        return
    set_clauses = []
    values: list = []
    for i, (col, val) in enumerate(cols, start=1):
        set_clauses.append(f"{col} = ${i}")
        values.append(val)
    set_clauses.append("updated_at = now()")
    sql = (f"UPDATE translations SET {', '.join(set_clauses)} "
           f"WHERE id = ${len(values) + 1}")
    values.append(trans_id)
    async with pool.acquire() as conn:
        await conn.execute(sql, *values)


async def put_translated_artifact(trans_id: _uuid.UUID, kind: str, body: bytes) -> str:
    if kind not in ARTIFACT_KEY:
        raise ValueError(f"unknown artifact kind: {kind}")
    key = _artifact_object_name(trans_id, kind)
    await _put_object(key, body, ARTIFACT_CONTENT_TYPE[kind])
    return key


async def fetch_translation(trans_id: _uuid.UUID) -> Optional[dict]:
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM translations WHERE id = $1", trans_id)
    return dict(row) if row else None


async def list_translations(limit: int = 50, offset: int = 0,
                            owner_id: Optional[_uuid.UUID] = None,
                            status: Optional[str] = None) -> list[dict]:
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    cols = ("id, source_document_id, original_name, target_lang, status, "
            "elapsed_sec, error, owner_id, created_at, updated_at")
    async with pool.acquire() as conn:
        if status and owner_id is not None:
            rows = await conn.fetch(
                f"SELECT {cols} FROM translations "
                f"WHERE status = $1 AND owner_id = $2 "
                f"ORDER BY created_at DESC LIMIT $3 OFFSET $4",
                status, owner_id, limit, offset,
            )
        elif owner_id is not None:
            rows = await conn.fetch(
                f"SELECT {cols} FROM translations WHERE owner_id = $1 "
                f"ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                owner_id, limit, offset,
            )
        else:
            rows = await conn.fetch(
                f"SELECT {cols} FROM translations "
                f"ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                limit, offset,
            )
    return [dict(r) for r in rows]


async def count_translations(owner_id: Optional[_uuid.UUID] = None,
                             status: Optional[str] = None) -> int:
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    async with pool.acquire() as conn:
        if status and owner_id is not None:
            return await conn.fetchval(
                "SELECT count(*)::int FROM translations "
                "WHERE status = $1 AND owner_id = $2",
                status, owner_id,
            )
        if owner_id is not None:
            return await conn.fetchval(
                "SELECT count(*)::int FROM translations WHERE owner_id = $1",
                owner_id,
            )
        return await conn.fetchval("SELECT count(*)::int FROM translations")


def _delete_minio_prefix(prefix: str) -> int:
    assert minio is not None
    keys = [DeleteObject(obj.object_name) for obj in
            minio.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True)]
    if not keys:
        return 0
    errors = list(minio.remove_objects(MINIO_BUCKET, keys))
    for err in errors:
        log.warning("minio delete error: %s", err)
    return len(keys) - len(errors)


async def delete_translation(trans_id: _uuid.UUID, owner_id: _uuid.UUID) -> bool:
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM translations WHERE id = $1 AND owner_id = $2",
            trans_id, owner_id,
        )
    if not result.endswith(" 1"):
        return False
    try:
        await asyncio.to_thread(_delete_minio_prefix, f"translations/{trans_id}/")
    except Exception as e:
        log.warning("MinIO cleanup failed for %s: %s", trans_id, e)
    return True


# ── OCR-side reads ─────────────────────────────────────────────────────────
# In swap mode the OCR container is stopped while the translator runs, so we
# can't go through ocr_service's HTTP API to fetch layout JSON or Picture
# PNGs. Read them straight from MinIO ourselves. Owner-scoping is enforced
# by joining against the `documents` table.

def _ocr_layout_object_name(doc_id: _uuid.UUID) -> str:
    return f"documents/{doc_id}/layout.json"


def _ocr_image_object_name(doc_id: _uuid.UUID, filename: str) -> str:
    safe = os.path.basename(filename)
    return f"documents/{doc_id}/images/{safe}"


async def list_ocr_documents(owner_id: _uuid.UUID, limit: int = 200,
                              offset: int = 0) -> list[dict]:
    """List OCR'd documents the caller can translate. Only status='ok' rows
    with a layout JSON key are returned. Owner-scoped."""
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, original_name, status, scan_type, page_count,
                   created_at, json_key
            FROM documents
            WHERE owner_id = $1 AND status = 'ok' AND json_key IS NOT NULL
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            owner_id, limit, offset,
        )
    return [dict(r) for r in rows]


async def count_ocr_documents(owner_id: _uuid.UUID) -> int:
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT count(*)::int FROM documents "
            "WHERE owner_id = $1 AND status = 'ok' AND json_key IS NOT NULL",
            owner_id,
        )


async def fetch_ocr_document_row(doc_id: _uuid.UUID,
                                  owner_id: _uuid.UUID) -> Optional[dict]:
    """Look up a documents row, scoped to owner. Returns None if missing or
    not owned by the caller."""
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, original_name, status, json_key, owner_id
            FROM documents
            WHERE id = $1 AND owner_id = $2
            """,
            doc_id, owner_id,
        )
    return dict(row) if row else None


async def fetch_ocr_layout(doc_id: _uuid.UUID) -> list[dict]:
    """Read the persisted OCR layout JSON for a document from MinIO.
    Caller is responsible for verifying ownership first."""
    import json as _j
    key = _ocr_layout_object_name(doc_id)
    body = await asyncio.to_thread(_get_object_bytes, key)
    obj = _j.loads(body.decode("utf-8"))
    if not isinstance(obj, list):
        raise RuntimeError(
            f"unexpected OCR layout shape for {doc_id}: {type(obj)}"
        )
    return obj


async def fetch_ocr_image(doc_id: _uuid.UUID, filename: str) -> bytes:
    """Read a Picture PNG from MinIO. Caller verifies ownership."""
    key = _ocr_image_object_name(doc_id, filename)
    return await asyncio.to_thread(_get_object_bytes, key)


# ── Checkpointing (long translations) ────────────────────────────────────────

async def put_checkpoint(trans_id: _uuid.UUID, layout: list[dict]) -> str:
    """Persist the partially-translated layout so the job can resume.

    Written every page-window, so it must stay cheap relative to a window's
    translation cost — for a 25-page window that is many model calls, so one
    JSON serialization is noise.
    """
    import json as _j
    body = _j.dumps(layout, ensure_ascii=False).encode("utf-8")
    return await put_translated_artifact(trans_id, "checkpoint", body)


async def fetch_checkpoint(trans_id: _uuid.UUID) -> Optional[list[dict]]:
    """Read a partially-translated layout back, or None if unusable.

    Never raises: a missing or corrupt checkpoint must degrade to "start over",
    never to a failed translation.
    """
    import json as _j
    key = _artifact_object_name(trans_id, "checkpoint")
    try:
        body = await asyncio.to_thread(_get_object_bytes, key)
        obj = _j.loads(body.decode("utf-8"))
    except Exception as exc:
        log.warning("checkpoint unreadable for %s (%s); starting over",
                    trans_id, exc)
        return None
    if not isinstance(obj, list):
        log.warning("unexpected checkpoint shape for %s: %s; starting over",
                    trans_id, type(obj))
        return None
    return obj


async def delete_checkpoint(trans_id: _uuid.UUID) -> None:
    """Drop the checkpoint once the translation has completed."""
    key = _artifact_object_name(trans_id, "checkpoint")
    try:
        await asyncio.to_thread(minio.remove_object, MINIO_BUCKET, key)
    except Exception as exc:  # best-effort cleanup
        log.warning("could not delete checkpoint %s: %s", key, exc)


async def get_artifact_bytes(trans_id: _uuid.UUID, kind: str
                              ) -> tuple[str, str, bytes]:
    """Return (filename, content_type, body) for a translated artifact.

    `kind` is one of {translated_json, translated_docx}. Raises KeyError if
    the translation row is unknown or FileNotFoundError if the key column
    is still null.
    """
    row = await fetch_translation(trans_id)
    if row is None:
        raise KeyError(f"translation not found: {trans_id}")
    col = {
        "translated_json": "translated_json_key",
        "translated_docx": "translated_docx_key",
    }[kind]
    key = row[col]
    if not key:
        raise FileNotFoundError(f"translation {trans_id} has no {kind} yet")
    content_type = ARTIFACT_CONTENT_TYPE[kind]
    stem = os.path.splitext(row["original_name"])[0] or str(trans_id)
    suffix = {"translated_json": ".json", "translated_docx": ".docx"}[kind]
    filename = f"{stem}_pt-BR{suffix}"
    body = await asyncio.to_thread(_get_object_bytes, key)
    return filename, content_type, body
