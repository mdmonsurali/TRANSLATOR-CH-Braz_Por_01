"""Read-only storage layer for orchestrator_service.

Lets the orchestrator serve translation artifacts (DOCX/JSON) directly from
MinIO + Postgres so users can download finished translations even when
translator_service has been stopped between batches.

Owner-scoping mirrors translator_service: each row carries owner_id and we
join the caller's identity against it before returning bytes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid as _uuid
from typing import Optional

import asyncpg
from minio import Minio

log = logging.getLogger("orchestrator.storage")

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


_pool: Optional[asyncpg.Pool] = None
_minio: Optional[Minio] = None


async def init_pool() -> None:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=POSTGRES_DSN, min_size=1, max_size=5)
        log.info("postgres pool ready: %s", POSTGRES_DSN.rsplit("@", 1)[-1])


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def init_minio() -> None:
    global _minio
    _minio = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
        region=MINIO_REGION,
    )
    if not _minio.bucket_exists(MINIO_BUCKET):
        log.warning("minio bucket %s not found", MINIO_BUCKET)
    log.info("minio ready: bucket=%s endpoint=%s", MINIO_BUCKET, MINIO_ENDPOINT)


# ── Reads ────────

async def translations_by_source(owner_id: _uuid.UUID,
                                  document_ids: list[_uuid.UUID]) -> dict:
    """Return {source_document_id_str: {translation_id, target_lang, status}}
    for every successful translation the caller owns whose source document is
    in `document_ids`. Empty docs list returns an empty dict.

    Used by Task History to render translation-download buttons next to the
    OCR documents that have a translation available.
    """
    if not document_ids:
        return {}
    if _pool is None:
        raise RuntimeError("postgres pool not initialised")
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, source_document_id, target_lang, status, original_name
            FROM translations
            WHERE owner_id = $1
              AND source_document_id = ANY($2::uuid[])
              AND status = 'ok'
            ORDER BY created_at DESC
            """,
            owner_id, document_ids,
        )

    out: dict[str, dict] = {}
    for r in rows:
        sid = str(r["source_document_id"])
        if sid in out:
            continue
        out[sid] = {
            "translation_id": str(r["id"]),
            "target_lang":    r["target_lang"],
            "status":         r["status"],
            "original_name":  r["original_name"],
        }
    return out


async def fetch_translation(trans_id: _uuid.UUID,
                             owner_id: _uuid.UUID) -> Optional[dict]:
    """Owner-scoped lookup. Returns None on miss or wrong owner so we don't
    leak existence of other users' rows."""
    if _pool is None:
        raise RuntimeError("postgres pool not initialised")
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, source_document_id, original_name, target_lang, status,
                   translated_json_key, translated_docx_key, elapsed_sec,
                   error, owner_id, created_at, updated_at
            FROM translations
            WHERE id = $1 AND owner_id = $2
            """,
            trans_id, owner_id,
        )
    return dict(row) if row else None


def _get_object_sync(name: str) -> bytes:
    assert _minio is not None
    resp = _minio.get_object(MINIO_BUCKET, name)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


_ARTIFACT_COLUMN = {
    "translated_json": "translated_json_key",
    "translated_docx": "translated_docx_key",
}
_ARTIFACT_CONTENT_TYPE = {
    "translated_json": "application/json",
    "translated_docx":
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_ARTIFACT_SUFFIX = {
    "translated_json": ".json",
    "translated_docx": ".docx",
}


async def get_artifact_bytes(row: dict, kind: str) -> tuple[str, str, bytes]:
    """Return (filename, content_type, body) for one translated artifact.

    Caller must have already verified ownership via fetch_translation().
    """
    if kind not in _ARTIFACT_COLUMN:
        raise ValueError(f"unknown artifact kind: {kind}")
    key = row.get(_ARTIFACT_COLUMN[kind])
    if not key:
        raise FileNotFoundError(f"translation has no {kind} yet")
    body = await asyncio.to_thread(_get_object_sync, key)
    stem = (row.get("original_name") or str(row.get("id")))
    # strip ext on the stem so we don't end up with foo.pdf.docx
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    # target_lang arrives from a user-supplied form field and lands in a
    # Content-Disposition header, so keep it to plain tag characters.
    lang = re.sub(r"[^A-Za-z0-9_-]", "", row.get("target_lang") or "") or "translated"
    filename = f"{stem}_{lang}{_ARTIFACT_SUFFIX[kind]}"
    return filename, _ARTIFACT_CONTENT_TYPE[kind], body


def health_postgres_sync() -> bool:
    """Cheap probe used by /health. Sync wrapper around a pool acquire."""
    return _pool is not None


def health_minio_sync() -> bool:
    if _minio is None:
        return False
    try:
        _minio.bucket_exists(MINIO_BUCKET)
        return True
    except Exception:
        return False
