from __future__ import annotations

import io
import json as _json
import logging
import os
import time
import uuid as _uuid
from typing import List, Optional

import httpx

import storage
from translation_reconstruction import json_to_docx
from translation import TRANSLATE_PAGE_WINDOW, translate_layout

log = logging.getLogger("translator_service.pipeline")

OCR_SERVICE_URL = os.environ.get("OCR_SERVICE_URL", "http://ocr_service:8001")
# Whole-document OCR call — must cover the entire job (hours for 1000 pages),
# not a single page. See the matching note in orchestrator_service/pipeline.py.
OCR_TIMEOUT_SEC = float(os.environ.get("OCR_TIMEOUT_SEC", "21600"))  # 6h
TRANSLATE_TARGET_LANG = os.environ.get("TRANSLATE_TARGET_LANG", "pt-BR")

SUPPORTED_EXTS = {".pdf", ".docx", ".jpg", ".jpeg", ".png"}

# Categories that carry a stored picture asset (image_url) to re-hydrate. Must
# match the labels ocr_service emits: standalone Image/Figure blocks, recovered
# table-cell photos and diagram crops (all "Image"), and legacy "Picture". The
# reconstruction renderer treats all of these as pictures, so hydrating only
# "Picture" silently dropped every Image/Figure from the translated DOCX.
PICTURE_CATEGORIES = {"Image", "Figure", "Picture"}


def _guard_translation(stats: dict, *, original_name: str) -> None:
    """Raise if translation had work to do but produced nothing usable.

    Three cases:
      * items_translated > 0                -> success (possibly partial); OK.
      * failed == 0 and items_translated 0  -> nothing translatable in the
                                               document (e.g. image-only pages).
                                               Legitimate; let it through.
      * failed > 0 and items_translated 0   -> every segment fell back to the
                                               source string. The DOCX would be
                                               100% untranslated. Hard-fail.

    A partial failure (some translated, some failed) stays 'ok' but is logged
    so it's visible in the service logs and the result row's `failed` count.
    """
    translated = int(stats.get("items_translated", 0) or 0)
    failed = int(stats.get("failed", 0) or 0)
    if failed and not translated:
        raise RuntimeError(
            f"translation produced no output for {original_name}: all {failed} "
            f"segment(s) fell back to source text. The translator could not "
            f"reach llama.cpp or every response was unparseable — refusing to "
            f"persist an untranslated document. Check translator_service "
            f"/health and the llama-server logs."
        )
    if failed:
        log.warning(
            "[translate] PARTIAL for %s: %d translated, %d failed (kept source)",
            original_name, translated, failed,
        )


def _guard_entry_counts(layout: List[dict], *, original_name: str) -> None:
    """Log a warning if any page has suspiciously few entries vs what's expected.

    translate_layout mutates entries in-place and never adds or removes them,
    so the count after translation should equal the count before. If entries
    are missing it means the layout JSON was malformed coming in, which is
    worth flagging before we persist a broken artifact.
    """
    for i, page in enumerate(layout):
        entries = page.get("entries") or []
        empty_text = sum(
            1 for e in entries
            if not (e.get("text") or "").strip()
            and e.get("category") not in ("Picture",)
        )
        if empty_text > len(entries) * 0.5 and len(entries) > 2:
            log.warning(
                "[translate] %s page %d: %d/%d non-picture entries have empty "
                "text after translation — possible scrambling",
                original_name, i + 1, empty_text, len(entries),
            )


def _identity_headers(user_id: _uuid.UUID, user_role: str) -> dict:
    return {"X-User-Id": str(user_id), "X-User-Role": user_role}


async def _translate_with_checkpoints(layout: List[dict], *,
                                      trans_id: _uuid.UUID,
                                      target_lang: str,
                                      original_name: str) -> dict:
    """Translate `layout` in page windows, persisting progress as it goes.

    A 1000-page document takes hours, so losing everything to a timeout, a GPU
    swap, or a container restart is unacceptable. After each window the
    partially-translated layout is written to MinIO and `pages_done` is
    advanced, so a re-run resumes from the first untranslated page.

    Returns the same stats dict as `translate_layout`. `layout` is mutated in
    place; on resume the already-translated leading pages are spliced back in,
    so the caller always ends up with a fully-translated list.
    """
    total_pages = len(layout)
    window = TRANSLATE_PAGE_WINDOW

    # Resume: splice in any leading pages a previous run already finished.
    resume_from = 0
    row = await storage.fetch_translation(trans_id)
    if row and row.get("checkpoint_key") and (row.get("pages_done") or 0) > 0:
        done = int(row["pages_done"])
        saved = await storage.fetch_checkpoint(trans_id)
        # Only trust a checkpoint that matches this document's shape; a
        # mismatch means the source changed under us, so start over.
        if saved is not None and len(saved) == total_pages and 0 < done <= total_pages:
            layout[:done] = saved[:done]
            resume_from = done
            log.info("[translate] %s: resuming at page %d/%d from checkpoint",
                     original_name, done, total_pages)
        elif saved is not None:
            log.warning("[translate] %s: checkpoint has %d pages, layout has "
                        "%d — ignoring and starting over",
                        original_name, len(saved), total_pages)

    await storage.update_translation(trans_id, page_count=total_pages)

    if resume_from >= total_pages:
        log.info("[translate] %s: checkpoint already covers every page",
                 original_name)
        return {"items_translated": 0, "chunks": 0, "failed": 0,
                "resumed_pages": resume_from}

    pending = layout[resume_from:]

    async def _on_progress(done_in_pending: int, _total: int, stats: dict) -> None:
        pages_done = resume_from + done_in_pending
        # Write the checkpoint BEFORE advancing pages_done: if we die between
        # the two, the next run re-translates one window rather than skipping
        # pages that were never persisted.
        try:
            key = await storage.put_checkpoint(trans_id, layout)
            await storage.update_translation(
                trans_id, checkpoint_key=key, pages_done=pages_done)
        except Exception as exc:
            # A checkpoint failure must not kill an otherwise-healthy job.
            log.warning("[translate] %s: could not checkpoint at page %d: %s",
                        original_name, pages_done, exc)

    stats = await translate_layout(
        pending, target_lang=target_lang,
        page_window=window if window > 0 else 0,
        on_progress=_on_progress,
    )
    # `pending` holds the same dict objects as `layout[resume_from:]`, so the
    # in-place writes are already visible in `layout`.
    if resume_from:
        stats = dict(stats)
        stats["resumed_pages"] = resume_from
    return stats


async def _run_ocr(content: bytes, original_name: str, content_type: Optional[str],
                   user_id: _uuid.UUID, user_role: str) -> dict:
    """POST one file to ocr_service /ocr and return the JSON row."""
    async with httpx.AsyncClient(timeout=OCR_TIMEOUT_SEC) as client:
        resp = await client.post(
            f"{OCR_SERVICE_URL}/ocr",
            files={"file": (original_name, content,
                            content_type or "application/octet-stream")},
            headers=_identity_headers(user_id, user_role),
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"ocr_service returned {resp.status_code}: {resp.text[:500]}"
        )
    return resp.json()


async def _fetch_layout_json(doc_id: str, user_id: _uuid.UUID,
                              user_role: str) -> List[dict]:
    """GET the persisted layout JSON for an OCR document."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.get(
            f"{OCR_SERVICE_URL}/documents/{doc_id}/json",
            headers=_identity_headers(user_id, user_role),
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"failed to fetch layout JSON for {doc_id}: "
            f"{resp.status_code} {resp.text[:500]}"
        )
    return resp.json()


async def _fetch_image_bytes(doc_id: str, filename: str, user_id: _uuid.UUID,
                              user_role: str) -> bytes:
    """Pull a stored Picture PNG back from ocr_service so the rebuilt DOCX
    can embed the original image rather than a broken reference."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(
            f"{OCR_SERVICE_URL}/documents/{doc_id}/images/{filename}",
            headers=_identity_headers(user_id, user_role),
        )
    resp.raise_for_status()
    return resp.content


async def _hydrate_pictures(layout: List[dict], doc_id: str, user_id: _uuid.UUID,
                             user_role: str) -> None:
    """`json_to_docx` will embed Picture entries that carry an `image_obj`
    (PIL.Image). The JSON we get from MinIO only has `image_url` strings, so
    we fetch each referenced image and load it back into PIL."""
    from PIL import Image

    for page in layout:
        for entry in page.get("entries") or []:
            if entry.get("category") not in PICTURE_CATEGORIES:
                continue
            url = entry.get("image_url")
            if not url:
                continue
            # url is "images/<filename>" as written by ocr_service.
            fname = url.split("/", 1)[-1] if "/" in url else url
            try:
                body = await _fetch_image_bytes(doc_id, fname, user_id, user_role)
                entry["image_obj"] = Image.open(io.BytesIO(body)).copy()
            except Exception as exc:
                log.warning("could not fetch picture %s for %s: %s",
                            fname, doc_id, exc)


async def _hydrate_pictures_from_minio(layout: List[dict],
                                        doc_id: _uuid.UUID) -> None:
    """Same as `_hydrate_pictures` but reads PNGs from MinIO directly.
    Used in swap mode when ocr_service is stopped."""
    from PIL import Image

    for page in layout:
        for entry in page.get("entries") or []:
            if entry.get("category") not in PICTURE_CATEGORIES:
                continue
            url = entry.get("image_url")
            if not url:
                continue
            fname = url.split("/", 1)[-1] if "/" in url else url
            try:
                body = await storage.fetch_ocr_image(doc_id, fname)
                entry["image_obj"] = Image.open(io.BytesIO(body)).copy()
            except Exception as exc:
                log.warning("could not fetch picture %s for %s: %s",
                            fname, doc_id, exc)


async def translate_one(*, content: bytes, original_name: str,
                        content_type: Optional[str],
                        user_id: _uuid.UUID, user_role: str,
                        target_lang: str = TRANSLATE_TARGET_LANG) -> dict:
    """Run the OCR → translate → rebuild → persist pipeline for one upload.

    Returns a result row compatible with the batch SSE contract:
        {translation_id, document_id, original_name, status, target_lang,
         page_count?, elapsed_sec, items_translated, chunks, failed}
    """
    t0 = time.perf_counter()

    # Bootstrap a translations row early so the user has something to find
    # in history even if OCR fails.
    trans_id = await storage.insert_translation(
        original_name=original_name,
        source_document_id=None,  # filled in after OCR succeeds
        target_lang=target_lang,
        owner_id=user_id,
    )

    try:
        await storage.update_translation(trans_id, status="ocr")
        ocr_row = await _run_ocr(content, original_name, content_type,
                                  user_id, user_role)
        doc_id = ocr_row["document_id"]
        await storage.update_translation(
            trans_id,
            source_document_id=_uuid.UUID(doc_id),
            status="translating",
        )

        layout = await _fetch_layout_json(doc_id, user_id, user_role)
        if not isinstance(layout, list):
            raise RuntimeError(
                f"unexpected layout JSON shape from ocr_service: {type(layout)}"
            )

        stats = await _translate_with_checkpoints(
            layout, trans_id=trans_id, target_lang=target_lang,
            original_name=original_name,
        )
        # Fail loudly if nothing translated (don't persist an all-source DOCX).
        _guard_translation(stats, original_name=original_name)
        _guard_entry_counts(layout, original_name=original_name)

        await storage.update_translation(trans_id, status="reconstructing")
        await _hydrate_pictures(layout, doc_id, user_id, user_role)

        translated_json_bytes = _json.dumps(
            [_strip_image_objs(p) for p in layout],
            indent=2, ensure_ascii=False,
        ).encode("utf-8")

        docx_buf = io.BytesIO()
        json_to_docx(layout, output_path=docx_buf)
        docx_bytes = docx_buf.getvalue()

        json_key = await storage.put_translated_artifact(
            trans_id, "translated_json", translated_json_bytes,
        )
        docx_key = await storage.put_translated_artifact(
            trans_id, "translated_docx", docx_bytes,
        )

        elapsed = round(time.perf_counter() - t0, 2)
        await storage.update_translation(
            trans_id,
            status="ok",
            translated_json_key=json_key,
            translated_docx_key=docx_key,
            elapsed_sec=elapsed,
            error=None,
        )
        # The finished artifacts supersede the checkpoint.
        await storage.delete_checkpoint(trans_id)
        await storage.update_translation(trans_id, checkpoint_key=None)

        return {
            "translation_id": str(trans_id),
            "document_id":    doc_id,
            "original_name":  original_name,
            "status":         "ok",
            "target_lang":    target_lang,
            "page_count":     ocr_row.get("page_count"),
            "elapsed_sec":    elapsed,
            "items_translated": stats.get("items_translated", 0),
            "chunks":           stats.get("chunks", 0),
            "failed":           stats.get("failed", 0),
        }

    except Exception as exc:
        log.exception("translation pipeline failed for %s", original_name)
        await storage.update_translation(
            trans_id, status="error", error=str(exc)[:1000],
        )
        raise


async def translate_from_doc_id(*, doc_id: _uuid.UUID,
                                 user_id: _uuid.UUID,
                                 target_lang: str = TRANSLATE_TARGET_LANG) -> dict:
    """Translate an already-OCR'd document. Reads layout JSON + pictures
    directly from MinIO so this works when ocr_service is stopped (swap mode).

    Returns the same shape as `translate_one`, minus `page_count` (we don't
    re-derive it).
    """
    t0 = time.perf_counter()

    doc_row = await storage.fetch_ocr_document_row(doc_id, user_id)
    if doc_row is None:
        raise RuntimeError(f"document not found: {doc_id}")
    if doc_row.get("status") != "ok":
        raise RuntimeError(
            f"document {doc_id} is not OCR-ok (status={doc_row.get('status')})"
        )
    if not doc_row.get("json_key"):
        raise RuntimeError(f"document {doc_id} has no layout JSON yet")

    original_name = doc_row["original_name"]

    trans_id = await storage.insert_translation(
        original_name=original_name,
        source_document_id=doc_id,
        target_lang=target_lang,
        owner_id=user_id,
    )

    try:
        await storage.update_translation(trans_id, status="translating")

        layout = await storage.fetch_ocr_layout(doc_id)
        stats = await _translate_with_checkpoints(
            layout, trans_id=trans_id, target_lang=target_lang,
            original_name=original_name,
        )
        # Fail loudly if nothing translated (don't persist an all-source DOCX).
        _guard_translation(stats, original_name=original_name)
        _guard_entry_counts(layout, original_name=original_name)

        await storage.update_translation(trans_id, status="reconstructing")
        await _hydrate_pictures_from_minio(layout, doc_id)

        translated_json_bytes = _json.dumps(
            [_strip_image_objs(p) for p in layout],
            indent=2, ensure_ascii=False,
        ).encode("utf-8")

        docx_buf = io.BytesIO()
        json_to_docx(layout, output_path=docx_buf)
        docx_bytes = docx_buf.getvalue()

        json_key = await storage.put_translated_artifact(
            trans_id, "translated_json", translated_json_bytes,
        )
        docx_key = await storage.put_translated_artifact(
            trans_id, "translated_docx", docx_bytes,
        )

        elapsed = round(time.perf_counter() - t0, 2)
        await storage.update_translation(
            trans_id,
            status="ok",
            translated_json_key=json_key,
            translated_docx_key=docx_key,
            elapsed_sec=elapsed,
            error=None,
        )
        # The finished artifacts supersede the checkpoint.
        await storage.delete_checkpoint(trans_id)
        await storage.update_translation(trans_id, checkpoint_key=None)

        return {
            "translation_id":   str(trans_id),
            "document_id":      str(doc_id),
            "original_name":    original_name,
            "status":           "ok",
            "target_lang":      target_lang,
            "elapsed_sec":      elapsed,
            "items_translated": stats.get("items_translated", 0),
            "chunks":           stats.get("chunks", 0),
            "failed":           stats.get("failed", 0),
        }
    except Exception as exc:
        log.exception("translate_from_doc_id failed for %s", doc_id)
        await storage.update_translation(
            trans_id, status="error", error=str(exc)[:1000],
        )
        raise


_NON_JSON_ENTRY_FIELDS = {
    "image_obj",      # PIL.Image — not JSON-serializable
    "_table_chain",   # back-reference into other entries; would loop
    "_shared_col_weights",  # rebuilt on the next reconstruction pass
}


def _strip_image_objs(page: dict) -> dict:
    """Remove non-serializable / non-JSON-safe fields before persisting JSON."""
    out_entries = []
    for e in page.get("entries") or []:
        out_entries.append({k: v for k, v in e.items() if k not in _NON_JSON_ENTRY_FIELDS})
    return {
        "page_index":     page.get("page_index", 0),
        "page_width_pt":  page.get("page_width_pt"),
        "page_height_pt": page.get("page_height_pt"),
        "zoom":           page.get("zoom"),
        "entries":        out_entries,
    }