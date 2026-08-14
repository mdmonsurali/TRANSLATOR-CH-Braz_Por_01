"""RFC 6266 Content-Disposition builder that survives non-ASCII filenames.

HTTP header values must be latin-1 encodable (RFC 7230); Starlette enforces
this in Response.init_headers and raises UnicodeEncodeError otherwise. This
project OCRs Chinese source documents, so the original filename routinely
contains CJK characters and interpolating it straight into the header 500s
the download endpoint.

We emit the two-form header from RFC 6266: a latin-1-safe ``filename=`` for
old clients, plus the exact UTF-8 name in ``filename*=`` (RFC 5987 encoding)
which every modern browser prefers.

NOTE: there is no shared package across services (each has its own src/ and
its own Docker build context), so this module is duplicated verbatim into
orchestrator_service, ocr_service and translator_service. Keep the copies in
sync -- same convention already used for document_reconstruction.
"""

from __future__ import annotations

import os
import re
import unicodedata
from urllib.parse import quote

# Characters that would terminate or split the header if left in the quoted
# filename= form: control chars (CR/LF header injection), the quote itself,
# backslash (quoted-pair escape) and the parameter separator.
_UNSAFE = re.compile(r'[\x00-\x1f\x7f"\\;]')

# Left over when a CJK tail is stripped, e.g. "81 GK3-STP-GY-011-03-" -- keep
# the meaningful document code but drop the dangling separators.
_TRAILING_SEP = " \t-_.,"

# A language tag this pipeline appends itself: "_pt-BR", "_en", "_zh-Hans".
_LANG_TAG = re.compile(r"_[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")


def _ascii_fallback(name: str) -> str:
    """Best-effort ASCII transliteration of ``name``.

    NFKD splits accented letters into base + combining mark, so dropping
    non-ASCII keeps the base letter: "Relatório" -> "Relatorio". CJK has no
    ASCII decomposition and disappears entirely, which the caller detects by
    checking for an empty stem.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return _UNSAFE.sub("", ascii_only)


def content_disposition(disposition: str, filename: str,
                        fallback_id: str) -> str:
    """Build a ``Content-Disposition`` value safe for latin-1 transport.

    ``disposition`` is "attachment" or "inline". ``fallback_id`` is used as the
    ASCII stem when the name transliterates to nothing (a wholly non-ASCII
    name) -- pass the record UUID so the fallback stays unique per document.
    """
    filename = _UNSAFE.sub("", filename).strip() or f"{fallback_id}"

    orig_stem, ext = os.path.splitext(filename)

    # Judge survival on the ORIGINAL stem minus any "_<lang>" tag we appended
    # ourselves. Suffixes like "_pt-BR" are ASCII and would otherwise look like
    # surviving content, degrading "质量手册_pt-BR.docx" to the shared name
    # "pt-BR.docx" -- identical for every Chinese document, so downloads would
    # collide in the browser's folder.
    doc_part = re.sub(_LANG_TAG, "", orig_stem)
    tag = orig_stem[len(doc_part):]

    stem = _ascii_fallback(orig_stem).strip(_TRAILING_SEP)
    if not any(c.isalnum() for c in _ascii_fallback(doc_part)):
        # Wholly non-ASCII document name: fall back to the record id so the
        # ASCII form stays unique per document, keeping the language tag.
        fb = _ascii_fallback(str(fallback_id)).strip(_TRAILING_SEP) or "download"
        stem = f"{fb}{_ascii_fallback(tag)}".strip(_TRAILING_SEP) or fb
    ascii_name = f"{stem}{_ascii_fallback(ext)}"

    header = f'{disposition}; filename="{ascii_name}"'
    if ascii_name != filename:
        # Only add filename* when it actually carries more information, so
        # plain-ASCII names keep the exact header they had before.
        header += f"; filename*=UTF-8''{quote(filename, safe='')}"
    return header


def safe_zip_name(name: str, fallback: str) -> str:
    """Sanitize one path segment used as a ZIP entry name.

    ZIP entries are UTF-8 capable so CJK is kept as-is; this only strips path
    separators and control characters so an odd filename cannot produce a
    traversal-style entry.
    """
    cleaned = re.sub(r"[\x00-\x1f\x7f/\\]", "", name).replace("..", "").strip()
    return cleaned.strip(_TRAILING_SEP) or str(fallback)
