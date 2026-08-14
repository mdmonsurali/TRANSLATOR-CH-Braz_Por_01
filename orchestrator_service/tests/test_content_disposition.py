"""Content-Disposition encoding tests.

Regression cover for a 500 on every download of a Chinese-named document:
HTTP header values must be latin-1 encodable (RFC 7230, enforced by Starlette
in Response.init_headers), but the filename is built from the uploaded
document name, which in this project is routinely CJK. Interpolating it raised

    UnicodeEncodeError: 'latin-1' codec can't encode characters in position
    43-55: ordinal not in range(256)

The fix emits the RFC 6266 two-form header: a transliterated ASCII filename=
plus the exact name in filename*=UTF-8''.

Run:  python -m pytest orchestrator_service/tests/test_content_disposition.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import unquote

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from content_disposition import content_disposition, safe_zip_name  # noqa: E402

# The document that actually produced the reported 500. 21 ASCII characters
# followed by 13 CJK characters, which is exactly where the traceback pointed.
REAL_NAME = "81 GK3-STP-GY-011-03-医用器械类材料进货检验规程_pt-BR.docx"
UID = "27d8b31c-08fe-42e5-b1eb-9f926b918584"


def _filename(header: str) -> str:
    return header.split('filename="', 1)[1].split('"', 1)[0]


def _ext_filename(header: str) -> str:
    return unquote(header.split("filename*=UTF-8''", 1)[1])


@pytest.mark.parametrize("disposition,name", [
    ("attachment", REAL_NAME),
    ("inline", REAL_NAME.replace(".docx", ".json")),
    ("attachment", "质量手册_pt-BR.docx"),
    ("attachment", "医用.docx"),
    ("attachment", "Relatório_Técnico_pt-BR.docx"),
    ("attachment", "Report_2024_pt-BR.docx"),
    ("inline", "report.md"),
])
def test_header_is_latin1_encodable(disposition, name):
    """The actual bug: the header must survive transport encoding."""
    header = content_disposition(disposition, name, UID)
    header.encode("latin-1")  # would raise UnicodeEncodeError before the fix


def test_real_failing_document_round_trips():
    header = content_disposition("attachment", REAL_NAME, UID)
    header.encode("latin-1")
    # The true name is preserved for modern clients...
    assert _ext_filename(header) == REAL_NAME
    # ...while the ASCII form keeps the meaningful document code.
    assert _filename(header) == "81 GK3-STP-GY-011-03-_pt-BR.docx"


def test_wholly_non_ascii_name_falls_back_to_record_id():
    """Nothing survives transliteration, so the id keeps the name unique."""
    header = content_disposition("attachment", "质量手册_pt-BR.docx", UID)
    assert _filename(header) == f"{UID}_pt-BR.docx"
    assert _ext_filename(header) == "质量手册_pt-BR.docx"


def test_distinct_cjk_documents_do_not_collide():
    """Two Chinese names must not both degrade to a bare "pt-BR.docx"."""
    other = "99999999-1111-2222-3333-444444444444"
    a = content_disposition("attachment", "质量手册_pt-BR.docx", UID)
    b = content_disposition("attachment", "另一个文件_pt-BR.docx", other)
    assert _filename(a) != _filename(b)


def test_accents_transliterate_rather_than_vanish():
    header = content_disposition("attachment", "Relatório_Técnico.docx", UID)
    assert _filename(header) == "Relatorio_Tecnico.docx"


@pytest.mark.parametrize("name", [
    "Report_2024_pt-BR.docx",
    "my_report_v2.docx",
    "report.md",
])
def test_pure_ascii_header_unchanged(name):
    """Existing English documents keep byte-identical headers (no filename*)."""
    header = content_disposition("attachment", name, UID)
    assert header == f'attachment; filename="{name}"'


@pytest.mark.parametrize("evil", [
    'ev"il.docx',
    "ev;il.docx",
    "ev\r\nX-Injected: 1.docx",
    "ev\\il.docx",
])
def test_header_injection_is_stripped(evil):
    """CR/LF, quotes, backslash and ';' must not escape the quoted string."""
    header = content_disposition("attachment", evil, UID)
    header.encode("latin-1")
    assert "\r" not in header and "\n" not in header
    body = _filename(header)
    assert '"' not in body and ";" not in body and "\\" not in body


def test_empty_name_uses_fallback():
    header = content_disposition("attachment", "", UID)
    assert UID in _filename(header)


def test_safe_zip_name_strips_traversal_but_keeps_cjk():
    assert "/" not in safe_zip_name("../../etc/passwd", "fb")
    assert ".." not in safe_zip_name("../../etc/passwd", "fb")
    # CJK is legal in a zip entry and should be preserved.
    assert safe_zip_name("医用器械", "fb") == "医用器械"
    assert safe_zip_name("", "fb") == "fb"
