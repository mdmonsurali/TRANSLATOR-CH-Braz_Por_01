from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import re
from typing import Any, Callable, Iterable, List, Optional, Tuple

import httpx

from llama_client import chat_completion

log = logging.getLogger("translator_service.translation")

TRANSLATE_CHUNK_ITEMS = int(os.getenv("TRANSLATE_CHUNK_ITEMS", "20"))
TRANSLATE_CHUNK_BYTES = int(os.getenv("TRANSLATE_CHUNK_BYTES", "4000"))
TRANSLATE_CONCURRENCY = int(os.getenv("TRANSLATE_CONCURRENCY", "2"))
TRANSLATE_RETRY_ONCE = os.getenv("TRANSLATE_RETRY_ONCE", "true").lower() == "true"
# Pages translated per group. Bounds every intermediate (units/sources/results/
# flat/retry copies) to the window instead of the whole document, and gives the
# pipeline a natural checkpoint boundary. 0 disables windowing.
TRANSLATE_PAGE_WINDOW = int(os.getenv("TRANSLATE_PAGE_WINDOW", "25"))

# Separator token injected between prose entries in a chunk so we can
# detect when the LLM merges or splits entries (count mismatch → per-item retry).
_ENTRY_SEP = "<<<ENTRY_SEP>>>"

# Form-checkbox glyphs (from the OCR ``<input type=checkbox>`` conversion) that
# appear in standalone prose blocks like "是否合格： ☑是 ☐否". They are symbols,
# not words, and an LLM asked to translate the prose can silently drop or
# reorder them. We MASK each glyph to an ASCII sentinel the model treats as a
# code (kept verbatim) before translation and RESTORE it after — so the checked/
# unchecked state always survives. Keyed on glyph TYPE, so restore is
# order-independent (every checked sentinel → ☑ regardless of reordering).
_CHECKBOX_GLYPHS = {"☑": "[[CBX_ON]]", "☐": "[[CBX_OFF]]"}
_CHECKBOX_UNMASK = {v: k for k, v in _CHECKBOX_GLYPHS.items()}


def _mask_checkbox_glyphs(s: str) -> str:
    for g, tok in _CHECKBOX_GLYPHS.items():
        s = s.replace(g, tok)
    return s


def _unmask_checkbox_glyphs(s: str) -> str:
    for tok, g in _CHECKBOX_UNMASK.items():
        s = s.replace(tok, g)
    return s

# Plain-prose categories: the whole `text` field is translated as-is.
TRANSLATABLE_CATEGORIES = {
    "Text", "Title", "Page-header", "Page-footer", "Section-header",
    "Caption", "List-item", "Footnote",
}
# Categories needing structure-aware handling (HTML / LaTeX): only the
# human-readable substrings are translated, markup is preserved verbatim.
TABLE_CATEGORIES = {"Table"}
# Chandra emits block math as "Equation-Block"; "Formula" is the legacy label.
# Both need structure-aware handling so only human-readable substrings (e.g.
# \text{...}) are translated and the LaTeX markup (\frac, ^{}, variable names)
# survives verbatim — otherwise the model rewrites the whole formula as prose
# and it renders as broken raw LaTeX.
FORMULA_CATEGORIES = {"Equation-Block", "Formula"}
# Categories whose text is NOT translated (they carry no natural-language
# prose). Everything else — including any future/unknown category the OCR
# emits — is treated as plain prose so nothing slips through untranslated.
NON_TEXT_CATEGORIES = {"Picture"}
# Set TRANSLATE_FORMULA_TEXT=false to leave formulas 100% untouched.
TRANSLATE_FORMULA_TEXT = os.getenv("TRANSLATE_FORMULA_TEXT", "true").lower() == "true"

# Targets the model into a strict JSON contract so we can parse the response
# without regex slop. Qwen3 follows this format reliably.
#
# NOTE: we intentionally do NOT ask the model to "preserve proper nouns" — that
# made the model leave Chinese company names ("山东威高骨科…"), product names and
# section labels untranslated. The only things that must survive verbatim are
# ASCII codes / numbers / units and LaTeX/HTML markup.
_SYSTEM_PROMPT_TEMPLATE = (
    "You are a professional translator. Translate EVERY input string into {lang}, "
    "including company names, brand and product names, section titles, table "
    "headers and short labels. Preserve line breaks, punctuation, and any inline "
    "math, LaTeX, or HTML exactly as written. Keep unchanged ONLY ASCII letter/"
    "digit product codes or model numbers (e.g. WGF8Z02-02, GK4-SOR-YF-012), "
    "numerals, and units of measurement. The output MUST NOT contain any Chinese, "
    "Japanese, or Korean characters. Do not add commentary, explanations, or "
    "quotation marks. Return ONLY a JSON object of the form "
    '{{"translations": ["...", "...", ...]}} '
    "with the same length and order as the input array. "
    "CRITICAL: the input items are delimited by the literal token <<<ENTRY_SEP>>> — "
    "you MUST return exactly one translation per input item, preserving that exact "
    "count. Never merge two items into one or split one item into two."
)

# Stricter prompt used by the residual retry pass for strings the first pass
# left with CJK characters still in them.
_STRICT_PROMPT_TEMPLATE = (
    "Translate the following text into {lang}. Translate ABSOLUTELY EVERY word, "
    "including names of companies, products, and people, and any titles or "
    "labels. If a name has no common {lang} form, transliterate it into the "
    "Latin alphabet. The result MUST NOT contain a single Chinese, Japanese, or "
    "Korean character. Keep unchanged ONLY ASCII alphanumeric product/model codes, "
    "numbers, units, and LaTeX/HTML markup. Output ONLY a JSON object "
    '{{"translations": ["..."]}} of the same length and order as the input.'
)

# CJK / Japanese kana / Korean Hangul ranges — used to detect text the model
# failed to translate so we can retry it.
_CJK_RESIDUAL_RE = re.compile(
    r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\uAC00-\uD7AF]"
)


def _contains_cjk(s: str) -> bool:
    return bool(_CJK_RESIDUAL_RE.search(s or ""))


# ── Response decoding (unchanged) ──────────────────────────────────────────

# Qwen3 (and other reasoning models) can emit a leading <think>...</think>
# block before the JSON answer. The llama-server `--reasoning-budget 0` flag
# suppresses this at the source, but strip it here too as a belt-and-suspenders
# guard so a stray reasoning block never breaks JSON parsing on any build.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def _strip_think_block(text: str) -> str:
    s = _THINK_BLOCK_RE.sub("", text)
    # An unterminated <think> (truncated / budget-clamped) leaves an open tag
    # with no closer — drop everything up to and including it.
    if "<think>" in s.lower():
        idx = s.lower().rfind("<think>")
        s = s[idx + len("<think>"):]
    return s.strip()


def _strip_code_fences(text: str) -> str:
    s = _strip_think_block(text)
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    return s


def _parse_translations(raw: str, expected_len: int) -> List[str] | None:
    text = _strip_code_fences(raw)
    try:
        obj = _json.loads(text)
    except _json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return None
        try:
            obj = _json.loads(m.group(0))
        except _json.JSONDecodeError:
            return None
    if isinstance(obj, dict):
        arr = obj.get("translations") or obj.get("items") or obj.get("output")
    elif isinstance(obj, list):
        arr = obj
    else:
        return None
    if not isinstance(arr, list) or len(arr) != expected_len:
        return None
    return [str(x) if x is not None else "" for x in arr]


async def _translate_chunk(items: List[str], target_lang: str,
                            client: httpx.AsyncClient) -> List[str]:
    if not items:
        return []
    system = _SYSTEM_PROMPT_TEMPLATE.format(lang=target_lang)
    # Embed the separator between items so the model sees them as distinct
    # units. We also send the JSON array so _parse_translations can still
    # validate the count from the structured response.
    sep_joined = _ENTRY_SEP.join(items)
    user = _json.dumps({"items": items, "_sep_hint": sep_joined}, ensure_ascii=False)

    async def _ask(payload: str, expected_len: int) -> List[str] | None:
        try:
            raw = await chat_completion(
                system=system,
                user=payload,
                client=client,
                response_format_json=True,
            )
        except (httpx.HTTPError, RuntimeError) as exc:
            log.warning("translation chunk call failed: %s", exc)
            return None
        return _parse_translations(raw, expected_len)

    parsed = await _ask(user, len(items))
    if parsed is None and TRANSLATE_RETRY_ONCE:
        log.info("chunk parse failed (count mismatch or bad JSON), retrying once")
        parsed = await _ask(user, len(items))

    if parsed is not None:
        # Sanity-check: none of the returned strings should contain the raw
        # separator (would mean the model echoed our internal token verbatim).
        if any(_ENTRY_SEP in s for s in parsed):
            log.warning("model echoed ENTRY_SEP token; falling back per-item")
            parsed = None

    if parsed is not None:
        return parsed

    # Per-item fallback: translate each entry individually. This is slower but
    # guarantees 1-to-1 mapping between source entries and translations.
    log.warning("falling back to per-item translation for chunk of %d", len(items))
    out: List[str] = []
    for it in items:
        single = await _ask(_json.dumps({"items": [it]}, ensure_ascii=False), 1)
        if single and len(single) == 1:
            out.append(single[0])
        else:
            log.warning("per-item fallback failed for item; keeping source text")
            out.append(it)
    return out


class _Unit:
    __slots__ = ("text", "_write")

    def __init__(self, text: str, write: Callable[[str], None]):
        self.text = text
        self._write = write

    def write(self, value: str) -> None:
        self._write(value)


_HAS_LETTER = None  # not used; kept for clarity


def _worth_translating(s: str) -> bool:
    """True if the string holds at least one letter (Latin OR CJK — both
    report True from str.isalpha). Pure numbers / codes / punctuation are
    skipped to save calls and avoid the model rewriting identifiers."""
    return any(ch.isalpha() for ch in (s or ""))


# Split an HTML string into a list of parts where every tag (<...>) is its own
# element and text between tags is its own element. Rejoining with "".join is
# lossless. Robust for the flat <table><tr><td>… markup the OCR emits.
_TAG_SPLIT = re.compile(r"(<[^>]+>)")

# Capture `\text{...}` groups in a formula (no nested braces in practice).
_LATEX_TEXT = re.compile(r"(\\text\{[^{}]*\})")


def _add_table_units(entry: dict, units: List[_Unit],
                     finalizers: List[Tuple[dict, List[str]]]) -> None:
    parts = _TAG_SPLIT.split(entry["text"])
    touched = False
    for i, part in enumerate(parts):
        if part.startswith("<"):
            continue                       # markup — never translate
        if not _worth_translating(part):
            continue                       # numbers / empty whitespace
        touched = True

        def _writer(value: str, _p=parts, _i=i) -> None:
            _p[_i] = value

        units.append(_Unit(part, _writer))
    if touched:
        finalizers.append((entry, parts))


def _add_formula_units(entry: dict, units: List[_Unit],
                       finalizers: List[Tuple[dict, List[str]]]) -> None:
    parts = _LATEX_TEXT.split(entry["text"])
    touched = False
    for i, part in enumerate(parts):
        m = re.fullmatch(r"\\text\{([^{}]*)\}", part)
        if not m:
            continue                       # raw math — leave alone
        inner = m.group(1)
        if not _worth_translating(inner):
            continue

        def _writer(value: str, _p=parts, _i=i) -> None:
            _p[_i] = "\\text{" + value + "}"

        units.append(_Unit(inner, _writer))
        touched = True
    if touched:
        finalizers.append((entry, parts))


def _collect_units(layout: List[dict]
                   ) -> Tuple[List[_Unit], List[Tuple[dict, List[str]]]]:
    """Walk every page envelope and build the flat list of translatable units.

    `finalizers` is a list of (entry, parts) for structured entries (tables,
    formulas); after the per-unit writers fire, each entry's `text` is rebuilt
    by joining its parts."""
    units: List[_Unit] = []
    finalizers: List[Tuple[dict, List[str]]] = []

    for page in layout:
        for entry in page.get("entries") or []:
            cat = entry.get("category")

            text = entry.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            if cat in TABLE_CATEGORIES:
                _add_table_units(entry, units, finalizers)

            elif cat in FORMULA_CATEGORIES:
                if TRANSLATE_FORMULA_TEXT:
                    _add_formula_units(entry, units, finalizers)

            elif cat in NON_TEXT_CATEGORIES:
                continue                       # no natural-language text

            else:
                # Plain-prose default: every remaining category (the known
                # TRANSLATABLE_CATEGORIES *and* any future/unknown category the
                # OCR starts emitting) has its whole `text` translated, so
                # nothing carrying prose is ever silently skipped.
                if _worth_translating(text):
                    def _writer(value: str, _e=entry) -> None:
                        _e["text"] = value
                    units.append(_Unit(text, _writer))

    return units, finalizers


def _chunk_indices(sources: List[str], max_items: int,
                   max_bytes: int) -> Iterable[Tuple[int, int]]:
    n = len(sources)
    i = 0
    while i < n:
        j = i
        size = 0
        while j < n and (j - i) < max_items:
            s_bytes = len(sources[j].encode("utf-8"))
            if j > i and size + s_bytes > max_bytes:
                break
            size += s_bytes
            j += 1
        yield i, j
        i = j


async def _force_translate(items: List[str], target_lang: str,
                           client: httpx.AsyncClient) -> List[str] | None:
    """One strict translation call for residual (still-CJK) strings."""
    if not items:
        return []
    system = _STRICT_PROMPT_TEMPLATE.format(lang=target_lang)
    user = _json.dumps({"items": items}, ensure_ascii=False)
    try:
        raw = await chat_completion(
            system=system, user=user, client=client, response_format_json=True,
        )
    except (httpx.HTTPError, RuntimeError) as exc:
        log.warning("strict retry call failed: %s", exc)
        return None
    return _parse_translations(raw, len(items))


async def _retry_residual_cjk(sources: List[str], flat: List[str],
                              target_lang: str, rounds: int = 2) -> tuple[int, int]:
    """Re-translate any unit whose result still contains CJK characters.

    Mutates `flat` in place, only accepting a retried value if it actually
    removed the CJK (never makes a string worse). Returns
    (num_units_retried, num_still_cjk_after)."""
    targets = [i for i in range(len(flat))
               if _contains_cjk(sources[i]) and _contains_cjk(flat[i])]
    if not targets:
        return 0, 0

    log.info("[translate] residual CJK retry: %d string(s) left untranslated",
             len(targets))
    async with httpx.AsyncClient(timeout=600.0) as client:
        for _round in range(rounds):
            pending = [i for i in targets if _contains_cjk(flat[i])]
            if not pending:
                break
            # Small batches first; then per-item for anything still stuck.
            for start in range(0, len(pending), 10):
                batch = pending[start:start + 10]
                items = [sources[i] for i in batch]
                out = await _force_translate(items, target_lang, client)
                if out and len(out) == len(items):
                    for i, val in zip(batch, out):
                        if val and not _contains_cjk(val):
                            flat[i] = val
                else:
                    for i in batch:
                        single = await _force_translate([sources[i]], target_lang, client)
                        if single and single[0] and not _contains_cjk(single[0]):
                            flat[i] = single[0]

    still = sum(1 for i in targets if _contains_cjk(flat[i]))
    if still:
        log.warning("[translate] %d string(s) still contain CJK after retry", still)
    return len(targets), still


async def translate_layout(layout: List[dict], target_lang: str = "pt-BR",
                           page_window: int = 0,
                           on_progress: Optional[Callable] = None) -> dict:
                           
    if page_window and page_window > 0 and len(layout) > page_window:
        return await _translate_layout_windowed(
            layout, target_lang, page_window, on_progress)
    return await _translate_layout_once(layout, target_lang)


async def _translate_layout_windowed(layout: List[dict], target_lang: str,
                                     page_window: int,
                                     on_progress: Optional[Callable]) -> dict:
    """Run `_translate_layout_once` over successive page groups, accumulating
    stats. Each group's intermediates are freed before the next one starts."""
    totals = {"items_translated": 0, "chunks": 0, "failed": 0,
              "retried": 0, "untranslated_cjk": 0}
    total_pages = len(layout)

    for start in range(0, total_pages, page_window):
        group = layout[start:start + page_window]
        stats = await _translate_layout_once(group, target_lang)
        for k in totals:
            totals[k] += int(stats.get(k, 0) or 0)

        done = min(start + page_window, total_pages)
        log.info("[translate] pages %d-%d done (%d/%d), %d item(s) so far",
                 start, done, done, total_pages, totals["items_translated"])
        if on_progress is not None:
            await on_progress(done, total_pages, dict(totals))

    return totals


async def _translate_layout_once(layout: List[dict], target_lang: str) -> dict:
    """Translate one batch of page envelopes in a single pass."""
    units, finalizers = _collect_units(layout)
    # Mask form-checkbox glyphs so the model preserves them verbatim (see
    # _mask_checkbox_glyphs). Restored on every result before write-back below.
    sources = [_mask_checkbox_glyphs(u.text) for u in units]
    if not sources:
        return {"items_translated": 0, "chunks": 0, "failed": 0}

    ranges = list(_chunk_indices(sources, TRANSLATE_CHUNK_ITEMS, TRANSLATE_CHUNK_BYTES))
    sem = asyncio.Semaphore(max(1, TRANSLATE_CONCURRENCY))
    results: List[List[str] | None] = [None] * len(ranges)

    async with httpx.AsyncClient(timeout=600.0) as client:
        async def run_one(slot: int, start: int, end: int) -> None:
            async with sem:
                results[slot] = await _translate_chunk(
                    sources[start:end], target_lang, client,
                )

        await asyncio.gather(*[
            run_one(slot, s, e) for slot, (s, e) in enumerate(ranges)
        ])

    failed = 0
    flat: List[str] = []
    for slot, (start, end) in enumerate(ranges):
        chunk = results[slot]
        expected = end - start
        if chunk is None or len(chunk) != expected:
            failed += expected
            flat.extend(sources[start:end])      # keep originals on failure
        else:
            flat.extend(chunk)

    # Second pass: re-translate anything the model echoed back with CJK still
    # in it (proper nouns, codes, short labels, or chunk failures). This is what
    # guarantees no Chinese is left behind in any category.
    retried, untranslated = await _retry_residual_cjk(sources, flat, target_lang)

    # Write each translated string back to its origin, restoring any masked
    # checkbox glyphs (☑/☐) the model kept as ASCII sentinels.
    for unit, translated in zip(units, flat):
        unit.write(_unmask_checkbox_glyphs(translated))

    # Rebuild structured entries (tables / formulas) from their parts.
    for entry, parts in finalizers:
        entry["text"] = "".join(parts)

    return {
        "items_translated": len(flat) - failed,
        "chunks": len(ranges),
        "failed": failed,
        "retried": retried,
        "untranslated_cjk": untranslated,
    }