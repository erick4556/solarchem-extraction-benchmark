"""Article text for claim extraction: pages, sentences and the numbers they state.

Pages come from the LightOn OCR cache when it exists (Markdown with HTML
tables and ``#`` headings, the same text the silver ground truth was built
from) and otherwise from the PDF text layer via pypdfium2. Either way the text
goes through :func:`normalize_scientific_text`, so a species or unit in a
sentence is spelled exactly as in the silver table cells it may match.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from solarchem_benchmark.gt.context import (
    CAPTION_RE,
    TABLE_REFERENCE_RE,
    is_label_only,
    is_section_heading,
    normalize_table_number,
    referenced_table_numbers,
)
from solarchem_benchmark.gt.normalize import normalize_scientific_text
from solarchem_benchmark.gt.tables import HTML_TABLE_RE

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_BLANK_LINE_RE = re.compile(r"\n\s*\n+")
_END_OF_BODY_RE = re.compile(
    r"^(?:\d+\.?\s*)?(?:References|Bibliography|Acknowledg(?:e)?ments?|"
    r"Appendix [A-Z]\.? Supplementary)\b",
    re.IGNORECASE,
)
_FIGURE_CAPTION_RE = re.compile(r"^(?:\*\*)?(?:Fig\.?|Figure|Scheme)\s*\d+[a-z]?(?:\*\*)?\s*[.:|]", re.I)
_PLAIN_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z][^.]{2,120}$")
_SOFT_HYPHEN_BREAK_RE = re.compile(r"[\ufffe\u00ad]\s*")
_DIGITS_RE = re.compile(r"\d+")

_ABBREVIATIONS = (
    "Fig", "Figs", "Eq", "Eqs", "Ref", "Refs", "Tab", "No", "Nos", "vs", "ca", "approx",
    "wt", "at", "e.g", "i.e", "cf", "resp", "min", "max", "ref", "al", "etc", "Vol", "pp",
)
_ABBREVIATION_END_RE = re.compile(
    r"(?:\b(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")|\b[A-Z])\.$"
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

_NUMBER_RE = re.compile(r"(?<![\w^.\-\u2212/])[-\u2212]?\d+(?:\.\d+)?")
_REFERENCE_SPAN_RE = re.compile(
    r"\b(?:Supplementary\s+)?(?:Tables?|Tab\.|Figs?\.|Figures?|Eqs?\.|Equations?|Schemes?|"
    r"Refs?\.|Sections?|Sect\.)\s*\(?[A-Z]?\d+[a-z]?\)?"
    r"(?:\s*(?:,|and|&|to|\u2013|-)\s*\(?[A-Z]?\d+[a-z]?\)?)*",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*[,\u2013-]\s*\d+)*\s*\]")
# Integer loadings that name a sample (``10 wt.% In-doped TiO2``, ``5%In-TiO2``)
# rather than state a measured value.
_SAMPLE_LOADING_RE = re.compile(
    r"(?<![\d.])\d+\s*(?:wt\.?|mol\.?|at\.?)?\s*%\s*(?=[A-Z(]|-[A-Za-z])"
)
_UNIT_AFTER_RE = re.compile(r"\s?([%\u00b0\u25e6\u00ba]\s?C?|[A-Za-z\u00b5\u03bc][\w/^\-]*)")
# Longest spellings first so ``mw/cm2`` wins over ``mw`` and ``umole`` over ``umol``.
_KNOWN_UNIT_RE = re.compile(
    r"^(mw/cm2|m2/g|m2g-1|cm3/g|umole|mmol|umol|mole|degc|cm3|kpa|mpa|atm|bar|ppm|rpm|mev|"
    r"min|mol|nm|um|mm|cm|ev|ml|hr|pa|mw|mg|kg|mv|ma|deg|%|k|h|s|w|g|v)(?:[-/.\d].*)?$"
)
_NUMERIC_TOKEN_RE = re.compile(r"^[-\u2212(]?\d+(?:\.\d+)?[)%,;]?$")
_FRONT_MATTER_RE = re.compile(
    r"(?:E-mail|Tel\.|Fax|https?://|www\.|\u00a9|Received \d|Accepted \d|Available online|"
    r"Contents lists available|journal homepage|All rights reserved|Corresponding author|"
    r"ARTICLE INFO|Article history|Keywords:)",
    re.IGNORECASE,
)

_CUE_RE = re.compile(
    r"\b(?:increas|decreas|higher|lower|enhanc|improv|reduc|compar|than|maximum|minimum|"
    r"optim|highest|lowest|yield|rate|selectiv|band ?gap|surface area|efficien|activit|"
    r"produc|evolution|shift|attribut|due to|indicat|show|exhibit|reveal|observ|confirm|"
    r"suggest|measur|obtain|achiev|conver)",
    re.IGNORECASE,
)

MIN_SENTENCE_CHARS = 40
MAX_SENTENCE_CHARS = 900


@dataclass(frozen=True)
class Sentence:
    """One sentence of the article body."""

    sentence_id: str
    page: int
    section: str
    text: str

    @property
    def table_refs(self) -> set[str]:
        """Canonical numbers of the tables this sentence names explicitly."""
        return referenced_table_numbers(self.text)


@dataclass(frozen=True)
class StatedNumber:
    """A number written in a sentence, with the precision it was written at."""

    value: float
    decimals: int
    raw: str
    unit: str = ""

    @property
    def informative(self) -> bool:
        """False for zero, years and one-digit integers, which match cells by chance."""
        if self.value == 0:
            return False
        if self.decimals == 0 and 1900 <= abs(self.value) <= 2100:
            return False
        return self.decimals > 0 or abs(self.value) >= 10


def load_pages(pdf_path: Path, *, ocr_cache_dir: Path | None = None) -> tuple[list[str], str]:
    """Return the page texts of a PDF and which source they came from.

    Args:
        pdf_path: Source PDF.
        ocr_cache_dir: ``data/intermediate/ocr_cache/<engine>/``. The cached
            transcription ``<stem>.json`` is preferred when present.

    Returns:
        ``(pages, source)`` where ``source`` is ``"ocr_cache"`` or ``"pdf_text"``.
    """
    if ocr_cache_dir is not None:
        cache_file = ocr_cache_dir / f"{pdf_path.stem}.json"
        if cache_file.is_file():
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            return list(payload["pages"]), "ocr_cache"

    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        pages = []
        for page in document:
            textpage = page.get_textpage()
            pages.append(textpage.get_text_range())
            textpage.close()
            page.close()
    finally:
        document.close()
    return pages, "pdf_text"


def _repair_hyphen_breaks(text: str) -> str:
    text = _SOFT_HYPHEN_BREAK_RE.sub("", text)
    return re.sub(r"(\w)-\s*\n\s*([a-z][\w'-]*)", r"\1\2", text)


def _running_lines(pages: list[str]) -> set[str]:
    """Header/footer lines of the PDF text layer, i.e. lines repeated across pages.

    Digits are masked before counting so ``... 162 (2015) 98-109 101`` and
    ``... 162 (2015) 98-109 103`` count as the same running header.
    """
    if len(pages) < 3:
        return set()
    counts: dict[str, int] = {}
    for page in pages:
        for key in {_DIGITS_RE.sub("#", line.strip()) for line in page.splitlines() if line.strip()}:
            counts[key] = counts.get(key, 0) + 1
    threshold = max(3, int(0.3 * len(pages)))
    return {key for key, count in counts.items() if count >= threshold}


def _heading_text(block: str, *, markdown: bool) -> str | None:
    """Return the heading text when ``block`` is an article-section heading."""
    stripped = block.strip()
    if markdown:
        match = _HEADING_RE.match(stripped)
        if match is None:
            return None
        title = match.group(1)
        return title if is_section_heading(title) or _END_OF_BODY_RE.match(title) else ""
    if "\n" in stripped or len(stripped) > 130:
        return None
    if _PLAIN_HEADING_RE.match(stripped) or _END_OF_BODY_RE.match(stripped):
        return stripped
    return None


def _is_caption(block: str) -> bool:
    stripped = block.strip().lstrip("#>*_ ")
    return bool(CAPTION_RE.match(stripped) or _FIGURE_CAPTION_RE.match(stripped))


def _paragraphs(pages: list[str], *, markdown: bool) -> list[tuple[int, str, str]]:
    """Body paragraphs as ``(page, section, text)``, stopping at the references."""
    out: list[tuple[int, str, str]] = []
    section = ""
    running = set() if markdown else _running_lines(pages)
    for page_number, raw in enumerate(pages, start=1):
        text = _MARKDOWN_IMAGE_RE.sub(" ", HTML_TABLE_RE.sub("\n\n", raw))
        text = _repair_hyphen_breaks(text)
        blocks = _BLANK_LINE_RE.split(text) if markdown else text.splitlines()

        buffer: list[str] = []
        skipping_caption = False

        def flush() -> None:
            if buffer:
                out.append((page_number, section, " ".join(buffer)))
                buffer.clear()

        for block in blocks:
            stripped = block.strip()
            if not stripped or _DIGITS_RE.sub("#", stripped) in running:
                continue
            heading = _heading_text(stripped, markdown=markdown)
            if heading is not None:
                if markdown or heading:
                    flush()
                if heading and _END_OF_BODY_RE.match(heading):
                    return out
                if heading:
                    section = normalize_scientific_text(heading)
                continue
            if markdown:
                if _is_caption(stripped) or is_label_only(stripped) or stripped.startswith("|"):
                    continue
                out.append((page_number, section, stripped))
                continue
            # PDF text layer: one block per printed line, so captions span
            # several lines and paragraphs are rebuilt by joining lines.
            if _is_caption(stripped):
                flush()
                skipping_caption = not stripped.endswith(".")
                continue
            if skipping_caption:
                skipping_caption = not stripped.endswith(".")
                continue
            buffer.append(stripped)
        flush()
    return out


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences without breaking at ``Fig.``, ``wt.``, ``et al.``."""
    pieces = _SENTENCE_BOUNDARY_RE.split(paragraph)
    sentences: list[str] = []
    for piece in pieces:
        if sentences and _ABBREVIATION_END_RE.search(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return [s.strip() for s in sentences if s.strip()]


def document_sentences(pages: list[str], *, source: str, document_id: str) -> list[Sentence]:
    """Every body sentence of a document, in reading order."""
    sentences: list[Sentence] = []
    for page, section, paragraph in _paragraphs(pages, markdown=source == "ocr_cache"):
        for text in split_sentences(normalize_scientific_text(paragraph)):
            sentences.append(
                Sentence(
                    sentence_id=f"{document_id}_s{len(sentences) + 1:04d}",
                    page=page,
                    section=section,
                    text=text,
                )
            )
    return sentences


def is_candidate(sentence: Sentence) -> bool:
    """Cheap high-recall filter applied before the model sees a sentence."""
    if not MIN_SENTENCE_CHARS <= len(sentence.text) <= MAX_SENTENCE_CHARS:
        return False
    if looks_like_table_residue(sentence.text) or _FRONT_MATTER_RE.search(sentence.text):
        return False
    if sentence.table_refs:
        return True
    return bool(stated_numbers(sentence.text)) or bool(_CUE_RE.search(sentence.text))


def looks_like_table_residue(text: str) -> bool:
    """True for table bodies that the PDF text layer emits as running text.

    A prose sentence listing a series (``43, 61, 84, 98 and 123 m2/g``) stays
    below both limits; a flattened table row block does not.
    """
    tokens = text.split()
    numbers = len(stated_numbers(text))
    if numbers >= 12 and numbers / max(1, len(tokens)) >= 0.3:
        return True
    run = longest = 0
    for token in tokens:
        run = run + 1 if _NUMERIC_TOKEN_RE.match(token) else 0
        longest = max(longest, run)
    return longest >= 6


def stated_numbers(text: str) -> list[StatedNumber]:
    """Numbers a sentence states as values.

    Numbers that only label something -- ``Table 3``, ``Fig. 11``, ``Eq. (4)``,
    citation brackets -- are removed first, and digits bound to a formula or a
    unit exponent (``TiO2``, ``g^-1``) never match the pattern.
    """
    body = _CITATION_RE.sub(" ", _REFERENCE_SPAN_RE.sub(" ", text))
    body = _SAMPLE_LOADING_RE.sub(" ", body)
    numbers = []
    for match in _NUMBER_RE.finditer(body):
        raw = match.group(0).replace("\u2212", "-")
        decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
        unit_match = _UNIT_AFTER_RE.match(body, match.end())
        known = _KNOWN_UNIT_RE.match(normalize_unit_text(unit_match.group(1))) if unit_match else None
        numbers.append(
            StatedNumber(
                value=float(raw),
                decimals=decimals,
                raw=raw,
                unit=known.group(1) if known else "",
            )
        )
    return numbers


def normalize_unit_text(text: str) -> str:
    """Lower-case, spaceless unit spelling: ``m^2 /g`` -> ``m2/g``, ``°C`` -> ``degc``."""
    text = text.lower()
    for mark in ("\u00b0", "\u25e6", "\u00ba"):
        text = text.replace(mark, "deg")
    for micro in ("\u00b5", "\u03bc"):
        text = text.replace(micro, "u")
    return re.sub(r"[\s^{}]", "", text)


def mask_table_references(text: str) -> str:
    """Replace explicit table references so the model must link on content."""
    return TABLE_REFERENCE_RE.sub("the table", text)


def canonical_table_number(label: str) -> str | None:
    """``"Table 3"`` -> ``"3"``; ``None`` when the label carries no number."""
    numbers = referenced_table_numbers(label)
    if len(numbers) == 1:
        return next(iter(numbers))
    match = re.search(r"([A-Z]?\d+)", label or "")
    return normalize_table_number(match.group(1)) if match else None
