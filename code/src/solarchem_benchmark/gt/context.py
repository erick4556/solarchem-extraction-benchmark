"""Extraction of the narrative context surrounding a table.

Everything is derived from the OCR transcription of the page, using positional
heuristics only:

* a caption is the block starting a line with ``Table N``, paired to the
  nearest unclaimed table block on the page; when that block holds nothing but
  the label, the block after it is absorbed as the caption text;
* the section title is the closest article-section heading above the table
  float (positional). Paper titles emitted as ``# ...`` are rejected; caption
  labels typed as headings are rejected;
* mentions are paragraphs elsewhere in the document that reference the table's
  number.

The caption regular expressions follow GAP-KGE's context generator so that
mention detection behaves identically across both benchmarks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from solarchem_benchmark.gt.normalize import normalize_scientific_text
from solarchem_benchmark.gt.tables import HTML_TABLE_RE

_NUMBER = r"[A-Za-z]?\d+(?:\.\d+)?|[IVXLC]+"

CAPTION_RE = re.compile(
    rf"(?:\*\*)?(?:Table|Tab\.?|TABLE)\s+({_NUMBER})(?:\*\*)?\s*[:.\u2014\-]?\s+",
    re.IGNORECASE,
)
TABLE_REFERENCE_RE = re.compile(
    rf"\b(?:Supplementary\s+)?(?:Tables?|Tab\.?|TABLES?)\s+({_NUMBER})"
    rf"(?:\s*(?:,|and|&)\s*(?:{_NUMBER}))*",
    re.IGNORECASE,
)
_REFERENCE_KEYWORD_RE = re.compile(r"(?:Supplementary\s+)?(?:Tables?|Tab\.?|TABLES?)\s*", re.IGNORECASE)
_REFERENCE_NUMBER_RE = re.compile(_NUMBER)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_NUMBERED_SECTION_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")
_NAMED_SECTION_RE = re.compile(
    r"^(?:Abstract|Introduction|Experimental(?: section)?|Materials and methods|"
    r"Methods|Results(?: and discussion)?|Discussion|Conclusions?|"
    r"Acknowledgments?|Supplementary(?: information)?)\b",
    re.IGNORECASE,
)

_LABEL_ONLY_RE = re.compile(
    rf"^[\s#>*_]*(?:Tables?|Tab\.?|TABLES?|Figs?\.?|Figures?|Schemes?)\s+"
    rf"(?:{_NUMBER})[\s*_]*[:.\u2014\-]?[\s*_]*$",
    re.IGNORECASE,
)
_DECORATION_PREFIX_RE = re.compile(r"^[\s#>*_]+")
_EMPHASIS_RE = re.compile(r"\*\*|__")
_BLANK_LINE_RE = re.compile(r"\n\s*\n")
_LINE_PREFIX_RE = re.compile(r"^[\s#>*_]*$")


def normalize_table_number(raw: str) -> str:
    """Canonicalise a table number so ``s1``, ``S1`` and ``S 1`` compare equal."""
    return raw.strip().upper().replace(" ", "")


def _clean_text(text: str) -> str:
    """Normalise a caption, heading or paragraph the same way a cell is.

    Two things happen here. Whitespace, line breaks included, collapses to
    single spaces, so that two ground-truth files can be compared without the
    source document's line wrapping affecting the score. And the scientific
    notation goes through :func:`normalize_scientific_text`, so a species named
    in a caption is spelled exactly as in the cell that measures it.
    """
    return normalize_scientific_text(text)


def _repair_hyphen_breaks(text: str) -> str:
    """Rejoin words split by a hyphen at a line break."""
    return re.sub(r"(\w)-\s*\n\s*([a-z][\w'-]*)", r"\1\2", text)


def split_paragraphs(text: str) -> list[str]:
    """Split page text into paragraphs.

    Blank lines are used when present. OCR output sometimes has none, in which
    case lines are regrouped into sentence-terminated chunks.

    Markdown headings are dropped rather than kept as paragraphs: they are
    already recorded as ``section_title``, and removing them leaves a blank line
    that correctly separates the sections' paragraphs.

    Args:
        text: Page text with table blocks already removed.

    Returns:
        Non-empty paragraphs with whitespace collapsed.
    """
    cleaned = _HEADING_RE.sub("", _MARKDOWN_IMAGE_RE.sub(" ", text))
    cleaned = _repair_hyphen_breaks(cleaned)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", cleaned) if p.strip()]
    if len(paragraphs) > 1:
        return [_clean_text(p) for p in paragraphs]

    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    if len(lines) <= 1:
        collapsed = _clean_text(cleaned)
        return [collapsed] if collapsed else []

    chunks: list[str] = []
    buffer: list[str] = []
    for line in lines:
        buffer.append(line)
        if line.endswith((".", "!", "?", '."', ".\u201d")):
            chunks.append(" ".join(buffer))
            buffer = []
    if buffer:
        chunks.append(" ".join(buffer))
    return [_clean_text(chunk) for chunk in chunks if chunk.strip()]


def referenced_table_numbers(paragraph: str) -> set[str]:
    """Return every table number referenced in a paragraph."""
    numbers: set[str] = set()
    for match in TABLE_REFERENCE_RE.finditer(paragraph):
        body = _REFERENCE_KEYWORD_RE.sub("", match.group(0))
        for number in _REFERENCE_NUMBER_RE.findall(body):
            numbers.add(normalize_table_number(number))
    return numbers


@dataclass(frozen=True)
class Caption:
    """A caption paragraph located on a page."""

    number: str
    text: str
    start: int
    end: int


def is_label_only(text: str) -> bool:
    """True when a block of text is nothing but a ``Table 1``-style label.

    Such a block is a typographic fragment, not content: it is neither a
    usable caption on its own nor an in-text mention of the table.
    """
    return bool(_LABEL_ONLY_RE.match(text))


def _clean_caption(text: str) -> str:
    """Drop Markdown decoration from a caption block and put it on one line."""
    return _clean_text(_EMPHASIS_RE.sub("", _DECORATION_PREFIX_RE.sub("", text)))


def caption_title(caption: str) -> str:
    """Strip the label prefix from a caption.

    ``Table 1: Photocatalytic CO2 reduction`` becomes ``Photocatalytic CO2
    reduction``. A caption that is only a label has no title and yields an
    empty string; one that opens with no recognisable label is returned
    unchanged.

    The ground truth stores the caption alone, since the title is exactly this
    function of it. Callers that want the title -- to prompt a model, or to
    fill ``dcterms:title`` on the RDF export -- apply it at read time.

    Args:
        caption: Full caption text.

    Returns:
        The caption's title proper.
    """
    if is_label_only(caption):
        return ""
    match = CAPTION_RE.match(caption)
    return caption[match.end() :].strip() if match else caption.strip()


def _opens_a_line(page_text: str, position: int) -> bool:
    """True when only Markdown decoration precedes ``position`` on its line.

    Captions always start a line. Requiring that keeps sentences such as
    ``...as listed in Table 1.\\nEven more noteworthy...`` from being read as
    a caption merely because a line break follows the number.
    """
    line_start = page_text.rfind("\n", 0, position) + 1
    return bool(_LINE_PREFIX_RE.match(page_text[line_start:position]))


def _block_at(text: str, offset: int) -> tuple[str, int]:
    """Return the block starting at ``offset`` and the offset of the next one."""
    separator = _BLANK_LINE_RE.search(text, offset)
    if separator is None:
        return text[offset:], len(text)
    return text[offset : separator.start()], separator.end()


def _is_caption_body(block: str) -> bool:
    """True when a block can serve as the text of a bare ``Table N`` label."""
    stripped = block.strip()
    if not stripped or is_label_only(stripped):
        return False
    return not stripped.startswith(("<", "|", "#", "!"))


def find_captions(page_text: str) -> list[Caption]:
    """Find every caption paragraph on a page, in reading order.

    Many publishers, Elsevier among them, typeset the label and the caption
    text as two separate blocks::

        Table 1

        Optical properties after modification of the benzene ring in UiO-66.

    A block holding nothing but the label therefore absorbs the block that
    follows it, so long as that block is prose rather than the table itself,
    a heading or another caption.
    """
    captions: list[Caption] = []
    for match in CAPTION_RE.finditer(page_text):
        start = match.start()
        if not _opens_a_line(page_text, start):
            continue

        block, next_offset = _block_at(page_text, start)
        text = _clean_caption(block)
        end = start + len(block)

        if is_label_only(text):
            body, _ = _block_at(page_text, next_offset)
            if _is_caption_body(body):
                text = f"{text} {_clean_caption(body)}"
                end = next_offset + len(body)

        if text:
            captions.append(
                Caption(
                    number=normalize_table_number(match.group(1)),
                    text=text,
                    start=start,
                    end=end,
                )
            )
    return captions


@dataclass
class TableOccurrence:
    """A table block on a page together with its resolved context."""

    page: int
    fragment: str
    start: int
    end: int
    caption: str = ""
    number: str | None = None
    section_title: str = ""


def _pair_captions(
    blocks: list[re.Match[str]],
    captions: list[Caption],
) -> list[tuple[re.Match[str], Caption | None]]:
    """Assign at most one caption to each table block.

    When the page has as many captions as tables, they are paired in reading
    order. Otherwise each block claims the nearest unclaimed caption by
    character distance.
    """
    if len(captions) == len(blocks):
        return list(zip(blocks, sorted(captions, key=lambda caption: caption.start)))

    claimed: set[int] = set()
    pairs: list[tuple[re.Match[str], Caption | None]] = []
    for block in blocks:
        best_index: int | None = None
        best_distance = float("inf")
        for index, caption in enumerate(captions):
            if index in claimed:
                continue
            if caption.end <= block.start():
                distance = block.start() - caption.end
            elif caption.start >= block.end():
                distance = caption.start - block.end()
            else:
                distance = 0
            if distance < best_distance:
                best_distance, best_index = distance, index
        if best_index is None:
            pairs.append((block, None))
        else:
            claimed.add(best_index)
            pairs.append((block, captions[best_index]))
    return pairs


def is_section_heading(text: str) -> bool:
    """True when a Markdown heading looks like an article section, not a title.

    LightOnOCR often emits the paper title as ``# ...`` on page 1. Using that
    as ``section_title`` pollutes the ground truth; only numbered sections and
    standard named sections are kept.
    """
    cleaned = _clean_text(text)
    if not cleaned or is_label_only(cleaned):
        return False
    if _NUMBERED_SECTION_RE.match(cleaned):
        return True
    if _NAMED_SECTION_RE.match(cleaned):
        return True
    return False


def _headings(page_text: str) -> list[re.Match[str]]:
    """Article-section Markdown headings on a page."""
    return [
        match
        for match in _HEADING_RE.finditer(page_text)
        if is_section_heading(match.group(1))
    ]


def _section_title(headings: list[re.Match[str]], position: int, carried: str) -> str:
    """Return the closest heading above ``position``."""
    above = [match for match in headings if match.start() < position]
    return _clean_text(above[-1].group(1)) if above else carried


def locate_tables(page_texts: list[str]) -> list[TableOccurrence]:
    """Locate every table block in a document and resolve its local context.

    Args:
        page_texts: OCR transcription of each page, in order.

    Returns:
        One occurrence per table block, in document order.
    """
    occurrences: list[TableOccurrence] = []
    carried_section = ""

    for page_index, page_text in enumerate(page_texts):
        blocks = list(HTML_TABLE_RE.finditer(page_text))
        headings = _headings(page_text)
        if headings:
            carried_section_after_page = _clean_text(headings[-1].group(1))
        else:
            carried_section_after_page = carried_section

        for block, caption in _pair_captions(blocks, find_captions(page_text)):
            occurrences.append(
                TableOccurrence(
                    page=page_index + 1,
                    fragment=block.group(0),
                    start=block.start(),
                    end=block.end(),
                    caption=caption.text if caption is not None else "",
                    number=caption.number if caption is not None else None,
                    section_title=_section_title(headings, block.start(), carried_section),
                )
            )

        carried_section = carried_section_after_page

    return occurrences


def collect_mentions(
    page_texts: list[str],
    number: str | None,
    *,
    exclude: set[str] | None = None,
) -> list[dict[str, object]]:
    """Collect in-text references to a table from the whole document.

    Args:
        page_texts: OCR transcription of each page, in order.
        number: Canonical table number, as parsed from the caption. When
            ``None`` no mention can be attributed and the list is empty.
        exclude: Paragraph texts to skip, typically the table's own caption.

    Returns:
        One ``{"page": int, "text": str}`` entry per referencing paragraph,
        deduplicated and in document order.
    """
    if not number:
        return []

    skip = exclude or set()
    mentions: list[dict[str, object]] = []
    seen: set[tuple[int, str]] = set()

    for page_index, page_text in enumerate(page_texts):
        body = HTML_TABLE_RE.sub(" ", page_text)
        for paragraph in split_paragraphs(body):
            if paragraph in skip or is_label_only(paragraph) or CAPTION_RE.match(paragraph):
                continue
            if number not in referenced_table_numbers(paragraph):
                continue
            key = (page_index + 1, paragraph)
            if key in seen:
                continue
            seen.add(key)
            mentions.append({"page": page_index + 1, "text": paragraph})

    return mentions
