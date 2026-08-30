"""Ground-truth generation: one PDF in, one ground-truth document out.

The pipeline is::

    PDF -> rendered pages -> OCR transcription
        -> table blocks located and paired with captions
        -> tables flattened to columns + rows
        -> context resolved (section title, in-text mentions)
        -> validated GroundTruthDocument

Tables and their context are emitted together, under the same table entry.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

from solarchem_benchmark.gt import context as context_module
from solarchem_benchmark.gt.metadata import pdf_title
from solarchem_benchmark.gt.ocr import OCREngine, transcribe_document
from solarchem_benchmark.gt.schema import (
    GroundTruthDocument,
    Mention,
    Table,
    TableContext,
)
from solarchem_benchmark.gt.tables import (
    extract_tables,
    merge_caption_parts,
    parse_html_table,
    table_number_from_caption,
)

logger = logging.getLogger(__name__)

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")
_ELSEVIER_SUFFIX_RE = re.compile(r"-main$")


def derive_document_id(pdf_path: Path, *, max_length: int = 60) -> str:
    """Derive a stable, filesystem-safe identifier from a PDF filename.

    The identifier is deterministic: the same filename always yields the same
    identifier, independently of how many other documents are processed.

    Args:
        pdf_path: Path to the source PDF.
        max_length: Maximum length of the slug part.

    Returns:
        An identifier such as ``solarchem_1_s2_0_s0021979721022451``.
    """
    stem = _ELSEVIER_SUFFIX_RE.sub("", pdf_path.stem)
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    slug = _SLUG_STRIP_RE.sub("_", ascii_stem.lower()).strip("_")
    return f"solarchem_{slug[:max_length].rstrip('_') or 'document'}"


def _table_label(number: str | None) -> str:
    return f"Table {number}" if number else ""


def build_document(
    page_texts: list[str],
    *,
    document_id: str,
    source_pdf: str,
    title: str = "",
) -> GroundTruthDocument:
    """Assemble a ground-truth document from page transcriptions.

    Args:
        page_texts: OCR transcription of each page, in order.
        document_id: Identifier of the document.
        source_pdf: Path to the PDF, relative to the corpus directory.
        title: Document title, empty when not reliably recoverable.

    Returns:
        A validated ground-truth document. Tables whose block could not be
        flattened into a rectangular grid are skipped, so a document may hold
        fewer tables than there are table blocks.
    """
    occurrences = context_module.locate_tables(page_texts)

    tables: list[Table] = []
    for occurrence in occurrences:
        flattened = parse_html_table(occurrence.fragment)
        if flattened is None:
            logger.debug(
                "%s: skipping unparseable table block on page %d",
                document_id,
                occurrence.page,
            )
            continue

        caption = merge_caption_parts(occurrence.caption, flattened.embedded_caption)
        number = occurrence.number or (
            table_number_from_caption(caption) if caption else None
        )
        captions_to_exclude = {caption} if caption else set()
        mentions = context_module.collect_mentions(
            page_texts, number, exclude=captions_to_exclude
        )

        table_index = len(tables) + 1
        tables.append(
            Table(
                table_id=f"{document_id}_table_{table_index:02d}",
                table_label=_table_label(number),
                page=occurrence.page,
                caption=caption,
                columns=flattened.columns,
                rows=flattened.rows,
                context=TableContext(
                    # Positional: closest valid heading above the table float.
                    # Do not retarget to the first mention — that would bake the
                    # Gold curation choice (semantic vs float) into the extractor.
                    section_title=occurrence.section_title,
                    mentions=[Mention(**mention) for mention in mentions],
                ),
            )
        )

    tables.extend(_markdown_only_tables(page_texts, document_id, len(tables)))

    return GroundTruthDocument(
        document_id=document_id,
        source_pdf=source_pdf,
        title=title,
        num_tables=len(tables),
        tables=tables,
    )


def _markdown_only_tables(
    page_texts: list[str],
    document_id: str,
    offset: int,
) -> list[Table]:
    """Recover tables from pages where the engine emitted Markdown, not HTML.

    Caption pairing and section resolution are positional and rely on
    ``<table>`` blocks, so pages without HTML yield structure without context.
    """
    recovered: list[Table] = []
    for page_index, page_text in enumerate(page_texts):
        if "<table" in page_text.lower():
            continue
        for _, flattened in extract_tables(page_text):
            index = offset + len(recovered) + 1
            number = (
                table_number_from_caption(flattened.embedded_caption)
                if flattened.embedded_caption
                else None
            )
            recovered.append(
                Table(
                    table_id=f"{document_id}_table_{index:02d}",
                    table_label=_table_label(number),
                    page=page_index + 1,
                    caption=flattened.embedded_caption,
                    columns=flattened.columns,
                    rows=flattened.rows,
                )
            )
    return recovered


def generate_for_pdf(
    pdf_path: Path,
    engine: OCREngine,
    *,
    document_id: str | None = None,
    corpus_dir: Path | None = None,
    cache_dir: Path | None = None,
    max_pages: int | None = None,
    force_ocr: bool = False,
) -> GroundTruthDocument:
    """Generate the ground truth for a single PDF.

    Args:
        pdf_path: Path to the source PDF.
        engine: OCR engine used to transcribe the pages.
        document_id: Explicit identifier; derived from the filename when absent.
        corpus_dir: Directory the recorded ``source_pdf`` is relative to.
        cache_dir: Directory for cached transcriptions.
        max_pages: Stop after this many pages, for smoke tests.
        force_ocr: Ignore any cached transcription.

    Returns:
        A validated ground-truth document.
    """
    resolved_id = document_id or derive_document_id(pdf_path)
    if corpus_dir is not None:
        try:
            source = pdf_path.resolve().relative_to(corpus_dir.resolve()).as_posix()
        except ValueError:
            source = pdf_path.name
    else:
        source = pdf_path.name

    logger.info("Processing %s as %s", pdf_path.name, resolved_id)
    page_texts = transcribe_document(
        pdf_path,
        engine,
        cache_dir=cache_dir,
        max_pages=max_pages,
        force=force_ocr,
    )
    document = build_document(
        page_texts,
        document_id=resolved_id,
        source_pdf=source,
        title=pdf_title(pdf_path),
    )
    logger.info("  %s: %d tables", resolved_id, document.num_tables)
    return document


def write_document(document: GroundTruthDocument, output_dir: Path) -> Path:
    """Write one ground-truth document as pretty-printed JSON.

    Args:
        document: The document to write.
        output_dir: Destination directory, created when missing.

    Returns:
        The path written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{document.document_id}.json"
    path.write_text(
        document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return path
