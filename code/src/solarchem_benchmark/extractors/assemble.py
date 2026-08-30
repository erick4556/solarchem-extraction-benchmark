"""Turn raw extractor grids into the shared GroundTruthDocument schema."""

from __future__ import annotations

import logging
from pathlib import Path

from solarchem_benchmark.extractors.base import ExtractedGrid, NativeTableExtractor
from solarchem_benchmark.gt.generate import derive_document_id
from solarchem_benchmark.gt.metadata import pdf_title
from solarchem_benchmark.gt.schema import GroundTruthDocument, Mention, Table, TableContext
from solarchem_benchmark.gt.tables import flatten_cell_grid, table_number_from_caption

logger = logging.getLogger(__name__)


def _table_label(number: str | None) -> str:
    return f"Table {number}" if number else ""


def build_prediction_document(
    grids: list[ExtractedGrid],
    *,
    document_id: str,
    source_pdf: str,
    title: str = "",
) -> GroundTruthDocument:
    """Flatten grids into a document.

    Phase 5–6 leave caption/section/mentions empty unless a ``Table N``
    spanner sits in the grid. Phase 7 Ollama extractors may set those fields
    on the grid; they are copied onto the schema table after flatten.
    GROBID does the same from TEI (caption, section heading, table refs).
    """
    tables: list[Table] = []
    for grid in grids:
        flattened = flatten_cell_grid(grid.rows)
        if flattened is None:
            logger.debug(
                "%s: skipping unflattenable grid on page %d",
                document_id,
                grid.page,
            )
            continue
        caption = (grid.caption or flattened.embedded_caption or "").strip()
        number = table_number_from_caption(caption) if caption else None
        if number is None and grid.table_label:
            number = table_number_from_caption(grid.table_label)
        table_index = len(tables) + 1
        label = grid.table_label.strip() or _table_label(number)
        mentions = [
            Mention(page=mention.page, text=mention.text)
            for mention in grid.mentions
            if mention.text.strip()
        ]
        tables.append(
            Table(
                table_id=f"{document_id}_table_{table_index:02d}",
                table_label=label,
                page=grid.page,
                caption=caption,
                columns=flattened.columns,
                rows=flattened.rows,
                context=TableContext(
                    section_title=grid.section_title.strip(),
                    mentions=mentions,
                ),
            )
        )

    return GroundTruthDocument(
        document_id=document_id,
        source_pdf=source_pdf,
        title=title,
        num_tables=len(tables),
        tables=tables,
    )


def extract_document(
    pdf_path: Path,
    extractor: NativeTableExtractor,
    *,
    document_id: str | None = None,
    corpus_dir: Path | None = None,
    max_pages: int | None = None,
) -> GroundTruthDocument:
    """Run one native extractor on a PDF and return a schema-valid document."""
    resolved_id = document_id or derive_document_id(pdf_path)
    if corpus_dir is not None:
        try:
            source = pdf_path.resolve().relative_to(corpus_dir.resolve()).as_posix()
        except ValueError:
            source = pdf_path.name
    else:
        source = pdf_path.name

    logger.info("Extracting %s with %s as %s", pdf_path.name, extractor.tool_id, resolved_id)
    grids = extractor.extract_grids(pdf_path, max_pages=max_pages)
    document = build_prediction_document(
        grids,
        document_id=resolved_id,
        source_pdf=source,
        title=pdf_title(pdf_path),
    )
    logger.info("  %s: %d tables", resolved_id, document.num_tables)
    return document
