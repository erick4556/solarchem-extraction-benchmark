"""Adapters for pdfplumber, Camelot (lattice/stream) and PyMuPDF.

Third-party packages are imported inside the extract methods so listing tools
or running tests does not require them.
"""

from __future__ import annotations

import logging
from pathlib import Path

from solarchem_benchmark.extractors.base import ExtractedGrid, NativeTableExtractor

logger = logging.getLogger(__name__)


def _page_limit(n_pages: int, max_pages: int | None) -> int:
    if max_pages is None or max_pages <= 0:
        return n_pages
    return min(n_pages, max_pages)


class PdfPlumberExtractor(NativeTableExtractor):
    """Baseline native extractor: ``page.extract_tables()`` with default settings."""

    tool_id = "pdfplumber"

    def extract_grids(
        self,
        pdf_path: Path,
        *,
        max_pages: int | None = None,
    ) -> list[ExtractedGrid]:
        try:
            import pdfplumber
        except ImportError as error:
            raise ImportError(
                "pdfplumber is not installed. "
                'Install with: pip install -e ".[pdfplumber]"'
            ) from error

        grids: list[ExtractedGrid] = []
        with pdfplumber.open(pdf_path) as document:
            last = _page_limit(len(document.pages), max_pages)
            for index in range(last):
                page = document.pages[index]
                try:
                    raw_tables = page.extract_tables() or []
                except Exception as error:  # noqa: BLE001 - one bad page must not drop the PDF
                    logger.debug(
                        "%s page %d: pdfplumber extract_tables failed: %s",
                        pdf_path.name,
                        index + 1,
                        error,
                    )
                    continue
                for raw in raw_tables:
                    if raw:
                        grids.append(ExtractedGrid(page=index + 1, rows=raw))
        return grids


class CamelotExtractor(NativeTableExtractor):
    """Camelot lattice (ruled tables) or stream (whitespace-separated)."""

    def __init__(self, flavor: str) -> None:
        if flavor not in {"lattice", "stream"}:
            raise ValueError(f"Camelot flavor must be lattice or stream, got {flavor!r}")
        self.flavor = flavor
        self.tool_id = f"camelot_{flavor}"

    def extract_grids(
        self,
        pdf_path: Path,
        *,
        max_pages: int | None = None,
    ) -> list[ExtractedGrid]:
        try:
            import camelot
        except ImportError as error:
            raise ImportError(
                "camelot-py is not installed. "
                'Install with: pip install -e ".[camelot]". '
                "Lattice flavour also needs Ghostscript on PATH."
            ) from error

        pages = "all" if max_pages is None or max_pages <= 0 else f"1-{max_pages}"
        tables = camelot.read_pdf(str(pdf_path), flavor=self.flavor, pages=pages)
        grids: list[ExtractedGrid] = []
        for table in tables:
            raw = [list(row) for row in table.data]
            if raw:
                grids.append(ExtractedGrid(page=int(table.page), rows=raw))
        return grids


class PyMuPDFExtractor(NativeTableExtractor):
    """PyMuPDF ``page.find_tables()`` (MuPDF table finder)."""

    tool_id = "pymupdf"

    def extract_grids(
        self,
        pdf_path: Path,
        *,
        max_pages: int | None = None,
    ) -> list[ExtractedGrid]:
        try:
            import fitz
        except ImportError as error:
            raise ImportError(
                "PyMuPDF is not installed. "
                'Install with: pip install -e ".[pymupdf]"'
            ) from error

        grids: list[ExtractedGrid] = []
        document = fitz.open(pdf_path)
        try:
            last = _page_limit(document.page_count, max_pages)
            for index in range(last):
                page = document.load_page(index)
                try:
                    finder = page.find_tables()
                except Exception as error:  # noqa: BLE001
                    logger.debug(
                        "%s page %d: PyMuPDF find_tables failed: %s",
                        pdf_path.name,
                        index + 1,
                        error,
                    )
                    continue
                found = getattr(finder, "tables", None) or list(finder or [])
                for table in found:
                    try:
                        raw = table.extract()
                    except Exception as error:  # noqa: BLE001
                        logger.debug(
                            "%s page %d: PyMuPDF table.extract failed: %s",
                            pdf_path.name,
                            index + 1,
                            error,
                        )
                        continue
                    if raw:
                        grids.append(ExtractedGrid(page=index + 1, rows=raw))
        finally:
            document.close()
        return grids
