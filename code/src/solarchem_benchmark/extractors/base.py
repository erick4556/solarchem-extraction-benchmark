"""Shared interface for table extractors (Phase 5–7).

These tools emit a cell grid per table. The common assemble step then applies
the same flatten / coerce / caption-peel rules as the OCR ground-truth
pipeline so that predictions share the GT schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple

NATIVE_TOOL_IDS = (
    "pdfplumber",
    "camelot_lattice",
    "camelot_stream",
    "pymupdf",
)

LAYOUT_TOOL_IDS = (
    "docling",
    "tatr",
    "unstructured",
)

GROBID_TOOL_IDS = ("grobid",)

OLLAMA_TOOL_IDS = (
    "ollama_qwen3_vl",
    "ollama_gemma4",
    "ollama_mistral_small",
)

TOOL_IDS = NATIVE_TOOL_IDS + LAYOUT_TOOL_IDS + GROBID_TOOL_IDS + OLLAMA_TOOL_IDS


class ExtractedMention(NamedTuple):
    """One in-text table mention, before schema assembly."""

    page: int
    text: str


class ExtractedGrid(NamedTuple):
    """One table as a raw cell grid, before flattening.

    Caption / section / mentions stay empty for Phase 5–6 extractors.
    Phase 7 GROBID and Ollama VLMs may fill them.
    """

    page: int
    rows: list[list[object | None]]
    caption: str = ""
    table_label: str = ""
    section_title: str = ""
    mentions: tuple[ExtractedMention, ...] = ()


class NativeTableExtractor(ABC):
    """Adapter around a native-PDF table library.

    ``extract_grids`` must not flatten or coerce cells. Caption / section /
    mentions stay empty unless the extractor sets ``emits_context``.
    """

    tool_id: str
    emits_context: bool = False

    @abstractmethod
    def extract_grids(
        self,
        pdf_path: Path,
        *,
        max_pages: int | None = None,
    ) -> list[ExtractedGrid]:
        """Return every table grid found in ``pdf_path``, in reading order."""


def available_tools() -> list[str]:
    """Return Phase 5, Phase 6 and Phase 7 extractor identifiers."""
    return list(TOOL_IDS)


def build_extractor(tool_id: str, **options) -> NativeTableExtractor:
    """Construct an extractor. Third-party imports happen on first use."""
    from solarchem_benchmark.extractors.layout import (
        DoclingExtractor,
        TATRExtractor,
        UnstructuredExtractor,
    )
    from solarchem_benchmark.extractors.native import (
        CamelotExtractor,
        PdfPlumberExtractor,
        PyMuPDFExtractor,
    )
    from solarchem_benchmark.extractors.grobid import GrobidExtractor
    from solarchem_benchmark.extractors.ollama import OllamaExtractor, ollama_model_for

    mapping: dict[str, NativeTableExtractor] = {
        "pdfplumber": PdfPlumberExtractor(),
        "camelot_lattice": CamelotExtractor("lattice"),
        "camelot_stream": CamelotExtractor("stream"),
        "pymupdf": PyMuPDFExtractor(),
        "docling": DoclingExtractor(),
        "tatr": TATRExtractor(),
        "unstructured": UnstructuredExtractor(),
        "grobid": GrobidExtractor(host=options.get("grobid_host")),
    }
    if tool_id in OLLAMA_TOOL_IDS:
        return OllamaExtractor(
            tool_id=tool_id,
            model=options.get("ollama_model") or ollama_model_for(tool_id),
            host=options.get("ollama_host"),
        )
    if tool_id not in mapping:
        known = ", ".join(TOOL_IDS)
        raise ValueError(f"Unknown extractor {tool_id!r}. Choose one of: {known}")
    return mapping[tool_id]


_ENVIRONMENT_MARKERS = (
    "ghostscript",
    "gs not found",
    "numpy.dtype size changed",
    "binary incompatibility",
    "cannot reach ollama",
    "cannot reach grobid",
    "ollama http",
    "try pulling",
    "connection refused",
)


def is_environment_error(error: BaseException) -> bool:
    """True when the failure is a missing install / ABI mix / Ollama, not a bad PDF."""
    if isinstance(error, ImportError):
        return True
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    message = str(error).lower()
    return any(marker in message for marker in _ENVIRONMENT_MARKERS)
