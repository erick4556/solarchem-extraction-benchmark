"""Extract tables and in-text mentions from PDFs with LightOnOCR."""

from table_context_extract.context import caption_title
from table_context_extract.schema import (
    GroundTruthCorpus,
    GroundTruthDocument,
    Mention,
    Table,
    TableContext,
)

__version__ = "0.1.0"

__all__ = [
    "GroundTruthCorpus",
    "GroundTruthDocument",
    "Mention",
    "Table",
    "TableContext",
    "caption_title",
]
