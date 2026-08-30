"""Automatic generation of the SolarChem table + table-context ground truth."""

from solarchem_benchmark.gt.context import caption_title
from solarchem_benchmark.gt.schema import (
    GroundTruthDocument,
    Mention,
    Table,
    TableContext,
)

__all__ = [
    "GroundTruthDocument",
    "Mention",
    "Table",
    "TableContext",
    "caption_title",
]
