"""Ground-truth data model.

The model contains only fields that are compared against a tool prediction
during evaluation. Pipeline bookkeeping (hashes, engine names, timestamps,
review state) is deliberately absent so that two ground-truth files can be
diffed and scored without stripping metadata first; that information belongs to
the run manifest instead.

One file is produced per document, holding both the table structure and the
table context.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

CellValue = str | int | float
"""A flattened cell: a number when the raw text parses as one, else the text."""


class _Model(BaseModel):
    """Base model rejecting unknown fields so schema drift fails loudly."""

    model_config = ConfigDict(extra="forbid")


class Mention(_Model):
    """An in-text reference to the table, with the sentence(s) around it."""

    page: Annotated[int, Field(ge=1, description="1-based page number.")]
    text: str


class TableContext(_Model):
    """Narrative context attached to a single table.

    Context is restricted to what identifies where the table sits in the
    article and where it is referred to: the section it belongs to, and the
    paragraphs that name it explicitly.
    """

    section_title: str = Field(
        default="",
        description="Heading of the section the table appears under.",
    )
    mentions: list[Mention] = Field(
        default_factory=list,
        description="Paragraphs elsewhere in the article that reference the table.",
    )


class Table(_Model):
    """A table in its flattened, evaluable form.

    ``columns`` and ``rows`` form a rectangular grid: merged cells have been
    expanded and multi-level headers collapsed into one label per column.
    """

    table_id: str
    table_label: str = Field(default="", description='Identifier only, e.g. "Table 1".')
    page: Annotated[int, Field(ge=1, description="1-based page number.")]
    caption: str = Field(
        default="",
        description=(
            "Full caption as printed, label included. The title alone is "
            "solarchem_benchmark.gt.context.caption_title(caption)."
        ),
    )
    columns: list[str]
    rows: list[list[CellValue]]
    context: TableContext = Field(default_factory=TableContext)

    @model_validator(mode="after")
    def _check_rectangular(self) -> Table:
        width = len(self.columns)
        for index, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(
                    f"{self.table_id}: row {index} has {len(row)} cells "
                    f"but the table declares {width} columns"
                )
        return self


class GroundTruthDocument(_Model):
    """Ground truth for one source PDF."""

    document_id: str
    source_pdf: str = Field(description="Path relative to the corpus directory.")
    title: str = Field(default="", description="Title of the article.")
    num_tables: int = Field(ge=0)
    tables: list[Table] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_table_count(self) -> GroundTruthDocument:
        if self.num_tables != len(self.tables):
            raise ValueError(
                f"{self.document_id}: num_tables={self.num_tables} "
                f"but {len(self.tables)} tables are present"
            )
        table_ids = [table.table_id for table in self.tables]
        if len(set(table_ids)) != len(table_ids):
            raise ValueError(f"{self.document_id}: duplicate table_id values")
        return self


class GroundTruthCorpus(_Model):
    """Optional single-file view over the whole corpus."""

    documents: list[GroundTruthDocument] = Field(default_factory=list)


def json_schema() -> dict:
    """Return the JSON Schema of a ground-truth document."""
    return GroundTruthDocument.model_json_schema()
