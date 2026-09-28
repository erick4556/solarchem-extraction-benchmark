"""Ask Laya which number sits in a hidden silver cell.

The silver table is the answer key. Each numeric cell is hidden. The question
names the row and the column, and the options are other numbers from that
table. This checks whether the model can read a table. It does not extract a
grid and it does not call the extractor evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from solarchem_benchmark.claims.link import build_table_view, predict
from solarchem_benchmark.claims.model import DecisionModel
from solarchem_benchmark.gt.context import caption_title
from solarchem_benchmark.gt.schema import GroundTruthDocument

#: Distinct numbers offered for one cell. The true value is always included.
MAX_READING_OPTIONS = 12

_CELL_QUESTION = "Which number belongs in the hidden cell?"


@dataclass(frozen=True)
class CellQuery:
    """One hidden cell and the choice question that asks for its value."""

    table_id: str
    table_label: str
    row_label: str
    column_label: str
    raw: str
    value: float
    state: str
    criteria: dict[str, str]
    correct_key: str

    @property
    def questions(self) -> dict[str, dict]:
        return {
            "cell": {
                "type": "choice",
                "instructions": _CELL_QUESTION,
                "criteria": self.criteria,
            }
        }


def _number_in_text(raw: str, text: str) -> bool:
    return re.search(rf"(?<!\d){re.escape(raw)}(?!\d)", text) is not None


def _row_label(view, row: int) -> str:
    table = view.table
    if view.label_column is not None and row < len(table.rows):
        label = str(table.rows[row][view.label_column]).strip()
        if label:
            return label
    return f"row {row + 1}"


def cell_queries(document: GroundTruthDocument) -> list[CellQuery]:
    """Hidden-cell questions for every numeric cell whose value is not already in the label."""
    queries: list[CellQuery] = []
    for table in document.tables:
        view = build_table_view(table)
        by_value: dict[float, str] = {}
        for cell in view.cells:
            by_value.setdefault(cell.value, cell.raw)
        if len(by_value) < 2:
            continue
        title = caption_title(table.caption).strip() or table.table_label
        for cell in view.cells:
            row_label = _row_label(view, cell.row)
            column_label = str(table.columns[cell.column])
            visible = f"{title}\n{row_label}\n{column_label}"
            if _number_in_text(cell.raw, visible):
                continue
            criteria, correct_key = _options(cell.value, cell.column, view.cells, by_value)
            if correct_key is None or len(criteria) < 2:
                continue
            state = (
                f"Table: {title}\n"
                f"Row: {row_label}\n"
                f"Column: {column_label}\n"
                "The cell where this row and this column meet is hidden."
            )
            queries.append(
                CellQuery(
                    table_id=table.table_id,
                    table_label=table.table_label,
                    row_label=row_label,
                    column_label=column_label,
                    raw=cell.raw,
                    value=cell.value,
                    state=state,
                    criteria=criteria,
                    correct_key=correct_key,
                )
            )
    return queries


def _options(
    true_value: float,
    column: int,
    cells,
    by_value: dict[float, str],
) -> tuple[dict[str, str], str | None]:
    """At most ``MAX_READING_OPTIONS`` numbers, always including the hidden one."""
    chosen = dict(by_value)
    if len(chosen) > MAX_READING_OPTIONS:
        same_column = {cell.value for cell in cells if cell.column == column}
        same_column.add(true_value)
        rest = sorted((value for value in by_value if value not in same_column), key=lambda v: abs(v - true_value))
        keep = set(same_column)
        for value in rest:
            if len(keep) >= MAX_READING_OPTIONS:
                break
            keep.add(value)
        if len(keep) > MAX_READING_OPTIONS:
            ranked = sorted(keep, key=lambda v: (v != true_value, abs(v - true_value)))
            keep = set(ranked[:MAX_READING_OPTIONS])
        chosen = {value: by_value[value] for value in keep}
    criteria = {f"v{index}": raw for index, (_, raw) in enumerate(sorted(chosen.items()), start=1)}
    correct = next((key for key, raw in criteria.items() if raw == by_value[true_value]), None)
    return criteria, correct


def run_table_reading(document: GroundTruthDocument, model: DecisionModel) -> dict[str, Any]:
    """Score hidden-cell questions for one silver document."""
    queries = cell_queries(document)
    if not queries:
        return {
            "document_id": document.document_id,
            "cells": 0,
            "correct": 0,
            "accuracy": None,
            "random_accuracy": None,
            "items": [],
        }

    groups: dict[tuple, list[int]] = {}
    for index, query in enumerate(queries):
        signature = tuple(query.criteria.items())
        groups.setdefault(signature, []).append(index)

    chosen: list[tuple[str, float]] = [("", 0.0)] * len(queries)
    for indexes in groups.values():
        sample = queries[indexes[0]]
        answers = predict(model, [queries[i].state for i in indexes], sample.questions)
        for index, answer in zip(indexes, answers):
            cell = answer.get("cell", {})
            key = str(cell.get("choice", ""))
            probability = float(cell.get("probabilities", {}).get(queries[index].correct_key, 0.0))
            chosen[index] = (key, probability)

    items = []
    correct = 0
    random_total = 0.0
    for query, (key, probability) in zip(queries, chosen):
        hit = key == query.correct_key
        correct += int(hit)
        random_total += 1 / len(query.criteria)
        items.append(
            {
                "table_id": query.table_id,
                "table_label": query.table_label,
                "row": query.row_label,
                "column": query.column_label,
                "value": query.raw,
                "choice": query.criteria.get(key, key),
                "correct": hit,
                "n_options": len(query.criteria),
                "p_correct": round(probability, 4),
            }
        )
    cells = len(queries)
    return {
        "document_id": document.document_id,
        "cells": cells,
        "correct": correct,
        "accuracy": round(correct / cells, 4),
        "random_accuracy": round(random_total / cells, 4),
        "items": items,
    }
