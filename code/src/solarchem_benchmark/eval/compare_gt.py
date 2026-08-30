"""Score a prediction corpus against a Gold or silver reference.

Only documents present in the reference file are scored. A prediction corpus
may be smaller: missing documents still contribute their reference tables as
detection misses. Cell and column scores stay restricted to matched tables.

The summary is grouped by field. Each field exposes the same four numbers:

* **accuracy** — exact agreement (string, header list, or cell), or Jaccard
  for detection / mentions (there is no meaningful true-negative count);
* **precision / recall / f1** — token overlap for caption, columns and
  section; set overlap for mentions; table-level detection for detection;
  positional cell hits for cells.

Token scores use bag-of-tokens overlap after the same scientific
normalisation as exact match. A wrong number or a swapped header is never
rescued by embeddings.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from solarchem_benchmark.gt.normalize import normalize_scientific_text
from solarchem_benchmark.gt.schema import GroundTruthDocument, Table

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./^-][a-z0-9]+)*", re.IGNORECASE)
_SOFT_MENTION_THRESHOLD = 0.5


@dataclass(frozen=True)
class PRF:
    """Precision / recall / F1 for one comparison."""

    precision: float
    recall: float
    f1: float


def _norm(value: object) -> str:
    return normalize_scientific_text(str(value))


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_norm(text).lower())


def token_prf(gold: str, predicted: str) -> PRF:
    """Bag-of-tokens precision, recall and F1 after scientific normalisation."""
    gold_tokens = _tokens(gold)
    pred_tokens = _tokens(predicted)
    if not gold_tokens and not pred_tokens:
        return PRF(1.0, 1.0, 1.0)
    if not gold_tokens or not pred_tokens:
        return PRF(0.0, 0.0, 0.0)

    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum(min(gold_counts[token], pred_counts[token]) for token in gold_counts)
    if overlap == 0:
        return PRF(0.0, 0.0, 0.0)

    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(gold_counts.values())
    return PRF(precision, recall, _harmonic(precision, recall))


def token_f1(gold: str, predicted: str) -> float:
    """Bag-of-tokens F1 between two strings after scientific normalisation."""
    return token_prf(gold, predicted).f1


def token_recall(gold: str, predicted: str) -> float:
    """Fraction of gold tokens found in ``predicted`` (order-invariant)."""
    return token_prf(gold, predicted).recall


def _harmonic(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _cell_equal(gold: object, pred: object) -> bool:
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return abs(float(gold) - float(pred)) <= 1e-6 * max(1.0, abs(float(gold)))
    return _norm(gold) == _norm(pred)


def _index_by_source(documents: list[GroundTruthDocument]) -> dict[str, GroundTruthDocument]:
    indexed: dict[str, GroundTruthDocument] = {}
    for document in documents:
        key = document.source_pdf.split("/")[-1]
        indexed[key] = document
        indexed[document.document_id] = document
    return indexed


def _match_table_index(gold: Table, predicted: list[Table]) -> int | None:
    for index, table in enumerate(predicted):
        if gold.table_label and table.table_label == gold.table_label:
            return index
    for index, table in enumerate(predicted):
        if table.page == gold.page:
            return index
    return None


def _match_table(gold: Table, predicted: list[Table]) -> Table | None:
    index = _match_table_index(gold, predicted)
    return None if index is None else predicted[index]


def _soft_hit(gold_text: str, pred_text: str, *, threshold: float) -> bool:
    scores = token_prf(gold_text, pred_text)
    return scores.f1 >= threshold or scores.recall >= threshold


@dataclass(frozen=True)
class MentionScores:
    exact_recall: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    mean_token_f1: float


def _mention_scores(
    gold_texts: list[str],
    pred_texts: list[str],
    *,
    soft_threshold: float = _SOFT_MENTION_THRESHOLD,
) -> MentionScores:
    """Set-level mention scores (soft) plus exact-set recall and mean token F1."""
    if not gold_texts and not pred_texts:
        return MentionScores(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    if not gold_texts or not pred_texts:
        return MentionScores(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    gold_norm = [_norm(text) for text in gold_texts]
    pred_norm = [_norm(text) for text in pred_texts]
    exact_recall = len(set(gold_norm) & set(pred_norm)) / len(gold_norm)

    best_f1s: list[float] = []
    gold_hits = 0
    for gold_text in gold_texts:
        best_f1s.append(max(token_f1(gold_text, pred_text) for pred_text in pred_texts))
        if any(_soft_hit(gold_text, pred_text, threshold=soft_threshold) for pred_text in pred_texts):
            gold_hits += 1

    pred_hits = sum(
        1
        for pred_text in pred_texts
        if any(_soft_hit(gold_text, pred_text, threshold=soft_threshold) for gold_text in gold_texts)
    )

    recall = gold_hits / len(gold_texts)
    precision = pred_hits / len(pred_texts)
    false_positives = len(pred_texts) - pred_hits
    false_negatives = len(gold_texts) - gold_hits
    union = gold_hits + false_positives + false_negatives
    accuracy = gold_hits / union if union else 1.0
    return MentionScores(
        exact_recall=exact_recall,
        precision=precision,
        recall=recall,
        f1=_harmonic(precision, recall),
        accuracy=accuracy,
        mean_token_f1=sum(best_f1s) / len(best_f1s),
    )


def _columns_token_prf(gold_columns: list[str], pred_columns: list[str]) -> PRF | None:
    """Mean token P/R/F1 over paired headers when both sides have the same width."""
    if len(gold_columns) != len(pred_columns):
        return None
    if not gold_columns:
        return PRF(1.0, 1.0, 1.0)
    scores = [token_prf(gold, pred) for gold, pred in zip(gold_columns, pred_columns)]
    count = len(scores)
    return PRF(
        sum(item.precision for item in scores) / count,
        sum(item.recall for item in scores) / count,
        sum(item.f1 for item in scores) / count,
    )


def _cell_scores(
    gold: Table,
    predicted: Table,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Positional cell accuracy / precision / recall / F1.

    Same grid shape: every gold cell has a counterpart, so the four numbers
    coincide. Different shape: the table scores 0 on all four (no alignment).
    """
    gold_cells = sum(len(row) for row in gold.rows)
    if len(gold.rows) == len(predicted.rows) and len(gold.columns) == len(predicted.columns):
        hits = 0
        total = 0
        for gold_row, pred_row in zip(gold.rows, predicted.rows):
            for gold_cell, pred_cell in zip(gold_row, pred_row):
                total += 1
                if _cell_equal(gold_cell, pred_cell):
                    hits += 1
        if not total:
            return None, None, None, None
        value = hits / total
        return value, value, value, value

    if not gold_cells:
        return 0.0, 0.0, 0.0, 0.0
    return 0.0, 0.0, 0.0, 0.0


@dataclass
class TableScore:
    table_id: str
    matched: bool
    caption_exact: bool = False
    caption_token_precision: float | None = None
    caption_token_recall: float | None = None
    caption_token_f1: float | None = None
    columns_exact: bool = False
    columns_token_precision: float | None = None
    columns_token_recall: float | None = None
    columns_token_f1: float | None = None
    cell_accuracy: float | None = None
    cell_precision: float | None = None
    cell_recall: float | None = None
    cell_f1: float | None = None
    section_title_exact: bool = False
    section_title_token_precision: float | None = None
    section_title_token_recall: float | None = None
    section_title_token_f1: float | None = None
    mention_recall_exact: float | None = None
    mention_precision: float | None = None
    mention_recall: float | None = None
    mention_f1: float | None = None
    mention_accuracy: float | None = None
    mention_mean_token_f1: float | None = None
    detail: str = ""

    # Back-compat aliases used by the first pilot report field names.
    @property
    def caption(self) -> bool:
        return self.caption_exact

    @property
    def columns(self) -> bool:
        return self.columns_exact

    @property
    def section_title(self) -> bool:
        return self.section_title_exact

    @property
    def mention_recall_soft(self) -> float | None:
        return self.mention_recall

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DocumentScore:
    source_pdf: str
    present_in_prediction: bool
    tables: list[TableScore] = field(default_factory=list)


def _round4(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _mean(values) -> float | None:
    values = list(values)
    if not values:
        return None
    return round(sum(float(value) for value in values) / len(values), 4)


def _metrics(
    *,
    accuracy: float | None,
    precision: float | None,
    recall: float | None,
    f1: float | None,
) -> dict[str, float | None]:
    return {
        "accuracy": _round4(accuracy),
        "precision": _round4(precision),
        "recall": _round4(recall),
        "f1": _round4(f1),
    }


def _detection_metrics(
    *,
    gold_tables: int,
    tables_matched: int,
    predicted_tables: int,
    predicted_tables_matched: int,
) -> dict[str, float | None]:
    if gold_tables == 0 and predicted_tables == 0:
        return _metrics(accuracy=1.0, precision=1.0, recall=1.0, f1=1.0)
    recall = None if gold_tables == 0 else tables_matched / gold_tables
    precision = (
        1.0
        if predicted_tables == 0 and gold_tables == 0
        else 0.0
        if predicted_tables == 0
        else predicted_tables_matched / predicted_tables
    )
    false_positives = predicted_tables - predicted_tables_matched
    false_negatives = gold_tables - tables_matched
    union = tables_matched + false_positives + false_negatives
    accuracy = None if union == 0 else tables_matched / union
    f1 = None if precision is None or recall is None else _harmonic(precision, recall)
    return _metrics(accuracy=accuracy, precision=precision, recall=recall, f1=f1)


@dataclass
class EvaluationReport:
    documents: list[DocumentScore]
    missing_from_prediction: list[str]
    predicted_tables: int = 0
    predicted_tables_matched: int = 0

    def summary(self) -> dict[str, object]:
        table_scores = [
            score
            for document in self.documents
            for score in document.tables
        ]
        matched = [score for score in table_scores if score.matched]
        return {
            "gold_documents": len(self.documents),
            "gold_documents_found_in_prediction": sum(
                1 for document in self.documents if document.present_in_prediction
            ),
            "missing_from_prediction": self.missing_from_prediction,
            "gold_tables": len(table_scores),
            "predicted_tables": self.predicted_tables,
            "tables_matched": len(matched),
            "detection": _detection_metrics(
                gold_tables=len(table_scores),
                tables_matched=len(matched),
                predicted_tables=self.predicted_tables,
                predicted_tables_matched=self.predicted_tables_matched,
            ),
            "columns": _metrics(
                accuracy=_mean(score.columns_exact for score in matched),
                precision=_mean(
                    score.columns_token_precision
                    for score in matched
                    if score.columns_token_precision is not None
                ),
                recall=_mean(
                    score.columns_token_recall
                    for score in matched
                    if score.columns_token_recall is not None
                ),
                f1=_mean(
                    score.columns_token_f1
                    for score in matched
                    if score.columns_token_f1 is not None
                ),
            ),
            "cells": _metrics(
                accuracy=_mean(
                    score.cell_accuracy
                    for score in matched
                    if score.cell_accuracy is not None
                ),
                precision=_mean(
                    score.cell_precision
                    for score in matched
                    if score.cell_precision is not None
                ),
                recall=_mean(
                    score.cell_recall
                    for score in matched
                    if score.cell_recall is not None
                ),
                f1=_mean(
                    score.cell_f1
                    for score in matched
                    if score.cell_f1 is not None
                ),
            ),
            "caption": _metrics(
                accuracy=_mean(score.caption_exact for score in matched),
                precision=_mean(
                    score.caption_token_precision
                    for score in matched
                    if score.caption_token_precision is not None
                ),
                recall=_mean(
                    score.caption_token_recall
                    for score in matched
                    if score.caption_token_recall is not None
                ),
                f1=_mean(
                    score.caption_token_f1
                    for score in matched
                    if score.caption_token_f1 is not None
                ),
            ),
            "section": _metrics(
                accuracy=_mean(score.section_title_exact for score in matched),
                precision=_mean(
                    score.section_title_token_precision
                    for score in matched
                    if score.section_title_token_precision is not None
                ),
                recall=_mean(
                    score.section_title_token_recall
                    for score in matched
                    if score.section_title_token_recall is not None
                ),
                f1=_mean(
                    score.section_title_token_f1
                    for score in matched
                    if score.section_title_token_f1 is not None
                ),
            ),
            "mentions": _metrics(
                accuracy=_mean(
                    score.mention_accuracy
                    for score in matched
                    if score.mention_accuracy is not None
                ),
                precision=_mean(
                    score.mention_precision
                    for score in matched
                    if score.mention_precision is not None
                ),
                recall=_mean(
                    score.mention_recall
                    for score in matched
                    if score.mention_recall is not None
                ),
                f1=_mean(
                    score.mention_f1
                    for score in matched
                    if score.mention_f1 is not None
                ),
            ),
        }

    def structure_summary(self) -> dict[str, object]:
        """Detection / columns / cells only — for structure-only tools (Phase 5)."""
        full = self.summary()
        return {
            "gold_documents": full["gold_documents"],
            "gold_documents_found_in_prediction": full["gold_documents_found_in_prediction"],
            "missing_from_prediction": full["missing_from_prediction"],
            "gold_tables": full["gold_tables"],
            "predicted_tables": full["predicted_tables"],
            "tables_matched": full["tables_matched"],
            "detection": full["detection"],
            "columns": full["columns"],
            "cells": full["cells"],
        }


def score_table(gold: Table, predicted: Table | None) -> TableScore:
    if predicted is None:
        return TableScore(table_id=gold.table_id, matched=False, detail="not found in prediction")

    caption_exact = _norm(gold.caption) == _norm(predicted.caption)
    columns_exact = [_norm(c) for c in gold.columns] == [_norm(c) for c in predicted.columns]
    caption_prf = token_prf(gold.caption, predicted.caption)
    columns_prf = _columns_token_prf(gold.columns, predicted.columns)
    cell_accuracy, cell_precision, cell_recall, cell_f1 = _cell_scores(gold, predicted)
    section_exact = _norm(gold.context.section_title) == _norm(predicted.context.section_title)
    section_prf = token_prf(gold.context.section_title, predicted.context.section_title)

    gold_mentions = [mention.text for mention in gold.context.mentions if mention.text]
    pred_mentions = [mention.text for mention in predicted.context.mentions if mention.text]
    mentions = _mention_scores(gold_mentions, pred_mentions)

    return TableScore(
        table_id=gold.table_id,
        matched=True,
        caption_exact=caption_exact,
        caption_token_precision=_round4(caption_prf.precision),
        caption_token_recall=_round4(caption_prf.recall),
        caption_token_f1=_round4(caption_prf.f1),
        columns_exact=columns_exact,
        columns_token_precision=_round4(None if columns_prf is None else columns_prf.precision),
        columns_token_recall=_round4(None if columns_prf is None else columns_prf.recall),
        columns_token_f1=_round4(None if columns_prf is None else columns_prf.f1),
        cell_accuracy=_round4(cell_accuracy),
        cell_precision=_round4(cell_precision),
        cell_recall=_round4(cell_recall),
        cell_f1=_round4(cell_f1),
        section_title_exact=section_exact,
        section_title_token_precision=_round4(section_prf.precision),
        section_title_token_recall=_round4(section_prf.recall),
        section_title_token_f1=_round4(section_prf.f1),
        mention_recall_exact=_round4(mentions.exact_recall),
        mention_precision=_round4(mentions.precision),
        mention_recall=_round4(mentions.recall),
        mention_f1=_round4(mentions.f1),
        mention_accuracy=_round4(mentions.accuracy),
        mention_mean_token_f1=_round4(mentions.mean_token_f1),
    )


def evaluate(
    gold_documents: list[GroundTruthDocument],
    prediction_documents: list[GroundTruthDocument],
) -> EvaluationReport:
    """Score each Gold document against a silver / tool prediction corpus."""
    predicted = _index_by_source(prediction_documents)
    scores: list[DocumentScore] = []
    missing: list[str] = []
    predicted_tables = 0
    predicted_tables_matched = 0

    for gold in gold_documents:
        key = gold.source_pdf.split("/")[-1]
        pred_doc = predicted.get(key) or predicted.get(gold.document_id)
        if pred_doc is None:
            missing.append(key)
            scores.append(
                DocumentScore(
                    source_pdf=key,
                    present_in_prediction=False,
                    tables=[score_table(table, None) for table in gold.tables],
                )
            )
            continue

        used: set[int] = set()
        table_scores: list[TableScore] = []
        for table in gold.tables:
            index = _match_table_index(table, pred_doc.tables)
            if index is None:
                table_scores.append(score_table(table, None))
                continue
            used.add(index)
            table_scores.append(score_table(table, pred_doc.tables[index]))

        predicted_tables += len(pred_doc.tables)
        predicted_tables_matched += len(used)
        scores.append(
            DocumentScore(source_pdf=key, present_in_prediction=True, tables=table_scores)
        )

    return EvaluationReport(
        documents=scores,
        missing_from_prediction=missing,
        predicted_tables=predicted_tables,
        predicted_tables_matched=predicted_tables_matched,
    )
