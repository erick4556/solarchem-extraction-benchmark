"""Compare an automatic (silver) ground truth against a Gold reference.

Only documents present in the Gold / reference file are scored. A prediction
corpus may be smaller: missing documents still contribute their Gold tables as
detection misses, so ``detection_recall`` uses the full reference denominator.
Cell and column scores stay restricted to matched tables.

Two families of text metrics are reported side by side:

* **exact** — normalised string equality (strict, what the first pilot used);
* **token_f1** — bag-of-tokens F1 after the same normalisation (softer, for
  captions, section titles and mentions). Cells and column *identity* stay
  exact: a wrong number or a swapped header must not be rescued by embeddings.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from solarchem_benchmark.gt.normalize import normalize_scientific_text
from solarchem_benchmark.gt.schema import GroundTruthDocument, Table

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./^-][a-z0-9]+)*", re.IGNORECASE)


def _norm(value: object) -> str:
    return normalize_scientific_text(str(value))


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_norm(text).lower())


def token_f1(gold: str, predicted: str) -> float:
    """Bag-of-tokens F1 between two strings after scientific normalisation."""
    gold_tokens = _tokens(gold)
    pred_tokens = _tokens(predicted)
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum(min(gold_counts[token], pred_counts[token]) for token in gold_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(gold_counts.values())
    return 2 * precision * recall / (precision + recall)


def token_recall(gold: str, predicted: str) -> float:
    """Fraction of gold tokens found in ``predicted`` (order-invariant).

    Useful for mentions: the OCR paragraph is often longer than the curated
    Gold sentence, so F1 alone under-penalises a correct longer span.
    """
    gold_tokens = _tokens(gold)
    pred_tokens = _tokens(predicted)
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum(min(gold_counts[token], pred_counts[token]) for token in gold_counts)
    return overlap / sum(gold_counts.values())


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


def _match_table(gold: Table, predicted: list[Table]) -> Table | None:
    for table in predicted:
        if gold.table_label and table.table_label == gold.table_label:
            return table
    for table in predicted:
        if table.page == gold.page:
            return table
    return None


def _mention_scores(
    gold_texts: list[str],
    pred_texts: list[str],
    *,
    soft_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Return exact-set recall, soft recall, and mean best-token-F1."""
    if not gold_texts:
        empty = 1.0 if not pred_texts else 0.0
        return empty, empty, empty

    gold_norm = [_norm(text) for text in gold_texts]
    pred_norm = [_norm(text) for text in pred_texts]
    exact = len(set(gold_norm) & set(pred_norm)) / len(gold_norm)

    best_f1s: list[float] = []
    soft_hits = 0
    for gold_text in gold_texts:
        if not pred_texts:
            best_f1s.append(0.0)
            continue
        best_f1 = max(token_f1(gold_text, pred_text) for pred_text in pred_texts)
        best_recall = max(token_recall(gold_text, pred_text) for pred_text in pred_texts)
        best_f1s.append(best_f1)
        if best_f1 >= soft_threshold or best_recall >= soft_threshold:
            soft_hits += 1

    soft_recall = soft_hits / len(gold_texts)
    mean_f1 = sum(best_f1s) / len(best_f1s)
    return exact, soft_recall, mean_f1


def _columns_token_f1(gold_columns: list[str], pred_columns: list[str]) -> float | None:
    """Mean token-F1 over paired headers when both sides have the same width."""
    if len(gold_columns) != len(pred_columns):
        return None
    if not gold_columns:
        return 1.0
    return round(
        sum(token_f1(g, p) for g, p in zip(gold_columns, pred_columns)) / len(gold_columns),
        4,
    )


@dataclass
class TableScore:
    table_id: str
    matched: bool
    caption_exact: bool = False
    caption_token_f1: float | None = None
    columns_exact: bool = False
    columns_token_f1: float | None = None
    cell_accuracy: float | None = None
    section_title_exact: bool = False
    section_title_token_f1: float | None = None
    mention_recall_exact: float | None = None
    mention_recall_soft: float | None = None
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
    def mention_recall(self) -> float | None:
        return self.mention_recall_exact

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DocumentScore:
    source_pdf: str
    present_in_prediction: bool
    tables: list[TableScore] = field(default_factory=list)


@dataclass
class EvaluationReport:
    documents: list[DocumentScore]
    missing_from_prediction: list[str]

    def summary(self) -> dict[str, object]:
        table_scores = [
            score
            for document in self.documents
            for score in document.tables
        ]
        matched = [score for score in table_scores if score.matched]
        cells = [score.cell_accuracy for score in matched if score.cell_accuracy is not None]
        return {
            "gold_documents": len(self.documents),
            "gold_documents_found_in_prediction": sum(
                1 for document in self.documents if document.present_in_prediction
            ),
            "missing_from_prediction": self.missing_from_prediction,
            "gold_tables": len(table_scores),
            "tables_matched": len(matched),
            "exact": {
                "caption_accuracy": _mean(score.caption_exact for score in matched),
                "columns_accuracy": _mean(score.columns_exact for score in matched),
                "mean_cell_accuracy": _mean(cells),
                "section_title_accuracy": _mean(score.section_title_exact for score in matched),
                "mean_mention_recall": _mean(
                    score.mention_recall_exact
                    for score in matched
                    if score.mention_recall_exact is not None
                ),
            },
            "soft": {
                "mean_caption_token_f1": _mean(
                    score.caption_token_f1
                    for score in matched
                    if score.caption_token_f1 is not None
                ),
                "mean_columns_token_f1": _mean(
                    score.columns_token_f1
                    for score in matched
                    if score.columns_token_f1 is not None
                ),
                "mean_section_title_token_f1": _mean(
                    score.section_title_token_f1
                    for score in matched
                    if score.section_title_token_f1 is not None
                ),
                "mean_mention_recall_soft": _mean(
                    score.mention_recall_soft
                    for score in matched
                    if score.mention_recall_soft is not None
                ),
                "mean_mention_token_f1": _mean(
                    score.mention_mean_token_f1
                    for score in matched
                    if score.mention_mean_token_f1 is not None
                ),
            },
        }

    def structure_summary(self) -> dict[str, object]:
        """Detection / columns / cells only — for structure-only tools (Phase 5)."""
        full = self.summary()
        gold_tables = full["gold_tables"]
        matched = full["tables_matched"]
        n_gold = int(gold_tables)  # type: ignore[arg-type]
        n_matched = int(matched)  # type: ignore[arg-type]
        detection = None if n_gold == 0 else round(n_matched / n_gold, 4)
        return {
            "gold_documents": full["gold_documents"],
            "gold_documents_found_in_prediction": full["gold_documents_found_in_prediction"],
            "missing_from_prediction": full["missing_from_prediction"],
            "gold_tables": gold_tables,
            "tables_matched": matched,
            "detection_recall": detection,
            "exact": {
                "columns_accuracy": full["exact"]["columns_accuracy"],
                "mean_cell_accuracy": full["exact"]["mean_cell_accuracy"],
            },
            "soft": {
                "mean_columns_token_f1": full["soft"]["mean_columns_token_f1"],
            },
        }


def _mean(values) -> float | None:
    values = list(values)
    if not values:
        return None
    return round(sum(float(value) for value in values) / len(values), 4)


def score_table(gold: Table, predicted: Table | None) -> TableScore:
    if predicted is None:
        return TableScore(table_id=gold.table_id, matched=False, detail="not found in prediction")

    caption_exact = _norm(gold.caption) == _norm(predicted.caption)
    columns_exact = [_norm(c) for c in gold.columns] == [_norm(c) for c in predicted.columns]

    cell_hits = cell_total = 0
    if len(gold.rows) == len(predicted.rows) and len(gold.columns) == len(predicted.columns):
        for gold_row, pred_row in zip(gold.rows, predicted.rows):
            for gold_cell, pred_cell in zip(gold_row, pred_row):
                cell_total += 1
                if _cell_equal(gold_cell, pred_cell):
                    cell_hits += 1
    else:
        cell_total = sum(len(row) for row in gold.rows) or 1
        cell_hits = 0

    section_exact = _norm(gold.context.section_title) == _norm(predicted.context.section_title)

    gold_mentions = [mention.text for mention in gold.context.mentions if mention.text]
    pred_mentions = [mention.text for mention in predicted.context.mentions if mention.text]
    mention_exact, mention_soft, mention_mean_f1 = _mention_scores(gold_mentions, pred_mentions)

    return TableScore(
        table_id=gold.table_id,
        matched=True,
        caption_exact=caption_exact,
        caption_token_f1=round(token_f1(gold.caption, predicted.caption), 4),
        columns_exact=columns_exact,
        columns_token_f1=_columns_token_f1(gold.columns, predicted.columns),
        cell_accuracy=round(cell_hits / cell_total, 4) if cell_total else None,
        section_title_exact=section_exact,
        section_title_token_f1=round(
            token_f1(gold.context.section_title, predicted.context.section_title), 4
        ),
        mention_recall_exact=round(mention_exact, 4),
        mention_recall_soft=round(mention_soft, 4),
        mention_mean_token_f1=round(mention_mean_f1, 4),
    )


def evaluate(
    gold_documents: list[GroundTruthDocument],
    prediction_documents: list[GroundTruthDocument],
) -> EvaluationReport:
    """Score each Gold document against a silver / tool prediction corpus."""
    predicted = _index_by_source(prediction_documents)
    scores: list[DocumentScore] = []
    missing: list[str] = []

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

        table_scores = [
            score_table(table, _match_table(table, pred_doc.tables)) for table in gold.tables
        ]
        scores.append(
            DocumentScore(source_pdf=key, present_in_prediction=True, tables=table_scores)
        )

    return EvaluationReport(documents=scores, missing_from_prediction=missing)
