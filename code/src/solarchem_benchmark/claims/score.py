"""Scores for the Laya claims trial.

Two blocks, written into the run JSON. Nothing in this module calls
``solarchem_benchmark.eval``.

* **link** — masked table choice, plus row and column when a number pins one line.
* **claims** — precision and recall against a label CSV. Without labels this block stays empty.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from solarchem_benchmark.claims.text import canonical_table_number

GOLD_FIELDS = ("gold_is_claim", "gold_role", "gold_type", "gold_table", "gold_status")
# Kept with the row so a person can read the key, and so a changed sentence
# split cannot be graded as if it were still the labeled sentence. ``gold_source``
# records a table, a figure, the literature or a method. Laya does not predict it.
LABEL_CONTEXT = ("text", "gold_source")
_YES = {"yes", "y", "true", "1", "si", "sí"}
_NO = {"no", "n", "false", "0"}
_SKIP_JSON = {"summary.json", "table_reading.json", "scores.json"}


def pool_link_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool per-document link scores, each rate weighted by its own count."""
    def pool(rate: str, weight: str) -> tuple[float | None, int]:
        total = 0
        acc = 0.0
        for evaluation in evaluations:
            count = evaluation.get(weight) or 0
            value = evaluation.get(rate)
            if not count or value is None:
                continue
            total += count
            acc += float(value) * count
        if not total:
            return None, 0
        return round(acc / total, 4), total

    top1, n = pool("laya_top1", "evaluable_sentences")
    mrr, _ = pool("laya_mrr", "evaluable_sentences")
    none_rate, _ = pool("laya_none_rate", "evaluable_sentences")
    numeric, _ = pool("numeric_top1", "evaluable_sentences")
    random_top1, _ = pool("random_top1", "evaluable_sentences")
    row_top1, row_n = pool("row_top1", "row_evaluable")
    column_top1, column_n = pool("column_top1", "column_evaluable")
    return {
        "evaluable_sentences": n,
        "laya_top1": top1,
        "laya_mrr": mrr,
        "laya_none_rate": none_rate,
        "numeric_top1": numeric,
        "random_top1": random_top1,
        "row_evaluable": row_n,
        "row_top1": row_top1,
        "column_evaluable": column_n,
        "column_top1": column_top1,
    }


def load_labels(paths: list[Path]) -> dict[tuple[str, str], dict[str, str]]:
    """Label rows keyed by ``(document_id, sentence_id)``. Later files override earlier ones."""
    labels: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
        for csv_path in files:
            if not csv_path.is_file():
                continue
            with csv_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    document_id = (row.get("document_id") or "").strip()
                    sentence_id = (row.get("sentence_id") or "").strip()
                    if not document_id or not sentence_id:
                        continue
                    current = labels.setdefault((document_id, sentence_id), {})
                    for field in (*GOLD_FIELDS, *LABEL_CONTEXT):
                        value = (row.get(field) or "").strip()
                        if value:
                            current[field] = value
    return labels


def _norm_text(value: str | None) -> str:
    return " ".join((value or "").split())


def _as_bool(value: str | None) -> bool | None:
    if not value:
        return None
    text = value.strip().lower()
    if text in _YES:
        return True
    if text in _NO:
        return False
    return None


def _table_key(label: str | None) -> str:
    if not label or label.strip().lower() in {"none", "ninguna", "no table"}:
        return "none"
    number = canonical_table_number(label)
    return number or label.strip().lower()


def predicted_table(record: dict[str, Any]) -> str:
    """Table the model ranked first, or ``none`` when that probability does not beat ``none``."""
    links = record.get("links") or []
    if not links:
        return "none"
    top = links[0]
    if float(record.get("p_no_table") or 0) >= float(top.get("p_link") or 0):
        return "none"
    return str(top.get("table_label") or "none")


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None or precision + recall == 0:
        f1 = None if precision is None else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1": None if f1 is None else round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def score_claims(
    documents: list[dict[str, Any]],
    labels: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    """Precision/recall of the claim bit, and accuracy of role, type, table and status.

    ``table_accuracy`` counts only rows whose gold table is a real table.
    Rows labeled ``none`` are scored apart, as ``no_table_accuracy``: the hit
    is predicting no table.
    """
    records: dict[tuple[str, str], dict[str, Any]] = {}
    by_text: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for document in documents:
        for record in document.get("sentences") or []:
            records[(document["document_id"], record["sentence_id"])] = record
            by_text.setdefault((document["document_id"], _norm_text(record.get("text"))), []).append(record)

    tp = fp = fn = 0
    labeled = role_hit = role_n = type_hit = type_n = 0
    table_hit = table_n = no_table_hit = no_table_n = status_hit = status_n = 0
    missing = text_mismatches = recovered_by_text = 0
    for key, gold in labels.items():
        record = records.get(key)
        gold_text = _norm_text(gold.get("text"))
        if record is None and gold_text:
            hits = by_text.get((key[0], gold_text), [])
            if len(hits) == 1:
                record = hits[0]
                recovered_by_text += 1
        if record is None:
            missing += 1
            continue
        if gold_text and _norm_text(record.get("text")) != gold_text:
            text_mismatches += 1
            continue
        claim = record.get("claim") or {}
        is_claim = _as_bool(gold.get("gold_is_claim"))
        if is_claim is not None:
            labeled += 1
            predicted = bool(claim.get("is_claim"))
            if predicted and is_claim:
                tp += 1
            elif predicted and not is_claim:
                fp += 1
            elif is_claim:
                fn += 1
        if gold.get("gold_role"):
            role_n += 1
            role_hit += int(claim.get("role") == gold["gold_role"])
        if gold.get("gold_type"):
            type_n += 1
            type_hit += int(claim.get("type") == gold["gold_type"])
        if gold.get("gold_table"):
            predicted = _table_key(predicted_table(record))
            target = _table_key(gold["gold_table"])
            if target == "none":
                no_table_n += 1
                no_table_hit += int(predicted == "none")
            else:
                table_n += 1
                table_hit += int(predicted == target)
        if gold.get("gold_status") and gold.get("gold_table"):
            target = _table_key(gold["gold_table"])
            if target != "none":
                status = next(
                    (
                        link.get("status")
                        for link in record.get("links") or []
                        if _table_key(link.get("table_label")) == target
                    ),
                    None,
                )
                if status is not None:
                    status_n += 1
                    status_hit += int(status == gold["gold_status"])

    def accuracy(hits: int, count: int) -> float | None:
        return round(hits / count, 4) if count else None

    return {
        "labeled": labeled,
        "missing_sentences": missing,
        "text_mismatches": text_mismatches,
        "recovered_by_text": recovered_by_text,
        **_prf(tp, fp, fn),
        "role_labeled": role_n,
        "role_accuracy": accuracy(role_hit, role_n),
        "type_labeled": type_n,
        "type_accuracy": accuracy(type_hit, type_n),
        "table_labeled": table_n,
        "table_accuracy": accuracy(table_hit, table_n),
        "no_table_labeled": no_table_n,
        "no_table_accuracy": accuracy(no_table_hit, no_table_n),
        "status_labeled": status_n,
        "status_accuracy": accuracy(status_hit, status_n),
    }


def document_jsons(output_dir: Path) -> list[Path]:
    return sorted(
        path for path in output_dir.glob("*.json") if path.name not in _SKIP_JSON and not path.name.startswith(".")
    )


def load_documents(output_dir: Path) -> list[dict[str, Any]]:
    documents = []
    for path in document_jsons(output_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("document_id") and "sentences" in payload:
            documents.append(payload)
    return documents


def build_scores(
    documents: list[dict[str, Any]],
    *,
    labels: dict[tuple[str, str], dict[str, str]],
    label_paths: list[str],
) -> dict[str, Any]:
    return {
        "link": pool_link_evaluations([doc.get("evaluation") or {} for doc in documents]),
        "claims": score_claims(documents, labels),
        "labels": label_paths,
        "documents": [doc["document_id"] for doc in documents],
    }
