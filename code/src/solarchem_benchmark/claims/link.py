"""Claim detection and claim-to-table linking over one silver document.

Stages
------
1. **Candidates** -- body sentences passing a cheap filter (a stated number, a
   result verb, or an explicit ``Table N``); see :mod:`.text`.
2. **Claim classification** (model) -- the sentence's role (result, method,
   background, other) and what it reports (performance, material property,
   condition, explanation).
3. **Which table** (model) -- one ``choice`` per sentence whose options are the
   paper's table captions plus ``none``. Probabilities are therefore comparable
   across tables; asking "is it about this table?" once per table was not (on
   the pilot, top-1 0.40 against 0.66 for this form, English checkpoint).
4. **Where in the table** (model) -- for every table: relation (repeats a
   value, trend, contradicts, explains), row and column. Row and column
   options are that table's own labels.
5. **Numeric check** (deterministic) -- numbers stated in the sentence are
   matched against the table cells (exact, rounded, within 1 %), with the unit
   required to appear in the column, row label or caption.
6. **Decision** -- a link is kept when the sentence names the table, or the
   model picks the table, or a number matches a table the model ranks second.

Explicit ``Table N`` references are masked before stages 3-4 so the model links
on content. Sentences that did name exactly one table are then a free label:
:func:`evaluate_links` reports how often the model ranks that table first.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from solarchem_benchmark.claims.model import DecisionModel
from solarchem_benchmark.claims.text import (
    Sentence,
    StatedNumber,
    canonical_table_number,
    is_candidate,
    mask_table_references,
    normalize_unit_text,
    stated_numbers,
)
from solarchem_benchmark.gt.context import caption_title
from solarchem_benchmark.gt.schema import GroundTruthDocument, Table

logger = logging.getLogger(__name__)

NONE_OPTION = "none"
RESULT_ROLE = "result"
MAX_OPTIONS = 24
MAX_LABEL_CHARS = 60
MAX_CAPTION_CHARS = 150
MAX_TABLE_ROWS = 40
MAX_CELL_CHARS = 40
RELATIVE_TOLERANCE = 0.01
#: Option lengths tried in turn when a question's options exceed ``head_max_len``.
SHRINK_STEPS = (40, 20, 10)

# A four-way role separated results from methods/introduction sentences
# better than a yes/no "is this a finding?" on the pilot (English checkpoint).
CLAIM_QUESTIONS: dict[str, dict] = {
    "role": {
        "type": "choice",
        "instructions": "What kind of sentence is this in a research article?",
        "criteria": {
            RESULT_ROLE: "result: reports what the authors measured, observed or concluded about their own samples",
            "method": "method: describes how the samples were prepared, characterised or tested",
            "background": "background: general knowledge, motivation or findings of other studies",
            "other": "other: title, authors, affiliation, caption, equation or text fragment",
        },
    },
    "claim_type": {
        "type": "choice",
        "instructions": "What does the sentence mainly report?",
        "criteria": {
            "performance": (
                "photocatalytic performance: product yield, production rate, selectivity, "
                "conversion, quantum efficiency or stability"
            ),
            "property": (
                "a property of the material: composition, surface area, pore size, band gap, "
                "crystallite size, XRD, XPS, PL or light absorption"
            ),
            "condition": (
                "how the experiment was run: light source, wavelength, irradiation time, catalyst "
                "mass, temperature, pressure, feed ratio or reaction medium"
            ),
            "explanation": "why a result happens: mechanism, charge separation, recombination, active sites",
            "other": "none of the above",
        },
    },
}

_RELATION_QUESTION = {
    "type": "choice",
    "instructions": "How does the sentence relate to the table?",
    "criteria": {
        "repeats_value": "the sentence quotes a number that appears in the table",
        "describes_trend": "the sentence describes a comparison or trend visible across the table",
        "contradicts": "the sentence states a value or trend that disagrees with the table",
        "explains": "the sentence gives the conditions, origin or explanation of the table data",
        "unrelated": "the sentence is not about this table",
    },
}


@dataclass(frozen=True)
class LinkConfig:
    #: A sentence is a claim when its role is ``result`` or p(result) reaches this.
    claim_threshold: float = 0.5
    #: Minimum p(table) for a model-only link; the table must also be the argmax.
    link_threshold: float = 0.0
    mask_references: bool = True
    max_candidates: int = 0
    only_referenced: bool = False


@dataclass(frozen=True)
class TableCell:
    row: int
    column: int
    value: float
    raw: str
    decimals: int


@dataclass
class TableView:
    """A silver table prepared for the model: text, options and numeric cells."""

    table: Table
    number: str | None
    text: str
    questions: dict[str, dict]
    row_labels: dict[str, str]
    column_labels: dict[str, str]
    label_column: int | None
    row_index: dict[str, int] = field(default_factory=dict)
    column_index: dict[str, int] = field(default_factory=dict)
    cells: list[TableCell] = field(default_factory=list)


def _short(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _label_column(table: Table) -> int | None:
    """Index of the column holding row labels (sample or product names), if any."""
    if not table.rows or not table.columns:
        return None
    first = [row[0] for row in table.rows]
    text_cells = sum(1 for cell in first if isinstance(cell, str) and cell.strip())
    return 0 if text_cells >= max(1, math.ceil(len(first) / 2)) else None


_CELL_NUMBER_RE = re.compile(r"(?<![A-Za-z_^.\d])[-\u2212]?\d+(?:\.\d+)?")


def _cell_numbers(cell: Any) -> list[tuple[float, str, int]]:
    if isinstance(cell, bool):
        return []
    if isinstance(cell, (int, float)):
        raw = repr(cell) if isinstance(cell, float) else str(cell)
        if "e" in raw.lower():
            raw = f"{cell:.12f}".rstrip("0").rstrip(".")
        decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
        return [(float(cell), raw, decimals)]
    out = []
    for match in _CELL_NUMBER_RE.finditer(str(cell)):
        raw = match.group(0).replace("\u2212", "-")
        decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
        out.append((float(raw), raw, decimals))
    return out


def serialize_table(table: Table, *, max_rows: int = MAX_TABLE_ROWS) -> str:
    """Pipe-separated rendering: caption, header, then rows."""
    label = table.table_label.strip() or "Table"
    title = caption_title(table.caption).strip()
    lines = [f"{label}: {title}" if title else label]
    if table.context.section_title.strip():
        lines.append(f"Table section: {table.context.section_title.strip()}")
    lines.append(" | ".join(_short(column, MAX_CELL_CHARS) for column in table.columns))
    for row in table.rows[:max_rows]:
        lines.append(" | ".join(_short(cell, MAX_CELL_CHARS) for cell in row))
    if len(table.rows) > max_rows:
        lines.append(f"... {len(table.rows) - max_rows} more rows")
    return "\n".join(lines)


def _options(labels: list[str], prefix: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for index, label in enumerate(labels[:MAX_OPTIONS], start=1):
        options[f"{prefix}{index}"] = _short(label, MAX_LABEL_CHARS) or f"{prefix}{index}"
    return options


def build_table_view(table: Table) -> TableView:
    label_column = _label_column(table)
    if label_column is None:
        row_names = [f"row {index}" for index in range(1, len(table.rows) + 1)]
        column_indices = list(range(len(table.columns)))
    else:
        row_names = [str(row[label_column]) for row in table.rows]
        column_indices = [i for i in range(len(table.columns)) if i != label_column]
    if len(row_names) > MAX_OPTIONS or len(column_indices) > MAX_OPTIONS:
        logger.info("%s: options truncated to %d", table.table_id, MAX_OPTIONS)

    row_labels = _options(row_names, "r")
    column_labels = _options([table.columns[i] for i in column_indices], "c")
    row_criteria = {**row_labels, NONE_OPTION: "no single row"}
    column_criteria = {**column_labels, NONE_OPTION: "no single column"}
    questions = {
        "relation": _RELATION_QUESTION,
        "row": {
            "type": "choice",
            "instructions": "Which table row (sample or item) does the sentence talk about?",
            "criteria": row_criteria,
        },
        "column": {
            "type": "choice",
            "instructions": "Which table column (measured quantity) does the sentence talk about?",
            "criteria": column_criteria,
        },
    }

    cells = []
    for r, row in enumerate(table.rows):
        for c, cell in enumerate(row):
            if c == label_column:
                continue
            for value, raw, decimals in _cell_numbers(cell):
                cells.append(TableCell(row=r, column=c, value=value, raw=raw, decimals=decimals))

    return TableView(
        table=table,
        number=canonical_table_number(table.table_label),
        text=serialize_table(table),
        questions=questions,
        row_labels=row_labels,
        column_labels=column_labels,
        label_column=label_column,
        row_index={key: i for i, key in enumerate(row_labels)},
        column_index={key: column_indices[i] for i, key in enumerate(column_labels)},
        cells=cells,
    )


def _significant_digits(value: float, decimals: int) -> int:
    digits = f"{abs(value):.{decimals}f}".replace(".", "").lstrip("0")
    return len(digits)


def match_kind(number: StatedNumber, cell: TableCell) -> str | None:
    """How a stated number agrees with a cell: exact, rounded, approx, or not at all.

    ``rounded`` only when the less precise side is the other rounded to its
    own precision and keeps at least two significant digits, so ``42.98`` vs
    ``43`` matches but ``4`` vs ``3.764`` or ``0.2`` vs ``0`` does not.
    """
    a, b = number.value, cell.value
    if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
        return "exact"
    if a == 0 or b == 0:
        return None
    if number.decimals < cell.decimals:
        coarse, fine, places = a, b, number.decimals
    elif cell.decimals < number.decimals:
        coarse, fine, places = b, a, cell.decimals
    else:
        coarse = None
    if coarse is not None and round(fine, places) == coarse and _significant_digits(coarse, places) >= 2:
        return "rounded"
    if abs(a - b) / abs(b) <= RELATIVE_TOLERANCE:
        return "approx"
    return None


def _unit_compatible(number: StatedNumber, view: TableView, cell: TableCell) -> bool:
    """A stated unit must appear in the cell, its row label, its column header or the caption."""
    if not number.unit:
        return True
    table = view.table
    row = table.rows[cell.row]
    row_label = row[view.label_column] if view.label_column is not None else ""
    haystack = normalize_unit_text(
        f"{table.columns[cell.column]} {row_label} {row[cell.column]} {table.caption}"
    )
    return number.unit in haystack


def numeric_matches(numbers: list[StatedNumber], view: TableView) -> list[dict[str, Any]]:
    table = view.table
    matches = []
    seen: set[tuple[str, int, int]] = set()
    for number in numbers:
        if not number.informative:
            continue
        for cell in view.cells:
            kind = match_kind(number, cell)
            if kind is None or not _unit_compatible(number, view, cell):
                continue
            key = (number.raw, cell.row, cell.column)
            if key in seen:
                continue
            seen.add(key)
            row_label = (
                str(table.rows[cell.row][view.label_column])
                if view.label_column is not None
                else f"row {cell.row + 1}"
            )
            matches.append(
                {
                    "stated": number.raw,
                    "cell": cell.raw,
                    "kind": kind,
                    "row": cell.row,
                    "column": cell.column,
                    "row_label": row_label,
                    "column_label": table.columns[cell.column],
                }
            )
    return matches


def is_table_dump(sentence: Sentence, views: list[TableView]) -> bool:
    """True when the PDF text layer emitted a table body as a sentence.

    Such text contains most of the table's row labels *and* many of its
    cells; a prose summary of the same table names one or two samples.
    """
    numbers = stated_numbers(sentence.text)
    for view in views:
        if view.label_column is None:
            continue
        labels = {
            str(row[view.label_column]).strip()
            for row in view.table.rows
            if len(str(row[view.label_column]).strip()) >= 3
        }
        if len(labels) < 3:
            continue
        present = sum(1 for label in labels if label in sentence.text)
        if present / len(labels) >= 0.5 and len(numeric_matches(numbers, view)) >= 6:
            return True
    return False


def table_option_key(view: TableView, index: int) -> str:
    return f"T{view.number}" if view.number else f"T_{index + 1}"


def table_choice_question(views: list[TableView]) -> dict[str, dict]:
    """One question over the paper's tables, each option being a caption title."""
    criteria: dict[str, str] = {}
    for index, view in enumerate(views):
        title = caption_title(view.table.caption).strip()
        if not title:
            title = "; ".join(str(column) for column in view.table.columns[:6])
        criteria[table_option_key(view, index)] = _short(title, MAX_CAPTION_CHARS)
    criteria[NONE_OPTION] = "none of these tables"
    return {
        "table": {
            "type": "choice",
            "instructions": "Which table reports the data this sentence talks about?",
            "criteria": criteria,
        }
    }


def _shrink(questions: dict[str, dict], limit: int) -> dict[str, dict]:
    shrunk = {}
    for qid, question in questions.items():
        criteria = question["criteria"]
        if isinstance(criteria, dict):
            criteria = {key: _short(text, limit) for key, text in criteria.items()}
        shrunk[qid] = {**question, "criteria": criteria}
    return shrunk


def predict(model: DecisionModel, states: list[str], questions: dict[str, dict]) -> list[dict]:
    """``model.predict_batch`` that shortens option texts when they overflow the head budget.

    Keys are unchanged by shrinking, so answers map back to rows, columns and
    tables exactly as with the full labels.
    """
    try:
        return model.predict_batch(states, questions)
    except ValueError as exc:
        if "head_max_len" not in str(exc):
            raise
        last = exc
    for limit in SHRINK_STEPS:
        logger.info("Options exceed the head budget; retrying at %d characters", limit)
        try:
            return model.predict_batch(states, _shrink(questions, limit))
        except ValueError as exc:
            if "head_max_len" not in str(exc):
                raise
            last = exc
    raise last


def claim_state(sentence: Sentence, *, mask: bool = False) -> str:
    text = mask_table_references(sentence.text) if mask else sentence.text
    parts = [f"Sentence: {text}"]
    if sentence.section:
        parts.append(f"Section: {sentence.section}")
    return "\n".join(parts)


def link_state(sentence: Sentence, view: TableView, *, mask: bool) -> str:
    text = mask_table_references(sentence.text) if mask else sentence.text
    head = f"Sentence: {text}"
    if sentence.section:
        head += f"\nSentence section: {sentence.section}"
    return f"{head}\n\n{view.text}"


def _probability(answer: dict[str, Any], key: str) -> float:
    return float(answer.get("probabilities", {}).get(key, 0.0))


def _decide_status(
    matches: list[dict[str, Any]],
    relation: str,
    relation_p: float,
    has_numbers: bool,
) -> str:
    kinds = {match["kind"] for match in matches}
    if kinds & {"exact", "rounded"}:
        return "consistent"
    if "approx" in kinds:
        return "approximatelyConsistent"
    if relation == "contradicts" and relation_p >= 0.5:
        return "inconsistent_suspected"
    return "unverified" if has_numbers else "qualitative"


def _link_record(
    sentence: Sentence,
    view: TableView,
    link_p: float,
    answers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    relation = answers["relation"]["choice"]
    relation_p = _probability(answers["relation"], relation)
    row_key = answers["row"]["choice"]
    column_key = answers["column"]["choice"]

    numbers = stated_numbers(sentence.text)
    matches = numeric_matches(numbers, view)
    row_index = view.row_index.get(row_key)
    column_index = view.column_index.get(column_key)
    for match in matches:
        match["in_selected_row"] = row_index is not None and match["row"] == row_index
        match["in_selected_cell"] = match["in_selected_row"] and match["column"] == column_index
    focused = [m for m in matches if m["in_selected_row"]] if row_index is not None else matches
    status = _decide_status(
        focused or matches, relation, relation_p, any(n.informative for n in numbers)
    )

    explicit = view.number is not None and view.number in sentence.table_refs
    return {
        "table_id": view.table.table_id,
        "table_label": view.table.table_label,
        "p_link": round(link_p, 4),
        "relation": relation,
        "relation_probabilities": answers["relation"].get("probabilities", {}),
        "row": view.row_labels.get(row_key, NONE_OPTION),
        "row_p": round(_probability(answers["row"], row_key), 4),
        "column": view.column_labels.get(column_key, NONE_OPTION),
        "column_p": round(_probability(answers["column"], column_key), 4),
        "row_index": row_index,
        "column_index": column_index,
        "explicit_reference": explicit,
        "numeric_matches": matches,
        "status": status,
        "linked": False,
        "link_sources": [],
    }


def _decide_links(
    links: list[dict[str, Any]],
    is_claim: bool,
    p_no_table: float,
    config: LinkConfig,
) -> None:
    """Set ``linked`` / ``link_sources`` on a sentence's links, ranked by ``p_link``.

    The model links a table only when it is the overall argmax (``none``
    included). A number matching the table (exact or rounded) also admits
    the table ranked second. An explicit ``Table N`` always links; the other
    two sources require the sentence to be a claim.
    """
    for rank, link in enumerate(links, start=1):
        supported = any(m["kind"] in {"exact", "rounded"} for m in link["numeric_matches"])
        sources = []
        if link["explicit_reference"]:
            sources.append("explicit_reference")
        if is_claim:
            if rank == 1 and link["p_link"] > p_no_table and link["p_link"] >= config.link_threshold:
                sources.append("laya")
            elif supported and rank <= 2:
                sources.append("laya+numeric")
        link["rank"] = rank
        link["link_sources"] = sources
        link["linked"] = bool(sources)


def run_document(
    document: GroundTruthDocument,
    sentences: list[Sentence],
    model: DecisionModel,
    config: LinkConfig = LinkConfig(),
) -> dict[str, Any]:
    """Classify the claims of one document and link them to its tables."""
    views = [build_table_view(table) for table in document.tables]
    candidates = [s for s in sentences if is_candidate(s) and not is_table_dump(s, views)]
    if config.only_referenced:
        candidates = [s for s in candidates if s.table_refs]
    if config.max_candidates > 0:
        candidates = candidates[: config.max_candidates]
    logger.info(
        "%s: %d sentences, %d candidates, %d tables",
        document.document_id, len(sentences), len(candidates), len(views),
    )

    logger.info("Stage 1/3: claim role and type")
    claim_answers = predict(model, [claim_state(s) for s in candidates], CLAIM_QUESTIONS)
    logger.info("Stage 2/3: which table")
    table_answers = (
        predict(
            model,
            [claim_state(s, mask=config.mask_references) for s in candidates],
            table_choice_question(views),
        )
        if views
        else [{} for _ in candidates]
    )
    per_table: list[list[dict[str, dict]]] = []
    for number, view in enumerate(views, start=1):
        logger.info("Stage 3/3: row and column in %s (%d/%d)", view.table.table_label, number, len(views))
        states = [link_state(s, view, mask=config.mask_references) for s in candidates]
        per_table.append(predict(model, states, view.questions))

    records = []
    for index, sentence in enumerate(candidates):
        claim = claim_answers[index]
        role = claim["role"]["choice"]
        p_claim = _probability(claim["role"], RESULT_ROLE)
        is_claim = role == RESULT_ROLE or p_claim >= config.claim_threshold
        choice = table_answers[index].get("table", {})
        p_no_table = _probability(choice, NONE_OPTION)
        links = [
            _link_record(
                sentence,
                view,
                _probability(choice, table_option_key(view, t)),
                per_table[t][index],
            )
            for t, view in enumerate(views)
        ]
        links.sort(key=lambda link: link["p_link"], reverse=True)
        _decide_links(links, is_claim, p_no_table, config)
        records.append(
            {
                "sentence_id": sentence.sentence_id,
                "page": sentence.page,
                "section": sentence.section,
                "text": sentence.text,
                "explicit_tables": sorted(sentence.table_refs),
                "numbers": [n.raw for n in stated_numbers(sentence.text) if n.informative],
                "claim": {
                    "p_claim": round(p_claim, 4),
                    "is_claim": is_claim,
                    "role": role,
                    "role_probabilities": claim["role"].get("probabilities", {}),
                    "type": claim["claim_type"]["choice"],
                    "type_probabilities": claim["claim_type"].get("probabilities", {}),
                },
                "p_no_table": round(p_no_table, 4),
                "links": links,
            }
        )

    linked = [(r, l) for r in records for l in r["links"] if l["linked"]]
    return {
        "document_id": document.document_id,
        "source_pdf": document.source_pdf,
        "title": document.title,
        "model": model.describe(),
        "config": asdict(config),
        "tables": [
            {"table_id": v.table.table_id, "table_label": v.table.table_label,
             "caption": v.table.caption, "page": v.table.page}
            for v in views
        ],
        "counts": {
            "sentences": len(sentences),
            "candidates": len(candidates),
            "claims": sum(1 for r in records if r["claim"]["is_claim"]),
            "links": len(linked),
            "links_by_status": _count(l["status"] for _, l in linked),
            "links_by_source": _count(s for _, l in linked for s in l["link_sources"]),
        },
        "evaluation": evaluate_links(records, views),
        "sentences": records,
    }


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _pinned_index(link: dict[str, Any], axis: str) -> int | None:
    """Row or column index when every exact/rounded match of this table sits on one line."""
    indexes = {
        match[axis]
        for match in link.get("numeric_matches") or []
        if match.get("kind") in {"exact", "rounded"} and isinstance(match.get(axis), int)
    }
    if len(indexes) != 1:
        return None
    return next(iter(indexes))


def _empty_link_evaluation(*, note: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "evaluable_sentences": 0,
        "row_evaluable": 0,
        "row_top1": None,
        "column_evaluable": 0,
        "column_top1": None,
    }
    if note:
        out["note"] = note
    return out


def evaluate_links(records: list[dict[str, Any]], views: list[TableView]) -> dict[str, Any]:
    """Self-supervised check on sentences that name exactly one table of the paper.

    The model saw those sentences with the reference masked, so ranking the
    named table first is evidence it links on content. ``laya_none_rate`` is
    how often it preferred ``none`` over every table. ``numeric_top1`` is
    the rule-only baseline (table with most exact/rounded matches, unique
    maximum only); ``random_top1`` is the expected score of guessing.

    Row and column use the same sentences. A number that lands on exactly one
    row (or one column) of the named table is the label; ``row_top1`` and
    ``column_top1`` are how often the model picked that line.
    """
    numbers = {view.number for view in views if view.number}
    if len(views) < 2:
        return _empty_link_evaluation(note="fewer than two tables")

    laya_hits = numeric_hits = said_none = 0
    row_hits = row_n = column_hits = column_n = 0
    reciprocal = 0.0
    evaluable = 0
    for record in records:
        refs = [ref for ref in record["explicit_tables"] if ref in numbers]
        if len(refs) != 1:
            continue
        evaluable += 1
        target = refs[0]
        ranked = [canonical_table_number(link["table_label"]) for link in record["links"]]
        if ranked and ranked[0] == target:
            laya_hits += 1
        if record["links"] and record.get("p_no_table", 0.0) >= record["links"][0]["p_link"]:
            said_none += 1
        if target in ranked:
            reciprocal += 1.0 / (ranked.index(target) + 1)
        support = {
            canonical_table_number(link["table_label"]): sum(
                1 for m in link["numeric_matches"] if m["kind"] in {"exact", "rounded"}
            )
            for link in record["links"]
        }
        best = max(support.values(), default=0)
        winners = [number for number, count in support.items() if count == best]
        if best > 0 and winners == [target]:
            numeric_hits += 1
        cited = next(
            (link for link in record["links"] if canonical_table_number(link["table_label"]) == target),
            None,
        )
        if cited is None:
            continue
        pinned_row = _pinned_index(cited, "row")
        if pinned_row is not None:
            row_n += 1
            if cited.get("row_index") == pinned_row:
                row_hits += 1
        pinned_column = _pinned_index(cited, "column")
        if pinned_column is not None:
            column_n += 1
            if cited.get("column_index") == pinned_column:
                column_hits += 1

    if not evaluable:
        return _empty_link_evaluation()
    return {
        "evaluable_sentences": evaluable,
        "tables": len(views),
        "laya_top1": round(laya_hits / evaluable, 4),
        "laya_mrr": round(reciprocal / evaluable, 4),
        "laya_none_rate": round(said_none / evaluable, 4),
        "numeric_top1": round(numeric_hits / evaluable, 4),
        "random_top1": round(1 / len(views), 4),
        "row_evaluable": row_n,
        "row_top1": round(row_hits / row_n, 4) if row_n else None,
        "column_evaluable": column_n,
        "column_top1": round(column_hits / column_n, 4) if column_n else None,
    }
