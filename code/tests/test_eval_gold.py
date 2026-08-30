"""Tests for Gold-vs-silver ground-truth comparison."""

from __future__ import annotations

from solarchem_benchmark.eval.compare_gt import evaluate, score_table, token_f1
from solarchem_benchmark.gt.schema import GroundTruthDocument, Mention, Table, TableContext


def _table(**kwargs) -> Table:
    base = dict(
        table_id="t1",
        table_label="Table 1",
        page=2,
        caption="Table 1 Lattice parameters of ATiO2 and O2-TiO2",
        columns=["Sample", "a (A)"],
        rows=[["ATiO2", 3.7748], ["O2-TiO2", 3.7852]],
        context=TableContext(
            section_title="2 Results and discussion",
            mentions=[Mention(page=2, text="observed for oxygen-modified TiO2 (Table 1).")],
        ),
    )
    base.update(kwargs)
    return Table(**base)


def _doc(source: str, table: Table) -> GroundTruthDocument:
    return GroundTruthDocument(
        document_id=f"id_{source}",
        source_pdf=source,
        title="Demo",
        num_tables=1,
        tables=[table],
    )


def test_perfect_match_scores_one() -> None:
    gold = _doc("a.pdf", _table())
    report = evaluate([gold], [gold])
    summary = report.summary()
    assert summary["tables_matched"] == 1
    assert summary["exact"]["caption_accuracy"] == 1.0
    assert summary["exact"]["mean_cell_accuracy"] == 1.0
    assert summary["soft"]["mean_caption_token_f1"] == 1.0


def test_missing_gold_document_is_a_detection_miss() -> None:
    gold = _doc("a.pdf", _table())
    other = _doc("b.pdf", _table(table_id="t2"))
    report = evaluate([gold], [other])
    summary = report.summary()
    assert report.missing_from_prediction == ["a.pdf"]
    assert summary["gold_documents_found_in_prediction"] == 0
    assert summary["gold_tables"] == 1
    assert summary["tables_matched"] == 0
    assert report.structure_summary()["detection_recall"] == 0.0


def test_cell_mismatch_lowers_accuracy() -> None:
    gold = _table()
    pred = _table(rows=[["ATiO2", 3.7748], ["O2-TiO2", 9.9999]])
    score = score_table(gold, pred)
    assert score.matched
    assert score.cell_accuracy == 0.75


def test_unmatched_table_is_a_detection_miss() -> None:
    score = score_table(_table(), None)
    assert not score.matched


def test_token_f1_is_high_for_near_paraphrases() -> None:
    gold = "Table 1 Lattice parameters of ATiO2 and O2-TiO2"
    pred = "Table 1: Lattice parameters of ATiO2 and O2–TiO2."
    assert token_f1(gold, pred) > 0.85


def test_soft_caption_survives_punctuation_where_exact_fails() -> None:
    gold = _table()
    pred = _table(caption="Table 1: Lattice parameters of ATiO2 and O2-TiO2.")
    score = score_table(gold, pred)
    assert score.caption_exact is False
    assert score.caption_token_f1 is not None and score.caption_token_f1 > 0.85


def test_soft_mention_recall_accepts_overlapping_paragraph() -> None:
    gold = _table()
    pred = _table(
        context=TableContext(
            section_title="2. Results and discussion",
            mentions=[
                Mention(
                    page=2,
                    text=(
                        "Furthermore, a decrease in the c/a ratio was observed for "
                        "oxygen-modified TiO2 (Table 1). The change can be explained."
                    ),
                )
            ],
        )
    )
    score = score_table(gold, pred)
    assert score.mention_recall_exact == 0.0
    assert score.mention_recall_soft == 1.0
    assert score.mention_mean_token_f1 is not None and score.mention_mean_token_f1 > 0.4


def test_section_title_token_f1_tolerates_dot_after_number() -> None:
    gold = _table(context=TableContext(section_title="2 Results and discussion", mentions=[]))
    pred = _table(context=TableContext(section_title="2. Results and discussion", mentions=[]))
    score = score_table(gold, pred)
    assert score.section_title_exact is False
    assert score.section_title_token_f1 == 1.0


def test_empty_prediction_document_counts_as_detection_miss() -> None:
    gold = _doc("a.pdf", _table())
    empty = GroundTruthDocument(
        document_id="id_a.pdf",
        source_pdf="a.pdf",
        num_tables=0,
        tables=[],
    )
    summary = evaluate([gold], [empty]).structure_summary()
    assert summary["gold_documents_found_in_prediction"] == 1
    assert summary["gold_tables"] == 1
    assert summary["tables_matched"] == 0
    assert summary["detection_recall"] == 0.0
    assert summary["exact"]["mean_cell_accuracy"] is None


def test_structure_summary_omits_context_metrics() -> None:
    gold = _doc("a.pdf", _table())
    pred = _doc(
        "a.pdf",
        _table(caption="", context=TableContext()),
    )
    summary = evaluate([gold], [pred]).structure_summary()
    assert summary["tables_matched"] == 1
    assert summary["detection_recall"] == 1.0
    assert "caption_accuracy" not in summary["exact"]
    assert "mean_caption_token_f1" not in summary["soft"]
    assert set(summary["exact"]) == {"columns_accuracy", "mean_cell_accuracy"}
