"""Tests for silver JSON → sc: Turtle export (Gold is never an input)."""

from __future__ import annotations

from pathlib import Path

import pytest

from solarchem_benchmark.gt.schema import GroundTruthDocument, Mention, Table, TableContext
from solarchem_benchmark.rdf.cli import main
from solarchem_benchmark.rdf.export import corpus_to_turtle, document_to_turtle, is_gold_path


def _silver_doc() -> GroundTruthDocument:
    return GroundTruthDocument(
        document_id="solarchem_paper_demo",
        source_pdf="demo.pdf",
        title="Photocatalytic reduction of CO2",
        num_tables=1,
        tables=[
            Table(
                table_id="solarchem_paper_demo_table_01",
                table_label="Table 1",
                page=5,
                caption="Table 1 Comparison of BET surface area.",
                columns=["Sample", "BET (m^2 g^-1)"],
                rows=[["HZSM-5", 277.6]],
                context=TableContext(
                    section_title="3.3. Texture of samples",
                    mentions=[
                        Mention(page=4, text="Texture data are listed in Table 1."),
                    ],
                ),
            )
        ],
    )


def test_document_emits_article_table_and_mention() -> None:
    ttl = document_to_turtle(_silver_doc())
    assert "inst:article_solarchem_paper_demo a sc:Article" in ttl
    assert "schema:name \"Photocatalytic reduction of CO2\"" in ttl
    assert "inst:table_solarchem_paper_demo_table_01 a sc:TableRepresentation" in ttl
    assert "sc:belongsToArticle inst:article_solarchem_paper_demo" in ttl
    assert "schema:articleSection \"3.3. Texture of samples\"" in ttl
    assert "schema:pageStart 5" in ttl
    assert "csvw:name \"BET (m^2 g^-1)\"" in ttl
    assert "a sc:TextMention" in ttl
    assert "schema:text \"Texture data are listed in Table 1.\"" in ttl


def test_corpus_header_names_silver_source() -> None:
    ttl = corpus_to_turtle([_silver_doc()], source_label="ground_truth_lighton_ocr_302.json")
    assert "ground_truth_lighton_ocr_302.json" in ttl
    assert "Not derived from Gold" in ttl
    assert "@prefix sc:" in ttl


def test_gold_directory_is_detected() -> None:
    assert is_gold_path(Path("data/ground_truth/gold/ground_truth_gold_pilot.json"))
    assert not is_gold_path(Path("data/ground_truth/ground_truth_lighton_ocr_302.json"))


def test_cli_refuses_gold(tmp_path: Path) -> None:
    gold = tmp_path / "gold" / "pilot.json"
    gold.parent.mkdir()
    gold.write_text("{}", encoding="utf-8")
    assert main(["--input", str(gold), "--output", str(tmp_path / "out.ttl")]) == 1
