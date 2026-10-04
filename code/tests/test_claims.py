"""Tests for claim detection and claim-to-table linking (model faked)."""

from __future__ import annotations

import json
from pathlib import Path

from solarchem_benchmark.claims.cli import _run_file, main
from solarchem_benchmark.claims.link import (
    LinkConfig,
    build_table_view,
    match_kind,
    numeric_matches,
    predict,
    run_document,
    table_choice_question,
)
from solarchem_benchmark.claims.score import score_claims
from solarchem_benchmark.claims.text import (
    Sentence,
    StatedNumber,
    document_sentences,
    is_candidate,
    looks_like_table_residue,
    mask_table_references,
    split_sentences,
    stated_numbers,
)
from solarchem_benchmark.gt.schema import GroundTruthCorpus, GroundTruthDocument, Table, TableContext


def _document() -> GroundTruthDocument:
    return GroundTruthDocument(
        document_id="solarchem_demo",
        source_pdf="demo.pdf",
        title="In-doped TiO2",
        num_tables=2,
        tables=[
            Table(
                table_id="solarchem_demo_table_01",
                table_label="Table 1",
                page=4,
                caption="Table 1 BET surface area and band gap.",
                columns=["Sample", "BET surface area (m^2/g)", "Band gap (eV)"],
                rows=[["TiO2", 43, 3.12], ["10% In-TiO2", 84, 3.2]],
                context=TableContext(section_title="3.2. Texture"),
            ),
            Table(
                table_id="solarchem_demo_table_02",
                table_label="Table 2",
                page=9,
                caption="Table 2 Product yield rates (umol g^-1 h^-1).",
                columns=["Products", "TiO2", "10 wt.% In/TiO2"],
                rows=[["CH4", 31, 244], ["CO", 46, 81]],
            ),
        ],
    )


class KeywordModel:
    """Picks the table whose caption shares a keyword with the sentence; refuses unmasked refs."""

    name = "keyword"
    keywords = ("yield", "BET")

    def describe(self) -> dict:
        return {"model": self.name}

    def predict_batch(self, states, questions):
        out = []
        for state in states:
            answers = {}
            for qid, question in questions.items():
                keys = list(question["criteria"])
                chosen = keys[0]
                if qid == "table":
                    assert "Table 2" not in state, "references must be masked"
                    chosen = next(
                        (
                            key
                            for key, caption in question["criteria"].items()
                            if any(word in state and word in caption for word in self.keywords)
                        ),
                        "none",
                    )
                if qid == "row" and "In" in state.split("\n\n")[0]:
                    chosen = next((k for k, v in question["criteria"].items() if "In" in v), keys[0])
                probabilities = {k: (0.9 if k == chosen else 0.1 / (len(keys) - 1)) for k in keys}
                answers[qid] = {"type": "choice", "choice": chosen, "probabilities": probabilities}
            out.append(answers)
        return out


def test_stated_numbers_skip_labels_formulas_and_years() -> None:
    text = "In Table 3 and Fig. 11, CH4 over TiO2 reached 244 umol g^-1 h^-1 in 2015 [12]."
    numbers = [n.raw for n in stated_numbers(text)]
    assert numbers == ["244", "2015"]
    assert [n.raw for n in stated_numbers(text) if n.informative] == ["244"]


def test_split_sentences_keeps_abbreviations() -> None:
    text = "As shown in Fig. 3 the 10 wt. % sample is best. Smith et al. agree. Next sentence."
    assert split_sentences(text) == [
        "As shown in Fig. 3 the 10 wt. % sample is best.",
        "Smith et al. agree.",
        "Next sentence.",
    ]


def test_table_residue_is_not_a_candidate() -> None:
    residue = "CH4 31 244 40 69.29 CO 46 81 60 29.66 C2H4 0.00 0.06 0.0 0.022 C2H6 0.00 2.78"
    assert looks_like_table_residue(residue)
    prose = Sentence("s1", 5, "", "The BET surface area of TiO2 was 42.98 m2/g, increased to 61, 84, 98 and 123 m2/g.")
    assert is_candidate(prose)


def test_match_kind_handles_rounding() -> None:
    view = build_table_view(_document().tables[0])
    cell_43 = next(c for c in view.cells if c.value == 43)
    assert match_kind(StatedNumber(42.98, 2, "42.98"), cell_43) == "rounded"
    assert match_kind(StatedNumber(43.0, 0, "43"), cell_43) == "exact"
    assert match_kind(StatedNumber(50.0, 0, "50"), cell_43) is None


def test_numeric_matches_require_compatible_units() -> None:
    view = build_table_view(_document().tables[1])
    assert numeric_matches(stated_numbers("CH4 reached 244 umol g^-1 h^-1."), view)
    assert not numeric_matches(stated_numbers("The lamp was placed 244 mm above."), view)
    assert numeric_matches(stated_numbers("It produced 244 over the doped sample."), view)
    assert not numeric_matches(stated_numbers("The 10 wt.% In-doped TiO2 was best."), view)


def test_table_view_uses_row_and_column_labels_as_options() -> None:
    view = build_table_view(_document().tables[1])
    assert list(view.row_labels.values()) == ["CH4", "CO"]
    assert list(view.column_labels.values()) == ["TiO2", "10 wt.% In/TiO2"]
    assert "none" in view.questions["row"]["criteria"]
    assert "CH4 | 31 | 244" in view.text


def test_mask_table_references() -> None:
    assert mask_table_references("as listed in Tables 3 and 4.") == "as listed in the table."


def test_predict_shrinks_options_that_overflow_the_head_budget() -> None:
    class HeadLimited(KeywordModel):
        def predict_batch(self, states, questions):
            for qid, question in questions.items():
                if any(len(text) > 20 for text in question["criteria"].values()):
                    raise ValueError(f"question {qid!r} options exceed head_max_len=192")
            return super().predict_batch(states, questions)

    question = {"q": {"type": "choice", "instructions": "?", "criteria": {"a": "x" * 80, "b": "y"}}}
    answers = predict(HeadLimited(), ["state"], question)
    assert set(answers[0]["q"]["probabilities"]) == {"a", "b"}


def test_table_choice_offers_every_table_and_none() -> None:
    views = [build_table_view(table) for table in _document().tables]
    criteria = table_choice_question(views)["table"]["criteria"]
    assert criteria == {
        "T1": "BET surface area and band gap.",
        "T2": "Product yield rates (umol g^-1 h^-1).",
        "none": "none of these tables",
    }


def test_run_document_links_and_checks_numbers() -> None:
    sentences = [
        Sentence("s1", 9, "3.5", "The yield rate of CH4 over In-doped TiO2 is 244 umol g^-1 h^-1 (Table 2)."),
        Sentence("s2", 5, "3.2", "The BET surface area of TiO2 was 42.98 m2/g and increased with In doping."),
    ]
    result = run_document(_document(), sentences, KeywordModel(), LinkConfig())

    first = result["sentences"][0]
    top = first["links"][0]
    assert top["table_label"] == "Table 2"
    assert top["explicit_reference"] and "laya" in top["link_sources"]
    assert top["status"] == "consistent"
    assert top["row"] == "CH4" or top["column"] == "10 wt.% In/TiO2"

    bet = result["sentences"][1]["links"][0]
    assert bet["table_label"] == "Table 1"
    assert any(m["stated"] == "42.98" and m["kind"] == "rounded" for m in bet["numeric_matches"])

    evaluation = result["evaluation"]
    assert evaluation["evaluable_sentences"] == 1
    assert evaluation["laya_top1"] == 1.0
    assert evaluation["row_evaluable"] == 1
    assert evaluation["row_top1"] == 1.0
    assert evaluation["column_evaluable"] == 1
    assert evaluation["column_top1"] == 0.0


def test_document_sentences_from_ocr_markdown() -> None:
    pages = [
        "# 3. Results and discussion\n\nThe CH4 rate was 244 umol/g/h. It is 7.9-fold higher.\n\n"
        "<table><tr><td>CH4</td><td>244</td></tr></table>\n\nTable 1 Yield rates.\n\n# References\n\n"
        "[1] A. Author, J. Catal. 2015.",
    ]
    sentences = document_sentences(pages, source="ocr_cache", document_id="d")
    assert [s.text for s in sentences] == ["The CH4 rate was 244 umol/g/h.", "It is 7.9-fold higher."]
    assert sentences[0].section == "3. Results and discussion"


def test_laya_results_filename(tmp_path: Path) -> None:
    from argparse import Namespace

    args = Namespace(output_dir=None)
    assert _run_file(args, tmp_path, "laya-english").name == "laya-results.json"
    assert _run_file(Namespace(output_dir=tmp_path / "out"), tmp_path, "laya-multilingual").name == "laya-results.json"
    assert _run_file(Namespace(output_dir=tmp_path / "out"), tmp_path, "uniform").name == "uniform.json"


def test_cli_dry_run_writes_outputs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    papers = data / "analysis" / "kg_pilot_10" / "papers"
    papers.mkdir(parents=True)
    (papers / "demo.pdf").write_bytes(b"%PDF-1.4 placeholder")
    silver = data / "ground_truth" / "ground_truth_lighton_ocr_302.json"
    silver.parent.mkdir(parents=True)
    silver.write_text(GroundTruthCorpus(documents=[_document()]).model_dump_json(), encoding="utf-8")
    cache = data / "intermediate" / "ocr_cache" / "lighton_ocr"
    cache.mkdir(parents=True)
    (cache / "demo.json").write_text(
        json.dumps({"pages": ["# 3. Results\n\nThe yield of CH4 reached 244 umol/g/h as listed in Table 2."]}),
        encoding="utf-8",
    )

    out = tmp_path / "out"
    assert main(["--data-root", str(data), "--dry-run", "--output-dir", str(out)]) == 0

    run = json.loads((out / "uniform.json").read_text(encoding="utf-8"))
    result = run["documents"][0]
    assert result["text_source"] == "ocr_cache"
    assert result["counts"]["candidates"] == 1
    assert run["model"]["model"] == "uniform"
    assert list(out.iterdir()) == [out / "uniform.json"]


def test_score_claims_precision_and_table() -> None:
    document = {
        "document_id": "solarchem_demo",
        "sentences": [
            {
                "sentence_id": "s1",
                "p_no_table": 0.2,
                "claim": {"is_claim": True, "role": "result", "type": "performance"},
                "links": [
                    {"table_label": "Table 2", "p_link": 0.7, "status": "consistent"},
                    {"table_label": "Table 1", "p_link": 0.1, "status": "qualitative"},
                ],
            },
            {
                "sentence_id": "s2",
                "p_no_table": 0.2,
                "claim": {"is_claim": True, "role": "result", "type": "property"},
                "links": [{"table_label": "Table 1", "p_link": 0.6, "status": "qualitative"}],
            },
        ],
    }
    labels = {
        ("solarchem_demo", "s1"): {
            "gold_is_claim": "yes",
            "gold_role": "result",
            "gold_type": "performance",
            "gold_table": "Table 2",
            "gold_status": "consistent",
        },
        ("solarchem_demo", "s2"): {
            "gold_is_claim": "no",
            "gold_role": "method",
            "gold_table": "none",
        },
    }
    scores = score_claims([document], labels)
    assert scores["labeled"] == 2
    assert scores["tp"] == 1 and scores["fp"] == 1 and scores["fn"] == 0
    assert scores["precision"] == 0.5
    assert scores["recall"] == 1.0
    assert scores["role_accuracy"] == 0.5
    assert scores["table_labeled"] == 1
    assert scores["table_accuracy"] == 1.0
    assert scores["no_table_labeled"] == 1
    assert scores["no_table_accuracy"] == 0.0
    assert scores["status_accuracy"] == 1.0
    assert scores["text_mismatches"] == 0
    assert scores["recovered_by_text"] == 0


def test_score_skips_a_row_whose_text_no_longer_matches() -> None:
    document = {
        "document_id": "solarchem_demo",
        "sentences": [
            {
                "sentence_id": "s1",
                "text": "The yield of CH4 reached 244.",
                "p_no_table": 0.2,
                "claim": {"is_claim": True, "role": "result", "type": "performance"},
                "links": [{"table_label": "Table 2", "p_link": 0.7, "status": "consistent"}],
            },
            {
                "sentence_id": "s9",
                "text": "The BET surface area was 84 m2/g.",
                "p_no_table": 0.4,
                "claim": {"is_claim": True, "role": "result", "type": "property"},
                "links": [{"table_label": "Table 2", "p_link": 0.6, "status": "consistent"}],
            },
        ],
    }
    labels = {
        ("solarchem_demo", "s1"): {
            "gold_is_claim": "yes",
            "gold_role": "result",
            "text": "A different sentence now sits at this id.",
        },
        ("solarchem_demo", "moved"): {
            "gold_is_claim": "yes",
            "gold_role": "result",
            "gold_table": "Table 2",
            "text": "The BET surface area was 84 m2/g.",
        },
    }
    scores = score_claims([document], labels)
    assert scores["text_mismatches"] == 1
    assert scores["recovered_by_text"] == 1
    assert scores["labeled"] == 1
    assert scores["missing_sentences"] == 0
    assert scores["table_accuracy"] == 1.0


def test_cli_link_and_score(tmp_path: Path) -> None:
    data = tmp_path / "data"
    papers = data / "analysis" / "kg_pilot_10" / "papers"
    papers.mkdir(parents=True)
    (papers / "demo.pdf").write_bytes(b"%PDF-1.4 placeholder")
    silver = data / "ground_truth" / "ground_truth_lighton_ocr_302.json"
    silver.parent.mkdir(parents=True)
    silver.write_text(GroundTruthCorpus(documents=[_document()]).model_dump_json(), encoding="utf-8")
    cache = data / "intermediate" / "ocr_cache" / "lighton_ocr"
    cache.mkdir(parents=True)
    (cache / "demo.json").write_text(
        json.dumps({"pages": ["# 3. Results\n\nThe yield of CH4 reached 244 umol/g/h as listed in Table 2."]}),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    base = ["--data-root", str(data), "--dry-run", "--output-dir", str(out)]
    assert main(["link", *base]) == 0
    run = json.loads((out / "uniform.json").read_text(encoding="utf-8"))
    assert "table_reading" not in run["documents"][0]

    labels = data / "analysis" / "kg_pilot_10" / "claims_labels" / "demo.csv"
    labels.parent.mkdir(parents=True)
    labels.write_text(
        "document_id,sentence_id,gold_is_claim,gold_role,gold_table\n"
        f"solarchem_demo,{run['documents'][0]['sentences'][0]['sentence_id']},yes,result,none\n",
        encoding="utf-8",
    )
    assert main(["score", "--data-root", str(data), "--output-dir", str(out)]) == 0
    run = json.loads((out / "uniform.json").read_text(encoding="utf-8"))
    assert "table_reading" not in run["scores"]
    assert run["scores"]["claims"]["labeled"] == 1
    assert run["scores"]["claims"]["precision"] == 1.0
    assert run["scores"]["link"]["evaluable_sentences"] == 1
    assert list(out.iterdir()) == [out / "uniform.json"]

    out_run = tmp_path / "out-run"
    assert main(["run", "--data-root", str(data), "--dry-run", "--output-dir", str(out_run)]) == 0
    combined = json.loads((out_run / "uniform.json").read_text(encoding="utf-8"))
    assert "table_reading" not in combined["documents"][0]
    assert "table_reading" not in combined["scores"]
    assert combined["scores"]["claims"]["labeled"] == 1
    assert combined["scores"]["link"]["evaluable_sentences"] == 1
