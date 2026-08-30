"""End-to-end tests for ground-truth assembly from page transcriptions.

These tests exercise the whole pipeline downstream of the OCR engine, using
synthetic transcriptions so that no model or GPU is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from solarchem_benchmark.gt.context import (
    caption_title,
    collect_mentions,
    find_captions,
    locate_tables,
)
from solarchem_benchmark.gt.generate import build_document, derive_document_id, write_document

PAGE_WITH_TABLE = """
## 3. Results and discussion

The photocatalytic performance of the prepared samples was evaluated under
simulated solar light for 4 h.

Table 1: Photocatalytic CO2 reduction activity over TiO2-based catalysts.

<table>
  <thead><tr><th>Catalyst</th><th>CH4 (&#181;mol g<sup>-1</sup> h<sup>-1</sup>)</th></tr></thead>
  <tbody>
    <tr><td>TiO<sub>2</sub></td><td>12.5</td></tr>
    <tr><td>g-C<sub>3</sub>N<sub>4</sub></td><td>8.2</td></tr>
  </tbody>
</table>

Compared with pristine g-C3N4, the TiO2 sample showed a higher methane yield.

The apparent quantum efficiency was measured at 365 nm.
"""

PAGE_WITH_MENTION = """
As summarised in Table 1, the TiO2 catalyst outperformed every other sample.

Further characterisation is discussed below.
"""

# Elsevier typesets the label and the caption text as two separate blocks, and
# the label often reaches the OCR as a heading or in bold.
PAGE_ELSEVIER_CAPTION = """
## 3.2. Optical properties

The absorption edge shifted towards the visible region.

## Table 1

Optical properties after modification of the benzene ring on H2BDC in UiO-66.

<table>
  <thead><tr><th>Organic linker</th><th>Band gap (eV)</th></tr></thead>
  <tbody><tr><td>H2BDC</td><td>3.91</td></tr></tbody>
</table>
"""


def _document():
    return build_document(
        [PAGE_WITH_TABLE, PAGE_WITH_MENTION],
        document_id="solarchem_test_001",
        source_pdf="test.pdf",
    )


def test_one_table_is_extracted_with_its_caption() -> None:
    document = _document()
    assert document.num_tables == 1
    table = document.tables[0]
    assert table.table_id == "solarchem_test_001_table_01"
    assert table.table_label == "Table 1"
    assert table.page == 1
    assert table.caption.startswith("Table 1: Photocatalytic CO2 reduction")


def test_the_title_is_derivable_from_the_stored_caption() -> None:
    table = _document().tables[0]
    expected = "Photocatalytic CO2 reduction activity over TiO2-based catalysts."
    assert caption_title(table.caption) == expected
    assert table.caption == f"{table.table_label}: {expected}"


def test_the_caption_is_the_only_stored_form_of_the_title() -> None:
    assert "title" not in _document().tables[0].model_dump()


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("Table 1: Reduction over TiO2.", "Reduction over TiO2."),
        ("Table 2. Band gap values.", "Band gap values."),
        ("Tab. 3 - Reaction conditions.", "Reaction conditions."),
        ("TABLE S1: Supplementary data.", "Supplementary data."),
        ("No label at all.", "No label at all."),
        ("", ""),
    ],
)
def test_caption_title_strips_every_label_style(caption: str, expected: str) -> None:
    assert caption_title(caption) == expected


def test_caption_split_over_two_blocks_is_rejoined() -> None:
    table = build_document(
        [PAGE_ELSEVIER_CAPTION], document_id="solarchem_elsevier", source_pdf="e.pdf"
    ).tables[0]
    expected = "Optical properties after modification of the benzene ring on H2BDC in UiO-66."
    assert table.table_label == "Table 1"
    assert table.caption == f"Table 1 {expected}"
    assert caption_title(table.caption) == expected


def test_a_caption_typeset_as_a_heading_is_not_a_section_title() -> None:
    table = build_document(
        [PAGE_ELSEVIER_CAPTION], document_id="solarchem_elsevier", source_pdf="e.pdf"
    ).tables[0]
    assert table.context.section_title == "3.2. Optical properties"


def test_a_bare_label_is_not_reported_as_a_mention() -> None:
    table = build_document(
        [PAGE_ELSEVIER_CAPTION], document_id="solarchem_elsevier", source_pdf="e.pdf"
    ).tables[0]
    assert [mention.text for mention in table.context.mentions] == []


@pytest.mark.parametrize(
    "label_block",
    ["## Table 1", "**Table 1**", "Table 1", "TABLE 1.", "Table 1:"],
)
def test_every_bare_label_style_absorbs_the_block_below(label_block: str) -> None:
    page = f"{label_block}\n\nCaption text here.\n\n<table><tr><td>A</td></tr><tr><td>1</td></tr></table>"
    assert caption_title(find_captions(page)[0].text) == "Caption text here."


def test_a_label_is_not_absorbed_into_the_table_or_a_heading() -> None:
    page = "Table 1\n\n<table><tr><td>A</td></tr><tr><td>1</td></tr></table>"
    caption = find_captions(page)[0]
    assert caption.text == "Table 1"
    assert caption_title(caption.text) == ""


def test_a_sentence_ending_before_a_line_break_is_not_a_caption() -> None:
    page = "The trend is summarised in Table 1.\nEven more noteworthy is the shift.\n\nNext."
    assert find_captions(page) == []


def test_continuation_caption_is_kept_verbatim() -> None:
    caption = find_captions("Table 2 (continued)\n\nPhotocatalysts BET Band gap")[0]
    assert caption.text == "Table 2 (continued)"


def test_table_without_a_caption_is_still_valid() -> None:
    document = build_document(
        ["<table><tr><td>A</td></tr><tr><td>1</td></tr></table>"],
        document_id="solarchem_nocaption",
        source_pdf="x.pdf",
    )
    assert document.tables[0].caption == ""
    assert caption_title(document.tables[0].caption) == ""


def test_a_species_is_spelled_alike_in_the_cell_the_caption_and_the_section() -> None:
    page = (
        "## 3.1. Functionalize organic linker H $_2$ BDC\n\n"
        "Table 1\n\n"
        "Optical properties of H$_2$BDC in UiO-66.\n\n"
        "<table><thead><tr><th>Linker</th></tr></thead>"
        "<tbody><tr><td>H<sub>2</sub>BDC</td></tr></tbody></table>\n"
    )
    table = build_document([page], document_id="d", source_pdf="x.pdf").tables[0]
    assert table.rows[0][0] == "H2BDC"
    assert table.caption == "Table 1 Optical properties of H2BDC in UiO-66."
    assert table.context.section_title == "3.1. Functionalize organic linker H2BDC"


def test_a_mention_is_normalised_like_the_rest() -> None:
    mentions = collect_mentions(["Reduction of CO$_2$ is reported in Table 1 for TiO $_2$."], "1")
    assert mentions == [{"page": 1, "text": "Reduction of CO2 is reported in Table 1 for TiO2."}]


def test_structure_is_flattened() -> None:
    table = _document().tables[0]
    assert table.columns == ["Catalyst", "CH4 (umol g^-1 h^-1)"]
    assert table.rows == [["TiO2", 12.5], ["g-C3N4", 8.2]]


def test_section_title_is_attached_to_the_same_table() -> None:
    assert _document().tables[0].context.section_title == "3. Results and discussion"


def test_document_title_heading_is_not_used_as_section_title() -> None:
    page = """
# Band gap engineered, oxygen-rich TiO2 for visible light induced photocatalytic reduction of CO2

Furthermore, a decrease in the c/a ratio was observed for oxygen-modified TiO2 (Table 1).

<table>
  <thead><tr><th>Sample</th><th>a (Å)</th></tr></thead>
  <tbody><tr><td>ATiO2</td><td>3.7748</td></tr></tbody>
</table>
"""
    table = build_document([page], document_id="d", source_pdf="x.pdf").tables[0]
    assert table.context.section_title == ""
    assert table.columns == ["Sample", "a (A)"]


def test_section_title_stays_positional_at_the_table_float() -> None:
    """Gold may prefer the discussing section; silver must not chase that."""
    page = """
## 1 Experimental

Samples were prepared as follows.

Table 1 Characterization data of the prepared samples.

<table>
  <thead><tr><th>Sample</th><th>SBET</th></tr></thead>
  <tbody><tr><td>TiO2</td><td>50</td></tr></tbody>
</table>

## 2 Results and discussion

The textural properties of the prepared samples are given in Table 1, together with the Ag content.
"""
    table = build_document([page], document_id="d", source_pdf="x.pdf").tables[0]
    assert table.context.section_title == "1 Experimental"


def test_mention_text_is_single_line() -> None:
    texts = [mention.text for mention in _document().tables[0].context.mentions]
    assert texts
    assert all("\n" not in text and "  " not in text for text in texts)


def test_mentions_come_from_other_pages_and_exclude_the_caption() -> None:
    context = _document().tables[0].context
    assert len(context.mentions) == 1
    assert context.mentions[0].page == 2
    assert "outperformed" in context.mentions[0].text
    assert not any(m.text.startswith("Table 1:") for m in context.mentions)


def test_context_holds_only_section_title_and_mentions() -> None:
    assert set(_document().tables[0].context.model_dump()) == {"section_title", "mentions"}


def test_no_mentions_without_a_caption_number() -> None:
    assert collect_mentions([PAGE_WITH_MENTION], None) == []


def test_page_without_tables_yields_no_occurrence() -> None:
    assert locate_tables([PAGE_WITH_MENTION]) == []


def test_document_with_no_tables_is_still_valid() -> None:
    document = build_document(
        [PAGE_WITH_MENTION], document_id="solarchem_empty", source_pdf="empty.pdf"
    )
    assert document.num_tables == 0
    assert document.tables == []


def test_document_id_is_deterministic_and_filesystem_safe() -> None:
    first = derive_document_id(Path("1-s2.0-S0021979721022451-main.pdf"))
    second = derive_document_id(Path("1-s2.0-S0021979721022451-main.pdf"))
    assert first == second == "solarchem_1_s2_0_s0021979721022451"
    assert derive_document_id(Path("10.1021@ja101318k.pdf")) == "solarchem_10_1021_ja101318k"


def test_written_file_round_trips(tmp_path: Path) -> None:
    from solarchem_benchmark.gt.validate import validate_file

    path = write_document(_document(), tmp_path)
    assert path.name == "solarchem_test_001.json"
    documents = validate_file(path)
    assert documents[0].tables[0].columns == ["Catalyst", "CH4 (umol g^-1 h^-1)"]


def test_the_merged_corpus_file_validates_too(tmp_path: Path) -> None:
    from solarchem_benchmark.gt.schema import GroundTruthCorpus
    from solarchem_benchmark.gt.validate import validate_file

    path = tmp_path / "ground_truth_all.json"
    corpus = GroundTruthCorpus(documents=[_document(), _document()])
    path.write_text(corpus.model_dump_json(indent=2), encoding="utf-8")
    assert len(validate_file(path)) == 2
