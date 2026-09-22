"""Parser tests that do not need a GPU or a model download."""

from __future__ import annotations

from pathlib import Path

from table_context_extract.context import caption_title, collect_mentions, find_captions
from table_context_extract.generate import build_document, derive_document_id
from table_context_extract.tables import parse_html_table

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
"""

PAGE_WITH_MENTION = """
As summarised in Table 1, the TiO2 catalyst outperformed every other sample.

Further characterisation is discussed below.
"""


def _document():
    return build_document(
        [PAGE_WITH_TABLE, PAGE_WITH_MENTION],
        document_id="doc_test_001",
        source_pdf="test.pdf",
    )


def test_table_grid_caption_and_section() -> None:
    table = _document().tables[0]
    assert table.table_label == "Table 1"
    assert table.page == 1
    assert table.caption.startswith("Table 1: Photocatalytic CO2 reduction")
    assert table.columns == ["Catalyst", "CH4 (umol g^-1 h^-1)"]
    assert table.rows == [["TiO2", 12.5], ["g-C3N4", 8.2]]
    assert table.context.section_title == "3. Results and discussion"


def test_mentions_are_paragraphs_that_name_the_table() -> None:
    mentions = _document().tables[0].context.mentions
    assert len(mentions) == 1
    assert mentions[0].page == 2
    assert "outperformed" in mentions[0].text
    assert not any(m.text.startswith("Table 1:") for m in mentions)


def test_caption_title_strips_the_label() -> None:
    assert caption_title("Table 1: Reduction over TiO2.") == "Reduction over TiO2."
    assert caption_title(_document().tables[0].caption).startswith("Photocatalytic")


def test_split_elsevier_caption_is_rejoined() -> None:
    page = """
## 3.2. Optical properties

## Table 1

Optical properties after modification of the benzene ring.

<table>
  <thead><tr><th>Linker</th><th>Band gap (eV)</th></tr></thead>
  <tbody><tr><td>H2BDC</td><td>3.91</td></tr></tbody>
</table>
"""
    table = build_document([page], document_id="doc_elsevier", source_pdf="e.pdf").tables[0]
    assert table.table_label == "Table 1"
    assert "Optical properties" in table.caption
    assert table.context.section_title == "3.2. Optical properties"
    assert table.context.mentions == []


def test_sentence_ending_before_line_break_is_not_a_caption() -> None:
    page = "The trend is summarised in Table 1.\nEven more noteworthy is the shift.\n\nNext."
    assert find_captions(page) == []


def test_html_colspan_is_flattened() -> None:
    html = """
    <table>
      <thead>
        <tr><th rowspan="2">Catalyst</th><th colspan="2">Production rate</th></tr>
        <tr><th>CH4</th><th>CO</th></tr>
      </thead>
      <tbody><tr><td>TiO2</td><td>12.5</td><td>3.1</td></tr></tbody>
    </table>
    """
    flattened = parse_html_table(html)
    assert flattened is not None
    assert flattened.columns == ["Catalyst", "Production rate_CH4", "Production rate_CO"]
    assert flattened.rows == [["TiO2", 12.5, 3.1]]


def test_document_id_is_filesystem_safe() -> None:
    first = derive_document_id(Path("1-s2.0-S0021979721022451-main.pdf"))
    second = derive_document_id(Path("1-s2.0-S0021979721022451-main.pdf"))
    assert first == second == "doc_1_s2_0_s0021979721022451"


def test_no_mentions_without_a_table_number() -> None:
    assert collect_mentions([PAGE_WITH_MENTION], None) == []
