"""Tests for table flattening."""

from __future__ import annotations

from solarchem_benchmark.gt.generate import build_document
from solarchem_benchmark.gt.tables import (
    extract_tables,
    find_html_tables,
    flatten_cell_grid,
    flatten_headers,
    parse_html_table,
    parse_markdown_table,
    peel_embedded_caption,
)

SIMPLE_HTML = """
<table>
  <thead><tr><th>Catalyst</th><th>CH<sub>4</sub> (&#181;mol g<sup>-1</sup> h<sup>-1</sup>)</th></tr></thead>
  <tbody>
    <tr><td>TiO<sub>2</sub></td><td>12.5</td></tr>
    <tr><td>g-C<sub>3</sub>N<sub>4</sub></td><td>8.2</td></tr>
  </tbody>
</table>
"""

MERGED_HTML = """
<table>
  <thead>
    <tr><th rowspan="2">Catalyst</th><th colspan="2">Production rate</th></tr>
    <tr><th>CH4</th><th>CO</th></tr>
  </thead>
  <tbody>
    <tr><td>TiO2</td><td>12.5</td><td>3.1</td></tr>
  </tbody>
</table>
"""

# LightOnOCR often puts the caption in the first header row as a colspan spanner.
CAPTION_IN_HEADER_HTML = """
<table>
  <thead>
    <tr><th colspan="5">Table 1 Characterization data of the prepared samples</th></tr>
    <tr>
      <th>Sample</th>
      <th>Ag content (wt.%)</th>
      <th>ABET (m<sup>2</sup>/g)</th>
      <th>Pore size rmax (nm)</th>
      <th>Absorption edge (eV)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>TiO2</td><td>0.00</td><td>67.6</td><td>1.48</td><td>2.98</td></tr>
    <tr><td>Ag/TiO2</td><td>5.19</td><td>79.7</td><td>1.65</td><td>2.74</td></tr>
  </tbody>
</table>
"""


def test_parse_simple_html_table() -> None:
    flattened = parse_html_table(SIMPLE_HTML)
    assert flattened is not None
    assert flattened.columns == ["Catalyst", "CH4 (umol g^-1 h^-1)"]
    assert flattened.rows == [["TiO2", 12.5], ["g-C3N4", 8.2]]
    assert flattened.embedded_caption == ""


def test_colspan_and_rowspan_are_expanded_into_a_rectangular_grid() -> None:
    flattened = parse_html_table(MERGED_HTML)
    assert flattened is not None
    assert flattened.columns == ["Catalyst", "Production rate_CH4", "Production rate_CO"]
    assert flattened.rows == [["TiO2", 12.5, 3.1]]
    assert all(len(row) == len(flattened.columns) for row in flattened.rows)


def test_table_without_thead_uses_first_row_as_header() -> None:
    html = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
    flattened = parse_html_table(html)
    assert flattened is not None
    assert flattened.columns == ["A", "B"]
    assert flattened.rows == [[1, 2]]


def test_repeated_header_level_contributes_once() -> None:
    assert flatten_headers([["Rate", "Rate"], ["CH4", "CO"]]) == ["Rate_CH4", "Rate_CO"]


def test_header_only_table_is_rejected() -> None:
    assert parse_html_table("<table><thead><tr><th>A</th></tr></thead></table>") is None


def test_markdown_table_is_flattened() -> None:
    lines = [
        "| Catalyst | CH4 |",
        "| --- | --- |",
        "| TiO2 | 12.5 |",
    ]
    result = parse_markdown_table(lines)
    assert result is not None
    assert result.columns == ["Catalyst", "CH4"]
    assert result.rows == [["TiO2", 12.5]]


def test_find_html_tables_returns_every_block() -> None:
    assert len(find_html_tables(SIMPLE_HTML + MERGED_HTML)) == 2


def test_markdown_is_ignored_when_the_page_has_html() -> None:
    page = SIMPLE_HTML + "\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    results = extract_tables(page)
    assert len(results) == 1
    assert results[0][1].columns == ["Catalyst", "CH4 (umol g^-1 h^-1)"]


def test_empty_rows_are_dropped() -> None:
    html = """
    <table><thead><tr><th>A</th><th>B</th></tr></thead>
    <tbody><tr><td></td><td>-</td></tr><tr><td>x</td><td>1</td></tr></tbody></table>
    """
    flattened = parse_html_table(html)
    assert flattened is not None
    assert flattened.columns == ["A", "B"]
    assert flattened.rows == [["x", 1]]


def test_caption_spanner_is_peeled_from_headers() -> None:
    row = ["Table 1 Characterization data of the prepared samples"] * 5
    remaining, caption = peel_embedded_caption([row, ["Sample", "Ag", "A", "B", "C"]])
    assert caption == "Table 1 Characterization data of the prepared samples"
    assert remaining == [["Sample", "Ag", "A", "B", "C"]]


def test_lighton_caption_inside_table_is_not_joined_into_columns() -> None:
    flattened = parse_html_table(CAPTION_IN_HEADER_HTML)
    assert flattened is not None
    assert flattened.embedded_caption == (
        "Table 1 Characterization data of the prepared samples"
    )
    assert flattened.columns == [
        "Sample",
        "Ag content (wt.%)",
        "ABET (m^2/g)",
        "Pore size rmax (nm)",
        "Absorption edge (eV)",
    ]
    assert flattened.rows == [
        ["TiO2", 0.0, 67.6, 1.48, 2.98],
        ["Ag/TiO2", 5.19, 79.7, 1.65, 2.74],
    ]


CAPTION_BODY_SPANNER_HTML = """
<table>
  <thead>
    <tr><th colspan="7">Table 1</th></tr>
    <tr><th colspan="7">Specific surface areas and CO2 adsorption for TiO2 with different calcination time.</th></tr>
    <tr>
      <th>Calcined Time (h)</th><th>0</th><th>0.25</th><th>1</th><th>2</th><th>3</th><th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Surface Areas (m2/g)</td><td>9.43</td><td>9.15</td><td>5.39</td><td>5.41</td><td>5.52</td><td>5.07</td></tr>
    <tr><td>CO2 abs (mol/g)</td><td>67.75</td><td>51.61</td><td>6.68</td><td>8.26</td><td>6.67</td><td>5.40</td></tr>
  </tbody>
</table>
"""

UNIT_SUBHEADER_HTML = """
<table>
  <thead>
    <tr>
      <th>Entry</th><th>Catalyst<sup>a</sup></th><th>pH</th><th>Light source<sup>b</sup></th>
      <th colspan="2">HCO<sub>2</sub>H yield</th>
    </tr>
  </thead>
  <tbody>
    <tr><td></td><td></td><td></td><td></td><td>mg l<sup>-1</sup> g cat<sup>-1</sup></td><td>mol g cat<sup>-1</sup></td></tr>
    <tr><td>1</td><td>TiO<sub>2</sub></td><td>3</td><td>S/H</td><td>120.5</td><td>131.0</td></tr>
    <tr><td>2</td><td>TiO<sub>2</sub>-H<sub>2</sub>Pc</td><td>3</td><td>S/H</td><td>69.0</td><td>75.0</td></tr>
  </tbody>
</table>
"""


def test_caption_body_spanner_is_peeled_and_not_joined_into_columns() -> None:
    flattened = parse_html_table(CAPTION_BODY_SPANNER_HTML)
    assert flattened is not None
    assert flattened.embedded_caption.startswith("Table 1 Specific surface areas")
    assert flattened.columns == ["Calcined Time (h)", "0", "0.25", "1", "2", "3", "4"]
    assert flattened.rows[0][0] == "Surface Areas (m2/g)"
    assert not any("Specific surface" in column for column in flattened.columns)


def test_unit_subheader_row_is_merged_into_columns() -> None:
    flattened = parse_html_table(UNIT_SUBHEADER_HTML)
    assert flattened is not None
    assert flattened.columns == [
        "Entry",
        "Catalyst",
        "pH",
        "Light source",
        "HCO2H yield (mg l^-1 g cat^-1)",
        "HCO2H yield (mol g cat^-1)",
    ]
    assert flattened.rows == [
        [1, "TiO2", 3, "S/H", 120.5, 131.0],
        [2, "TiO2-H2Pc", 3, "S/H", 69.0, 75.0],
    ]


def test_empty_header_under_unit_row_reuses_previous_stem() -> None:
    """OCR sometimes drops the right cell of a measure spanner."""
    html = """
    <table>
      <thead>
        <tr><th>Entry</th><th>HCO2H yield</th><th></th></tr>
      </thead>
      <tbody>
        <tr><td></td><td>mg l^-1</td><td>mol g^-1</td></tr>
        <tr><td>1</td><td>1.0</td><td>2.0</td></tr>
      </tbody>
    </table>
    """
    flattened = parse_html_table(html)
    assert flattened is not None
    assert flattened.columns == [
        "Entry",
        "HCO2H yield (mg l^-1)",
        "HCO2H yield (mol g^-1)",
    ]


def test_embedded_caption_fills_label_and_enables_mentions() -> None:
    page = f"""
## 2 Results and discussion

The textural properties are given in Table 1, together with the Ag content.

{CAPTION_IN_HEADER_HTML}
"""
    document = build_document([page], document_id="doc", source_pdf="x.pdf")
    table = document.tables[0]
    assert table.table_label == "Table 1"
    assert table.caption.startswith("Table 1 Characterization")
    assert not any(column.startswith("Table 1") for column in table.columns)
    assert table.context.mentions
    assert "Table 1" in table.context.mentions[0].text


def test_flatten_cell_grid_matches_html_without_thead() -> None:
    flattened = flatten_cell_grid([["Catalyst", "CH4"], ["TiO2", "12.5"], [None, "3.1"]])
    assert flattened is not None
    assert flattened.columns == ["Catalyst", "CH4"]
    assert flattened.rows == [["TiO2", 12.5], ["-", 3.1]]
    assert flattened.embedded_caption == ""


def test_flatten_cell_grid_peels_caption_spanner() -> None:
    caption = "Table 1 Characterization data of the prepared samples"
    flattened = flatten_cell_grid(
        [
            [caption, caption, caption],
            ["Sample", "Ag", "ABET"],
            ["TiO2", "0.00", "67.6"],
        ]
    )
    assert flattened is not None
    assert flattened.embedded_caption == caption
    assert flattened.columns == ["Sample", "Ag", "ABET"]
    assert flattened.rows == [["TiO2", 0.0, 67.6]]


def test_flatten_cell_grid_peels_stacked_caption_body() -> None:
    flattened = flatten_cell_grid(
        [
            ["Table 1", "Table 1", "Table 1"],
            [
                "Specific surface areas of the samples.",
                "Specific surface areas of the samples.",
                "Specific surface areas of the samples.",
            ],
            ["Sample", "A", "B"],
            ["TiO2", "1", "2"],
        ]
    )
    assert flattened is not None
    assert flattened.embedded_caption.startswith("Table 1 Specific surface")
    assert flattened.columns == ["Sample", "A", "B"]
    assert flattened.rows == [["TiO2", 1, 2]]


def test_flatten_cell_grid_merges_unit_row() -> None:
    flattened = flatten_cell_grid(
        [
            ["Entry", "HCO2H yield", ""],
            ["", "mg l^-1", "mol g^-1"],
            ["1", "1.0", "2.0"],
        ]
    )
    assert flattened is not None
    assert flattened.columns == [
        "Entry",
        "HCO2H yield (mg l^-1)",
        "HCO2H yield (mol g^-1)",
    ]
    assert flattened.rows == [[1, 1.0, 2.0]]


def test_flatten_cell_grid_rejects_header_only() -> None:
    assert flatten_cell_grid([["A", "B"]]) is None
    assert flatten_cell_grid([]) is None

