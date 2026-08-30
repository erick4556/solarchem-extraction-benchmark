"""Tests for scientific text normalisation."""

from __future__ import annotations

import pytest

from solarchem_benchmark.gt.normalize import (
    MISSING_MARKER,
    coerce_cell,
    is_missing,
    normalize_header_label,
    normalize_scientific_text,
    normalize_unit,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("TiO\u2082", "TiO2"),
        ("CO\u2082", "CO2"),
        ("g\u207b\u00b9", "g^-1"),
        ("cm\u00b2", "cm^2"),
        ("m^2/g", "m^2/g"),
        ("10\u207b\u00b3", "10^-3"),
        ("\u00b5mol", "umol"),
        ("\u03bcmol", "umol"),
        ("\u00c5", "A"),
        ("a (\u00c5)", "a (A)"),
        ("CO _2", "CO2"),
        ("E_g (eV)", "Eg (eV)"),
        ("\\mu mol", "umol"),
        ("m ^2 /g", "m^2 /g"),
        ("g ^-1", "g^-1"),
        ("\u22120.5", "-0.5"),
        ("  spaced   out  ", "spaced out"),
    ],
)
def test_normalize_scientific_text(raw: str, expected: str) -> None:
    assert normalize_scientific_text(raw) == expected


def test_case_is_preserved_because_co_and_cobalt_differ() -> None:
    assert normalize_scientific_text("CO") == "CO"
    assert normalize_scientific_text("Co") == "Co"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("H$_2$BDC", "H2BDC"),
        ("CO$_2$", "CO2"),
        ("g$^{-1}$", "g^-1"),
        ("g$^{-1}$ h$^{-1}$", "g^-1 h^-1"),
        ("10$^{-3}$ mol", "10^-3 mol"),
        (r"$\mu$mol", "umol"),
        (r"5 $\mu$m", "5 um"),
        (r"$\lambda$ > 400 nm", "\u03bb > 400 nm"),
        (r"1.2 $\times$ 10$^{-3}$", "1.2 \u00d7 10^-3"),
        ("price is $5 and $9", "price is $5 and $9"),
        (r"$\unknownmacro$", r"\unknownmacro"),
        # ``\( ... \)`` / ``\[ ... \]`` — same rules, any OCR engine.
        (r"\(ATiO2 \)", "ATiO2"),
        (r"TiO\(2 \)", "TiO2"),
        (r"ATiO \( _{2} \)", "ATiO2"),
        (r"HCO\(2 \)H yield", "HCO2H yield"),
        (r"mg l\(^-1 \) g cat\(^-1 \)", "mg l^-1 g cat^-1"),
        (r"CO\( _{2} \) reduction", "CO2 reduction"),
        (r"\[ TiO_2 \]", "TiO2"),
    ],
)
def test_latex_math_markup_is_expanded(raw: str, expected: str) -> None:
    assert normalize_scientific_text(raw) == expected


def test_paren_and_dollar_math_read_the_same() -> None:
    forms = [r"CO$_2$", r"CO\( _{2} \)", "CO\u2082", "CO2"]
    assert {normalize_scientific_text(form) for form in forms} == {"CO2"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"Catalyst\(^a \)", "Catalyst"),
        (r"Light source\(^b \)", "Light source"),
        (r"Catalyst ^a", "Catalyst"),
        ("Light source^b", "Light source"),
        ("A(CO2) (cm^3 g^-1)^a", "A(CO2) (cm^3 g^-1)"),
        ("Sample", "Sample"),
    ],
)
def test_normalize_header_label_strips_footnote_markers(raw: str, expected: str) -> None:
    assert normalize_header_label(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # A script binds to its base, so the space the engine inserted goes.
        ("H $_2$ BDC", "H2BDC"),
        ("H $_2$ O and CO $_2$", "H2O and CO2"),
        # ...but a following word is not part of the formula.
        ("CO $_2$ reduction", "CO2 reduction"),
        ("CO $_2$ Reduction of TiO$_2$", "CO2 Reduction of TiO2"),
        ("TiO $_2$ Nanoparticles", "TiO2 Nanoparticles"),
        ("CO$_2$ RGO composite", "CO2 RGO composite"),
    ],
)
def test_a_script_reattaches_to_its_base_without_welding_words(raw: str, expected: str) -> None:
    assert normalize_scientific_text(raw) == expected


def test_a_species_reads_the_same_however_the_engine_wrote_it() -> None:
    forms = ["CO$_2$", "CO\u2082", "CO2", "CO $_2$"]
    assert {normalize_scientific_text(form) for form in forms} == {"CO2"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("mmol g\u207b\u00b9 h\u207b\u00b9", "mmol g^-1 h^-1"),
        ("\u00b5mol\u00b7g\u207b\u00b9\u00b7h\u207b\u00b9", "umol g^-1 h^-1"),
        ("%", "%"),
    ],
)
def test_normalize_unit(raw: str, expected: str) -> None:
    assert normalize_unit(raw) == expected


@pytest.mark.parametrize("raw", ["", "-", "\u2013", "\u2014", "   "])
def test_dashes_and_blanks_are_missing(raw: str) -> None:
    assert is_missing(raw)


@pytest.mark.parametrize("raw", ["ND", "N/A", "n.d.", "0"])
def test_textual_markers_are_not_missing(raw: str) -> None:
    assert not is_missing(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12.5", 12.5),
        ("10", 10),
        ("40,943", 40943),
        ("0,5", 0.5),
        ("-3.2", -3.2),
        ("", MISSING_MARKER),
        ("\u2014", MISSING_MARKER),
    ],
)
def test_coerce_cell_numeric(raw: str, expected: object) -> None:
    assert coerce_cell(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12.5 \u00b1 0.8", "12.5 \u00b1 0.8"),
        ("<0.01", "<0.01"),
        ("ND", "ND"),
        ("N/A", "N/A"),
        ("1.2 \u00d7 10\u207b\u00b3", "1.2 \u00d7 10^-3"),
        ("TiO\u2082", "TiO2"),
    ],
)
def test_coerce_cell_preserves_composite_values(raw: str, expected: str) -> None:
    assert coerce_cell(raw) == expected
