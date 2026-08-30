"""Tests for article titles read from PDF metadata.

Every rejected sample below is a real ``/Title`` value found in the SolarChem
corpus, so the filter is tested against the artefacts it exists to remove.
"""

from __future__ import annotations

import pytest

from solarchem_benchmark.gt.metadata import clean_pdf_title

REAL_TITLES = [
    "Construction of NiO/g-C3N4 p-n heterojunctions for enhanced photocatalytic CO2 reduction",
    "Titanosilicates enhance carbon dioxide photocatalytic reduction",
    "Manganese carbonyl complexes for CO2 reduction",
]

ARTEFACTS = [
    "",
    "   ",
    "No Job Name",
    "untitled",
    "doi:",
    "doi:10.1016/j.catcom.2005.01.011",
    "PII: S1010-6030(97)00082-8",
    "ja304075b 1..6",
    "c6gc03527b 1..5 ++",
    "C2NR31718D 262..268",
    "CC 10 1147-1155..b317004g chapter .. Page1147",
    "manuscript 1..12",
    "001.docx",
    "b613098d.dvi",
    "01 kocemba.p65",
    "899370_File000005_16904184.pdf",
    "Template for Electronic Submission to ACS Journals - ACS_Catal_accepted.pdf",
    "C:\\PS\\RCF3F5.PS",
    "MJAS Vol 21 No 1 (2017)",
    "Microsoft Word - 14-2-04-122-130-Oman Zuas",
]


@pytest.mark.parametrize("title", REAL_TITLES)
def test_a_real_title_survives_unchanged(title: str) -> None:
    assert clean_pdf_title(title) == title


@pytest.mark.parametrize("artefact", ARTEFACTS)
def test_typesetting_artefacts_are_rejected(artefact: str) -> None:
    assert clean_pdf_title(artefact) == ""


def test_a_title_behind_a_word_prefix_is_recovered() -> None:
    raw = "Microsoft Word - 2-A Study on The Photoreduction of Green House CO2 Gas"
    assert clean_pdf_title(raw) == "2-A Study on The Photoreduction of Green House CO2 Gas"


def test_html_entities_are_decoded() -> None:
    raw = "CH-&#x03C0; interaction boosts CO2 reduction"
    assert clean_pdf_title(raw) == "CH-\u03c0 interaction boosts CO2 reduction"


def test_line_wrapping_is_removed() -> None:
    assert clean_pdf_title("Photocatalytic\nreduction   of\tCO2") == "Photocatalytic reduction of CO2"
