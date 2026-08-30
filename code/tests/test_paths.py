"""Tests for data-root path helpers."""

from __future__ import annotations

from pathlib import Path

from solarchem_benchmark.paths import (
    default_merged_ground_truth_path,
    default_prediction_path,
    default_working_silver_path,
    merged_ground_truth_filename,
)


def test_merged_filename_includes_engine_id() -> None:
    assert merged_ground_truth_filename("lighton_ocr") == "ground_truth_lighton_ocr.json"
    assert merged_ground_truth_filename("unlimited_ocr") == "ground_truth_unlimited_ocr.json"


def test_default_merged_path_under_ground_truth(tmp_path: Path) -> None:
    path = default_merged_ground_truth_path(tmp_path, "lighton_ocr")
    assert path == tmp_path / "ground_truth" / "ground_truth_lighton_ocr.json"


def test_ocr_cache_dir_is_per_engine(tmp_path: Path) -> None:
    from solarchem_benchmark.paths import default_ocr_cache_dir

    assert default_ocr_cache_dir(tmp_path, "lighton_ocr") == (
        tmp_path / "intermediate" / "ocr_cache" / "lighton_ocr"
    )
    assert default_ocr_cache_dir(tmp_path, "unlimited_ocr") == (
        tmp_path / "intermediate" / "ocr_cache" / "unlimited_ocr"
    )
    assert default_ocr_cache_dir(tmp_path) == tmp_path / "intermediate" / "ocr_cache"


def test_prediction_path_is_under_predictions(tmp_path: Path) -> None:
    assert default_prediction_path(tmp_path, "pdfplumber") == (
        tmp_path / "predictions" / "pdfplumber.json"
    )
    assert default_prediction_path(tmp_path, "camelot_lattice") == (
        tmp_path / "predictions" / "camelot_lattice.json"
    )


def test_working_silver_snapshot_filename(tmp_path: Path) -> None:
    assert default_working_silver_path(tmp_path) == (
        tmp_path / "ground_truth" / "ground_truth_lighton_ocr_302.json"
    )
