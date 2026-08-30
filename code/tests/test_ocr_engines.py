"""Tests for OCR engine adapters (no GPU / no model download)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from solarchem_benchmark.gt.ocr import (
    UnlimitedOCREngine,
    _ensure_transformers_remote_code_compat,
    available_engines,
    build_engine,
    strip_unlimited_det_markers,
)


def test_unlimited_ocr_is_registered() -> None:
    assert "unlimited_ocr" in available_engines()
    assert "lighton_ocr" in available_engines()
    engine = build_engine("unlimited_ocr")
    assert isinstance(engine, UnlimitedOCREngine)
    assert engine.engine_id == "unlimited_ocr"


def test_configure_unlimited_ocr_sdpa_disables_cudnn() -> None:
    torch = pytest.importorskip("torch")
    if not hasattr(torch.backends, "cuda") or not hasattr(
        torch.backends.cuda, "enable_cudnn_sdp"
    ):
        pytest.skip("torch.backends.cuda.enable_cudnn_sdp unavailable")

    from solarchem_benchmark.gt.ocr import _configure_unlimited_ocr_sdpa

    torch.backends.cuda.enable_cudnn_sdp(True)
    _configure_unlimited_ocr_sdpa()
    # Query via the enable API's companion when available.
    is_enabled = getattr(torch.backends.cuda, "cudnn_sdp_enabled", None)
    if callable(is_enabled):
        assert is_enabled() is False
    else:
        # Older torch: just ensure the call does not raise.
        assert True


def test_transformers_remote_code_compat_shim() -> None:
    pytest.importorskip("transformers")
    import transformers.utils.import_utils as import_utils

    _ensure_transformers_remote_code_compat()
    assert callable(import_utils.is_torch_fx_available)


def test_unlimited_ocr_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="gundam"):
        UnlimitedOCREngine(mode="turbo")


def test_strip_unlimited_det_markers_keeps_table_html() -> None:
    raw = (
        "<|det|>title [10, 10, 100, 40]<|/det|>Results\n"
        "<|det|>table [10, 50, 400, 200]<|/det|><table><tr><td>A</td></tr></table>\n"
        "<|det|>image [0, 0, 1, 1]<|/det|>ignore me\n"
    )
    cleaned = strip_unlimited_det_markers(raw)
    assert "<table>" in cleaned
    assert "Results" in cleaned
    assert "ignore me" not in cleaned
    assert "<|det|>" not in cleaned


def test_strip_passthrough_when_no_markers() -> None:
    text = "<table><tr><td>1</td></tr></table>"
    assert strip_unlimited_det_markers(text) == text


def test_transcribe_page_calls_infer_eval_mode_and_cleans_markers() -> None:
    engine = UnlimitedOCREngine(mode="gundam")
    engine._model = MagicMock()
    engine._tokenizer = MagicMock()
    engine._device = "cuda"
    engine._model.infer.return_value = (
        "<|det|>table [0,0,1,1]<|/det|><table><tr><td>TiO2</td></tr></table>"
    )

    image = Image.new("RGB", (64, 64), color=(255, 255, 255))
    text = engine.transcribe_page(image)

    assert text == "<table><tr><td>TiO2</td></tr></table>"
    kwargs = engine._model.infer.call_args.kwargs
    assert kwargs["eval_mode"] is True
    assert kwargs["save_results"] is False
    assert kwargs["crop_mode"] is True
    assert kwargs["image_size"] == 640
    assert kwargs["prompt"] == "<image>document parsing."


def test_describe_includes_mode() -> None:
    engine = UnlimitedOCREngine(mode="base")
    meta = engine.describe()
    assert meta["engine_id"] == "unlimited_ocr"
    assert meta["mode"] == "base"
