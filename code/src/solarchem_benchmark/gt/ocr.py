"""OCR engines used to transcribe pages before ground-truth extraction.

The generator is engine-agnostic on purpose. Which engine produced a draft is a
property of the run, not of the ground truth, and keeping the choice
configurable is what allows a later experiment to compare engines without the
draft generator silently favouring one of them.

Raw transcriptions are cached per document so that re-running the generator
after a parser change costs no GPU time.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pypdfium2 as pdfium

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from PIL.Image import Image

logger = logging.getLogger(__name__)

DEFAULT_RENDER_DPI = 200
DEFAULT_TARGET_LONGEST = 1540
DEFAULT_MAX_NEW_TOKENS = 8192


def _ensure_transformers_remote_code_compat() -> None:
    """Shim APIs that Hub ``trust_remote_code`` models still import.

    ``baidu/Unlimited-OCR`` ships DeepSeek-V2 modeling that imports
    ``is_torch_fx_available``, removed in transformers 5.0. Prefer pinning
    ``transformers>=4.57.1,<5``; the shim only unblocks that obsolete import.
    """
    import transformers.utils.import_utils as import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: True


def _configure_unlimited_ocr_sdpa() -> None:
    """Prefer non-cuDNN SDPA backends for Unlimited-OCR vision attention.

    The Hub ``deepencoder`` path calls ``scaled_dot_product_attention``; on some
    CUDA/cuDNN stacks the cuDNN SDPA backend raises
    ``No valid execution plans built``. Flash / mem-efficient / math still work.
    """
    import torch

    enable_cudnn = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
    if callable(enable_cudnn):
        enable_cudnn(False)
    # Keep other backends available when present.
    for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
        fn = getattr(torch.backends.cuda, name, None)
        if callable(fn) and name == "enable_math_sdp":
            fn(True)


class OCREngine(ABC):
    """Transcribes document pages to text."""

    #: Stable identifier used in cache filenames and CLI arguments.
    engine_id: str

    @abstractmethod
    def transcribe_page(self, image: Image) -> str:
        """Transcribe one rendered page.

        Args:
            image: The rendered page.

        Returns:
            The transcription, expected to contain tables as HTML or as
            Markdown pipe tables.
        """

    def load(self) -> None:
        """Load weights. Called once before the first transcription."""

    def describe(self) -> dict[str, Any]:
        """Return a short description of the engine, for run manifests."""
        return {"engine_id": self.engine_id}


class LightOnOCREngine(OCREngine):
    """LightOnOCR-2 via Hugging Face Transformers.

    The model is natively supported by Transformers, so no remote code is
    executed. The rendering resolution and the token budget follow the model
    card's recommendation of a 1540 px longest side.
    """

    engine_id = "lighton_ocr"

    def __init__(
        self,
        model_id: str = "lightonai/LightOnOCR-2-1B",
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None
        self._device: str = "cpu"
        self._dtype: Any = None

    def load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

        if torch.cuda.is_available():
            self._device = "cuda"
            self._dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif torch.backends.mps.is_available():
            self._device = "mps"
            self._dtype = torch.float32
        else:
            self._device = "cpu"
            self._dtype = torch.float32

        logger.info("Loading %s on %s (%s)", self.model_id, self._device, self._dtype)
        self._processor = LightOnOcrProcessor.from_pretrained(self.model_id)
        self._model = LightOnOcrForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self._dtype,
            attn_implementation="eager",
        ).to(self._device)

    def transcribe_page(self, image: Image) -> str:
        if self._model is None:
            self.load()

        import torch

        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(handle, format="PNG")
        handle.close()
        try:
            conversation = [{"role": "user", "content": [{"type": "image", "url": handle.name}]}]
            inputs = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(device=self._device, dtype=self._dtype)
                if value.is_floating_point()
                else value.to(self._device)
                for key, value in inputs.items()
            }
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            generated = output[0, inputs["input_ids"].shape[1] :]
            return self._processor.decode(generated, skip_special_tokens=True)
        finally:
            os.unlink(handle.name)

    def describe(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "model_id": self.model_id,
            "device": self._device,
            "max_new_tokens": self.max_new_tokens,
        }


class UnlimitedOCREngine(OCREngine):
    """baidu/Unlimited-OCR via Hugging Face ``AutoModel`` (custom code).

    Uses the single-image ``infer(..., eval_mode=True)`` entry point from the
    model card so the decoded page text is returned in-process (no result.md
    round-trip). Default image config is **gundam** (crop tiling), which the
    card recommends for single pages; ``base`` is available for ablations.

    LightOnOCR is untouched: this class only runs when ``--engine unlimited_ocr``
    is selected, and OCR caches are keyed by ``engine_id``.
    """

    engine_id = "unlimited_ocr"

    def __init__(
        self,
        model_id: str = "baidu/Unlimited-OCR",
        *,
        mode: str = "gundam",
        prompt: str = "<image>document parsing.",
        max_length: int = 8192,
        no_repeat_ngram_size: int = 35,
        ngram_window: int | None = None,
    ) -> None:
        if mode not in {"gundam", "base"}:
            raise ValueError(f"Unlimited-OCR mode must be 'gundam' or 'base', got {mode!r}")
        self.model_id = model_id
        self.mode = mode
        self.prompt = prompt
        self.max_length = max_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
        # Card defaults: gundam → 128, multi-page base → 1024. Single-page base
        # stays at 128 unless the caller overrides.
        self.ngram_window = 128 if ngram_window is None else ngram_window
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"

    def load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Unlimited-OCR requires CUDA: its published infer() path "
                "hardcodes .cuda() tensors. Use --engine lighton_ocr on CPU/MPS."
            )

        # Remote modeling_deepseekv2.py still imports is_torch_fx_available,
        # which transformers removed in 5.0. Prefer transformers 4.57.x;
        # keep a shim so a stray 5.x install fails later for real API drift,
        # not on this obsolete helper.
        _ensure_transformers_remote_code_compat()
        _configure_unlimited_ocr_sdpa()

        self._device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info("Loading %s on %s (%s, mode=%s)", self.model_id, self._device, dtype, self.mode)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=dtype,
        )
        self._model = self._model.eval().cuda()

    def transcribe_page(self, image: Image) -> str:
        if self._model is None:
            self.load()

        import shutil

        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(handle, format="PNG")
        handle.close()
        output_dir = tempfile.mkdtemp(prefix="unlimited_ocr_")
        try:
            if self.mode == "gundam":
                base_size, image_size, crop_mode = 1024, 640, True
            else:
                base_size, image_size, crop_mode = 1024, 1024, False

            raw = self._model.infer(
                self._tokenizer,
                prompt=self.prompt,
                image_file=handle.name,
                output_path=output_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                max_length=self.max_length,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                ngram_window=self.ngram_window,
                save_results=False,
                eval_mode=True,
                temperature=0.0,
            )
            if not isinstance(raw, str):
                raise RuntimeError(
                    "Unlimited-OCR infer(eval_mode=True) did not return a string; "
                    f"got {type(raw).__name__}. Check the installed model card revision."
                )
            return strip_unlimited_det_markers(raw)
        finally:
            os.unlink(handle.name)
            shutil.rmtree(output_dir, ignore_errors=True)

    def describe(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "model_id": self.model_id,
            "device": self._device,
            "mode": self.mode,
            "prompt": self.prompt,
            "max_length": self.max_length,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "ngram_window": self.ngram_window,
        }


_DET_LINE_RE = re.compile(
    r"<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)",
    re.DOTALL,
)


def strip_unlimited_det_markers(raw: str) -> str:
    """Drop Unlimited-OCR / DeepSeek-style ``<|det|>`` layout markers.

    The model card's OmniDocBench post-process keeps block text and drops image
    detections. HTML tables and Markdown stay in the content so our flatteners
    still see them.
    """
    if "<|det|>" not in raw:
        return raw.strip()

    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        match = _DET_LINE_RE.match(line)
        if match:
            category, content = match.group(1).strip(), match.group(2).strip()
            if category == "image":
                continue
            if current is not None:
                blocks.append(current)
            current = [content] if content else []
            continue
        if current is None:
            current = []
        current.append(line)
    if current is not None:
        blocks.append(current)
    return "\n\n".join("\n".join(block) for block in blocks).strip()


_ENGINES: dict[str, type[OCREngine]] = {
    LightOnOCREngine.engine_id: LightOnOCREngine,
    UnlimitedOCREngine.engine_id: UnlimitedOCREngine,
}


def available_engines() -> list[str]:
    """Return the registered engine identifiers."""
    return sorted(_ENGINES)


def build_engine(engine_id: str, **kwargs: Any) -> OCREngine:
    """Instantiate an engine by identifier.

    Args:
        engine_id: One of :func:`available_engines`.
        **kwargs: Forwarded to the engine constructor.

    Raises:
        KeyError: If the identifier is unknown.
    """
    try:
        engine_class = _ENGINES[engine_id]
    except KeyError:
        raise KeyError(
            f"Unknown OCR engine {engine_id!r}; available: {', '.join(available_engines())}"
        ) from None
    return engine_class(**kwargs)


def render_page(
    document: pdfium.PdfDocument,
    page_index: int,
    *,
    dpi: int = DEFAULT_RENDER_DPI,
    target_longest: int = DEFAULT_TARGET_LONGEST,
) -> Image:
    """Render one page as an RGB image.

    Args:
        document: An open PDF document.
        page_index: 0-based page index.
        dpi: Rendering resolution before downscaling.
        target_longest: Maximum length of the longest side, in pixels.

    Returns:
        The rendered page.
    """
    from PIL import Image as PILImage

    image = document[page_index].render(scale=dpi / 72).to_pil()
    width, height = image.size
    longest = max(width, height)
    if longest > target_longest:
        ratio = target_longest / longest
        image = image.resize((int(width * ratio), int(height * ratio)), PILImage.LANCZOS)
    return image if image.mode == "RGB" else image.convert("RGB")


def transcribe_document(
    pdf_path: Path,
    engine: OCREngine,
    *,
    cache_dir: Path | None = None,
    dpi: int = DEFAULT_RENDER_DPI,
    target_longest: int = DEFAULT_TARGET_LONGEST,
    max_pages: int | None = None,
    force: bool = False,
) -> list[str]:
    """Transcribe every page of a PDF, caching the result.

    Args:
        pdf_path: Path to the source PDF.
        engine: The OCR engine to use.
        cache_dir: Directory for cached transcriptions. Caching is disabled when
            ``None``.
        dpi: Rendering resolution.
        target_longest: Maximum length of the longest side, in pixels.
        max_pages: Stop after this many pages, for smoke tests.
        force: Ignore any cached transcription and recompute.

    Returns:
        One transcription per page, in order.
    """
    cache_file = None
    if cache_dir is not None:
        cache_file = cache_dir / f"{pdf_path.stem}.json"
        if not cache_file.exists() and not force:
            # Legacy flat layout from before per-engine subfolders:
            # intermediate/ocr_cache/<stem>__<engine_id>.json
            legacy = cache_dir.parent / f"{pdf_path.stem}__{engine.engine_id}.json"
            if legacy.is_file():
                cache_file = legacy
        if cache_file.exists() and not force:
            logger.debug("OCR cache hit: %s", cache_file)
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            pages: list[str] = payload["pages"]
            return pages[:max_pages] if max_pages else pages

    engine.load()
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        total = len(document)
        limit = min(total, max_pages) if max_pages else total
        pages = []
        for page_index in range(limit):
            logger.info("  OCR page %d/%d", page_index + 1, limit)
            image = render_page(
                document, page_index, dpi=dpi, target_longest=target_longest
            )
            pages.append(engine.transcribe_page(image))
    finally:
        document.close()

    if cache_dir is not None:
        write_path = cache_dir / f"{pdf_path.stem}.json"
        write_path.parent.mkdir(parents=True, exist_ok=True)
        write_path.write_text(
            json.dumps(
                {"source_pdf": pdf_path.name, "engine": engine.describe(), "pages": pages},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return pages
