"""LightOnOCR-2 adapter: transcribe PDF pages before table extraction.

Raw transcriptions are cached per document so that re-running after a parser
change costs no GPU time.
"""

from __future__ import annotations

import json
import logging
import os
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


class OCREngine(ABC):
    """Transcribes document pages to text."""

    engine_id: str

    @abstractmethod
    def transcribe_page(self, image: Image) -> str:
        """Transcribe one rendered page to HTML/Markdown text."""

    def load(self) -> None:
        """Load weights. Called once before the first transcription."""

    def describe(self) -> dict[str, Any]:
        return {"engine_id": self.engine_id}


class LightOnOCREngine(OCREngine):
    """LightOnOCR-2 via Hugging Face Transformers.

    Needs ``transformers>=5.0`` (the LightOn OCR classes are not in 4.x).
    Rendering follows the model card: 200 DPI, longest side 1540 px.
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


def build_engine(model_id: str | None = None, **kwargs: Any) -> LightOnOCREngine:
    """Instantiate LightOnOCR, optionally overriding the Hugging Face model id."""
    if model_id:
        kwargs["model_id"] = model_id
    return LightOnOCREngine(**kwargs)


def render_page(
    document: pdfium.PdfDocument,
    page_index: int,
    *,
    dpi: int = DEFAULT_RENDER_DPI,
    target_longest: int = DEFAULT_TARGET_LONGEST,
) -> Image:
    """Render one page as an RGB image, downscaled to ``target_longest``."""
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
    """Transcribe every page of a PDF, caching the result."""
    cache_file = None
    if cache_dir is not None:
        cache_file = cache_dir / f"{pdf_path.stem}.json"
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
