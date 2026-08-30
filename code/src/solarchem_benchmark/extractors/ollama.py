"""Phase 7 Ollama VLMs: table grids plus caption / section / mentions.

Each page is rendered like the OCR GT pipeline (200 DPI, longest side 1540)
and sent to a local Ollama chat endpoint. Mentions are filled in a second
text-only pass over the PDF text layer so they are not limited to the table
page. Stdlib HTTP only — no extra pip packages.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from solarchem_benchmark.extractors.base import (
    ExtractedGrid,
    ExtractedMention,
    NativeTableExtractor,
)
from solarchem_benchmark.gt.ocr import (
    DEFAULT_RENDER_DPI,
    DEFAULT_TARGET_LONGEST,
    render_page,
)

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_S = 600
# Vision pages need headroom for image tokens; 8192 fills the window and
# qwen3-vl then returns HTTP 200 with empty content after prompt eval.
PAGE_NUM_CTX = 32768
MENTION_NUM_CTX = 32768
PAGE_NUM_PREDICT = 4096
ARTICLE_TEXT_CHARS = 14_000

OLLAMA_MODELS: dict[str, str] = {
    "ollama_qwen3_vl": "qwen3-vl:32b",
    "ollama_gemma4": "gemma4:31b",
    "ollama_mistral_small": "mistral-small3.2:24b",
}

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

PAGE_PROMPT = """You extract tables from one page of a scientific PDF about photocatalysis / solar fuels.
Return JSON only, no markdown.

Schema:
{"tables":[{"table_label":"Table 1","caption":"full caption including Table N","section_title":"heading this table sits under","grid":[["header1","header2"],["cell","cell"]]}]}

Rules:
- grid[0] is the column headers. Further rows are the body. Repeat merged cells.
- If this page has no data table, return {"tables":[]}.
- Ignore figures, equations, and page headers/footers.
- table_label is only the identifier (Table 1, Table 2, ...).
- caption is the printed caption, or empty if none.
- section_title is the nearest section heading above the table on this page, or empty.
- Preserve numbers, units, and chemical formulas as printed.
"""

MENTION_PROMPT = """You are given tables already extracted from a scientific article, and the article text with page markers.
For each table, list in-text mentions: sentences or short paragraphs that refer to that table (e.g. "as shown in Table 1").
Return JSON only:
{"mentions_by_label":[{"table_label":"Table 1","mentions":[{"page":2,"text":"..."}]}]}
If a table is not mentioned, use an empty mentions list. Do not invent quotations.
"""

def _ollama_http_error(status: int, model: str, detail: str) -> str:
    lowered = detail.lower()
    if "cannot get current path" in lowered:
        return (
            f"Ollama HTTP {status} for {model}: llama-server cannot getcwd "
            "(its working directory was deleted — common if `ollama serve` "
            "was started from a Jupyter cwd that vanished). "
            "Stop Ollama, then: cd /tmp && ollama serve"
        )
    return f"Ollama HTTP {status} for {model}: {detail}"


PAGE_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "table_label": {"type": "string"},
                    "caption": {"type": "string"},
                    "section_title": {"type": "string"},
                    "grid": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "required": ["grid"],
            },
        }
    },
    "required": ["tables"],
}

MENTION_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mentions_by_label": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "table_label": {"type": "string"},
                    "mentions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "page": {"type": "integer"},
                                "text": {"type": "string"},
                            },
                            "required": ["page", "text"],
                        },
                    },
                },
                "required": ["table_label", "mentions"],
            },
        }
    },
    "required": ["mentions_by_label"],
}


def ollama_model_for(tool_id: str) -> str:
    try:
        return OLLAMA_MODELS[tool_id]
    except KeyError as error:
        raise ValueError(f"Unknown Ollama tool {tool_id!r}") from error


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    return ""


def ollama_message_text(body: Any) -> str:
    """Assistant text from an Ollama /api/chat object.

    Qwen3-VL often puts the JSON in ``message.thinking`` and leaves
    ``message.content`` empty when thinking or structured output is on.
    """
    if not isinstance(body, dict):
        return ""
    message_out = body.get("message")
    if isinstance(message_out, dict):
        for key in ("content", "thinking", "reasoning"):
            text = _as_text(message_out.get(key)).strip()
            if text:
                return text
    return _as_text(body.get("response")).strip()


def ollama_body_summary(body: Any) -> str:
    if not isinstance(body, dict):
        return f"non-object {type(body).__name__}"
    message_out = body.get("message") if isinstance(body.get("message"), dict) else {}
    content = _as_text(message_out.get("content") if isinstance(message_out, dict) else "")
    thinking = _as_text(message_out.get("thinking") if isinstance(message_out, dict) else "")
    return (
        f"done={body.get('done')!r} done_reason={body.get('done_reason')!r} "
        f"eval_count={body.get('eval_count')!r} "
        f"prompt_eval_count={body.get('prompt_eval_count')!r} "
        f"content_chars={len(content)} thinking_chars={len(thinking)}"
    )


def _loads_json(text: str) -> Any:
    stripped = text.strip()
    fenced = _JSON_FENCE_RE.search(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def _as_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _as_grid(raw: object) -> list[list[object | None]]:
    if not isinstance(raw, list) or not raw:
        return []
    rows: list[list[object | None]] = []
    width = 1
    for row in raw:
        if not isinstance(row, list):
            continue
        cells = [_as_cell(cell) for cell in row]
        width = max(width, len(cells), 1)
        rows.append(cells)
    padded: list[list[object | None]] = []
    for row in rows:
        padded.append(list(row) + [""] * (width - len(row)))
    return padded


def _norm_label(text: str) -> str:
    return " ".join(text.lower().split())


def parse_page_tables(payload: Any, page: int) -> list[ExtractedGrid]:
    """Turn one Ollama JSON object into grids for ``page`` (1-based)."""
    if isinstance(payload, list):
        payload = {"tables": payload}
    if not isinstance(payload, dict):
        return []
    tables = payload.get("tables")
    if not isinstance(tables, list):
        return []
    grids: list[ExtractedGrid] = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        grid = _as_grid(table.get("grid"))
        if len(grid) < 2:
            continue
        grids.append(
            ExtractedGrid(
                page=page,
                rows=grid,
                caption=_as_cell(table.get("caption")).strip(),
                table_label=_as_cell(table.get("table_label")).strip(),
                section_title=_as_cell(table.get("section_title")).strip(),
            )
        )
    return grids


def attach_mentions(grids: list[ExtractedGrid], payload: Any) -> list[ExtractedGrid]:
    """Copy mention lists onto grids by ``table_label``."""
    if not isinstance(payload, dict):
        return grids
    raw = payload.get("mentions_by_label")
    if not isinstance(raw, list):
        return grids
    by_label: dict[str, tuple[ExtractedMention, ...]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = _norm_label(_as_cell(item.get("table_label")))
        mentions: list[ExtractedMention] = []
        for mention in item.get("mentions") or []:
            if not isinstance(mention, dict):
                continue
            text = _as_cell(mention.get("text")).strip()
            if not text:
                continue
            try:
                page = int(mention.get("page") or 1)
            except (TypeError, ValueError):
                page = 1
            mentions.append(ExtractedMention(page=max(page, 1), text=text))
        if label:
            by_label[label] = tuple(mentions)
    attached: list[ExtractedGrid] = []
    for grid in grids:
        key = _norm_label(grid.table_label or grid.caption)
        mentions = by_label.get(key, ())
        attached.append(grid._replace(mentions=mentions))
    return attached


def _image_jpeg_b64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _pdf_text_by_page(pdf_path: Path, last_page: int) -> str:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    chunks: list[str] = []
    try:
        n_pages = min(len(document), last_page)
        for index in range(n_pages):
            page = document[index]
            textpage = page.get_textpage()
            try:
                text = textpage.get_text_range() or ""
            finally:
                close = getattr(textpage, "close", None)
                if close:
                    close()
                page_close = getattr(page, "close", None)
                if page_close:
                    page_close()
            chunks.append(f"[page {index + 1}]\n{text.strip()}")
    finally:
        close = getattr(document, "close", None)
        if close:
            close()
    joined = "\n\n".join(chunk for chunk in chunks if chunk.strip())
    if len(joined) > ARTICLE_TEXT_CHARS:
        joined = joined[:ARTICLE_TEXT_CHARS] + "\n[truncated]"
    return joined


class OllamaExtractor(NativeTableExtractor):
    """Vision-language table + context extractor via a local Ollama server."""

    emits_context = True

    def __init__(
        self,
        *,
        tool_id: str,
        model: str | None = None,
        host: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.tool_id = tool_id
        self.model = model or ollama_model_for(tool_id)
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip(
            "/"
        )
        self.timeout = timeout

    def extract_grids(
        self,
        pdf_path: Path,
        *,
        max_pages: int | None = None,
    ) -> list[ExtractedGrid]:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(pdf_path))
        grids: list[ExtractedGrid] = []
        try:
            n_pages = len(document)
            last = n_pages if max_pages is None or max_pages <= 0 else min(n_pages, max_pages)
            for index in range(last):
                image = render_page(
                    document,
                    index,
                    dpi=DEFAULT_RENDER_DPI,
                    target_longest=DEFAULT_TARGET_LONGEST,
                )
                page_no = index + 1
                logger.info(
                    "  Ollama %s page %d/%d",
                    self.model,
                    page_no,
                    last,
                )
                payload = self._chat_page(image, page_no)
                grids.extend(parse_page_tables(payload, page_no))
        finally:
            close = getattr(document, "close", None)
            if close:
                close()

        if grids:
            try:
                article = _pdf_text_by_page(
                    pdf_path,
                    last_page=max(grid.page for grid in grids),
                )
                mention_payload = self._chat_mentions(grids, article)
                grids = attach_mentions(grids, mention_payload)
            except Exception as error:  # noqa: BLE001 - keep tables if mentions fail
                logger.warning("Ollama mention pass failed (%s); keeping tables", error)
        return grids

    def _chat_page(self, image, page: int) -> Any:
        prompt = f"{PAGE_PROMPT}\nThis is page {page} of the PDF."
        return self._chat(
            prompt,
            images=[_image_jpeg_b64(image)],
            json_format=PAGE_FORMAT,
            num_ctx=PAGE_NUM_CTX,
        )

    def _chat_mentions(self, grids: list[ExtractedGrid], article: str) -> Any:
        found = [
            {
                "table_label": grid.table_label or f"page {grid.page} table",
                "caption": grid.caption,
                "page": grid.page,
            }
            for grid in grids
        ]
        prompt = (
            f"{MENTION_PROMPT}\nTables:\n{json.dumps(found, ensure_ascii=False)}\n\n"
            f"Article text:\n{article}"
        )
        return self._chat(
            prompt,
            images=None,
            json_format=MENTION_FORMAT,
            num_ctx=MENTION_NUM_CTX,
        )

    def _chat(
        self,
        prompt: str,
        *,
        images: list[str] | None,
        json_format: dict[str, Any],
        num_ctx: int,
    ) -> Any:
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            message["images"] = images
        # JSON Schema + Qwen3-VL often yields HTTP 200 with empty content.
        # format=json first; schema then unconstrained as fallbacks.
        attempts: list[dict[str, Any]] = [
            {"format": "json", "think": False},
            {"format": json_format, "think": False},
            {"think": False},
        ]
        last_summary = "no response"
        last_content = ""
        for extra in attempts:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [message],
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_ctx": num_ctx,
                    "num_predict": PAGE_NUM_PREDICT,
                },
                **extra,
            }
            body = self._post(payload)
            content = ollama_message_text(body)
            last_summary = ollama_body_summary(body)
            last_content = content
            if not content:
                logger.warning(
                    "Ollama empty content from %s (%s); %s",
                    self.model,
                    extra,
                    last_summary,
                )
                continue
            try:
                return _loads_json(content)
            except json.JSONDecodeError as error:
                logger.warning("Ollama JSON parse failed (%s): %s", extra, error)
                continue
        if last_content:
            return {"tables": []}
        raise RuntimeError(
            f"Ollama HTTP empty response from {self.model} ({last_summary}). "
            "qwen3-vl often does this when num_ctx is smaller than the image "
            "tokens or when JSON Schema format is used. Restart ollama from "
            "/tmp if llama-server is still crashing."
        )

    def _post(self, payload: dict[str, Any]) -> Any:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code in {400, 422} and payload.get("think") is False:
                retry = dict(payload)
                retry.pop("think", None)
                logger.info("Ollama rejected think=false; retrying without it")
                return self._post(retry)
            if error.code in {400, 422} and isinstance(payload.get("format"), dict):
                retry = dict(payload)
                retry.pop("think", None)
                retry["format"] = "json"
                logger.info("Ollama rejected JSON schema; retrying with format=json")
                return self._post(retry)
            raise RuntimeError(_ollama_http_error(error.code, self.model, detail)) from error
        except urllib.error.URLError as error:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.host} ({error.reason}). "
                "Start it with: ollama serve"
            ) from error
        except TimeoutError as error:
            raise TimeoutError(
                f"Ollama timed out after {self.timeout}s talking to {self.model}"
            ) from error
