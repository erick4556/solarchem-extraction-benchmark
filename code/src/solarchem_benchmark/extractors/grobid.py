"""GROBID full-text TEI: table grids plus caption / section / mentions.

Talks to a running GROBID HTTP server (``/api/processFulltextDocument``).
Stdlib only — no extra pip packages. Same ``ExtractedGrid`` as the other
Phase 5–7 tools so assemble + eval stay unchanged.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from solarchem_benchmark.extractors.base import (
    ExtractedGrid,
    ExtractedMention,
    NativeTableExtractor,
)

logger = logging.getLogger(__name__)

DEFAULT_GROBID_HOST = "http://127.0.0.1:8070"
DEFAULT_TIMEOUT_S = 180
TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def _local(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _xml_id(node: ET.Element) -> str:
    return (node.get(XML_ID) or node.get("id") or "").strip()


def _page_from_coords(coords: str | None) -> int:
    """GROBID ``coords`` start with the 1-based page: ``p,x,y,w,h``."""
    if not coords:
        return 1
    first = coords.replace(";", " ").split()[0]
    token = first.split(",")[0].strip()
    try:
        return max(int(float(token)), 1)
    except ValueError:
        return 1


def _pad_rows(rows: list[list[str]]) -> list[list[object | None]]:
    if not rows:
        return []
    width = max(len(row) for row in rows)
    if width == 0:
        return []
    return [list(row) + [""] * (width - len(row)) for row in rows]


def _row_cells(row: ET.Element) -> list[str]:
    cells: list[str] = []
    for child in row:
        if _local(child.tag) != "cell":
            continue
        text = _text(child)
        raw_span = child.get("cols") or child.get("colspan") or "1"
        try:
            span = max(int(raw_span), 1)
        except ValueError:
            span = 1
        cells.extend([text] * span)
    return cells


def _table_rows(figure: ET.Element) -> list[list[str]]:
    table = figure.find(f"{{{TEI_NS}}}table")
    if table is None:
        return []
    rows: list[list[str]] = []
    for row in table:
        if _local(row.tag) != "row":
            continue
        cells = _row_cells(row)
        if cells:
            rows.append(cells)
    return rows


def _table_label(figure: ET.Element) -> str:
    head = _text(figure.find(f"{{{TEI_NS}}}head"))
    label = _text(figure.find(f"{{{TEI_NS}}}label"))
    if head.lower().startswith("table"):
        return head.split(".")[0].strip()
    if label:
        if label.lower().startswith("table"):
            return label
        return f"Table {label}"
    return ""


def _caption(figure: ET.Element) -> str:
    desc = _text(figure.find(f"{{{TEI_NS}}}figDesc"))
    head = _text(figure.find(f"{{{TEI_NS}}}head"))
    if desc:
        return desc
    return head


def _collect_mentions(root: ET.Element) -> dict[str, list[ExtractedMention]]:
    """Map figure xml:id → in-text ``<ref type="table">`` paragraphs."""
    by_id: dict[str, list[ExtractedMention]] = {}

    def walk(node: ET.Element) -> None:
        if _local(node.tag) == "p":
            page = _page_from_coords(node.get("coords"))
            paragraph = _text(node)
            for ref in node.iter(f"{{{TEI_NS}}}ref"):
                if (ref.get("type") or "").lower() != "table":
                    continue
                target = (ref.get("target") or "").lstrip("#").strip()
                if not target or not paragraph:
                    continue
                by_id.setdefault(target, []).append(
                    ExtractedMention(page=page, text=paragraph)
                )
        for child in list(node):
            walk(child)

    walk(root)
    return by_id


def _is_table_figure(node: ET.Element) -> bool:
    if _local(node.tag) != "figure":
        return False
    if (node.get("type") or "").lower() == "table":
        return True
    return node.find(f"{{{TEI_NS}}}table") is not None


def parse_grobid_tei(xml_text: str, *, max_pages: int | None = None) -> list[ExtractedGrid]:
    """Turn GROBID TEI into grids (page, cells, caption, section, mentions)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise RuntimeError(f"GROBID returned invalid TEI: {error}") from error

    mentions_by_id = _collect_mentions(root)
    grids: list[ExtractedGrid] = []

    def walk(node: ET.Element, section: str) -> None:
        next_section = section
        if _local(node.tag) == "div":
            head = node.find(f"{{{TEI_NS}}}head")
            if head is not None:
                heading = _text(head)
                if heading:
                    next_section = heading
        if _is_table_figure(node):
            rows = _pad_rows(_table_rows(node))
            if len(rows) >= 2:
                page = _page_from_coords(node.get("coords"))
                if max_pages is None or max_pages <= 0 or page <= max_pages:
                    figure_id = _xml_id(node)
                    grids.append(
                        ExtractedGrid(
                            page=page,
                            rows=rows,
                            caption=_caption(node),
                            table_label=_table_label(node),
                            section_title=next_section,
                            mentions=tuple(mentions_by_id.get(figure_id, ())),
                        )
                    )
        for child in list(node):
            walk(child, next_section)

    walk(root, "")
    return grids


class GrobidExtractor(NativeTableExtractor):
    """Table + context extractor via a local GROBID full-text server."""

    tool_id = "grobid"
    emits_context = True

    def __init__(
        self,
        *,
        host: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.host = (host or os.environ.get("GROBID_HOST") or DEFAULT_GROBID_HOST).rstrip(
            "/"
        )
        self.timeout = timeout

    def extract_grids(
        self,
        pdf_path: Path,
        *,
        max_pages: int | None = None,
    ) -> list[ExtractedGrid]:
        tei = self._process_fulltext(pdf_path)
        grids = parse_grobid_tei(tei, max_pages=max_pages)
        logger.info("  GROBID %s: %d tables", pdf_path.name, len(grids))
        return grids

    def _process_fulltext(self, pdf_path: Path) -> str:
        query = urllib.parse.urlencode(
            {
                "consolidateHeader": "0",
                "consolidateCitations": "0",
                "teiCoordinates": "figure,ref,p,s,head,div,table",
            }
        )
        url = f"{self.host}/api/processFulltextDocument?{query}"
        body, content_type = _pdf_multipart(pdf_path)
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            if error.code in {502, 503, 504}:
                raise ConnectionError(
                    f"Cannot reach GROBID at {self.host} (HTTP {error.code}). "
                    "Wait until the service is up, or restart the GROBID container."
                ) from error
            raise RuntimeError(
                f"GROBID HTTP {error.code} for {pdf_path.name}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise ConnectionError(
                f"Cannot reach GROBID at {self.host} ({error.reason}). "
                "Start it with: docker run --rm -p 8070:8070 grobid/grobid:0.8.2"
            ) from error
        except TimeoutError as error:
            raise TimeoutError(
                f"GROBID timed out after {self.timeout}s on {pdf_path.name}"
            ) from error


def _pdf_multipart(pdf_path: Path) -> tuple[bytes, str]:
    boundary = "----SolarChemGrobidBoundary"
    filename = pdf_path.name.replace('"', "")
    preamble = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="input"; filename="{filename}"\r\n'
        "Content-Type: application/pdf\r\n"
        "\r\n"
    ).encode("utf-8")
    closing = f"\r\n--{boundary}--\r\n".encode("ascii")
    return preamble + pdf_path.read_bytes() + closing, (
        f"multipart/form-data; boundary={boundary}"
    )
