"""Table flattening.

An OCR engine emits tables as HTML (preferred) or as a Markdown pipe table.
Both are reduced to the same rectangular form used by the ground truth:

* merged cells are expanded, so a ``rowspan``/``colspan`` region repeats its
  value across every grid position it covers;
* when the top header row is a caption spanner (``Table N ...`` repeated across
  columns), it is peeled out instead of being joined into every column label;
  a following full-width row is peeled only when it continues that caption
  (stacked ``Table N`` + title), not by length heuristics;
* multi-level headers are collapsed to one label per column, joining the
  distinct levels top to bottom with ``_`` (``Product`` over ``CH4`` becomes
  ``Product_CH4``);
* a leading body row that only carries unit strings is merged into the column
  labels (``HCO2H yield`` + ``mg l^-1 ...`` → ``HCO2H yield (mg l^-1 ...)``);
* body cells are coerced to numbers when unambiguous.

The header-joining and span-expansion strategy follows the GAP-KGE extractor
(``table_extraction/extract_tables_lightonocr.py``) so that both benchmarks
produce comparable flattened tables. It deviates in one respect: spaces inside a
header level are preserved rather than replaced with ``_``, which keeps units
legible in labels such as ``CH4 (umol g^-1 h^-1)``.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from bs4 import BeautifulSoup, NavigableString, Tag

from solarchem_benchmark.gt.normalize import (
    MISSING_MARKER,
    coerce_cell,
    expand_latex_math_delimiters,
    normalize_header_label,
    normalize_scientific_text,
)

CellValue = str | int | float

HTML_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)

_LATEX_COMMANDS = ("textbf", "textit", "text", "mathbf", "mathrm", "mathit", "ce")
_SEPARATOR_CELL_RE = re.compile(r"^:?-{1,}:?$")
_NUMBER = r"[A-Za-z]?\d+(?:\.\d+)?|[IVXLC]+"
#: Caption text that LightOnOCR sometimes embeds as the top ``<th colspan>``.
_CAPTION_SPANNER_RE = re.compile(
    rf"^(?:\*\*)?(?:Table|Tab\.?|TABLE)\s+({_NUMBER})(?:\*\*)?\b",
    re.IGNORECASE,
)
#: Unit-like tokens used to detect a sub-header row under the main headers.
#: These are generic SI / catalysis unit fragments, not paper-specific labels.
_UNIT_HINT_RE = re.compile(
    r"(?:\b(?:mol|mmol|umol|cat|wt)\b|m\^?2|cm\^?3|/g|/h|eV|\bnm\b|%\b|h\^-|l\^-|min\b)",
    re.IGNORECASE,
)


class FlattenedTable(NamedTuple):
    """Result of flattening one HTML or Markdown table."""

    columns: list[str]
    rows: list[list[CellValue]]
    embedded_caption: str = ""



def strip_inline_formatting(text: str) -> str:
    """Remove Markdown and LaTeX decoration, keeping the scientific payload.

    Args:
        text: Raw cell text as produced by the OCR engine.

    Returns:
        The plain text, normalised with
        :func:`~solarchem_benchmark.gt.normalize.normalize_scientific_text`.
    """
    if not text:
        return ""
    result = text.strip()
    result = re.sub(r"\*\*(.+?)\*\*", r"\1", result)
    result = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", result)
    result = re.sub(r"`(.+?)`", r"\1", result)
    # Expand ``\( ... \)`` / ``$...$`` before brace-stripping so scripts stay
    # attached to their base (same path for every OCR engine).
    result = expand_latex_math_delimiters(result)
    result = re.sub(r"\$\$?([^$]+?)\$\$?", r"\1", result)
    for command in _LATEX_COMMANDS:
        result = re.sub(rf"\\{command}\{{(.+?)\}}", r"\1", result)
    result = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", result)
    result = re.sub(r"\\(left|right)\b", "", result)
    result = result.replace("\\times", "x").replace("\\cdot", "\u00b7")
    result = result.replace("\\pm", "\u00b1")
    result = result.replace("\\leq", "\u2264").replace("\\geq", "\u2265")
    result = re.sub(r"\\([#&$%_])", r"\1", result)
    result = re.sub(r"\\mu\b", "u", result)
    # LaTeX sub/superscript braces: ``g^{-1}`` -> ``g^-1``, ``H_{2}O`` -> ``H2O``.
    result = re.sub(r"\^\{([^{}]*)\}", r"^\1", result)
    result = re.sub(r"_\{([^{}]*)\}", r"\1", result)
    result = re.sub(r"(?<=[A-Za-z\)])_(\d)", r"\1", result)
    result = result.replace("{", "").replace("}", "")
    result = re.sub(r"\(\s+", "(", result)
    result = re.sub(r"\s+\)", ")", result)
    return normalize_scientific_text(result)


def find_html_tables(text: str) -> list[str]:
    """Return every ``<table>...</table>`` block in an OCR transcription."""
    return HTML_TABLE_RE.findall(text)


def cell_text(cell: Tag) -> str:
    """Read a table cell, keeping sub- and superscripts attached to their base.

    ``<td>CH<sub>4</sub></td>`` must read ``CH4`` and ``<td>g<sup>-1</sup></td>``
    must read ``g^-1``. Joining the cell's text nodes with a space would instead
    yield ``CH 4`` and ``g -1``, splitting formulas and units apart, so the
    markup is rewritten before the text is read and only genuine line or block
    breaks contribute whitespace.

    Args:
        cell: A ``<td>`` or ``<th>`` element. It is modified in place; the
            surrounding soup is discarded after parsing.

    Returns:
        The cell text, normalised.
    """
    for tag in cell.find_all(["br"]):
        tag.replace_with(NavigableString(" "))
    for tag in cell.find_all(["p", "div", "li"]):
        tag.insert_after(NavigableString(" "))
    for tag in cell.find_all(["sub", "sup"]):
        text = tag.get_text().strip()
        prefix = "^" if tag.name == "sup" and text else ""
        tag.replace_with(NavigableString(f"{prefix}{text}"))
    return strip_inline_formatting(cell.get_text())


def _span(cell: Tag, attribute: str) -> int:
    try:
        value = int(str(cell.get(attribute, 1)))
    except (TypeError, ValueError):
        return 1
    return max(value, 1)


def _place_row(cells: list[Tag], grid: dict[int, dict[int, str]], row_index: int) -> None:
    """Write one ``<tr>`` into a sparse grid, honouring row and column spans."""
    column = 0
    for cell in cells:
        while grid.get(row_index, {}).get(column) is not None:
            column += 1
        text = cell_text(cell)
        for row_offset in range(_span(cell, "rowspan")):
            for column_offset in range(_span(cell, "colspan")):
                grid.setdefault(row_index + row_offset, {})[column + column_offset] = text
        column += _span(cell, "colspan")


def _grid_to_rows(grid: dict[int, dict[int, str]]) -> list[list[str]]:
    """Densify a sparse grid into a rectangular list of rows."""
    if not grid:
        return []
    height = max(grid) + 1
    width = max((max(row) + 1 for row in grid.values() if row), default=0)
    return [[grid.get(r, {}).get(c, "") for c in range(width)] for r in range(height)]


def _pad(rows: list[list[str]], width: int) -> list[list[str]]:
    return [row + [""] * (width - len(row)) for row in rows]


def flatten_headers(header_rows: list[list[str]]) -> list[str]:
    """Collapse one or more header rows into a single label per column.

    Args:
        header_rows: Header rows of equal width, ordered top to bottom.

    Returns:
        One label per column. Consecutive repeated levels, which is how a
        horizontally merged spanner cell appears after expansion, contribute a
        single time.
    """
    if not header_rows:
        return []
    if len(header_rows) == 1:
        return [label.strip() for label in header_rows[0]]

    columns: list[str] = []
    for column in range(len(header_rows[0])):
        parts: list[str] = []
        previous: str | None = None
        for row in header_rows:
            value = row[column].strip()
            if value and value != previous:
                parts.append(value)
            previous = value
        columns.append("_".join(parts))
    return [normalize_header_label(column) for column in columns]


def _uniform_row_text(row: list[str]) -> str | None:
    """Return the shared cell text when every non-empty cell is identical."""
    values = [cell.strip() for cell in row if cell and cell.strip()]
    if not values:
        return None
    unique = set(values)
    if len(unique) != 1:
        return None
    return values[0]


def caption_spanner_text(row: list[str]) -> str | None:
    """Return the caption text when ``row`` is a ``Table N`` spanner, else ``None``.

    LightOnOCR often typesets the caption as a single ``colspan`` cell in the
    first header row. After span expansion every cell holds the same
    ``Table N ...`` string; joining that into column labels pollutes them and
    leaves ``caption`` empty in the ground truth.
    """
    if len(row) <= 1:
        return None
    text = _uniform_row_text(row)
    if text is None:
        return None
    if not _CAPTION_SPANNER_RE.match(text):
        return None
    return text


def peel_embedded_caption(
    header_rows: list[list[str]],
) -> tuple[list[list[str]], str]:
    """Remove leading caption spanners from the header block, if present.

    After a ``Table N`` spanner, one extra full-width row is absorbed when it
    is the caption body stacked underneath the label (same colspan pattern,
    no ``Table N`` prefix). That row is identified structurally (uniform across
    columns, immediately after a table-label spanner), not by paper-specific
    text or length cutoffs tuned to the Gold set.
    """
    if not header_rows:
        return header_rows, ""

    remaining = list(header_rows)
    captions: list[str] = []

    first = caption_spanner_text(remaining[0])
    if first is None:
        return header_rows, ""
    captions.append(first)
    remaining = remaining[1:]

    # Stacked caption body: only legal right after a Table-N label spanner.
    if remaining:
        body = _uniform_row_text(remaining[0])
        if (
            body is not None
            and len(remaining[0]) > 1
            and not _CAPTION_SPANNER_RE.match(body)
            and not body.startswith(("|", "<"))
        ):
            captions.append(body)
            remaining = remaining[1:]

    return remaining, merge_caption_parts(*captions)


def merge_caption_parts(*parts: str) -> str:
    """Join caption fragments without duplicating a shared ``Table N`` label."""
    cleaned = [normalize_scientific_text(part) for part in parts if part and part.strip()]
    if not cleaned:
        return ""
    merged = cleaned[0]
    for part in cleaned[1:]:
        if not part or part in merged:
            continue
        if merged in part:
            merged = part
            continue
        merged = f"{merged} {part}"
    return normalize_scientific_text(merged)


def table_number_from_caption(caption: str) -> str | None:
    """Return the canonical table number embedded in a caption, if any."""
    match = _CAPTION_SPANNER_RE.match(caption.strip())
    if match is None:
        return None
    return match.group(1).strip().upper().replace(" ", "")


def _is_unit_row(row: list[CellValue]) -> bool:
    """True when a body row only carries unit strings under the headers."""
    if any(isinstance(value, (int, float)) for value in row):
        return False
    non_empty = [
        str(value).strip()
        for value in row
        if value != MISSING_MARKER and str(value).strip()
    ]
    if not non_empty:
        return False
    unitish = sum(1 for value in non_empty if _UNIT_HINT_RE.search(value))
    return unitish >= max(1, (len(non_empty) + 1) // 2)


def _merge_unit_into_columns(columns: list[str], unit_row: list[CellValue]) -> list[str]:
    """Attach unit sub-header cells to the column labels above them.

    When a measure name was a horizontal spanner but the OCR left the right
    header cell empty, the previous non-empty label is reused as the stem so
    the unit row still produces ``Measure (unit)`` instead of a bare unit.
    """
    merged: list[str] = []
    last_stem = ""
    for column, unit in zip(columns, unit_row):
        label = normalize_header_label(column)
        unit_text = ""
        if unit != MISSING_MARKER:
            unit_text = normalize_header_label(str(unit))
        if label:
            last_stem = re.sub(r"\s*\([^()]*\)\s*$", "", label).strip() or label
        elif unit_text and last_stem:
            label = last_stem
        if not unit_text:
            merged.append(label)
            continue
        if not label:
            merged.append(unit_text)
            continue
        if unit_text.lower() in label.lower():
            merged.append(label)
            continue
        if label.endswith(")"):
            merged.append(f"{label} {unit_text}")
        else:
            merged.append(f"{label} ({unit_text})")
    if len(columns) > len(unit_row):
        merged.extend(columns[len(unit_row) :])
    return merged


def absorb_unit_header_row(
    columns: list[str],
    body_rows: list[list[CellValue]],
) -> tuple[list[str], list[list[CellValue]]]:
    """Merge a leading unit-only body row into ``columns`` when present."""
    if not body_rows or not _is_unit_row(body_rows[0]):
        return columns, body_rows
    return _merge_unit_into_columns(columns, body_rows[0]), body_rows[1:]


def _coerce_body(body_rows: list[list[str]]) -> list[list[CellValue]]:
    """Coerce body cells and drop rows that carry no content."""
    data_rows: list[list[CellValue]] = []
    for row in body_rows:
        coerced = [coerce_cell(value) for value in row]
        if any(value != MISSING_MARKER for value in coerced):
            data_rows.append(coerced)
    return data_rows


def parse_html_table(html: str) -> FlattenedTable | None:
    """Flatten one HTML table.

    Args:
        html: A single ``<table>...</table>`` fragment.

    Returns:
        A :class:`FlattenedTable`, or ``None`` when the fragment holds no
        header or no data row. ``embedded_caption`` is set when the OCR put
        the caption inside the table as a header spanner.
    """
    table = BeautifulSoup(html, "html.parser").find("table")
    if table is None:
        return None

    head = table.find("thead")
    body = table.find("tbody")

    header_grid: dict[int, dict[int, str]] = {}
    if head is not None:
        for index, row in enumerate(head.find_all("tr")):
            _place_row(row.find_all(["th", "td"]), header_grid, index)

    if body is not None:
        body_tags = body.find_all("tr")
    else:
        head_rows = set(head.find_all("tr")) if head is not None else set()
        body_tags = [row for row in table.find_all("tr") if row not in head_rows]

    body_grid: dict[int, dict[int, str]] = {}
    for index, row in enumerate(body_tags):
        _place_row(row.find_all(["th", "td"]), body_grid, index)

    header_rows = _grid_to_rows(header_grid)
    body_rows = _grid_to_rows(body_grid)

    # A table written without <thead> puts its header in the first <tr>.
    if not header_rows and body_rows:
        header_rows, body_rows = [body_rows[0]], body_rows[1:]

    header_rows, embedded_caption = peel_embedded_caption(header_rows)
    if not header_rows and body_rows:
        header_rows, body_rows = [body_rows[0]], body_rows[1:]

    if not header_rows or not body_rows:
        return None

    width = max(
        max((len(row) for row in header_rows), default=0),
        max((len(row) for row in body_rows), default=0),
    )
    columns = flatten_headers(_pad(header_rows, width))
    data_rows = _coerce_body(_pad(body_rows, width))
    columns, data_rows = absorb_unit_header_row(columns, data_rows)
    if not data_rows:
        return None
    return FlattenedTable(columns=columns, rows=data_rows, embedded_caption=embedded_caption)


def _is_separator_line(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().split("|") if cell.strip()]
    return bool(cells) and all(_SEPARATOR_CELL_RE.match(cell) for cell in cells)


def find_markdown_tables(text: str) -> list[list[str]]:
    """Return every Markdown pipe table, each as its list of lines."""
    tables: list[list[str]] = []
    block: list[str] = []

    def flush() -> None:
        if len(block) >= 3 and any(_is_separator_line(line) for line in block):
            tables.append(block.copy())
        block.clear()

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if line.startswith("|") and "|" in line[1:]:
            block.append(line)
        else:
            flush()
    flush()
    return tables


def _split_markdown_row(line: str) -> list[str]:
    cells = line.split("|")
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return [strip_inline_formatting(cell) for cell in cells]


def parse_markdown_table(lines: list[str]) -> FlattenedTable | None:
    """Flatten one Markdown pipe table.

    Args:
        lines: The table's lines, including the ``---`` separator.

    Returns:
        A :class:`FlattenedTable`, or ``None`` when the block has no header or
        no data row.
    """
    separator = next((i for i, line in enumerate(lines) if _is_separator_line(line)), None)
    if not separator:
        return None

    header_rows = [_split_markdown_row(line) for line in lines[:separator]]
    body_lines = [line for line in lines[separator + 1 :] if line.strip()]
    if not body_lines:
        return None
    body_rows = [_split_markdown_row(line) for line in body_lines]

    header_rows, embedded_caption = peel_embedded_caption(header_rows)
    if not header_rows and body_rows:
        header_rows, body_rows = [body_rows[0]], body_rows[1:]
    if not header_rows or not body_rows:
        return None

    width = max(len(row) for row in header_rows + body_rows)
    columns = flatten_headers(_pad(header_rows, width))
    columns += [""] * (width - len(columns))
    data_rows = _coerce_body(_pad(body_rows, width))
    columns, data_rows = absorb_unit_header_row(columns, data_rows)
    if not data_rows:
        return None
    return FlattenedTable(columns=columns, rows=data_rows, embedded_caption=embedded_caption)


def flatten_cell_grid(raw_rows: list[list[object | None]]) -> FlattenedTable | None:
    """Flatten a native-PDF extractor grid with the same rules as HTML tables.

    Used by pdfplumber / Camelot / PyMuPDF so Phase 5 predictions share the
    GT header flatten, caption peel, unit-row merge and numeric coerce.
    These extractors do not emit ``<thead>``: leading ``Table N`` spanner
    rows are peeled, then the next row is the header and the rest the body.
    """
    if not raw_rows:
        return None
    cleaned: list[list[str]] = []
    for row in raw_rows:
        if row is None:
            continue
        cleaned.append(
            [strip_inline_formatting("" if cell is None else str(cell)) for cell in row]
        )
    if len(cleaned) < 2:
        return None

    # Look at the first two rows so a stacked Table-N + caption-body pair
    # is peeled the same way as an HTML ``<thead>``.
    look_ahead = cleaned[:2]
    leftover_header, embedded_caption = peel_embedded_caption(look_ahead)
    if embedded_caption:
        rest = leftover_header + cleaned[len(look_ahead) :]
    else:
        rest = cleaned

    if len(rest) < 2:
        return None
    header_rows, body_rows = [rest[0]], rest[1:]
    if not header_rows or not body_rows:
        return None

    width = max(len(row) for row in header_rows + body_rows)
    if width == 0:
        return None
    columns = flatten_headers(_pad(header_rows, width))
    columns += [""] * (width - len(columns))
    data_rows = _coerce_body(_pad(body_rows, width))
    columns, data_rows = absorb_unit_header_row(columns, data_rows)
    if not data_rows:
        return None
    return FlattenedTable(
        columns=columns, rows=data_rows, embedded_caption=embedded_caption
    )


def extract_tables(page_text: str) -> list[tuple[str, FlattenedTable]]:
    """Flatten every table found in one page transcription.

    HTML is preferred; Markdown tables are only considered when the page
    contains no HTML table, mirroring how the OCR engines choose one syntax per
    page.

    Args:
        page_text: The OCR transcription of a single page.

    Returns:
        One ``(raw_fragment, flattened)`` pair per successfully flattened
        table, in reading order.
    """
    results: list[tuple[str, FlattenedTable]] = []

    html_tables = find_html_tables(page_text)
    for fragment in html_tables:
        parsed = parse_html_table(fragment)
        if parsed is not None:
            results.append((fragment, parsed))

    if not html_tables:
        for lines in find_markdown_tables(page_text):
            parsed = parse_markdown_table(lines)
            if parsed is not None:
                results.append(("\n".join(lines), parsed))

    return results
