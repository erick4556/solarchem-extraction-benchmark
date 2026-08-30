"""Normalisation of scientific text found in SolarChem documents.

Purpose
-------
Make the *same* chemical / physical string compare equal no matter how the OCR
or PDF layer wrote it. Rules are **general notation transforms**, applied
uniformly to every cell, header, caption, section title and mention. They are
not paper-specific and are not tuned to any Gold file.

What belongs here (general)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Unicode scientific characters → ASCII (sub/superscripts, minus, micro, thin
   spaces, soft hyphens, angstrom sign).
2. LaTeX math delimiters → the same ASCII form as the Unicode path:
   ``$...$``, ``\\(...\\)`` and ``\\[...\\]`` (engine-agnostic; no OCR preference).
3. Script attachment: a subscript/superscript binds to its base
   (``CO $_2$`` / ``CO _2`` / ``CO₂`` → ``CO2``).
4. Micro as a prefix attaches to the following token (``µ mol`` / ``\\mu mol``
   → ``umol``), without a closed list of unit names.
5. Spaces around the caret operator are removed (``m ^2`` / ``g ^-1`` →
   ``m^2`` / ``g^-1``), matching how powers are written in plain ASCII.
6. Whitespace collapsed to single spaces.

What does **not** belong here
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Rewriting section titles, captions or prose to match Gold wording.
* Paper-specific aliases or lexicon lookups.
* Evaluation softness (token-F1) — that lives in ``eval/``.

``normalize_unit`` adds only unit-multiplier cleanup (``·`` → space).
``normalize_header_label`` adds only publisher footnote marks on headers.
"""

from __future__ import annotations

import re

_SUBSCRIPTS = str.maketrans(
    "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎",
    "0123456789+-=()",
)

_SUPERSCRIPT_CHARS = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "ⁿ": "n",
}

_SUPERSCRIPT_RUN_RE = re.compile(f"[{''.join(_SUPERSCRIPT_CHARS)}]+")

# Lookalike / layout characters → a single ASCII standing. No domain lexicon.
# Micro / mu are handled by ``_MICRO_PREFIX_RE`` (they also eat a following space).
_CHAR_REPLACEMENTS = {
    "\u2212": "-",  # MINUS SIGN
    "\u2013": "-",  # EN DASH
    "\u2014": "-",  # EM DASH
    "\u00ad": "",  # SOFT HYPHEN (PDF line-break artefact)
    "\u00a0": " ",  # NO-BREAK SPACE
    "\u2009": " ",  # THIN SPACE
    "\u202f": " ",  # NARROW NO-BREAK SPACE
    "\u00c5": "A",  # LATIN CAPITAL A WITH RING (angstrom, as ASCII "A")
    "\u212b": "A",  # ANGSTROM SIGN
    "\u00e5": "a",
}

_UNIT_MULTIPLIERS = ("\u00b7", "\u22c5", "\u2022", "\u00d7", "*")

# LaTeX macros inside math spans. Unlisted macros are left alone.
_MATH_MACROS = {
    "mu": "u",
    "times": "\u00d7",
    "cdot": "\u00b7",
    "pm": "\u00b1",
    "approx": "\u2248",
    "sim": "~",
    "to": "\u2192",
    "rightarrow": "\u2192",
    "leftrightarrow": "\u2194",
    "circ": "\u00b0",
    "degree": "\u00b0",
    "alpha": "\u03b1",
    "beta": "\u03b2",
    "gamma": "\u03b3",
    "delta": "\u03b4",
    "eta": "\u03b7",
    "theta": "\u03b8",
    "lambda": "\u03bb",
    "nu": "\u03bd",
    "%": "%",
}
_MACRO_RE = re.compile(r"\\([a-zA-Z]+|%)")
_MATH_SUBSCRIPT_RE = re.compile(r"_\{([^}]*)\}|_(\S)")
_MATH_SUPERSCRIPT_RE = re.compile(r"\^\{([^}]*)\}|\^(\S)")
# Dollar math: only rewrite spans that look like math (see ``_MATH_BODY_RE``),
# so currency like ``$5`` is left alone.
_MATH_DOLLAR_RE = re.compile(r"([ \t]*)\$([^$\n]+)\$([ \t]*)(?=(\S{0,2}))")
# ``\( ... \)`` / ``\[ ... \]`` are always math delimiters (never currency).
_MATH_PAREN_RE = re.compile(r"([ \t]*)\\\((.+?)\\\)([ \t]*)(?=(\S{0,2}))")
_MATH_BRACKET_RE = re.compile(r"([ \t]*)\\\[(.+?)\\\]([ \t]*)(?=(\S{0,2}))", re.DOTALL)
_MATH_BODY_RE = re.compile(r"[_^\\]")
_FORMULA_CONTINUES_RE = re.compile(r"[0-9]|[A-Z](?![a-z])")

# Plain-text leftovers after math expansion / OCR: scripts written with ``_``.
_DIGIT_SUBSCRIPT_RE = re.compile(r"([A-Za-z])\s*_\s*(\d+)")
_LETTER_SUBSCRIPT_RE = re.compile(r"([A-Za-z])_([A-Za-z])\b")
# OCR / HTML often inserts spaces around ``^`` (``m ^2``, ``g ^-1``).
_CARET_SPACES_RE = re.compile(r"\s*\^\s*")

# Micro / mu as a *prefix*: the marker itself plus any following space become
# ``u``, so ``µ mol`` / ``\mu mol`` / ``$\mu$ mol`` collapse to ``umol``.
# Only real micro/mu markers are matched — never a bare Latin ``u``.
_MICRO_PREFIX_RE = re.compile(r"(?:\\mu|[\u00b5\u03bc])\s*")

# Publisher footnote marks on headers only (see ``normalize_header_label``).
_HEADER_FOOTNOTE_RE = re.compile(
    r"(?:(?<=[A-Za-z0-9)])\s*\^[a-zA-Z]\b|\s*[\u2020\u2021\u00a7\u00b6\*]+)\s*$"
)

_MISSING_TOKENS = frozenset({"", "-", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015"})

MISSING_MARKER = "-"
"""Value written into the flattened grid for an empty or dash-only cell."""


def _expand_superscripts(text: str) -> str:
    """Rewrite runs of Unicode superscripts as ``^`` followed by ASCII."""

    def replace(match: re.Match[str]) -> str:
        digits = "".join(_SUPERSCRIPT_CHARS[char] for char in match.group(0))
        return f"^{digits}"

    return _SUPERSCRIPT_RUN_RE.sub(replace, text)


def _render_math(body: str) -> str:
    """Rewrite the inside of a math span as plain scientific text."""
    rendered = _MACRO_RE.sub(lambda m: _MATH_MACROS.get(m.group(1), m.group(0)), body)
    rendered = _MATH_SUBSCRIPT_RE.sub(lambda m: m.group(1) or m.group(2) or "", rendered)
    rendered = _MATH_SUPERSCRIPT_RE.sub(lambda m: f"^{m.group(1) or m.group(2) or ''}", rendered)
    return rendered.replace("{", "").replace("}", "").strip()


def _replace_math_span(match: re.Match[str], *, always_unwrap: bool) -> str:
    """Shared replacement for ``$...$``, ``\\(...\\)`` and ``\\[...\\]`` spans."""
    before, body, after, following = match.groups()
    has_math = bool(_MATH_BODY_RE.search(body))
    if not has_math and not always_unwrap:
        return match.group(0)

    rendered = _render_math(body) if has_math else body.strip()
    if has_math and body.strip().startswith(("_", "^")):
        if before and after and _FORMULA_CONTINUES_RE.match(following or ""):
            return rendered
        return f"{rendered}{after}"
    return f"{before}{rendered}{after}"


def expand_latex_math_delimiters(text: str) -> str:
    """Expand LaTeX math delimiters used by OCR engines into plain text.

    Handles ``$...$`` (only when the body looks like math), plus ``\\(...\\)``
    and ``\\[...\\]`` (always unwrapped). Applied uniformly for every engine.
    """
    if not text:
        return ""
    result = _MATH_PAREN_RE.sub(
        lambda m: _replace_math_span(m, always_unwrap=True), text
    )
    result = _MATH_BRACKET_RE.sub(
        lambda m: _replace_math_span(m, always_unwrap=True), result
    )
    result = _MATH_DOLLAR_RE.sub(
        lambda m: _replace_math_span(m, always_unwrap=False), result
    )
    return result


def _expand_math(text: str) -> str:
    """Expand LaTeX math spans, reattaching scripts to the base they modify."""
    return expand_latex_math_delimiters(text)


def normalize_scientific_text(text: str) -> str:
    """Canonicalise a cell, header, caption or paragraph.

    Args:
        text: Raw text as transcribed from the document.

    Returns:
        The canonical form under the general notation rules documented in the
        module docstring.
    """
    if not text:
        return ""
    result = expand_latex_math_delimiters(text)
    # After math expansion, ``$\mu$`` is already ``u``; bare ``\mu`` / ``µ`` remain.
    result = _MICRO_PREFIX_RE.sub("u", result)
    result = result.translate(_SUBSCRIPTS)
    result = _expand_superscripts(result)
    for source, target in _CHAR_REPLACEMENTS.items():
        result = result.replace(source, target)
    result = _DIGIT_SUBSCRIPT_RE.sub(r"\1\2", result)
    result = _LETTER_SUBSCRIPT_RE.sub(r"\1\2", result)
    result = _CARET_SPACES_RE.sub("^", result)
    return re.sub(r"\s+", " ", result).strip()


def normalize_header_label(text: str) -> str:
    """Canonicalise a column header, dropping publisher footnote marks."""
    result = normalize_scientific_text(text)
    while True:
        stripped = _HEADER_FOOTNOTE_RE.sub("", result).strip()
        if stripped == result:
            return result
        result = stripped


def normalize_unit(text: str) -> str:
    """Canonicalise a unit expression (multipliers → spaces)."""
    result = normalize_scientific_text(text)
    for multiplier in _UNIT_MULTIPLIERS:
        result = result.replace(multiplier, " ")
    result = result.strip(" ,;:")
    return re.sub(r"\s+", " ", result).strip()


def is_missing(text: str) -> bool:
    """Report whether a cell is empty or contains only a dash placeholder."""
    return normalize_scientific_text(text) in _MISSING_TOKENS


def coerce_cell(text: str) -> str | int | float:
    """Convert a cell to a number when it is unambiguously numeric."""
    normalized = normalize_scientific_text(text)
    if normalized in _MISSING_TOKENS:
        return MISSING_MARKER

    candidate = normalized
    if re.search(r",\d{3}(\D|$)", candidate):
        candidate = candidate.replace(",", "")
    elif candidate.count(",") == 1 and "." not in candidate:
        candidate = candidate.replace(",", ".")

    try:
        value = float(candidate)
    except ValueError:
        return normalized
    if value.is_integer() and "e" not in candidate.lower() and "." not in candidate:
        return int(value)
    return value
