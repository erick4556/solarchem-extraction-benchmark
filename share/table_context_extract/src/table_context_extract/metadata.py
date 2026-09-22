"""Bibliographic metadata read from the PDF itself.

The article title is taken from the PDF's ``/Title`` entry rather than from the
OCR transcription: on a first page the title is typographically indistinguishable
from the running head, the journal name and the author list, so recovering it
from text alone is guesswork, whereas the metadata entry is exact when present.

Publishers do, however, leave typesetting artefacts in that entry -- job names
such as ``ja304075b 1..6``, source filenames, ``No Job Name``. Those are
rejected: for a ground truth an empty title is better than a wrong one.
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from pathlib import Path

import pypdfium2 as pdfium

logger = logging.getLogger(__name__)

_FILENAME_RE = re.compile(
    r"\.(?:pdf|ps|eps|doc|docx|rtf|tex|dvi|fm|indd|qxd|p65|xml|txt)$",
    re.IGNORECASE,
)
#: Word saves the source document name; the real title sometimes follows it.
_WORD_PREFIX_RE = re.compile(r"^microsoft\s+word\s*[-:]\s*", re.IGNORECASE)
#: ``MJAS Vol 21 No 1 (2017)``: the issue was stored instead of the article.
_JOURNAL_ISSUE_RE = re.compile(r"\bvol\.?\s*\d+\s+no\.?\s*\d+", re.IGNORECASE)
#: ``ja304075b 1..6``, ``CC 10 1147-1155..b317004g``: page range of a print job.
_PAGE_RANGE = ".."
_PLACEHOLDER_TITLES = frozenset({"no job name", "untitled", "manuscript", "doi:", "title"})
_MIN_WORDS = 3


def clean_pdf_title(raw: str) -> str:
    """Turn a raw ``/Title`` metadata entry into an article title.

    Args:
        raw: The value stored in the PDF, possibly with HTML entities.

    Returns:
        The title, or an empty string when the entry is a typesetting
        artefact rather than a real title.
    """
    title = unicodedata.normalize("NFKC", html.unescape(raw))
    title = re.sub(r"\s+", " ", title).strip().strip("*")
    title = _WORD_PREFIX_RE.sub("", title).strip()

    if not title or title.lower() in _PLACEHOLDER_TITLES:
        return ""
    if "\\" in title or _PAGE_RANGE in title:
        return ""
    if _FILENAME_RE.search(title) or _JOURNAL_ISSUE_RE.search(title):
        return ""
    if len(title.split()) < _MIN_WORDS:
        return ""
    return title


def pdf_title(pdf_path: Path) -> str:
    """Read the article title from a PDF's metadata.

    Args:
        pdf_path: Path to the source PDF.

    Returns:
        The cleaned title, empty when the PDF carries none worth trusting.
    """
    try:
        raw = pdfium.PdfDocument(pdf_path).get_metadata_dict().get("Title") or ""
    except Exception:  # noqa: BLE001 - a damaged PDF must not abort a corpus run
        logger.warning("Could not read metadata from %s", pdf_path.name)
        return ""
    return clean_pdf_title(raw)
