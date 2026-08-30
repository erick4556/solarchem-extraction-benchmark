"""Filesystem layout resolution.

The benchmark is developed locally but executed on a GPU server where the code
and the data live in sibling directories::

    <server_root>/code/    # this repository
    <server_root>/data/    # documents, caches and ground truth

Nothing in the codebase may hardcode an absolute path. Every location is
derived from a data root that is resolved, in order of precedence, from an
explicit CLI argument, the ``SOLARCHEM_DATA_ROOT`` environment variable, or a
small set of layout conventions.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV_VAR = "SOLARCHEM_DATA_ROOT"

_DOCUMENTS_DIR_NAME = "documents"
_GROUND_TRUTH_DIR_NAME = "ground_truth"
_PREDICTIONS_DIR_NAME = "predictions"
_OCR_CACHE_DIR_NAME = "intermediate/ocr_cache"

WORKING_SILVER_FILENAME = "ground_truth_lighton_ocr_302.json"


def repository_root() -> Path:
    """Return the repository root, i.e. the parent of ``src/``."""
    return Path(__file__).resolve().parents[2]


def resolve_data_root(explicit: Path | str | None = None) -> Path:
    """Resolve the data root directory.

    Args:
        explicit: Value supplied on the command line, if any.

    Returns:
        The resolved data root. The directory is not required to exist yet;
        callers that write into it are responsible for creating it.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    from_env = os.environ.get(DATA_ROOT_ENV_VAR)
    if from_env:
        return Path(from_env).expanduser().resolve()

    repo = repository_root()
    for candidate in (repo / "data", repo.parent / "data"):
        if candidate.is_dir():
            return candidate.resolve()
    return (repo / "data").resolve()


def resolve_documents_dir(
    explicit: Path | str | None = None,
    *,
    data_root: Path | None = None,
) -> Path:
    """Resolve the directory holding the source PDF corpus.

    Args:
        explicit: Value supplied on the command line, if any.
        data_root: Pre-resolved data root, used when ``explicit`` is ``None``.

    Returns:
        The first existing conventional location, or the default under the data
        root when none of them exist.

    Raises:
        NotADirectoryError: If ``explicit`` is given but is not a directory.
    """
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Documents directory not found: {path}")
        return path

    root = data_root if data_root is not None else resolve_data_root()
    for candidate in (root / _DOCUMENTS_DIR_NAME, repository_root() / _DOCUMENTS_DIR_NAME):
        if candidate.is_dir():
            return candidate.resolve()
    return (root / _DOCUMENTS_DIR_NAME).resolve()


def default_ground_truth_dir(data_root: Path) -> Path:
    """Return the default output directory for ground-truth documents."""
    return data_root / _GROUND_TRUTH_DIR_NAME


def merged_ground_truth_filename(engine_id: str) -> str:
    """Return the default merged silver filename for an OCR engine.

    Examples: ``ground_truth_lighton_ocr.json``, ``ground_truth_unlimited_ocr.json``.
    """
    slug = "".join(char if char.isalnum() or char in "-_" else "_" for char in engine_id).strip("_")
    return f"ground_truth_{slug or 'engine'}.json"


def default_merged_ground_truth_path(data_root: Path, engine_id: str) -> Path:
    """Return ``data/ground_truth/ground_truth_<engine_id>.json``."""
    return default_ground_truth_dir(data_root) / merged_ground_truth_filename(engine_id)


def default_working_silver_path(data_root: Path) -> Path:
    """Return the frozen LightOn silver snapshot used as the Phase 5 working set."""
    return default_ground_truth_dir(data_root) / WORKING_SILVER_FILENAME


def default_predictions_dir(data_root: Path) -> Path:
    """Return ``data/predictions/`` for tool outputs (kept separate from GT)."""
    return data_root / _PREDICTIONS_DIR_NAME


def default_prediction_path(data_root: Path, tool_id: str) -> Path:
    """Return ``data/predictions/<tool_id>.json``."""
    slug = "".join(char if char.isalnum() or char in "-_" else "_" for char in tool_id).strip(
        "_"
    )
    return default_predictions_dir(data_root) / f"{slug or 'tool'}.json"


def default_ocr_cache_dir(data_root: Path, engine_id: str | None = None) -> Path:
    """Return the default directory for cached raw OCR page transcriptions.

    With ``engine_id``, returns ``data/intermediate/ocr_cache/<engine_id>/`` so
    LightOn and Unlimited caches stay in separate folders. Without it, returns
    the shared ``ocr_cache/`` root (useful when the caller appends the engine).
    """
    root = data_root / _OCR_CACHE_DIR_NAME
    if engine_id:
        slug = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in engine_id
        ).strip("_")
        return root / (slug or "engine")
    return root
