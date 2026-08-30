"""Layout / end-to-end table extractors (Phase 6): Docling, TATR, Unstructured.

Third-party packages are imported inside the extract methods so listing tools
or running tests does not require them. Models load lazily on first PDF.
"""

from __future__ import annotations

import logging
from pathlib import Path

from solarchem_benchmark.extractors.base import ExtractedGrid, NativeTableExtractor
from solarchem_benchmark.gt.tables import parse_html_table

logger = logging.getLogger(__name__)

_NUMPY1_PANDAS_EXTRAS = ("numexpr", "bottleneck")


def _block_stale_numpy1_accelerators() -> None:
    """Stop pandas from importing conda numexpr/bottleneck built for NumPy 1.

    Those extras fail under NumPy 2 with ``AttributeError: _ARRAY_API not found``.
    Pandas only treats ``ImportError`` as optional, so the AttributeError aborts
    Docling. Putting a ``None`` placeholder in ``sys.modules`` makes the import
    fail as ImportError; pandas skips the extra and Docling can load.
    """
    import sys

    try:
        import numpy as np
    except ImportError:
        return
    try:
        major = int(str(np.__version__).split(".", 1)[0])
    except (TypeError, ValueError):
        return
    if major < 2:
        return
    for name in _NUMPY1_PANDAS_EXTRAS:
        sys.modules[name] = None


def _numpy2_warning(name: str):
    try:
        import numpy as np
    except ImportError:
        return Warning
    warning = getattr(np, name, None)
    if warning is not None:
        return warning
    try:
        from numpy import exceptions as numpy_exceptions
    except ImportError:
        return Warning
    return getattr(numpy_exceptions, name, Warning)


def _shim_numpy2_aliases() -> None:
    """Put back NumPy 1 names that SciPy / sklearn / Docling still import.

    ``from numpy.core.numeric import ComplexWarning`` raises ImportError on
    NumPy 2 because the warning moved to ``numpy.exceptions``. NumPy 2 also
    renamed ``numpy.core`` to ``numpy._core``; patch both modules.
    """
    aliases = {
        "ComplexWarning": _numpy2_warning("ComplexWarning"),
        "VisibleDeprecationWarning": _numpy2_warning("VisibleDeprecationWarning"),
    }
    for module_name in ("numpy.core.numeric", "numpy._core.numeric"):
        try:
            module = __import__(module_name, fromlist=["numeric"])
        except ImportError:
            continue
        for name, warning in aliases.items():
            if not hasattr(module, name):
                setattr(module, name, warning)


def _purge_modules(*prefixes: str) -> None:
    import sys

    names = [
        key
        for key in list(sys.modules)
        if any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes)
    ]
    for name in names:
        del sys.modules[name]


def _install_sklearn_stub() -> None:
    """Minimal sklearn so transformers generation can import without conda ABI.

    Docling layout detection only needs ``AutoModelForObjectDetection``. Newer
    transformers still do ``from sklearn.metrics import roc_curve`` when sklearn
    is installed; conda sklearn is built for NumPy 1 and crashes under NumPy 2.
    """
    import sys
    import types

    sklearn = types.ModuleType("sklearn")
    sklearn.__version__ = "0.0.0-stub"
    metrics = types.ModuleType("sklearn.metrics")

    def roc_curve(*args, **kwargs):
        raise RuntimeError("sklearn is stubbed; roc_curve is unavailable")

    metrics.roc_curve = roc_curve
    sklearn.metrics = metrics
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.metrics"] = metrics
    logger.warning(
        "Stubbed sklearn (conda build is NumPy 1, runtime is NumPy 2); "
        "Docling layout does not need it. Optional fix: "
        "pip install --user --upgrade scikit-learn"
    )


def _ensure_sklearn_safe_for_transformers() -> None:
    try:
        from sklearn.metrics import roc_curve  # noqa: F401
    except ImportError:
        return
    except Exception:  # noqa: BLE001 - ABI errors are ValueError, not ImportError
        _purge_modules("sklearn")
        _install_sklearn_stub()


def _is_numpy_abi_error(error: BaseException) -> bool:
    message = str(error).lower()
    return (
        "numpy.dtype size changed" in message
        or "binary incompatibility" in message
        or "_array_api not found" in message
    )


_ABI_PIP_PACKAGES = {
    "cv2": "opencv-python-headless",
    "pyarrow": "pyarrow",
    "onnxruntime": "onnxruntime",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "pandas": "pandas numexpr bottleneck",
    "numexpr": "numexpr",
    "bottleneck": "bottleneck",
    "h5py": "h5py",
    "spacy": "h5py",
    "thinc": "h5py",
}


def _pip_packages_for(module_name: str) -> str:
    top = module_name.split(".", 1)[0]
    return _ABI_PIP_PACKAGES.get(module_name) or _ABI_PIP_PACKAGES.get(
        top,
        "opencv-python-headless pyarrow numexpr bottleneck",
    )


def _numpy_abi_import_error(
    error: BaseException,
    module_name: str = "unknown",
) -> ImportError:
    packages = _pip_packages_for(module_name)
    return ImportError(
        f"{module_name} is compiled for NumPy 1 while this runtime is NumPy 2 "
        f"({error}). Do not reinstall torch or downgrade numpy. Fix with: "
        f"python -m pip install --upgrade --force-reinstall --no-cache-dir "
        f"--target <numpy-site-packages> {packages}"
    )


def _require_numpy2_extension(*module_names: str) -> None:
    """Fail fast when a C extension was built against NumPy 1.

    ``__import__('a.b')`` only loads ``a``; use ``import_module`` so
    ``scipy.ndimage`` / ``sklearn.metrics`` are actually imported.
    """
    import importlib

    for name in module_names:
        try:
            importlib.import_module(name)
        except ImportError:
            continue
        except Exception as error:  # noqa: BLE001 - C-extension ABI is ValueError
            if _is_numpy_abi_error(error):
                logger.exception("NumPy ABI failure while importing %s", name)
                raise _numpy_abi_import_error(error, name) from error
            raise


def _prepare_layout_runtime(*, stub_sklearn: bool = True) -> None:
    """Make the JupyterHub NumPy 2 + conda mix safe for Docling / TATR / Unstructured."""
    _block_stale_numpy1_accelerators()
    _shim_numpy2_aliases()
    if stub_sklearn:
        _ensure_sklearn_safe_for_transformers()


def _page_limit(n_pages: int, max_pages: int | None) -> int:
    if max_pages is None or max_pages <= 0:
        return n_pages
    return min(n_pages, max_pages)


def _as_cell(value: object) -> object | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except Exception:  # noqa: BLE001 - pandas NA compares raise
        return str(value)
    return value


def _dataframe_to_grid(frame) -> list[list[object | None]]:
    """Turn a pandas DataFrame into a header + body grid."""
    if frame is None or getattr(frame, "empty", True):
        return []
    columns = frame.columns
    if getattr(columns, "nlevels", 1) > 1:
        header_rows = [
            ["" if value is None else str(value) for value in columns.get_level_values(level)]
            for level in range(columns.nlevels)
        ]
        from solarchem_benchmark.gt.tables import flatten_headers

        headers = flatten_headers(header_rows)
    else:
        headers = ["" if value is None else str(value) for value in columns.tolist()]
    body: list[list[object | None]] = []
    for row in frame.itertuples(index=False, name=None):
        body.append([_as_cell(value) for value in row])
    return [headers] + body


def _html_to_grid(html: str) -> list[list[object | None]]:
    """Flatten an HTML table, then re-emit a grid for the shared assemble step."""
    parsed = parse_html_table(html)
    if parsed is None:
        return []
    body = [[cell for cell in row] for row in parsed.rows]
    grid: list[list[object | None]] = [list(parsed.columns)] + body
    if parsed.embedded_caption:
        width = max(len(parsed.columns), 2)
        caption_row: list[object | None] = [parsed.embedded_caption] * width
        grid = [caption_row] + grid
    return grid


class DoclingExtractor(NativeTableExtractor):
    """IBM Docling PDF pipeline with table structure (TableFormer). OCR off."""

    tool_id = "docling"

    def __init__(self) -> None:
        self._converter = None

    def _converter_instance(self):
        if self._converter is not None:
            return self._converter
        _prepare_layout_runtime()
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as error:
            raise ImportError(
                "docling is not installed. "
                'Install with: pip install -e ".[docling]"'
            ) from error
        except AttributeError as error:
            raise ImportError(
                "Docling failed to import pandas (NumPy 2 vs conda numexpr/"
                "bottleneck). Update the extras without touching torch: "
                "pip install --upgrade numexpr bottleneck"
            ) from error

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        return self._converter

    def extract_grids(
        self,
        pdf_path: Path,
        *,
        max_pages: int | None = None,
    ) -> list[ExtractedGrid]:
        converter = self._converter_instance()
        _prepare_layout_runtime()
        kwargs: dict = {}
        if max_pages is not None and max_pages > 0:
            kwargs["max_num_pages"] = max_pages

        def _convert():
            try:
                return converter.convert(str(pdf_path), **kwargs)
            except TypeError:
                return converter.convert(str(pdf_path))

        try:
            result = _convert()
        except ImportError as error:
            if "ComplexWarning" not in str(error):
                raise
            _shim_numpy2_aliases()
            try:
                result = _convert()
            except ImportError as retry_error:
                raise ImportError(
                    "Docling failed because NumPy 2 removed "
                    "numpy.core.numeric.ComplexWarning (used by sklearn/scipy). "
                    "Upgrade those packages without touching torch: "
                    "pip install --upgrade scipy scikit-learn"
                ) from retry_error
        except ValueError as error:
            message = str(error)
            if "numpy.dtype size changed" not in message:
                raise
            _ensure_sklearn_safe_for_transformers()
            _purge_modules("transformers.generation", "transformers.models.auto")
            try:
                result = _convert()
            except ValueError as retry_error:
                raise ImportError(
                    "Docling cannot load transformers: conda scikit-learn is built "
                    "for NumPy 1 (numpy.dtype size changed). Layout detection does "
                    "not need sklearn. Re-run this CLI after syncing the stub, or: "
                    "pip install --user --upgrade scikit-learn"
                ) from retry_error

        document = getattr(result, "document", result)
        tables = getattr(document, "tables", None) or []
        grids: list[ExtractedGrid] = []
        last_page = max_pages if max_pages and max_pages > 0 else None
        for table in tables:
            page = _docling_page(table)
            if last_page is not None and page > last_page:
                continue
            rows = _docling_table_rows(table)
            if rows:
                grids.append(ExtractedGrid(page=page, rows=rows))
        return grids


def _docling_page(table) -> int:
    prov = getattr(table, "prov", None) or []
    if not prov:
        return 1
    page_no = getattr(prov[0], "page_no", 1)
    try:
        number = int(page_no)
    except (TypeError, ValueError):
        return 1
    return number if number >= 1 else number + 1


def _docling_table_rows(table) -> list[list[object | None]]:
    data = getattr(table, "data", None)
    grid = getattr(data, "grid", None) if data is not None else None
    if grid:
        rows: list[list[object | None]] = []
        for row in grid:
            rows.append(
                [
                    None if cell is None else getattr(cell, "text", cell)
                    for cell in row
                ]
            )
        return rows
    if hasattr(table, "export_to_dataframe"):
        try:
            return _dataframe_to_grid(table.export_to_dataframe())
        except Exception as error:  # noqa: BLE001
            logger.debug("Docling export_to_dataframe failed: %s", error)
    return []


class TATRExtractor(NativeTableExtractor):
    """Table Transformer via gmft (detection + structure + PDF cell text)."""

    tool_id = "tatr"

    def __init__(self) -> None:
        self._detector = None
        self._formatter = None

    def _models(self):
        if self._detector is not None and self._formatter is not None:
            return self._detector, self._formatter
        _prepare_layout_runtime()
        try:
            from gmft.auto import AutoTableDetector, AutoTableFormatter
        except ImportError as error:
            raise ImportError(
                "gmft is not installed (Table Transformer pipeline). "
                'Install with: pip install -e ".[tatr]" --no-deps && pip install gmft'
            ) from error
        self._detector = AutoTableDetector()
        self._formatter = AutoTableFormatter()
        return self._detector, self._formatter

    def extract_grids(
        self,
        pdf_path: Path,
        *,
        max_pages: int | None = None,
    ) -> list[ExtractedGrid]:
        detector, formatter = self._models()
        try:
            from gmft.pdf_bindings import PyPDFium2Document
        except ImportError:
            try:
                from gmft.pdf_bindings.pdfium import PyPDFium2Document
            except ImportError as error:
                raise ImportError(
                    "gmft.pdf_bindings.PyPDFium2Document is missing. "
                    'Install with: pip install -e ".[tatr]"'
                ) from error

        document = PyPDFium2Document(str(pdf_path))
        grids: list[ExtractedGrid] = []
        try:
            pages = list(document)
            last = _page_limit(len(pages), max_pages)
            for index in range(last):
                page = pages[index]
                try:
                    cropped = detector.extract(page)
                except Exception as error:  # noqa: BLE001
                    logger.debug(
                        "%s page %d: TATR detect failed: %s",
                        pdf_path.name,
                        index + 1,
                        error,
                    )
                    continue
                for table in cropped or []:
                    try:
                        formatted = formatter.extract(table)
                        frame = formatted.df() if hasattr(formatted, "df") else None
                    except Exception as error:  # noqa: BLE001
                        logger.debug(
                            "%s page %d: TATR structure failed: %s",
                            pdf_path.name,
                            index + 1,
                            error,
                        )
                        continue
                    rows = _dataframe_to_grid(frame)
                    if rows:
                        grids.append(ExtractedGrid(page=index + 1, rows=rows))
        finally:
            close = getattr(document, "close", None)
            if close:
                close()
        return grids


class UnstructuredExtractor(NativeTableExtractor):
    """Unstructured hi_res PDF partition with inferred HTML table structure."""

    tool_id = "unstructured"

    def extract_grids(
        self,
        pdf_path: Path,
        *,
        max_pages: int | None = None,
    ) -> list[ExtractedGrid]:
        _prepare_layout_runtime(stub_sklearn=False)
        _require_numpy2_extension(
            "scipy.ndimage",
            "pyarrow",
            "cv2",
            "onnxruntime",
            "h5py",
            "sklearn.metrics",
            "unstructured.partition.pdf",
        )
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as error:
            if _is_numpy_abi_error(error):
                raise _numpy_abi_import_error(
                    error, "unstructured.partition.pdf"
                ) from error
            raise ImportError(
                "unstructured is not installed. "
                'Install with: pip install -e ".[unstructured]"'
            ) from error
        except Exception as error:  # noqa: BLE001 - C-extension ABI is ValueError
            if _is_numpy_abi_error(error):
                raise _numpy_abi_import_error(
                    error, "unstructured.partition.pdf"
                ) from error
            raise

        kwargs: dict = {
            "filename": str(pdf_path),
            "strategy": "hi_res",
            "infer_table_structure": True,
        }
        if max_pages is not None and max_pages > 0:
            kwargs["last_page"] = max_pages
        try:
            elements = partition_pdf(**kwargs)
        except TypeError:
            kwargs.pop("last_page", None)
            try:
                elements = partition_pdf(**kwargs)
            except Exception as error:  # noqa: BLE001
                if _is_numpy_abi_error(error):
                    raise _numpy_abi_import_error(error, "unstructured.partition.pdf") from error
                raise
        except Exception as error:  # noqa: BLE001 - C-extension ABI is ValueError
            if _is_numpy_abi_error(error):
                raise _numpy_abi_import_error(error, "unstructured.partition.pdf") from error
            raise

        grids: list[ExtractedGrid] = []
        last_page = max_pages if max_pages and max_pages > 0 else None
        for element in elements:
            if not _is_table_element(element):
                continue
            metadata = getattr(element, "metadata", None)
            page = getattr(metadata, "page_number", None) or 1
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = 1
            if last_page is not None and page > last_page:
                continue
            html = getattr(metadata, "text_as_html", None) or ""
            rows = _html_to_grid(html) if html else []
            if rows:
                grids.append(ExtractedGrid(page=page, rows=rows))
        return grids


def _is_table_element(element) -> bool:
    category = getattr(element, "category", "") or ""
    name = type(element).__name__
    return category == "Table" or name == "Table"
