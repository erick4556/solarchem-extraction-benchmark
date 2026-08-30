"""Tests for native-PDF extractors and schema assembly."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from solarchem_benchmark.extractors.assemble import (
    build_prediction_document,
    extract_document,
)
from solarchem_benchmark.extractors.base import (
    ExtractedGrid,
    ExtractedMention,
    NativeTableExtractor,
    available_tools,
    build_extractor,
    is_environment_error,
)
from solarchem_benchmark.extractors.layout import (
    DoclingExtractor,
    TATRExtractor,
    UnstructuredExtractor,
    _block_stale_numpy1_accelerators,
    _ensure_sklearn_safe_for_transformers,
    _numpy_abi_import_error,
    _require_numpy2_extension,
    _shim_numpy2_aliases,
)
from solarchem_benchmark.extractors.native import (
    CamelotExtractor,
    PdfPlumberExtractor,
    PyMuPDFExtractor,
)
from solarchem_benchmark.extractors.grobid import parse_grobid_tei
from solarchem_benchmark.extractors.ollama import (
    OllamaExtractor,
    attach_mentions,
    ollama_message_text,
    parse_page_tables,
)


class FakeExtractor(NativeTableExtractor):
    tool_id = "fake"

    def __init__(self, grids: list[ExtractedGrid]) -> None:
        self.grids = grids

    def extract_grids(self, pdf_path: Path, *, max_pages: int | None = None) -> list[ExtractedGrid]:
        return self.grids


def test_available_tools_include_native_and_layout_extractors() -> None:
    assert available_tools() == [
        "pdfplumber",
        "camelot_lattice",
        "camelot_stream",
        "pymupdf",
        "docling",
        "tatr",
        "unstructured",
        "grobid",
        "ollama_qwen3_vl",
        "ollama_gemma4",
        "ollama_mistral_small",
    ]


def test_build_extractor_rejects_unknown_tool() -> None:
    with pytest.raises(ValueError, match="Unknown extractor"):
        build_extractor("not_a_real_tool")


def test_build_extractor_does_not_import_third_party() -> None:
    extractor = build_extractor("pdfplumber")
    assert extractor.tool_id == "pdfplumber"
    extractor = build_extractor("camelot_stream")
    assert extractor.tool_id == "camelot_stream"
    extractor = build_extractor("docling")
    assert extractor.tool_id == "docling"
    extractor = build_extractor("tatr")
    assert extractor.tool_id == "tatr"
    extractor = build_extractor("unstructured")
    assert extractor.tool_id == "unstructured"
    extractor = build_extractor("grobid")
    assert extractor.tool_id == "grobid"
    assert extractor.emits_context is True
    extractor = build_extractor("ollama_qwen3_vl")
    assert extractor.tool_id == "ollama_qwen3_vl"
    assert extractor.emits_context is True


def test_prediction_document_is_structure_only() -> None:
    document = build_prediction_document(
        [
            ExtractedGrid(
                page=3,
                rows=[
                    ["Table 1 Rates", "Table 1 Rates"],
                    ["Catalyst", "CH4"],
                    ["TiO2", "12.5"],
                ],
            )
        ],
        document_id="doc",
        source_pdf="paper.pdf",
        title="A title",
    )
    assert document.num_tables == 1
    table = document.tables[0]
    assert table.page == 3
    assert table.table_label == "Table 1"
    assert table.caption == "Table 1 Rates"
    assert table.columns == ["Catalyst", "CH4"]
    assert table.rows == [["TiO2", 12.5]]
    assert table.context.section_title == ""
    assert table.context.mentions == []


def test_prediction_document_keeps_ollama_context() -> None:
    document = build_prediction_document(
        [
            ExtractedGrid(
                page=4,
                rows=[["Catalyst", "CH4"], ["TiO2", "12.5"]],
                caption="Table 1 Photocatalytic rates.",
                table_label="Table 1",
                section_title="3. Results",
                mentions=(ExtractedMention(page=2, text="as shown in Table 1"),),
            )
        ],
        document_id="doc",
        source_pdf="paper.pdf",
    )
    table = document.tables[0]
    assert table.caption == "Table 1 Photocatalytic rates."
    assert table.table_label == "Table 1"
    assert table.context.section_title == "3. Results"
    assert table.context.mentions[0].text == "as shown in Table 1"
    assert table.context.mentions[0].page == 2


def test_parse_page_tables_skips_empty_and_pad_rows() -> None:
    grids = parse_page_tables(
        {
            "tables": [
                {
                    "table_label": "Table 1",
                    "caption": "Table 1 Rates",
                    "section_title": "2. Experimental",
                    "grid": [["A", "B"], ["1"], ["x", "y", "z"]],
                },
                {"grid": [["only header"]]},
            ]
        },
        page=3,
    )
    assert len(grids) == 1
    assert grids[0].page == 3
    assert grids[0].rows[1] == ["1", "", ""]
    assert grids[0].caption == "Table 1 Rates"


def test_attach_mentions_matches_table_label() -> None:
    grids = [
        ExtractedGrid(page=5, rows=[["A", "B"], ["1", "2"]], table_label="Table 1"),
        ExtractedGrid(page=6, rows=[["A", "B"], ["3", "4"]], table_label="Table 2"),
    ]
    attached = attach_mentions(
        grids,
        {
            "mentions_by_label": [
                {
                    "table_label": "Table 1",
                    "mentions": [{"page": 2, "text": "see Table 1"}],
                }
            ]
        },
    )
    assert attached[0].mentions == (ExtractedMention(page=2, text="see Table 1"),)
    assert attached[1].mentions == ()


def test_unflattenable_grids_are_skipped() -> None:
    document = build_prediction_document(
        [ExtractedGrid(page=1, rows=[["only header"]])],
        document_id="doc",
        source_pdf="x.pdf",
    )
    assert document.num_tables == 0
    assert document.tables == []


def test_extract_document_uses_corpus_relative_source(tmp_path: Path) -> None:
    corpus = tmp_path / "documents"
    corpus.mkdir()
    pdf = corpus / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    extractor = FakeExtractor(
        [ExtractedGrid(page=1, rows=[["A", "B"], ["1", "2"]])]
    )
    document = extract_document(pdf, extractor, corpus_dir=corpus)
    assert document.source_pdf == "paper.pdf"
    assert document.num_tables == 1
    assert document.tables[0].rows == [[1, 2]]


def test_pdfplumber_adapter_reads_extract_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakePage:
        def extract_tables(self):
            return [[["A", "B"], ["1", "2"]]]

    class FakeDoc:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_module = MagicMock()
    fake_module.open.return_value = FakeDoc()
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_module)
    grids = PdfPlumberExtractor().extract_grids(tmp_path / "x.pdf")
    assert grids == [ExtractedGrid(page=1, rows=[["A", "B"], ["1", "2"]])]
    fake_module.open.assert_called_once()


def test_pdfplumber_missing_install_is_importerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    with pytest.raises(ImportError, match="pdfplumber is not installed"):
        PdfPlumberExtractor().extract_grids(tmp_path / "x.pdf")


def test_camelot_adapter_uses_table_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    table = SimpleNamespace(page=2, data=[["A", "B"], ["1", "2"]])
    fake_module = MagicMock()
    fake_module.read_pdf.return_value = [table]
    monkeypatch.setitem(sys.modules, "camelot", fake_module)
    grids = CamelotExtractor("lattice").extract_grids(tmp_path / "x.pdf", max_pages=3)
    assert grids == [ExtractedGrid(page=2, rows=[["A", "B"], ["1", "2"]])]
    fake_module.read_pdf.assert_called_once_with(
        str(tmp_path / "x.pdf"), flavor="lattice", pages="1-3"
    )


def test_pymupdf_adapter_uses_find_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    table = SimpleNamespace(extract=lambda: [["A", "B"], ["3", "4"]])
    page = SimpleNamespace(find_tables=lambda: SimpleNamespace(tables=[table]))
    document = MagicMock()
    document.page_count = 1
    document.load_page.return_value = page
    fake_fitz = MagicMock()
    fake_fitz.open.return_value = document
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)
    grids = PyMuPDFExtractor().extract_grids(tmp_path / "x.pdf")
    assert grids == [ExtractedGrid(page=1, rows=[["A", "B"], ["3", "4"]])]
    document.close.assert_called_once()


def test_ghostscript_message_is_an_environment_error() -> None:
    assert is_environment_error(ImportError("missing"))
    assert is_environment_error(RuntimeError("Please install Ghostscript"))
    assert is_environment_error(
        ValueError(
            "numpy.dtype size changed, may indicate binary incompatibility. "
            "Expected 96 from C header, got 88 from PyObject"
        )
    )
    assert is_environment_error(ConnectionError("Cannot reach Ollama at http://127.0.0.1:11434"))
    assert is_environment_error(ConnectionError("Cannot reach GROBID at http://127.0.0.1:8070"))
    assert is_environment_error(RuntimeError("Ollama HTTP empty response from qwen3-vl:32b"))
    assert not is_environment_error(ValueError("broken page stream"))


def test_ollama_message_text_reads_thinking_when_content_empty() -> None:
    assert (
        ollama_message_text(
            {"message": {"content": "", "thinking": '{"tables":[]}'}}
        )
        == '{"tables":[]}'
    )
    assert ollama_message_text({"response": '{"tables":[]}'}) == '{"tables":[]}'


def test_ollama_chat_retries_format_json_after_empty_content() -> None:
    extractor = OllamaExtractor(tool_id="ollama_qwen3_vl")
    bodies = [
        {
            "message": {"content": ""},
            "done": True,
            "done_reason": "stop",
            "eval_count": 0,
        },
        {"message": {"content": '{"tables":[]}'}},
    ]

    def fake_post(payload: dict) -> dict:
        return bodies.pop(0)

    extractor._post = fake_post  # type: ignore[method-assign]
    assert extractor._chat("prompt", images=None, json_format={"type": "object"}, num_ctx=128) == {
        "tables": []
    }


_GROBID_TEI = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <div>
        <head>3. Results</head>
        <p coords="2,10.0,10.0,200.0,12.0">as shown in <ref type="table" target="#fig_0">Table 1</ref>.</p>
        <figure xml:id="fig_0" type="table" coords="3,10.0,50.0,200.0,80.0">
          <head>Table 1</head>
          <label>1</label>
          <figDesc>Table 1 Photocatalytic rates.</figDesc>
          <table>
            <row><cell>Catalyst</cell><cell>CH4</cell></row>
            <row><cell>TiO2</cell><cell>12.5</cell></row>
          </table>
        </figure>
      </div>
    </body>
  </text>
</TEI>
"""


def test_parse_grobid_tei_fills_grid_and_context() -> None:
    grids = parse_grobid_tei(_GROBID_TEI)
    assert len(grids) == 1
    grid = grids[0]
    assert grid.page == 3
    assert grid.rows == [["Catalyst", "CH4"], ["TiO2", "12.5"]]
    assert grid.table_label == "Table 1"
    assert grid.caption == "Table 1 Photocatalytic rates."
    assert grid.section_title == "3. Results"
    assert grid.mentions == (ExtractedMention(page=2, text="as shown in Table 1."),)


def test_parse_grobid_tei_respects_max_pages() -> None:
    assert parse_grobid_tei(_GROBID_TEI, max_pages=2) == []


def test_parse_grobid_tei_repeats_colspan_cells() -> None:
    xml = """\
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <figure type="table" coords="1,0,0,1,1">
    <table>
      <row><cell cols="2">Rates</cell></row>
      <row><cell>A</cell><cell>B</cell></row>
    </table>
  </figure>
</TEI>
"""
    grids = parse_grobid_tei(xml)
    assert grids[0].rows == [["Rates", "Rates"], ["A", "B"]]


def test_numpy2_blocks_stale_numexpr_before_pandas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numpy_mod = SimpleNamespace(__version__="2.2.6")
    monkeypatch.setitem(sys.modules, "numpy", numpy_mod)
    sys.modules.pop("numexpr", None)
    sys.modules.pop("bottleneck", None)
    _block_stale_numpy1_accelerators()
    assert sys.modules["numexpr"] is None
    assert sys.modules["bottleneck"] is None
    sys.modules.pop("numexpr", None)
    sys.modules.pop("bottleneck", None)


def test_shim_numpy2_complex_warning_is_importable() -> None:
    _shim_numpy2_aliases()
    from numpy.core.numeric import ComplexWarning

    assert ComplexWarning is not None
    try:
        import numpy._core.numeric as core_numeric
    except ImportError:
        return
    assert hasattr(core_numeric, "ComplexWarning")


def test_broken_sklearn_is_stubbed_for_transformers() -> None:
    import types

    class Boom(types.ModuleType):
        def __getattr__(self, name):
            raise ValueError(
                "numpy.dtype size changed, may indicate binary incompatibility. "
                "Expected 96 from C header, got 88 from PyObject"
            )

    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "sklearn" or name.startswith("sklearn.")
    }
    try:
        sklearn = types.ModuleType("sklearn")
        metrics = Boom("sklearn.metrics")
        sklearn.metrics = metrics
        sys.modules["sklearn"] = sklearn
        sys.modules["sklearn.metrics"] = metrics
        _ensure_sklearn_safe_for_transformers()
        from sklearn.metrics import roc_curve

        assert callable(roc_curve)
    finally:
        for name in list(sys.modules):
            if name == "sklearn" or name.startswith("sklearn."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


def test_numpy_abi_error_becomes_import_error_with_pip_hint() -> None:
    error = _numpy_abi_import_error(
        ValueError(
            "numpy.dtype size changed, may indicate binary incompatibility. "
            "Expected 96 from C header, got 88 from PyObject"
        ),
        "cv2",
    )
    assert isinstance(error, ImportError)
    assert "cv2" in str(error)
    assert "opencv-python-headless" in str(error)


def test_require_numpy2_extension_raises_on_abi_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(name):
        raise ValueError(
            "numpy.dtype size changed, may indicate binary incompatibility. "
            "Expected 96 from C header, got 88 from PyObject"
        )

    monkeypatch.setattr("importlib.import_module", boom)
    with pytest.raises(ImportError, match="cv2"):
        _require_numpy2_extension("cv2")


def test_docling_adapter_reads_table_grid(tmp_path: Path) -> None:
    cell = lambda text: SimpleNamespace(text=text)
    table = SimpleNamespace(
        prov=[SimpleNamespace(page_no=2)],
        data=SimpleNamespace(
            grid=[[cell("A"), cell("B")], [cell("1"), cell("2")]]
        ),
    )
    converter = MagicMock()
    converter.convert.return_value = SimpleNamespace(
        document=SimpleNamespace(tables=[table])
    )
    extractor = DoclingExtractor()
    extractor._converter = converter
    grids = extractor.extract_grids(tmp_path / "x.pdf")
    assert grids == [ExtractedGrid(page=2, rows=[["A", "B"], ["1", "2"]])]


def test_docling_missing_install_is_importerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "docling", None)
    monkeypatch.setitem(sys.modules, "docling.datamodel", None)
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", None)
    monkeypatch.setitem(sys.modules, "docling.datamodel.pipeline_options", None)
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    with pytest.raises(ImportError, match="docling is not installed"):
        DoclingExtractor().extract_grids(tmp_path / "x.pdf")


def test_tatr_adapter_reads_gmft_dataframe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Columns(list):
        nlevels = 1

        def tolist(self):
            return list(self)

    class Frame:
        empty = False
        columns = Columns(["A", "B"])

        def itertuples(self, index=False, name=None):
            yield ("1", "2")

    page = object()
    document = MagicMock()
    document.__iter__ = MagicMock(return_value=iter([page]))
    formatted = SimpleNamespace(df=lambda: Frame())
    detector = MagicMock()
    detector.extract.return_value = [object()]
    formatter = MagicMock()
    formatter.extract.return_value = formatted
    pdf_bindings = MagicMock()
    pdf_bindings.PyPDFium2Document.return_value = document
    auto = MagicMock()
    auto.AutoTableDetector.return_value = detector
    auto.AutoTableFormatter.return_value = formatter
    monkeypatch.setitem(sys.modules, "gmft", MagicMock())
    monkeypatch.setitem(sys.modules, "gmft.auto", auto)
    monkeypatch.setitem(sys.modules, "gmft.pdf_bindings", pdf_bindings)
    extractor = TATRExtractor()
    grids = extractor.extract_grids(tmp_path / "x.pdf")
    assert grids == [ExtractedGrid(page=1, rows=[["A", "B"], ["1", "2"]])]
    document.close.assert_called_once()


def test_unstructured_adapter_parses_html_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    element = SimpleNamespace(
        category="Table",
        metadata=SimpleNamespace(
            page_number=3,
            text_as_html=(
                "<table><tr><td>A</td><td>B</td></tr>"
                "<tr><td>1</td><td>2</td></tr></table>"
            ),
        ),
    )
    pdf_mod = MagicMock()
    pdf_mod.partition_pdf.return_value = [element]
    monkeypatch.setitem(sys.modules, "unstructured", MagicMock())
    monkeypatch.setitem(sys.modules, "unstructured.partition", MagicMock())
    monkeypatch.setitem(sys.modules, "unstructured.partition.pdf", pdf_mod)
    grids = UnstructuredExtractor().extract_grids(tmp_path / "x.pdf")
    assert len(grids) == 1
    assert grids[0].page == 3
    assert grids[0].rows[0] == ["A", "B"]
    assert grids[0].rows[1] == [1, 2]
