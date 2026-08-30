"""Resume and reference-filter behaviour of the native-extractor CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from solarchem_benchmark.extractors.cli import main, pdfs_from_reference
from solarchem_benchmark.gt.schema import GroundTruthCorpus, GroundTruthDocument, Table


def _table(document_id: str) -> Table:
    return Table(
        table_id=f"{document_id}_table_01",
        page=1,
        columns=["A"],
        rows=[[1]],
    )


def _doc(
    source: str,
    document_id: str | None = None,
    *,
    with_table: bool = True,
) -> GroundTruthDocument:
    resolved_id = document_id or f"id_{Path(source).stem}"
    tables = [_table(resolved_id)] if with_table else []
    return GroundTruthDocument(
        document_id=resolved_id,
        source_pdf=source,
        num_tables=len(tables),
        tables=tables,
    )


def _corpus(tmp_path: Path, names: tuple[str, ...] = ("a.pdf", "b.pdf", "c.pdf")) -> Path:
    documents = tmp_path / "data" / "documents"
    documents.mkdir(parents=True)
    for name in names:
        (documents / name).write_bytes(b"%PDF-1.4")
    return tmp_path / "data"


def test_pdfs_from_reference_keeps_reference_order(tmp_path: Path) -> None:
    corpus = tmp_path / "documents"
    corpus.mkdir()
    for name in ("b.pdf", "a.pdf"):
        (corpus / name).write_bytes(b"%PDF-1.4")
    reference = tmp_path / "silver.json"
    reference.write_text(
        GroundTruthCorpus(documents=[_doc("a.pdf"), _doc("missing.pdf"), _doc("b.pdf")]).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    found, missing = pdfs_from_reference(reference, corpus)
    assert [path.name for path in found] == ["a.pdf", "b.pdf"]
    assert missing == ["missing.pdf"]


def test_default_output_is_named_after_the_tool(tmp_path: Path) -> None:
    data_root = _corpus(tmp_path, ("a.pdf",))

    def fake_extract(pdf, extractor, **kwargs):
        return _doc(pdf.name)

    with patch("solarchem_benchmark.extractors.cli.extract_document", side_effect=fake_extract):
        assert main(["--data-root", str(data_root), "--tool", "pdfplumber", "--input", str(data_root / "documents")]) == 0

    output = data_root / "predictions" / "pdfplumber.json"
    assert output.is_file()
    corpus = GroundTruthCorpus.model_validate_json(output.read_text(encoding="utf-8"))
    assert corpus.documents[0].source_pdf == "a.pdf"


def test_reference_restricts_the_working_set(tmp_path: Path) -> None:
    data_root = _corpus(tmp_path)
    reference = data_root / "ground_truth" / "ground_truth_lighton_ocr_302.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(
        GroundTruthCorpus(documents=[_doc("b.pdf")]).model_dump_json() + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_extract(pdf, extractor, **kwargs):
        calls.append(pdf.name)
        return _doc(pdf.name)

    output = data_root / "predictions" / "pdfplumber.json"
    with patch("solarchem_benchmark.extractors.cli.extract_document", side_effect=fake_extract):
        assert (
            main(
                [
                    "--data-root",
                    str(data_root),
                    "--tool",
                    "pdfplumber",
                    "--reference",
                    str(reference),
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
    assert calls == ["b.pdf"]


def test_default_reference_is_the_working_silver_snapshot(tmp_path: Path) -> None:
    data_root = _corpus(tmp_path)
    reference = data_root / "ground_truth" / "ground_truth_lighton_ocr_302.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(
        GroundTruthCorpus(documents=[_doc("a.pdf")]).model_dump_json() + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_extract(pdf, extractor, **kwargs):
        calls.append(pdf.name)
        return _doc(pdf.name)

    with patch("solarchem_benchmark.extractors.cli.extract_document", side_effect=fake_extract):
        assert main(["--data-root", str(data_root), "--tool", "pymupdf"]) == 0
    assert calls == ["a.pdf"]


def test_merged_run_skips_pdfs_already_in_the_output(tmp_path: Path) -> None:
    data_root = _corpus(tmp_path)
    output = data_root / "predictions" / "pdfplumber.json"
    output.parent.mkdir(parents=True)
    output.write_text(
        GroundTruthCorpus(documents=[_doc("a.pdf"), _doc("b.pdf")]).model_dump_json() + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_extract(pdf, extractor, **kwargs):
        calls.append(pdf.name)
        return _doc(pdf.name)

    with patch("solarchem_benchmark.extractors.cli.extract_document", side_effect=fake_extract):
        assert (
            main(
                [
                    "--data-root",
                    str(data_root),
                    "--tool",
                    "pdfplumber",
                    "--input",
                    str(data_root / "documents"),
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
    assert calls == ["c.pdf"]


def test_keeps_empty_documents_so_the_working_set_aligns(tmp_path: Path) -> None:
    data_root = _corpus(tmp_path, ("empty.pdf", "has_table.pdf"))
    output = data_root / "predictions" / "pdfplumber.json"

    def fake_extract(pdf, extractor, **kwargs):
        return _doc(pdf.name, with_table=pdf.name != "empty.pdf")

    with patch("solarchem_benchmark.extractors.cli.extract_document", side_effect=fake_extract):
        assert (
            main(
                [
                    "--data-root",
                    str(data_root),
                    "--tool",
                    "pdfplumber",
                    "--input",
                    str(data_root / "documents"),
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
    corpus = GroundTruthCorpus.model_validate_json(output.read_text(encoding="utf-8"))
    assert [document.source_pdf for document in corpus.documents] == [
        "empty.pdf",
        "has_table.pdf",
    ]
    by_source = {document.source_pdf: document for document in corpus.documents}
    assert by_source["empty.pdf"].num_tables == 0
    assert by_source["has_table.pdf"].num_tables == 1


def test_importerror_aborts_the_run(tmp_path: Path) -> None:
    data_root = _corpus(tmp_path, ("a.pdf", "b.pdf"))
    calls: list[str] = []

    def fake_extract(pdf, extractor, **kwargs):
        calls.append(pdf.name)
        raise ImportError("pdfplumber is not installed")

    with patch("solarchem_benchmark.extractors.cli.extract_document", side_effect=fake_extract):
        assert (
            main(
                [
                    "--data-root",
                    str(data_root),
                    "--tool",
                    "pdfplumber",
                    "--input",
                    str(data_root / "documents"),
                ]
            )
            == 1
        )
    assert calls == ["a.pdf"]


def test_file_is_written_after_each_success(tmp_path: Path) -> None:
    data_root = _corpus(tmp_path, ("a.pdf", "b.pdf"))
    output = data_root / "predictions" / "pdfplumber.json"
    seen_sizes: list[int] = []

    def fake_extract(pdf, extractor, **kwargs):
        if output.is_file():
            seen_sizes.append(
                len(GroundTruthCorpus.model_validate_json(output.read_text()).documents)
            )
        return _doc(pdf.name)

    with patch("solarchem_benchmark.extractors.cli.extract_document", side_effect=fake_extract):
        assert (
            main(
                [
                    "--data-root",
                    str(data_root),
                    "--tool",
                    "pdfplumber",
                    "--input",
                    str(data_root / "documents"),
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
    assert seen_sizes == [1]
    assert len(GroundTruthCorpus.model_validate_json(output.read_text()).documents) == 2
