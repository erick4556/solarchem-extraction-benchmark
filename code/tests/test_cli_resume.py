"""Resume behaviour of the ground-truth CLI: skip work already on disk."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from solarchem_benchmark.gt.cli import (
    load_merged_documents,
    main,
    merge_corpus_documents,
    source_pdf_key,
)
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


def test_source_pdf_key_is_relative_to_the_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "documents"
    corpus.mkdir()
    pdf = corpus / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    assert source_pdf_key(pdf, corpus) == "paper.pdf"


def test_load_merged_documents_indexes_by_source_pdf(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth_all.json"
    corpus = GroundTruthCorpus(documents=[_doc("a.pdf"), _doc("b.pdf")])
    path.write_text(corpus.model_dump_json(), encoding="utf-8")
    loaded = load_merged_documents(path)
    assert set(loaded) == {"a.pdf", "b.pdf"}


def test_load_merged_documents_returns_empty_when_missing(tmp_path: Path) -> None:
    assert load_merged_documents(tmp_path / "absent.json") == {}


def test_merge_keeps_documents_outside_the_current_pdf_list(tmp_path: Path) -> None:
    corpus = tmp_path / "documents"
    corpus.mkdir()
    pdfs = []
    for name in ("a.pdf", "b.pdf"):
        path = corpus / name
        path.write_bytes(b"%PDF")
        pdfs.append(path)
    existing = {
        "a.pdf": _doc("a.pdf"),
        "b.pdf": _doc("b.pdf"),
        "old.pdf": _doc("old.pdf"),
    }
    # A later --limit 1 must not erase b.pdf from a previous longer run.
    ordered = merge_corpus_documents(existing, pdfs[:1], corpus)
    assert [document.source_pdf for document in ordered] == ["a.pdf", "b.pdf", "old.pdf"]


def test_merged_run_skips_pdfs_already_in_the_output(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    documents = data_root / "documents"
    documents.mkdir(parents=True)
    pdfs = []
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        path = documents / name
        path.write_bytes(b"%PDF-1.4")
        pdfs.append(path)

    output = data_root / "ground_truth" / "ground_truth_all.json"
    output.parent.mkdir(parents=True)
    output.write_text(
        GroundTruthCorpus(documents=[_doc("a.pdf"), _doc("b.pdf")]).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_generate(pdf, engine, **kwargs):
        calls.append(pdf.name)
        return _doc(pdf.name, document_id=f"id_{pdf.stem}")

    fake_engine = MagicMock()
    with (
        patch("solarchem_benchmark.gt.cli.build_engine", return_value=fake_engine),
        patch("solarchem_benchmark.gt.cli.generate_for_pdf", side_effect=fake_generate),
    ):
        assert main(["--data-root", str(data_root), "--output", str(output)]) == 0

    assert calls == ["c.pdf"]
    corpus = GroundTruthCorpus.model_validate_json(output.read_text(encoding="utf-8"))
    assert [document.source_pdf for document in corpus.documents] == [
        "a.pdf",
        "b.pdf",
        "c.pdf",
    ]


def test_overwrite_regenerates_existing_documents(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    documents = data_root / "documents"
    documents.mkdir(parents=True)
    pdf = documents / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    output = data_root / "ground_truth" / "ground_truth_all.json"
    output.parent.mkdir(parents=True)
    output.write_text(
        GroundTruthCorpus(documents=[_doc("a.pdf", "old_id")]).model_dump_json() + "\n",
        encoding="utf-8",
    )

    def fake_generate(pdf_path, engine, **kwargs):
        return _doc(pdf_path.name, document_id="new_id")

    with (
        patch("solarchem_benchmark.gt.cli.build_engine", return_value=MagicMock()),
        patch("solarchem_benchmark.gt.cli.generate_for_pdf", side_effect=fake_generate),
    ):
        assert (
            main(
                [
                    "--data-root",
                    str(data_root),
                    "--output",
                    str(output),
                    "--overwrite",
                ]
            )
            == 0
        )

    corpus = GroundTruthCorpus.model_validate_json(output.read_text(encoding="utf-8"))
    assert corpus.documents[0].document_id == "new_id"


def test_per_document_skips_before_calling_generate(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    documents = data_root / "documents"
    documents.mkdir(parents=True)
    pdf = documents / "1-s2.0-S0010854522000236-main.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    from solarchem_benchmark.gt.generate import derive_document_id

    document_id = derive_document_id(pdf)
    output_dir = data_root / "ground_truth"
    output_dir.mkdir(parents=True)
    (output_dir / f"{document_id}.json").write_text(
        _doc(pdf.name, document_id).model_dump_json() + "\n",
        encoding="utf-8",
    )

    with (
        patch("solarchem_benchmark.gt.cli.build_engine", return_value=MagicMock()),
        patch("solarchem_benchmark.gt.cli.generate_for_pdf") as generate,
    ):
        assert main(["--data-root", str(data_root), "--per-document"]) == 0
        generate.assert_not_called()


def test_default_merged_output_is_named_after_the_engine(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    documents = data_root / "documents"
    documents.mkdir(parents=True)
    (documents / "a.pdf").write_bytes(b"%PDF-1.4")

    def fake_generate(pdf, engine, **kwargs):
        return _doc(pdf.name)

    with (
        patch("solarchem_benchmark.gt.cli.build_engine", return_value=MagicMock()),
        patch("solarchem_benchmark.gt.cli.generate_for_pdf", side_effect=fake_generate),
    ):
        assert main(["--data-root", str(data_root), "--engine", "lighton_ocr"]) == 0

    output = data_root / "ground_truth" / "ground_truth_lighton_ocr.json"
    assert output.is_file()
    corpus = GroundTruthCorpus.model_validate_json(output.read_text(encoding="utf-8"))
    assert corpus.documents[0].source_pdf == "a.pdf"


def test_merged_file_is_written_after_each_success(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    documents = data_root / "documents"
    documents.mkdir(parents=True)
    for name in ("a.pdf", "b.pdf"):
        (documents / name).write_bytes(b"%PDF-1.4")

    output = data_root / "ground_truth" / "ground_truth_lighton_ocr.json"
    seen_sizes: list[int] = []

    def fake_generate(pdf, engine, **kwargs):
        document = _doc(pdf.name)
        if output.is_file():
            seen_sizes.append(
                len(GroundTruthCorpus.model_validate_json(output.read_text()).documents)
            )
        return document

    with (
        patch("solarchem_benchmark.gt.cli.build_engine", return_value=MagicMock()),
        patch("solarchem_benchmark.gt.cli.generate_for_pdf", side_effect=fake_generate),
    ):
        assert main(["--data-root", str(data_root), "--output", str(output)]) == 0

    # After the first PDF is generated, the second call already sees one document on disk.
    assert seen_sizes == [1]
    assert len(GroundTruthCorpus.model_validate_json(output.read_text()).documents) == 2


def test_merged_run_omits_documents_with_no_tables(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    documents = data_root / "documents"
    documents.mkdir(parents=True)
    for name in ("empty.pdf", "has_table.pdf"):
        (documents / name).write_bytes(b"%PDF-1.4")

    output = data_root / "ground_truth" / "ground_truth_lighton_ocr.json"

    def fake_generate(pdf, engine, **kwargs):
        return _doc(pdf.name, with_table=pdf.name != "empty.pdf")

    with (
        patch("solarchem_benchmark.gt.cli.build_engine", return_value=MagicMock()),
        patch("solarchem_benchmark.gt.cli.generate_for_pdf", side_effect=fake_generate),
    ):
        assert main(["--data-root", str(data_root), "--output", str(output)]) == 0

    corpus = GroundTruthCorpus.model_validate_json(output.read_text(encoding="utf-8"))
    assert [document.source_pdf for document in corpus.documents] == ["has_table.pdf"]


def test_existing_empty_documents_are_dropped_from_the_merged_file(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    documents = data_root / "documents"
    documents.mkdir(parents=True)
    (documents / "empty.pdf").write_bytes(b"%PDF-1.4")
    (documents / "kept.pdf").write_bytes(b"%PDF-1.4")

    output = data_root / "ground_truth" / "ground_truth_all.json"
    output.parent.mkdir(parents=True)
    output.write_text(
        GroundTruthCorpus(
            documents=[
                _doc("empty.pdf", with_table=False),
                _doc("kept.pdf"),
            ]
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    with (
        patch("solarchem_benchmark.gt.cli.build_engine", return_value=MagicMock()),
        patch(
            "solarchem_benchmark.gt.cli.generate_for_pdf",
            side_effect=lambda pdf, engine, **kwargs: _doc(pdf.name, with_table=False),
        ),
    ):
        assert main(["--data-root", str(data_root), "--output", str(output)]) == 0

    corpus = GroundTruthCorpus.model_validate_json(output.read_text(encoding="utf-8"))
    assert [document.source_pdf for document in corpus.documents] == ["kept.pdf"]
