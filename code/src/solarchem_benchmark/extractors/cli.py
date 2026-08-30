"""CLI for table extractors (Phase 5 native + Phase 6 layout + Phase 7 GROBID/Ollama).

Predictions are written under ``data/predictions/``, never into the silver GT
files. By default only PDFs listed in the working silver snapshot
(``ground_truth_lighton_ocr_302.json``) are processed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from solarchem_benchmark.extractors.assemble import extract_document
from solarchem_benchmark.extractors.base import (
    available_tools,
    build_extractor,
    is_environment_error,
)
from solarchem_benchmark.gt.cli import (
    load_merged_documents,
    merge_corpus_documents,
    source_pdf_key,
)
from solarchem_benchmark.gt.generate import derive_document_id
from solarchem_benchmark.gt.schema import GroundTruthCorpus
from solarchem_benchmark.paths import (
    default_prediction_path,
    default_working_silver_path,
    resolve_data_root,
    resolve_documents_dir,
)

logger = logging.getLogger("solarchem_benchmark.extractors")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solarchem-extract-tables",
        description=(
            "Extract tables from PDFs (pdfplumber, Camelot, PyMuPDF, Docling, "
            "TATR, Unstructured, GROBID, Ollama VLMs) into the benchmark schema. "
            "Output goes to data/predictions/, not into the ground-truth files. "
            "Documents already present are skipped unless --overwrite is set. "
            "Documents with 0 recovered tables are kept (num_tables=0) so the "
            "prediction file aligns with the silver/Gold working set and "
            "detection misses are scored."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pdf", type=Path, help="Process a single PDF.")
    source.add_argument(
        "--input",
        type=Path,
        help="Directory of PDFs (defaults to the corpus directory).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help=(
            "Silver JSON whose source_pdf list is the working set. Defaults to "
            "data/ground_truth/ground_truth_lighton_ocr_302.json when that "
            "file exists and neither --pdf nor --input is given."
        ),
    )
    parser.add_argument("--data-root", type=Path, help="Root of the data directory.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON (default: data/predictions/<tool>.json).",
    )
    parser.add_argument(
        "--tool",
        required=True,
        choices=available_tools(),
        help="Table extractor to run.",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama base URL (default: $OLLAMA_HOST or http://127.0.0.1:11434).",
    )
    parser.add_argument(
        "--ollama-model",
        help="Override the default Ollama tag for an ollama_* tool.",
    )
    parser.add_argument(
        "--grobid-host",
        default=os.environ.get("GROBID_HOST", "http://127.0.0.1:8070"),
        help="GROBID base URL (default: $GROBID_HOST or http://127.0.0.1:8070).",
    )
    parser.add_argument("--limit", type=int, help="Process at most this many PDFs.")
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Process at most this many pages per PDF, for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract documents already present in the output.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def pdfs_from_reference(
    reference_path: Path,
    corpus_dir: Path,
) -> tuple[list[Path], list[str]]:
    """Resolve corpus PDFs listed in a silver / Gold JSON.

    Returns:
        ``(found_pdfs, missing_names)``. Order follows the reference file.
    """
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "documents" in payload:
        names = [document["source_pdf"] for document in payload["documents"]]
    elif isinstance(payload, dict) and "source_pdf" in payload:
        names = [payload["source_pdf"]]
    else:
        raise ValueError(f"Not a ground-truth JSON: {reference_path}")

    found: list[Path] = []
    missing: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = Path(name).name
        if key in seen:
            continue
        seen.add(key)
        path = corpus_dir / key
        if path.is_file():
            found.append(path)
        else:
            missing.append(name)
    return found, missing


def _collect_pdfs(args: argparse.Namespace, data_root: Path) -> tuple[list[Path], Path]:
    if args.pdf is not None:
        pdf = args.pdf.expanduser().resolve()
        if not pdf.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf}")
        return [pdf], pdf.parent

    corpus_dir = resolve_documents_dir(args.input, data_root=data_root)

    reference = args.reference
    if reference is None and args.input is None:
        default_ref = default_working_silver_path(data_root)
        if default_ref.is_file():
            reference = default_ref

    if reference is not None:
        reference = Path(reference).expanduser().resolve()
        if not reference.is_file():
            raise FileNotFoundError(f"Reference JSON not found: {reference}")
        pdfs, missing = pdfs_from_reference(reference, corpus_dir)
        logger.info("Reference:  %s (%d PDFs listed)", reference, len(pdfs) + len(missing))
        if missing:
            preview = ", ".join(Path(name).name for name in missing[:5])
            extra = f" … (+{len(missing) - 5})" if len(missing) > 5 else ""
            logger.warning(
                "Reference PDFs missing from corpus (%d): %s%s",
                len(missing),
                preview,
                extra,
            )
        if not pdfs:
            raise FileNotFoundError(f"No reference PDFs found under {corpus_dir}")
    else:
        pdfs = sorted(corpus_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDFs found in {corpus_dir}")

    if args.limit:
        pdfs = pdfs[: args.limit]
    return pdfs, corpus_dir


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    data_root = resolve_data_root(args.data_root)
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else default_prediction_path(data_root, args.tool)
    )

    try:
        pdfs, corpus_dir = _collect_pdfs(args, data_root)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        logger.error("%s", error)
        return 1

    existing = load_merged_documents(output_path)
    if existing and not args.overwrite:
        logger.info(
            "Resuming from %s (%d documents already present)",
            output_path,
            len(existing),
        )

    extractor = build_extractor(
        args.tool,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        grobid_host=args.grobid_host,
    )
    logger.info("Corpus:     %s (%d PDFs)", corpus_dir, len(pdfs))
    logger.info("Output:     %s", output_path)
    if args.tool == "grobid":
        logger.info(
            "Tool:       grobid (structure + context; GROBID at %s)",
            args.grobid_host,
        )
    elif extractor.emits_context:
        model = getattr(extractor, "model", "")
        logger.info(
            "Tool:       %s (structure + context; Ollama model %s at %s)",
            args.tool,
            model,
            args.ollama_host,
        )
    else:
        logger.info(
            "Tool:       %s (structure only; caption/section/mentions empty)",
            args.tool,
        )

    def _persist() -> None:
        ordered = merge_corpus_documents(existing, pdfs, corpus_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        corpus = GroundTruthCorpus(documents=ordered)
        output_path.write_text(corpus.model_dump_json(indent=2) + "\n", encoding="utf-8")

    failures: list[tuple[Path, str]] = []
    generated = 0
    skipped = 0
    empty_saved = 0

    for position, pdf in enumerate(pdfs, start=1):
        source_key = source_pdf_key(pdf, corpus_dir)
        logger.info("[%d/%d] %s", position, len(pdfs), pdf.name)

        if not args.overwrite and source_key in existing:
            logger.info("  exists, skipping (use --overwrite): %s", source_key)
            skipped += 1
            continue

        try:
            document = extract_document(
                pdf,
                extractor,
                document_id=derive_document_id(pdf),
                corpus_dir=corpus_dir,
                max_pages=args.max_pages,
            )
        except Exception as error:  # noqa: BLE001 - one bad PDF must not stop the corpus
            if is_environment_error(error):
                logger.exception(
                    "Tool %s cannot run in this environment; aborting the remaining "
                    "corpus so the same error is not repeated. %s",
                    args.tool,
                    error,
                )
                return 1
            logger.exception("  failed: %s", pdf.name)
            failures.append((pdf, str(error)))
            continue

        if document.num_tables == 0:
            logger.info("  no tables recovered, keeping empty document: %s", pdf.name)
            empty_saved += 1
        else:
            generated += 1

        existing[document.source_pdf] = document
        _persist()
        logger.info("  saved (%d documents in %s)", len(existing), output_path.name)

    if existing:
        ordered = merge_corpus_documents(existing, pdfs, corpus_dir)
        total_tables = sum(document.num_tables for document in ordered)
        logger.info(
            "Predictions at %s (%d documents, %d tables)",
            output_path,
            len(ordered),
            total_tables,
        )

    logger.info(
        "Done: %d with tables, %d empty (0 tables), %d skipped, %d PDFs in this run, %d failures",
        generated,
        empty_saved,
        skipped,
        len(pdfs),
        len(failures),
    )

    if failures:
        report = output_path.parent / f"{output_path.stem}_failures.json"
        report.write_text(
            json.dumps(
                [{"pdf": pdf.name, "error": message} for pdf, message in failures],
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.warning("Failures recorded in %s", report)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
