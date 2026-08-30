"""Command-line interface for automatic ground-truth generation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from solarchem_benchmark.gt.generate import derive_document_id, generate_for_pdf, write_document
from solarchem_benchmark.gt.ocr import available_engines, build_engine
from solarchem_benchmark.gt.schema import GroundTruthCorpus, GroundTruthDocument
from solarchem_benchmark.paths import (
    default_ground_truth_dir,
    default_merged_ground_truth_path,
    default_ocr_cache_dir,
    resolve_data_root,
    resolve_documents_dir,
)

logger = logging.getLogger("solarchem_benchmark.gt")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="solarchem-gt",
        description=(
            "Generate the SolarChem table and table-context ground truth. "
            "By default a single corpus-wide JSON is written; per-document files "
            "are optional. Documents already present in the output are skipped "
            "unless --overwrite is set. PDFs from which no table is recovered "
            "are omitted from the ground truth (OCR cache is still kept)."
        ),
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pdf", type=Path, help="Process a single PDF.")
    source.add_argument(
        "--input",
        type=Path,
        help="Directory of PDFs to process (defaults to the corpus directory).",
    )

    parser.add_argument("--data-root", type=Path, help="Root of the data directory.")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output location. Defaults to "
            "data/ground_truth/ground_truth_<engine>.json "
            "(e.g. ground_truth_lighton_ocr.json) or data/ground_truth/ "
            "with --per-document."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "Directory for cached OCR output. Defaults to "
            "data/intermediate/ocr_cache/<engine>/."
        ),
    )
    parser.add_argument(
        "--document-id",
        help="Explicit document identifier; only valid together with --pdf.",
    )
    parser.add_argument(
        "--engine",
        default="lighton_ocr",
        choices=available_engines(),
        help="OCR engine used to transcribe the pages (default: %(default)s).",
    )
    parser.add_argument(
        "--model-id",
        help="Override the engine's default model identifier.",
    )
    parser.add_argument("--limit", type=int, help="Process at most this many PDFs.")
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Process at most this many pages per PDF, for smoke tests.",
    )
    parser.add_argument(
        "--merged",
        type=Path,
        help=(
            "Deprecated alias for the single-file output path. "
            "Prefer --output <file>.json."
        ),
    )
    parser.add_argument(
        "--per-document",
        action="store_true",
        help="Write one JSON per document instead of one merged corpus JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate ground truth for documents that are already present "
            "in the output (default: skip them)."
        ),
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Ignore cached transcriptions and re-run the OCR engine.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable OCR caching.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s).",
    )
    return parser


def source_pdf_key(pdf_path: Path, corpus_dir: Path) -> str:
    """Return the ``source_pdf`` string used in the ground truth for a PDF."""
    try:
        return pdf_path.resolve().relative_to(corpus_dir.resolve()).as_posix()
    except ValueError:
        return pdf_path.name


def load_merged_documents(path: Path) -> dict[str, GroundTruthDocument]:
    """Load an existing merged corpus, keyed by ``source_pdf``.

    Args:
        path: Path to ``ground_truth_all.json`` (or equivalent).

    Returns:
        Documents already on disk. Empty when the file is missing or unreadable.
    """
    if not path.is_file():
        return {}
    try:
        corpus = GroundTruthCorpus.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 - a damaged file must not abort a resume
        logger.warning("Could not load existing ground truth at %s: %s", path, error)
        return {}
    return {document.source_pdf: document for document in corpus.documents}


def load_document_file(path: Path) -> GroundTruthDocument | None:
    """Load one per-document ground-truth file, or ``None`` when unusable."""
    if not path.is_file():
        return None
    try:
        return GroundTruthDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        logger.warning("Could not load existing ground truth at %s: %s", path, error)
        return None


def merge_corpus_documents(
    existing: dict[str, GroundTruthDocument],
    pdfs: list[Path],
    corpus_dir: Path,
) -> list[GroundTruthDocument]:
    """Order the corpus for writing: current PDFs first, then any leftover.

    Documents whose ``source_pdf`` is not in the current PDF list are kept, so
    a later ``--limit N`` run cannot erase work from a previous longer run.
    """
    current_keys = [source_pdf_key(pdf, corpus_dir) for pdf in pdfs]
    current_set = set(current_keys)
    ordered = [existing[key] for key in current_keys if key in existing]
    ordered.extend(
        document
        for key, document in existing.items()
        if key not in current_set
    )
    return ordered


def _collect_pdfs(args: argparse.Namespace, data_root: Path) -> tuple[list[Path], Path]:
    """Resolve the list of PDFs to process and the corpus directory."""
    if args.pdf is not None:
        pdf = args.pdf.expanduser().resolve()
        if not pdf.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf}")
        return [pdf], pdf.parent

    corpus_dir = resolve_documents_dir(args.input, data_root=data_root)
    pdfs = sorted(corpus_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {corpus_dir}")
    if args.limit:
        pdfs = pdfs[: args.limit]
    return pdfs, corpus_dir


def main(argv: list[str] | None = None) -> int:
    """Run the generator.

    Returns:
        ``0`` when every PDF was processed or skipped cleanly, ``1`` when at
        least one failed.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.document_id and args.pdf is None:
        logger.error("--document-id requires --pdf")
        return 1

    data_root = resolve_data_root(args.data_root)
    default_gt_dir = default_ground_truth_dir(data_root).expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else None
    merged_path = args.merged.expanduser().resolve() if args.merged else None
    if output_path is not None and merged_path is not None:
        logger.error("Use either --output or --merged, not both")
        return 1

    if args.per_document:
        output_dir = output_path or default_gt_dir
        if output_dir.suffix.lower() == ".json":
            logger.error("--per-document expects --output to be a directory, not a JSON file")
            return 1
        merged_output = None
    else:
        if output_path is not None:
            merged_output = output_path
        elif merged_path is not None:
            merged_output = merged_path
        else:
            merged_output = default_merged_ground_truth_path(data_root, args.engine)
        output_dir = default_gt_dir
    cache_dir = (
        None
        if args.no_cache
        else (
            args.cache_dir.expanduser().resolve()
            if args.cache_dir
            else default_ocr_cache_dir(data_root, args.engine).expanduser().resolve()
        )
    )

    try:
        pdfs, corpus_dir = _collect_pdfs(args, data_root)
    except (FileNotFoundError, NotADirectoryError) as error:
        logger.error("%s", error)
        return 1

    existing = (
        {}
        if args.per_document
        else load_merged_documents(merged_output)  # type: ignore[arg-type]
    )
    if existing and not args.overwrite:
        logger.info("Resuming from %s (%d documents already present)", merged_output, len(existing))

    engine_kwargs = {"model_id": args.model_id} if args.model_id else {}
    engine = build_engine(args.engine, **engine_kwargs)

    logger.info("Corpus:     %s (%d PDFs)", corpus_dir, len(pdfs))
    if args.per_document:
        logger.info("Output:     %s (one file per document)", output_dir)
    else:
        logger.info("Output:     %s (single merged file)", merged_output)
    logger.info("OCR cache:  %s", cache_dir or "disabled")
    logger.info("Engine:     %s", args.engine)

    def _persist_merged() -> None:
        """Write the merged corpus after each success so a crash can resume."""
        assert merged_output is not None
        ordered = merge_corpus_documents(existing, pdfs, corpus_dir)
        merged_output.parent.mkdir(parents=True, exist_ok=True)
        corpus = GroundTruthCorpus(documents=ordered)
        merged_output.write_text(corpus.model_dump_json(indent=2) + "\n", encoding="utf-8")

    if not args.per_document:
        empty_keys = [key for key, document in existing.items() if document.num_tables == 0]
        if empty_keys:
            for key in empty_keys:
                del existing[key]
            logger.info(
                "Removed %d document(s) with 0 tables from %s",
                len(empty_keys),
                merged_output.name if merged_output is not None else "corpus",
            )
            _persist_merged()

    failures: list[tuple[Path, str]] = []
    generated = 0
    skipped = 0
    omitted_empty = 0

    for position, pdf in enumerate(pdfs, start=1):
        document_id = args.document_id or derive_document_id(pdf)
        source_key = source_pdf_key(pdf, corpus_dir)
        logger.info("[%d/%d] %s", position, len(pdfs), pdf.name)

        if args.per_document:
            target = output_dir / f"{document_id}.json"
            if not args.overwrite:
                kept = load_document_file(target)
                if kept is not None:
                    if kept.num_tables == 0:
                        target.unlink()
                    else:
                        logger.info("  exists, skipping (use --overwrite): %s", target.name)
                        skipped += 1
                        continue
        elif not args.overwrite and source_key in existing:
            logger.info("  exists, skipping (use --overwrite): %s", source_key)
            skipped += 1
            continue

        try:
            document = generate_for_pdf(
                pdf,
                engine,
                document_id=document_id,
                corpus_dir=corpus_dir,
                cache_dir=cache_dir,
                max_pages=args.max_pages,
                force_ocr=args.force_ocr,
            )
        except NotImplementedError as error:
            logger.error("%s", error)
            return 1
        except ImportError as error:
            logger.error(
                "OCR engine %s cannot be imported; aborting the remaining "
                "corpus so the same environment error is not repeated. %s",
                args.engine,
                error,
            )
            return 1
        except Exception as error:  # noqa: BLE001 - one bad PDF must not stop the corpus
            logger.exception("  failed: %s", pdf.name)
            failures.append((pdf, str(error)))
            continue

        if document.num_tables == 0:
            logger.info("  no tables recovered, omitting from ground truth: %s", pdf.name)
            omitted_empty += 1
            if args.per_document:
                target = output_dir / f"{document_id}.json"
                if target.is_file():
                    target.unlink()
            elif source_key in existing:
                del existing[source_key]
                _persist_merged()
            continue

        generated += 1
        if args.per_document:
            write_document(document, output_dir)
        else:
            existing[document.source_pdf] = document
            _persist_merged()
            logger.info(
                "  saved (%d documents in %s)",
                len(existing),
                merged_output.name if merged_output is not None else "?",
            )

    if not args.per_document and existing:
        ordered = merge_corpus_documents(existing, pdfs, corpus_dir)
        total_tables = sum(document.num_tables for document in ordered)
        logger.info(
            "Merged corpus at %s (%d documents, %d tables)",
            merged_output,
            len(ordered),
            total_tables,
        )

    logger.info(
        "Done: %d generated, %d skipped, %d omitted (0 tables), %d PDFs in this run, %d failures",
        generated,
        skipped,
        omitted_empty,
        len(pdfs),
        len(failures),
    )

    if failures:
        report = output_dir / "generation_failures.json"
        report.parent.mkdir(parents=True, exist_ok=True)
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
