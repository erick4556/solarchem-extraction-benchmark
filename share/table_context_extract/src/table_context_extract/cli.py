"""CLI: extract tables and in-text mentions from PDFs with LightOnOCR."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from table_context_extract.generate import derive_document_id, generate_for_pdf, write_document
from table_context_extract.ocr import build_engine
from table_context_extract.schema import GroundTruthCorpus, GroundTruthDocument

logger = logging.getLogger("table_context_extract")

_DEFAULT_OUTPUT = Path("output") / "extracted.json"
_DEFAULT_CACHE = Path("output") / "ocr_cache"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract-tables",
        description=(
            "Extract tables (grid + caption) and the paragraphs that mention "
            "them from scientific PDFs, using LightOnOCR-2."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", type=Path, help="Process a single PDF.")
    source.add_argument("--input", type=Path, help="Directory of PDFs to process.")
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=(
            "JSON file for the merged corpus (default: %(default)s), "
            "or a directory with --per-document."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE,
        help="Directory for cached OCR pages (default: %(default)s).",
    )
    parser.add_argument(
        "--document-id",
        help="Explicit document identifier; only valid together with --pdf.",
    )
    parser.add_argument("--model-id", help="Override the Hugging Face model id.")
    parser.add_argument("--limit", type=int, help="Process at most this many PDFs.")
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Process at most this many pages per PDF (smoke tests).",
    )
    parser.add_argument(
        "--per-document",
        action="store_true",
        help="Write one JSON per document instead of one merged file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate documents already present in the output.",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Ignore cached transcriptions and re-run LightOnOCR.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable OCR caching.")
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep PDFs from which no table was recovered (default: omit them).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s).",
    )
    return parser


def source_pdf_key(pdf_path: Path, corpus_dir: Path) -> str:
    try:
        return pdf_path.resolve().relative_to(corpus_dir.resolve()).as_posix()
    except ValueError:
        return pdf_path.name


def load_merged_documents(path: Path) -> dict[str, GroundTruthDocument]:
    if not path.is_file():
        return {}
    try:
        corpus = GroundTruthCorpus.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        logger.warning("Could not load existing output at %s: %s", path, error)
        return {}
    return {document.source_pdf: document for document in corpus.documents}


def load_document_file(path: Path) -> GroundTruthDocument | None:
    if not path.is_file():
        return None
    try:
        return GroundTruthDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        logger.warning("Could not load existing output at %s: %s", path, error)
        return None


def merge_corpus_documents(
    existing: dict[str, GroundTruthDocument],
    pdfs: list[Path],
    corpus_dir: Path,
) -> list[GroundTruthDocument]:
    current_keys = [source_pdf_key(pdf, corpus_dir) for pdf in pdfs]
    current_set = set(current_keys)
    ordered = [existing[key] for key in current_keys if key in existing]
    ordered.extend(document for key, document in existing.items() if key not in current_set)
    return ordered


def _collect_pdfs(args: argparse.Namespace) -> tuple[list[Path], Path]:
    if args.pdf is not None:
        pdf = args.pdf.expanduser().resolve()
        if not pdf.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf}")
        return [pdf], pdf.parent

    corpus_dir = args.input.expanduser().resolve()
    if not corpus_dir.is_dir():
        raise NotADirectoryError(f"Documents directory not found: {corpus_dir}")
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

    if args.document_id and args.pdf is None:
        logger.error("--document-id requires --pdf")
        return 1

    output_path = args.output.expanduser().resolve()
    if args.per_document:
        output_dir = output_path
        if output_dir.suffix.lower() == ".json":
            logger.error("--per-document expects --output to be a directory, not a JSON file")
            return 1
        merged_output = None
    else:
        merged_output = output_path
        output_dir = merged_output.parent
        if merged_output.suffix.lower() != ".json":
            merged_output = merged_output / "extracted.json"
            output_dir = merged_output.parent

    cache_dir = None if args.no_cache else args.cache_dir.expanduser().resolve()

    try:
        pdfs, corpus_dir = _collect_pdfs(args)
    except (FileNotFoundError, NotADirectoryError) as error:
        logger.error("%s", error)
        return 1

    existing = {} if args.per_document else load_merged_documents(merged_output)  # type: ignore[arg-type]
    if existing and not args.overwrite:
        logger.info("Resuming from %s (%d documents already present)", merged_output, len(existing))

    engine = build_engine(args.model_id)

    logger.info("PDFs:       %s (%d)", corpus_dir, len(pdfs))
    if args.per_document:
        logger.info("Output:     %s (one file per document)", output_dir)
    else:
        logger.info("Output:     %s", merged_output)
    logger.info("OCR cache:  %s", cache_dir or "disabled")
    logger.info("Engine:     lighton_ocr")

    def _persist_merged() -> None:
        assert merged_output is not None
        ordered = merge_corpus_documents(existing, pdfs, corpus_dir)
        merged_output.parent.mkdir(parents=True, exist_ok=True)
        corpus = GroundTruthCorpus(documents=ordered)
        merged_output.write_text(corpus.model_dump_json(indent=2) + "\n", encoding="utf-8")

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
                    if kept.num_tables == 0 and not args.keep_empty:
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
        except ImportError as error:
            logger.error(
                "LightOnOCR cannot be imported. Install transformers>=5.0, torch "
                "and accelerate. %s",
                error,
            )
            return 1
        except Exception as error:  # noqa: BLE001
            logger.exception("  failed: %s", pdf.name)
            failures.append((pdf, str(error)))
            continue

        if document.num_tables == 0 and not args.keep_empty:
            logger.info("  no tables recovered, omitting: %s", pdf.name)
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
            "Wrote %s (%d documents, %d tables)",
            merged_output,
            len(ordered),
            total_tables,
        )

    logger.info(
        "Done: %d generated, %d skipped, %d omitted (0 tables), %d PDFs, %d failures",
        generated,
        skipped,
        omitted_empty,
        len(pdfs),
        len(failures),
    )

    if failures:
        report = output_dir / "extraction_failures.json"
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
