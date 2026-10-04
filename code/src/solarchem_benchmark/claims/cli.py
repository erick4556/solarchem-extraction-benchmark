"""CLI for the Laya claims trial.

One command writes the whole run into a single JSON file::

    solarchem-link-claims run --pdf paper.pdf --device cuda

``link`` and ``score`` still run one stage each.
Invoking the script with flags and no command still runs ``link``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

from solarchem_benchmark.claims.link import LinkConfig, run_document
from solarchem_benchmark.claims.model import CHECKPOINTS, LayaModel, UniformModel
from solarchem_benchmark.claims.score import build_scores, load_labels, pool_link_evaluations
from solarchem_benchmark.claims.text import document_sentences, load_pages
from solarchem_benchmark.gt.schema import GroundTruthCorpus, GroundTruthDocument
from solarchem_benchmark.paths import (
    default_ocr_cache_dir,
    default_predictions_dir,
    default_working_silver_path,
    resolve_data_root,
)

logger = logging.getLogger("solarchem_benchmark.claims")

PILOT_PAPERS = Path("analysis/kg_pilot_10/papers")
LABELS_DIR = Path("analysis/kg_pilot_10/claims_labels")
COMMANDS = {"run", "link", "score"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solarchem-link-claims",
        description=(
            "Laya trial: classify claims and link them to silver tables. "
            "Does not run the extractor evaluation."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Link claims to tables and score, in one step.")
    _add_run_args(run)

    link = sub.add_parser("link", help="Classify claims and link them to silver tables.")
    _add_run_args(link)

    score = sub.add_parser("score", help="Add the scores block to the run file.")
    score.add_argument("--data-root", type=Path, help="Root of the data directory.")
    score.add_argument("--checkpoint", default="english", choices=sorted(CHECKPOINTS), help=argparse.SUPPRESS)
    score.add_argument("--output-dir", type=Path, help="Run file or its directory (default: claims_laya/laya-results.json).")
    score.add_argument(
        "--labels",
        type=Path,
        help="Label CSV or directory (default: <data>/analysis/kg_pilot_10/claims_labels).",
    )
    score.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, help="Root of the data directory.")
    parser.add_argument("--output-dir", type=Path, help="Directory or JSON path. Default: <data>/predictions/claims_laya/laya-results.json.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def _add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", default="english", choices=sorted(CHECKPOINTS))
    parser.add_argument("--model-path", help="Local or Hub path of a fine-tuned Laya checkpoint.")
    parser.add_argument("--device", help="cuda, cpu or mps (default: Laya picks).")
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--head-max-len", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Uniform probabilities, no Laya.")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    _add_shared(parser)
    _add_model(parser)
    _add_papers(parser)
    parser.add_argument("--claim-threshold", type=float, default=0.5)
    parser.add_argument("--link-threshold", type=float, default=0.0)
    parser.add_argument("--no-mask-references", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--only-referenced", action="store_true")
    parser.add_argument(
        "--labels",
        type=Path,
        help="Label CSV or directory (default: <data>/analysis/kg_pilot_10/claims_labels).",
    )


def _add_papers(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pdf", type=Path, action="append", help="PDF to process (repeatable).")
    parser.add_argument("--papers-dir", type=Path, help=f"Directory of PDFs (default: <data>/{PILOT_PAPERS}).")
    parser.add_argument("--match", help="Only PDFs whose file name contains this text.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N PDFs.")
    parser.add_argument("--silver", type=Path, help="Silver JSON with the tables (default: LightOn 302).")
    parser.add_argument("--ocr-cache", type=Path)
    parser.add_argument("--no-ocr-cache", action="store_true")


def _load_silver(path: Path) -> dict[str, GroundTruthDocument]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "documents" in payload:
        documents = GroundTruthCorpus.model_validate(payload).documents
    else:
        documents = [GroundTruthDocument.model_validate(payload)]
    return {Path(document.source_pdf).name: document for document in documents}


def _collect_pdfs(args: argparse.Namespace, data_root: Path) -> list[Path]:
    if args.pdf:
        pdfs = [path.expanduser().resolve() for path in args.pdf]
    else:
        directory = (args.papers_dir or data_root / PILOT_PAPERS).expanduser().resolve()
        pdfs = sorted(directory.glob("*.pdf"))
    if args.match:
        pdfs = [pdf for pdf in pdfs if args.match.lower() in pdf.name.lower()]
    if args.limit and args.limit > 0:
        pdfs = pdfs[: args.limit]
    return pdfs


def _build_model(args: argparse.Namespace):
    if args.dry_run:
        return UniformModel()
    return LayaModel(
        args.checkpoint,
        model_path=args.model_path,
        device=args.device,
        max_len=args.max_len,
        head_max_len=args.head_max_len,
        batch_size=args.batch_size,
        fast=args.fast,
    )


RESULTS_FILE = "laya-results.json"


def _results_filename(model_name: str) -> str:
    """One results file. The dry-run stand-in keeps its own name."""
    if model_name == UniformModel.name:
        return f"{model_name}.json"
    return RESULTS_FILE


def _run_file(args: argparse.Namespace, data_root: Path, model_name: str) -> Path:
    """One JSON for the whole run. A directory argument gets the results file inside it."""
    filename = _results_filename(model_name)
    if args.output_dir:
        path = args.output_dir.expanduser().resolve()
        if path.suffix == ".json":
            return path
        return path / filename
    return default_predictions_dir(data_root) / "claims_laya" / filename


def _load_run(path: Path) -> dict:
    if not path.is_file():
        return {"documents": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("documents", [])
    return payload


def _write_run(path: Path, run: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")


def _upsert(run: dict, document_id: str, update: dict) -> None:
    """Replace one document, keeping sentences already stored by ``link``."""
    previous = next((doc for doc in run["documents"] if doc.get("document_id") == document_id), None)
    merged = dict(previous or {})
    if previous and "sentences" in previous and "sentences" not in update:
        merged.update(update)
        update = merged
    update["document_id"] = document_id
    run["documents"] = [doc for doc in run["documents"] if doc.get("document_id") != document_id]
    run["documents"].append(update)


def _label_paths(data_root: Path, explicit: Path | None) -> list[Path]:
    if explicit is not None:
        return [explicit.expanduser().resolve()]
    directory = data_root / LABELS_DIR
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.csv"))


def _prepare_papers(args: argparse.Namespace):
    data_root = resolve_data_root(args.data_root)
    silver_path = (args.silver or default_working_silver_path(data_root)).expanduser().resolve()
    if not silver_path.is_file():
        logger.error("Silver JSON not found: %s", silver_path)
        return None
    pdfs = _collect_pdfs(args, data_root)
    if not pdfs:
        logger.error("No PDFs selected.")
        return None
    return data_root, silver_path, _load_silver(silver_path), pdfs


def _cmd_link(args: argparse.Namespace) -> int:
    prepared = _prepare_papers(args)
    if prepared is None:
        return 1
    data_root, silver_path, silver, pdfs = prepared
    ocr_cache = None
    if not args.no_ocr_cache:
        ocr_cache = (args.ocr_cache or default_ocr_cache_dir(data_root, "lighton_ocr")).expanduser().resolve()
        if not ocr_cache.is_dir():
            logger.warning("OCR cache not found (%s); using the PDF text layer.", ocr_cache)
            ocr_cache = None

    model = _build_model(args)
    config = LinkConfig(
        claim_threshold=args.claim_threshold,
        link_threshold=args.link_threshold,
        mask_references=not args.no_mask_references,
        max_candidates=args.max_candidates,
        only_referenced=args.only_referenced,
    )
    run_path = _run_file(args, data_root, model.name)
    run = _load_run(run_path)
    results = []
    for pdf in pdfs:
        document = silver.get(pdf.name)
        if document is None:
            logger.warning("Not in silver, skipped (no tables): %s", pdf.name)
            continue
        if not document.tables:
            logger.warning("No tables in silver, skipped: %s", pdf.name)
            continue
        started = time.perf_counter()
        pages, text_source = load_pages(pdf, ocr_cache_dir=ocr_cache)
        sentences = document_sentences(pages, source=text_source, document_id=document.document_id)
        result = run_document(document, sentences, model, config)
        result["text_source"] = text_source
        result["seconds"] = round(time.perf_counter() - started, 2)
        _upsert(run, document.document_id, result)
        counts, evaluation = result["counts"], result["evaluation"]
        logger.info(
            "%s [%s] claims=%d links=%d laya_top1=%s row_top1=%s (n=%d, row_n=%d) %.1fs",
            document.document_id, text_source, counts["claims"], counts["links"],
            evaluation.get("laya_top1"), evaluation.get("row_top1"),
            evaluation.get("evaluable_sentences", 0), evaluation.get("row_evaluable", 0),
            result["seconds"],
        )
        results.append(result)

    if not results:
        logger.error("Nothing processed.")
        return 1

    run["model"] = model.describe()
    run["config"] = asdict(config)
    run["silver"] = str(silver_path)
    run["evaluation"] = pool_link_evaluations([r["evaluation"] for r in results])
    _write_run(run_path, run)
    logger.info("Pooled self-check: %s", run["evaluation"])
    logger.info("Output: %s", run_path)
    return 0


def _score_run_file(args: argparse.Namespace, data_root: Path) -> Path | None:
    if args.output_dir:
        path = args.output_dir.expanduser().resolve()
        if path.suffix == ".json":
            return path if path.is_file() else None
        if not path.is_dir():
            return None
        preferred = path / RESULTS_FILE
        if preferred.is_file():
            return preferred
        found = sorted(path.glob("*.json"))
        return found[0] if len(found) == 1 else None
    path = default_predictions_dir(data_root) / "claims_laya" / RESULTS_FILE
    return path if path.is_file() else None


def _cmd_score(args: argparse.Namespace) -> int:
    data_root = resolve_data_root(args.data_root)
    run_path = _score_run_file(args, data_root)
    if run_path is None:
        logger.error("Run file not found. Run link first.")
        return 1
    run = _load_run(run_path)
    documents = [doc for doc in run["documents"] if doc.get("sentences")]
    if not documents:
        logger.error("No linked documents in %s. Run link first.", run_path)
        return 1
    label_paths = _label_paths(data_root, args.labels)
    scores = build_scores(
        documents,
        labels=load_labels(label_paths),
        label_paths=[str(path) for path in label_paths],
    )
    run["scores"] = scores
    _write_run(run_path, run)
    logger.info("Link: %s", scores["link"])
    logger.info(
        "Claims: %s",
        {
            k: scores["claims"][k]
            for k in ("labeled", "precision", "recall", "f1", "table_accuracy", "no_table_accuracy", "text_mismatches")
        },
    )
    logger.info("Output: %s", run_path)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Link claims to tables, then score, into the same JSON file."""
    data_root = resolve_data_root(args.data_root)
    args.output_dir = _run_file(args, data_root, _build_model(args).name)
    for step in (_cmd_link, _cmd_score):
        code = step(args)
        if code:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS | {"-h", "--help"}:
        argv = ["link", *argv]
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)-7s %(message)s")
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "score":
        return _cmd_score(args)
    return _cmd_link(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
