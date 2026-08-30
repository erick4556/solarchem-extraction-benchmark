"""CLI: score a prediction JSON against Gold or a silver snapshot."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from solarchem_benchmark.eval.compare_gt import evaluate
from solarchem_benchmark.gt.schema import GroundTruthCorpus, GroundTruthDocument
from solarchem_benchmark.paths import (
    default_ground_truth_dir,
    default_merged_ground_truth_path,
    default_prediction_path,
    resolve_data_root,
)

logger = logging.getLogger("solarchem_benchmark.eval")


def _load_documents(path: Path) -> list[GroundTruthDocument]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "documents" in payload:
        return list(GroundTruthCorpus.model_validate(payload).documents)
    return [GroundTruthDocument.model_validate(payload)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solarchem-eval-gold",
        description=(
            "Evaluate a prediction JSON against a reference corpus. The "
            "reference is Gold by default; pass --reference to score a native "
            "extractor against the working silver snapshot. Only documents in "
            "the reference file are scored."
        ),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        help="Gold JSON (default: data/ground_truth/gold/ground_truth_gold_pilot.json).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Alias of --gold (e.g. the silver-302 snapshot). Cannot be combined with --gold.",
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        help=(
            "Prediction JSON to score. Defaults to data/predictions/<tool>.json "
            "when --tool is set, else data/ground_truth/ground_truth_<engine>.json."
        ),
    )
    parser.add_argument(
        "--tool",
        help="Native extractor id used only to pick data/predictions/<tool>.json.",
    )
    parser.add_argument(
        "--engine",
        default="lighton_ocr",
        help=(
            "OCR engine id used only to pick the default --prediction filename "
            "when --tool is not set (default: %(default)s → ground_truth_lighton_ocr.json)."
        ),
    )
    parser.add_argument(
        "--metrics",
        choices=["all", "structure"],
        default="all",
        help=(
            "all (default): structure + context. structure: detection, columns "
            "and cells only — use this for pdfplumber / Camelot / PyMuPDF / "
            "Docling / TATR / Unstructured. Use all for GROBID and Ollama."
        ),
    )
    parser.add_argument("--data-root", type=Path, help="Root of the data directory.")
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path to write the full JSON report.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-7s %(message)s",
    )

    if args.gold and args.reference:
        logger.error("Use either --gold or --reference, not both")
        return 1

    data_root = resolve_data_root(args.data_root)
    gt_dir = default_ground_truth_dir(data_root)
    reference_arg = args.gold or args.reference
    gold_path = (
        reference_arg.expanduser().resolve()
        if reference_arg
        else (gt_dir / "gold" / "ground_truth_gold_pilot.json").resolve()
    )
    if args.prediction:
        pred_path = args.prediction.expanduser().resolve()
    elif args.tool:
        pred_path = default_prediction_path(data_root, args.tool).resolve()
    else:
        pred_path = default_merged_ground_truth_path(data_root, args.engine).resolve()

    if not gold_path.is_file():
        logger.error("Reference file not found: %s", gold_path)
        logger.error("Copy data/ground_truth/gold/ from your laptop to the server first.")
        return 1
    if not pred_path.is_file():
        logger.error("Prediction file not found: %s", pred_path)
        return 1

    gold = _load_documents(gold_path)
    prediction = _load_documents(pred_path)
    report = evaluate(gold, prediction)
    summary = report.structure_summary() if args.metrics == "structure" else report.summary()

    logger.info("Reference:  %s (%d documents)", gold_path, len(gold))
    logger.info("Prediction: %s (%d documents)", pred_path, len(prediction))
    if args.metrics == "structure":
        logger.info("Metrics:    structure only (detection / columns / cells)")
    if summary["missing_from_prediction"]:
        logger.warning(
            "Reference PDFs absent from prediction (%d): %s",
            len(summary["missing_from_prediction"]),
            ", ".join(summary["missing_from_prediction"]),
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "documents": [
                {
                    "source_pdf": document.source_pdf,
                    "present_in_prediction": document.present_in_prediction,
                    "tables": [table.as_dict() for table in document.tables],
                }
                for document in report.documents
            ],
        }
        args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        logger.info("Full report written to %s", args.report)

    if summary["gold_documents_found_in_prediction"] == 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
