"""CLI: export LightOn silver JSON to Turtle (``sc:`` instances)."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from solarchem_benchmark.gt.schema import GroundTruthCorpus, GroundTruthDocument
from solarchem_benchmark.paths import (
    default_rdf_dir,
    default_working_silver_path,
    resolve_data_root,
)
from solarchem_benchmark.rdf.export import (
    corpus_to_turtle,
    is_gold_path,
    ontology_path,
)

logger = logging.getLogger("solarchem_benchmark.rdf")


def _load_documents(path: Path) -> list[GroundTruthDocument]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "documents" in payload:
        return list(GroundTruthCorpus.model_validate(payload).documents)
    return [GroundTruthDocument.model_validate(payload)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solarchem-export-rdf",
        description=(
            "Export the LightOn silver ground truth to Turtle using the sc: "
            "alignment ontology. Gold is refused: it was only used to select "
            "the OCR engine, not as the published graph."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "Silver JSON (default: data/ground_truth/ground_truth_lighton_ocr_302.json). "
            "Must not be the Gold pilot."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Turtle instances (default: data/rdf/silver302.ttl).",
    )
    parser.add_argument("--data-root", type=Path, help="Root of the data directory.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Export only the first N documents (0 = all).",
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

    data_root = resolve_data_root(args.data_root)
    input_path = (
        args.input.expanduser().resolve()
        if args.input
        else default_working_silver_path(data_root).resolve()
    )
    if is_gold_path(input_path):
        logger.error(
            "Refusing Gold (%s). RDF is built from LightOn silver, not from "
            "the OCR-selection pilot.",
            input_path,
        )
        return 1
    if not input_path.is_file():
        logger.error("Silver JSON not found: %s", input_path)
        return 1

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else (default_rdf_dir(data_root) / "silver302.ttl").resolve()
    )

    documents = _load_documents(input_path)
    if args.limit and args.limit > 0:
        documents = documents[: args.limit]

    turtle = corpus_to_turtle(documents, source_label=input_path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(turtle, encoding="utf-8")

    ontology_dest = output_path.parent / "solarchem-alignment.ttl"
    shutil.copyfile(ontology_path(), ontology_dest)

    logger.info("Documents: %d", len(documents))
    logger.info("Instances: %s", output_path)
    logger.info("Ontology:  %s", ontology_dest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
