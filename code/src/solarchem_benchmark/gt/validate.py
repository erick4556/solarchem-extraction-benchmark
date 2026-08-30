"""Validation and summary reporting for generated ground-truth files."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from solarchem_benchmark.gt.context import caption_title
from solarchem_benchmark.gt.schema import GroundTruthCorpus, GroundTruthDocument, json_schema
from solarchem_benchmark.paths import default_ground_truth_dir, resolve_data_root

logger = logging.getLogger("solarchem_benchmark.gt.validate")


def validate_file(path: Path) -> list[GroundTruthDocument]:
    """Load and validate one ground-truth file.

    Both layouts the generator writes are accepted: the merged corpus file,
    which wraps a ``documents`` list, and a single-document file.

    Args:
        path: Path to the JSON file.

    Returns:
        The validated documents it holds.

    Raises:
        pydantic.ValidationError: If the file does not match the schema.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "documents" in payload:
        return list(GroundTruthCorpus.model_validate(payload).documents)
    return [GroundTruthDocument.model_validate(payload)]


def summarise(documents: list[GroundTruthDocument]) -> dict[str, object]:
    """Compute corpus-level counts over validated documents."""
    tables = [table for document in documents for table in document.tables]
    return {
        "documents": len(documents),
        "tables": len(tables),
        "documents_without_tables": sum(1 for d in documents if d.num_tables == 0),
        "documents_with_title": sum(1 for d in documents if d.title),
        "coverage": {
            "caption": sum(1 for table in tables if table.caption),
            "caption_with_title": sum(1 for table in tables if caption_title(table.caption)),
            "table_label": sum(1 for table in tables if table.table_label),
            "section_title": sum(1 for table in tables if table.context.section_title),
            "mentions": sum(1 for table in tables if table.context.mentions),
        },
        "mean_rows_per_table": (
            round(sum(len(table.rows) for table in tables) / len(tables), 2) if tables else 0
        ),
        "mean_columns_per_table": (
            round(sum(len(table.columns) for table in tables) / len(tables), 2) if tables else 0
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="solarchem-gt-validate",
        description="Validate generated ground-truth files against the schema.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Ground-truth file or directory (defaults to the ground-truth directory).",
    )
    parser.add_argument("--data-root", type=Path, help="Root of the data directory.")
    parser.add_argument(
        "--dump-schema",
        type=Path,
        help="Write the JSON Schema to this path and exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate ground-truth files.

    Returns:
        ``0`` when every file is valid, ``1`` otherwise.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-7s %(message)s",
    )

    if args.dump_schema:
        args.dump_schema.parent.mkdir(parents=True, exist_ok=True)
        args.dump_schema.write_text(
            json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Schema written to %s", args.dump_schema)
        return 0

    target = args.input or default_ground_truth_dir(resolve_data_root(args.data_root))
    target = target.expanduser().resolve()

    if target.is_file():
        paths = [target]
    elif target.is_dir():
        paths = sorted(p for p in target.glob("*.json") if p.name != "generation_failures.json")
    else:
        logger.error("Not found: %s", target)
        return 1

    if not paths:
        logger.error("No ground-truth files found in %s", target)
        return 1

    documents: list[GroundTruthDocument] = []
    invalid = 0
    valid_files = 0
    for path in paths:
        try:
            documents.extend(validate_file(path))
            valid_files += 1
        except (ValidationError, json.JSONDecodeError) as error:
            invalid += 1
            logger.error("%s: %s", path.name, error)

    logger.info("Validated %d/%d files (%d documents)", valid_files, len(paths), len(documents))
    print(json.dumps(summarise(documents), indent=2, ensure_ascii=False))
    return 1 if invalid else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
