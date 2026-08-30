#!/usr/bin/env python3
"""Extract tables into the benchmark schema (Phase 5–7).

    python scripts/extract_native_tables.py --tool pdfplumber --limit 1
    python scripts/extract_native_tables.py --tool ollama_qwen3_vl \
        --reference ../data/ground_truth/gold/ground_truth_gold_pilot.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solarchem_benchmark.extractors.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
