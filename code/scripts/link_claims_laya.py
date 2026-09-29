#!/usr/bin/env python3
"""Laya trial. One JSON: link claims, read hidden cells, score. Kept apart from extractor eval.

    python scripts/link_claims_laya.py run --pdf paper.pdf --device cuda
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solarchem_benchmark.claims.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
