#!/usr/bin/env python3
"""Entry point for ground-truth validation.

    python scripts/validate_ground_truth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solarchem_benchmark.gt.validate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
