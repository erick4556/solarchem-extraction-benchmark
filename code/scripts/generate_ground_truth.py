#!/usr/bin/env python3
"""Entry point for ground-truth generation.

Usable without installing the package, which is convenient on a server where
the repository is checked out under ``code/``::

    python scripts/generate_ground_truth.py --limit 1 --max-pages 8
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solarchem_benchmark.gt.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
