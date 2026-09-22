#!/usr/bin/env python3
"""Run table + mention extraction without installing the package.

    python extract.py --pdf paper.pdf
    python extract.py --input ./pdfs --output ./output/extracted.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from table_context_extract.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
