#!/usr/bin/env python3
"""Backward-compatible entry. Prefer: ``python -m kaggle_lab``.

Old:
  python main.py 001 auth
  python main.py 014 run
  python main.py 015 run --preset debate --rounds 3

New (same thing, named after intent):
  python -m kaggle_lab auth
  python -m kaggle_lab workforce
  python -m kaggle_lab debate --preset debate --rounds 3
  python -m kaggle_lab list
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
