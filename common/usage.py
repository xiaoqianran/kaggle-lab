"""Shim — prefer ``from kaggle_lab.usage import ...``."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kaggle_lab.usage import (  # noqa: E402,F401
    USAGE_LOG,
    load_usage_rows,
    log_from_openai_usage,
    log_usage,
    summarize_usage,
)
