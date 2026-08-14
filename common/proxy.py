"""Shim — prefer ``from kaggle_lab.proxy import ...``."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kaggle_lab.proxy import (  # noqa: E402,F401
    DEFAULT_MODEL,
    ENV_FILE,
    ROOT,
    chat,
    chat_messages,
    ensure_kaggle_token,
    load_env,
    make_client,
    print_usage,
    refresh_token,
)
