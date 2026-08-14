"""Shim — prefer ``from kaggle_lab.camel_proxy import ...``."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kaggle_lab.camel_proxy import (  # noqa: E402,F401
    DEFAULT_CAMEL_MODEL,
    ensure_proxy,
    extract_text,
    make_camel_model,
    make_chat_agent,
    proxy_url_and_key,
)
