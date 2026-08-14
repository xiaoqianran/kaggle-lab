"""CAMEL ↔ Kaggle Model Proxy bridge.

Uses ModelPlatformType.OPENAI_COMPATIBLE_MODEL against:
  {MODEL_PROXY_URL}/openapi  +  MODEL_PROXY_API_KEY

Requires: camel-ai, and mcp==1.9.x (FastMCP; mcp 2.x breaks camel imports).
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from kaggle_lab.proxy import (
    DEFAULT_MODEL,
    ENV_FILE,
    ensure_kaggle_token,
    load_env,
    refresh_token,
)

DEFAULT_CAMEL_MODEL = DEFAULT_MODEL


def ensure_proxy(refresh: bool = False) -> None:
    ensure_kaggle_token()
    if refresh or not ENV_FILE.exists():
        refresh_token()
    load_env()


def proxy_url_and_key() -> tuple[str, str]:
    ensure_proxy(False)
    base = os.environ["MODEL_PROXY_URL"].rstrip("/")
    key = os.environ["MODEL_PROXY_API_KEY"]
    return f"{base}/openapi", key


def make_camel_model(
    model_type: str = DEFAULT_CAMEL_MODEL,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    refresh: bool = False,
    model_config_dict: Optional[dict[str, Any]] = None,
):
    """Return a CAMEL BaseModelBackend wired to Kaggle Model Proxy."""
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    ensure_proxy(refresh)
    url, key = proxy_url_and_key()
    cfg = {"temperature": temperature, "max_tokens": max_tokens}
    if model_config_dict:
        cfg.update(model_config_dict)
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=model_type,
        api_key=key,
        url=url,
        model_config_dict=cfg,
        max_retries=3,
    )


def make_chat_agent(
    system_message: str,
    *,
    model_type: str = DEFAULT_CAMEL_MODEL,
    tools: Optional[Sequence[Any]] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    refresh: bool = False,
    output_language: Optional[str] = None,
):
    from camel.agents import ChatAgent

    model = make_camel_model(
        model_type,
        temperature=temperature,
        max_tokens=max_tokens,
        refresh=refresh,
    )
    kwargs: dict[str, Any] = {
        "system_message": system_message,
        "model": model,
    }
    if tools:
        kwargs["tools"] = list(tools)
    if output_language:
        kwargs["output_language"] = output_language
    return ChatAgent(**kwargs)


def extract_text(response: Any) -> str:
    """Best-effort text from ChatAgent.step / RolePlaying turn."""
    if response is None:
        return ""
    msgs = getattr(response, "msgs", None)
    if msgs:
        parts = []
        for m in msgs:
            c = getattr(m, "content", None)
            if c:
                parts.append(str(c))
        if parts:
            return "\n".join(parts)
    if isinstance(response, (tuple, list)) and response:
        return extract_text(response[0])
    return str(response)
