"""Kaggle Model Proxy helpers (Daily $10 / Monthly $100 AI Models)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env.model-proxy"
DEFAULT_MODEL = "google/gemini-3-flash-preview"


def ensure_kaggle_token() -> None:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    token_path = Path.home() / ".kaggle" / "access_token"
    if token_path.exists():
        os.environ["KAGGLE_API_TOKEN"] = token_path.read_text().strip()


def load_env(path: Path | None = None) -> None:
    path = path or ENV_FILE
    if not path.exists():
        raise SystemExit(
            f"缺少 {path}。请先运行:\n"
            f"  python main.py 001 auth\n"
            f"  # 或: kaggle b auth -y --env-file {path}"
        )
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ[key.strip()] = val.strip().strip('"').strip("'")


def refresh_token(env_file: Path | None = None) -> Path:
    """用 KGAT 重新申请 Model Proxy 临时 key（约 1 小时过期）。"""
    env_file = env_file or ENV_FILE
    ensure_kaggle_token()
    env = os.environ.copy()
    subprocess.check_call(
        ["kaggle", "b", "auth", "-y", "--env-file", str(env_file)],
        env=env,
    )
    return env_file


def make_client(refresh: bool = False) -> OpenAI:
    if refresh or not ENV_FILE.exists():
        refresh_token()
    load_env()
    base = os.environ["MODEL_PROXY_URL"].rstrip("/")
    return OpenAI(
        api_key=os.environ["MODEL_PROXY_API_KEY"],
        base_url=f"{base}/openapi",
    )


def print_usage(usage, stream=None) -> None:
    stream = stream or sys.stderr
    if usage is None:
        return
    cost = getattr(usage, "cost", None) or {}
    if isinstance(cost, dict):
        total_nano = (cost.get("input_tokens_cost_nanodollars") or 0) + (
            cost.get("output_tokens_cost_nanodollars") or 0
        )
        print(
            f"[usage] prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} "
            f"cost≈${total_nano / 1e9:.8f}",
            file=stream,
        )
    else:
        print(
            f"[usage] prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens}",
            file=stream,
        )


def chat(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 256,
    refresh: bool = False,
) -> str:
    client = make_client(refresh=refresh)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
    except Exception as exc:
        if "401" in str(exc) or "403" in str(exc) or "auth" in str(exc).lower():
            print("凭证可能过期，正在刷新…", file=sys.stderr)
            client = make_client(refresh=True)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
        else:
            raise
    print_usage(resp.usage)
    return resp.choices[0].message.content or ""
