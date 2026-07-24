#!/usr/bin/env python3
"""通过 Kaggle Model Proxy 调用 AI Models（扣 Daily $10 / Monthly $100 额度）。

用法:
  source .venv/bin/activate
  export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"

  # 刷新临时凭证（约 1 小时过期）
  kaggle b auth -y --env-file .env.model-proxy

  python call_ai_model.py "你好，介绍一下你自己"
  python call_ai_model.py -m openai/gpt-5.4-nano-2026-03-17 "1+1=?"
  python call_ai_model.py -m google/gemini-3-flash-preview --max-tokens 256 "写一首短诗"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

LAB = Path(__file__).resolve().parent
ENV_FILE = LAB / ".env.model-proxy"
DEFAULT_MODEL = "google/gemini-3-flash-preview"


def load_env(path: Path) -> None:
    if not path.exists():
        raise SystemExit(
            f"缺少 {path}。请先运行:\n"
            f"  kaggle b auth -y --env-file {path}"
        )
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ[key.strip()] = val.strip().strip('"').strip("'")


def refresh_token() -> None:
    """用 KGAT 重新申请 Model Proxy 临时 key。"""
    env = os.environ.copy()
    if "KAGGLE_API_TOKEN" not in env:
        token_path = Path.home() / ".kaggle" / "access_token"
        if token_path.exists():
            env["KAGGLE_API_TOKEN"] = token_path.read_text().strip()
    subprocess.check_call(
        ["kaggle", "b", "auth", "-y", "--env-file", str(ENV_FILE)],
        env=env,
    )


def make_client() -> OpenAI:
    load_env(ENV_FILE)
    base = os.environ["MODEL_PROXY_URL"].rstrip("/")
    # OpenAI 兼容端点：{MODEL_PROXY_URL}/openapi
    return OpenAI(
        api_key=os.environ["MODEL_PROXY_API_KEY"],
        base_url=f"{base}/openapi",
    )


def chat(model: str, prompt: str, max_tokens: int = 256) -> str:
    client = make_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message.content or ""
    usage = resp.usage
    if usage is not None:
        cost = getattr(usage, "cost", None) or {}
        if isinstance(cost, dict):
            total_nano = (cost.get("input_tokens_cost_nanodollars") or 0) + (
                cost.get("output_tokens_cost_nanodollars") or 0
            )
            print(
                f"[usage] prompt={usage.prompt_tokens} "
                f"completion={usage.completion_tokens} "
                f"cost≈${total_nano / 1e9:.8f}",
                file=sys.stderr,
            )
        else:
            print(
                f"[usage] prompt={usage.prompt_tokens} "
                f"completion={usage.completion_tokens}",
                file=sys.stderr,
            )
    return msg


def main() -> None:
    parser = argparse.ArgumentParser(description="调用 Kaggle AI Models")
    parser.add_argument("prompt", help="用户提示词")
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"模型名（默认 {DEFAULT_MODEL}）",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="调用前先刷新 Model Proxy 凭证",
    )
    args = parser.parse_args()

    if args.refresh or not ENV_FILE.exists():
        refresh_token()

    try:
        print(chat(args.model, args.prompt, args.max_tokens))
    except Exception as exc:
        # 凭证过期时自动刷新一次再试
        if "401" in str(exc) or "403" in str(exc) or "auth" in str(exc).lower():
            print("凭证可能过期，正在刷新…", file=sys.stderr)
            refresh_token()
            print(chat(args.model, args.prompt, args.max_tokens))
        else:
            raise


if __name__ == "__main__":
    main()
