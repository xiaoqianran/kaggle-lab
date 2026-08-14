#!/usr/bin/env python3
"""013-camel-kaggle-crew — CAMEL 工具智能体 + 真实 Kaggle 信号（MCP/CLI）。

有意义默认任务：搜 LLM 相关竞赛 → 结合本机 GPU 配额 → 输出「本周可做」简报。

Usage:
  python main.py 013 scout
  python main.py 013 scout --q "llm agent" --n 5
  python main.py 013 ask "根据我的 GPU 余量和热门 LLM 竞赛，推荐一个周末 project"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.camel_proxy import (  # noqa: E402
    DEFAULT_CAMEL_MODEL,
    extract_text,
    make_chat_agent,
)
from common.proxy import ensure_kaggle_token  # noqa: E402
from common.usage import log_usage  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
EXP = "013-camel-kaggle-crew"

REAL_CMDS = """
本仓库真实命令（只能推荐这些，禁止编造不存在的脚本名）:
- python main.py 001 auth|chat
- python main.py 003 list|dump
- python main.py 005 auth|validate|push|run|download
- python main.py 006 run|show
- python main.py 007 tools|competitions|datasets|profile
- python main.py 008 run [--with-mcp]
- python main.py 010 show|gpu|usage
- python main.py 011 smoke|chat|tools
- python main.py 012 run [--rounds N]
- python main.py 013 scout|ask
禁止: SAE 开考 (004 start/submit)。notebook 训练在 notebooks/ 目录，另算 GPU 小时。
"""

SYSTEM = (
    "You are the Kaggle Lab Crew lead agent with tools for competitions, "
    "datasets, GPU quota, and model hints.\n"
    "Always call tools when facts are needed — never invent quota numbers or "
    "competition titles.\n"
    f"{REAL_CMDS}\n"
    "Write the final answer in clear Chinese Markdown with sections:\n"
    "结论 / 依据（工具结果摘要）/ 本周可执行步骤（真实命令）/ 额度风险.\n"
    "Be concrete and actionable."
)


def _mcp_call(name: str, arguments: dict[str, Any]) -> Any:
    import importlib.util

    path = ROOT / "labs" / "007-mcp-harness" / "run.py"
    spec = importlib.util.spec_from_file_location("mcp007", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.mcp_request(
        "tools/call",
        {"name": name, "arguments": arguments},
        req_id=50,
    )
    return mod._unwrap_text(r)


def build_tools():
    from camel.toolkits import FunctionTool

    def search_kaggle_competitions(query: str, n: int = 5) -> str:
        """Search Kaggle competitions by keyword. Returns JSON summary."""
        try:
            data = _mcp_call(
                "search_competitions",
                {"request": {"search": query, "pageSize": int(n)}},
            )
        except SystemExit as e:
            return json.dumps({"error": str(e)})
        comps = []
        raw = data.get("competitions") if isinstance(data, dict) else None
        if isinstance(raw, list):
            for c in raw[: int(n)]:
                comps.append(
                    {
                        "title": c.get("title"),
                        "url": c.get("url") or c.get("ref"),
                        "reward": c.get("reward"),
                        "deadline": c.get("deadline"),
                        "teamCount": c.get("team_count") or c.get("teamCount"),
                        "description": (c.get("description") or "")[:240],
                    }
                )
        else:
            return json.dumps(data, ensure_ascii=False)[:4000]
        return json.dumps({"query": query, "competitions": comps}, ensure_ascii=False)

    def get_gpu_quota() -> str:
        """Return current Kaggle GPU/TPU weekly accelerator quota as text."""
        ensure_kaggle_token()
        try:
            out = subprocess.check_output(
                ["kaggle", "quota"],
                text=True,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            return out.strip()
        except Exception as e:
            return f"error: {e}"

    def list_proxy_model_hints() -> str:
        """Return shortlist of cheap Model Proxy models for agents."""
        return json.dumps(
            {
                "cheap_flash": [
                    "google/gemini-3-flash-preview",
                    "google/gemini-3.5-flash",
                    "google/gemini-3.5-flash-lite",
                    "openai/gpt-5.4-nano-2026-03-17",
                ],
                "daily_budget_usd": 10,
                "monthly_budget_usd": 100,
            }
        )

    def search_kaggle_datasets(query: str, n: int = 3) -> str:
        """Search Kaggle datasets by keyword."""
        try:
            data = _mcp_call(
                "search_datasets",
                {"request": {"search": query, "pageSize": int(n)}},
            )
        except SystemExit as e:
            return json.dumps({"error": str(e)})
        return json.dumps(data, ensure_ascii=False)[:4000]

    return [
        FunctionTool(search_kaggle_competitions),
        FunctionTool(get_gpu_quota),
        FunctionTool(list_proxy_model_hints),
        FunctionTool(search_kaggle_datasets),
    ]


def cmd_ask(args: argparse.Namespace) -> int:
    tools = build_tools()
    agent = make_chat_agent(
        SYSTEM,
        model_type=args.model,
        tools=tools,
        max_tokens=args.max_tokens,
        temperature=0.2,
        refresh=args.refresh,
    )
    print(f"=== 013 ask / model={args.model} ===")
    print(f"prompt: {args.prompt[:200]}")
    resp = agent.step(args.prompt)
    text = extract_text(resp)
    print("\n" + text)
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / "last-ask.md"
    path.write_text(text + "\n", encoding="utf-8")
    (ART / "last-ask.json").write_text(
        json.dumps(
            {"prompt": args.prompt, "reply": text, "model": args.model},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log_usage(
        experiment=EXP,
        model=args.model,
        extra={"cmd": "ask", "chars": len(text)},
    )
    print(f"\nsaved → {path}")
    return 0


def cmd_scout(args: argparse.Namespace) -> int:
    prompt = (
        f"请用工具搜索 query={args.q!r} 的 Kaggle 竞赛（约 {args.n} 个），"
        "再查 GPU/TPU 配额与推荐的便宜 Proxy 模型。"
        "输出一份「本周 Agent 向 Kaggle 行动简报」，包含：\n"
        "1. 最值得关注的 1–2 个竞赛/方向及理由\n"
        "2. 是否该用 GPU 小时还是只用 Model Proxy $\n"
        "3. 用本仓库 011/012/005/006/008 如何落地（必须写真实 python main.py … 命令）\n"
        "4. 预估会烧掉的 AI $ 量级（粗估即可）"
    )
    args.prompt = prompt
    return cmd_ask(args)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="013-camel-kaggle-crew")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("-m", "--model", default=DEFAULT_CAMEL_MODEL)
        sp.add_argument("--max-tokens", type=int, default=1600)
        sp.add_argument("--refresh", action="store_true")

    pa = sub.add_parser("ask", help="自由提问（带 Kaggle 工具）")
    add_common(pa)
    pa.add_argument("prompt")
    pa.set_defaults(func=cmd_ask)

    ps = sub.add_parser("scout", help="竞赛+配额 行动简报（默认有意义任务）")
    add_common(ps)
    ps.add_argument("--q", default="llm")
    ps.add_argument("--n", type=int, default=5)
    ps.set_defaults(func=cmd_scout)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
