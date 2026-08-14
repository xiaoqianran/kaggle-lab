#!/usr/bin/env python3
"""007-mcp-harness — 调用 Kaggle 官方 MCP（~70 tools）的薄客户端。

Usage:
  python main.py 007 tools                     # tools/list
  python main.py 007 tools --grep competition
  python main.py 007 call get_user_profile --json '{"request":{"userName":"qixiaer"}}'
  python main.py 007 call get_accelerator_quota
  python main.py 007 call search_competitions --json '{"request":{"search":"llm","pageSize":5}}'
  python main.py 007 profile
  python main.py 007 competitions --q "llm" --n 5
  python main.py 007 datasets --q "titanic" --n 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.proxy import ensure_kaggle_token  # noqa: E402

MCP_URL = "https://www.kaggle.com/mcp"
ART = Path(__file__).resolve().parent / "artifacts"
PROTOCOL = "2024-11-05"


def _token() -> str:
    ensure_kaggle_token()
    tok = os.environ.get("KAGGLE_API_TOKEN") or ""
    if not tok:
        raise SystemExit("缺少 KAGGLE_API_TOKEN / ~/.kaggle/access_token")
    return tok


def _username() -> str | None:
    """Best-effort username from `kaggle config view`."""
    try:
        ensure_kaggle_token()
        out = subprocess.check_output(
            ["kaggle", "config", "view"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if "username" in line.lower():
                # "- username: qixiaer"
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip() or None
    except Exception:
        return None
    return None


def mcp_request(method: str, params: dict | None = None, *, req_id: int = 1) -> Any:
    body = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params or {},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        MCP_URL,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as e:
        err = e.read().decode(errors="replace")
        raise SystemExit(f"MCP HTTP {e.code}: {err[:800]}") from e

    payloads: list[Any] = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            payloads.append(json.loads(line[5:].strip()))
        elif line.strip().startswith("{"):
            payloads.append(json.loads(line.strip()))
    if not payloads:
        raise SystemExit(f"MCP empty response:\n{raw[:500]}")
    last = payloads[-1]
    if "error" in last:
        raise SystemExit(f"MCP error: {json.dumps(last['error'], ensure_ascii=False)}")
    return last.get("result", last)


def cmd_tools(args: argparse.Namespace) -> int:
    try:
        mcp_request(
            "initialize",
            {
                "protocolVersion": PROTOCOL,
                "capabilities": {},
                "clientInfo": {"name": "kaggle-lab-007", "version": "0.1"},
            },
            req_id=0,
        )
    except SystemExit:
        pass
    result = mcp_request("tools/list", {}, req_id=1)
    tools = result.get("tools") or []
    if args.grep:
        rx = re.compile(args.grep, re.I)
        tools = [
            t
            for t in tools
            if rx.search(t.get("name", "")) or rx.search(t.get("description", ""))
        ]
    print(f"tools: {len(tools)}")
    for t in tools:
        name = t.get("name", "?")
        desc = (t.get("description") or "").replace("\n", " ")
        if len(desc) > 100:
            desc = desc[:97] + "..."
        print(f"  {name:48}  {desc}")
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "tools.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved → {ART / 'tools.json'}")
    return 0


def cmd_call(args: argparse.Namespace) -> int:
    if args.json:
        arguments = json.loads(args.json)
    elif args.empty:
        arguments = {}
    else:
        arguments = {"request": {}}
    result = mcp_request(
        "tools/call",
        {"name": args.name, "arguments": arguments},
        req_id=2,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    ART.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.name)[:80]
    (ART / f"call-{safe}.json").write_text(text + "\n", encoding="utf-8")
    return 0


def _unwrap_text(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    content = result.get("content")
    if isinstance(content, list) and content:
        texts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                texts.append(c.get("text") or "")
        if len(texts) == 1:
            t = texts[0]
            try:
                return json.loads(t)
            except json.JSONDecodeError:
                return t
        if texts:
            return texts
    return result


def cmd_profile(_: argparse.Namespace) -> int:
    user = _username()
    print(f"username (from kaggle config): {user or '(unknown)'}")
    # profile
    print("=== get_user_profile ===")
    req: dict[str, Any] = {}
    if user:
        req["userName"] = user
    try:
        r = mcp_request(
            "tools/call",
            {"name": "get_user_profile", "arguments": {"request": req}},
            req_id=10,
        )
        print(json.dumps(_unwrap_text(r), ensure_ascii=False, indent=2)[:4000])
    except SystemExit as e:
        print(e, file=sys.stderr)

    print("=== get_accelerator_quota ===")
    try:
        r = mcp_request(
            "tools/call",
            {"name": "get_accelerator_quota", "arguments": {"request": {}}},
            req_id=11,
        )
        print(json.dumps(_unwrap_text(r), ensure_ascii=False, indent=2)[:4000])
    except SystemExit as e:
        print(e, file=sys.stderr)
    return 0


def cmd_competitions(args: argparse.Namespace) -> int:
    arguments = {"request": {"search": args.q, "pageSize": args.n}}
    last: BaseException | None = None
    for name in ("search_competitions", "list_competitions"):
        try:
            r = mcp_request(
                "tools/call",
                {"name": name, "arguments": arguments},
                req_id=20,
            )
            data = _unwrap_text(r)
            print(json.dumps(data, ensure_ascii=False, indent=2)[:6000])
            ART.mkdir(parents=True, exist_ok=True)
            (ART / "competitions.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return 0
        except SystemExit as e:
            last = e
            continue
    raise SystemExit(last or "search failed")


def cmd_datasets(args: argparse.Namespace) -> int:
    arguments = {"request": {"search": args.q, "pageSize": args.n}}
    last: BaseException | None = None
    for name in ("search_datasets", "list_datasets"):
        try:
            r = mcp_request(
                "tools/call",
                {"name": name, "arguments": arguments},
                req_id=30,
            )
            data = _unwrap_text(r)
            print(json.dumps(data, ensure_ascii=False, indent=2)[:6000])
            ART.mkdir(parents=True, exist_ok=True)
            (ART / "datasets.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return 0
        except SystemExit as e:
            last = e
            continue
    raise SystemExit(last or "search failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="007-mcp-harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tools", help="列出 MCP tools")
    p.add_argument("--grep", default=None, help="按名称/描述过滤")
    p.set_defaults(func=cmd_tools)

    p = sub.add_parser("call", help="调用任意 tool")
    p.add_argument("name", help="tool name")
    p.add_argument("--json", default=None, help='arguments JSON，如 {"request":{}}')
    p.add_argument("--empty", action="store_true", help="传空 arguments {}")
    p.set_defaults(func=cmd_call)

    p = sub.add_parser("profile", help="用户资料 + accelerator 额度")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("competitions", help="搜索竞赛")
    p.add_argument("--q", default="llm")
    p.add_argument("--n", type=int, default=5)
    p.set_defaults(func=cmd_competitions)

    p = sub.add_parser("datasets", help="搜索数据集")
    p.add_argument("--q", default="titanic")
    p.add_argument("--n", type=int, default=3)
    p.set_defaults(func=cmd_datasets)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
