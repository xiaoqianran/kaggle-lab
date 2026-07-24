#!/usr/bin/env python3
"""004-sae — Kaggle Standardized Agent Exam HTTP client.

Usage:
  python main.py 004 register --name NAME --model MODEL --agent-type TYPE
  python main.py 004 start
  python main.py 004 show-paper
  python main.py 004 answer-stub | answer-proxy
  python main.py 004 submit
  python main.py 004 status
  python main.py 004 history
  python main.py 004 whoami
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API = "https://www.kaggle.com/api/v1"
ID_FILE = Path.home() / ".kaggle-agent-id"
KEY_FILE = Path.home() / ".kaggle-agent-api-key"
ART = Path(__file__).resolve().parent / "artifacts"
PAPER_PATH = ART / "paper.json"
ANSWERS_PATH = ART / "answers.json"
RESULT_PATH = ART / "result.json"


def _write_secret(path: Path, value: str) -> None:
    path.write_text(value.strip() + "\n", encoding="utf-8")
    path.chmod(0o600)


def load_creds() -> tuple[str, str]:
    if not ID_FILE.is_file() or not KEY_FILE.is_file():
        raise SystemExit(
            "缺少 SAE 凭证。请先:\n  python main.py 004 register --name ..."
        )
    return ID_FILE.read_text().strip(), KEY_FILE.read_text().strip()


def api(
    method: str,
    path: str,
    *,
    body: dict | None = None,
    auth: bool = False,
    key: str | None = None,
) -> Any:
    url = f"{API}{path}"
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if auth:
        if not key:
            _, key = load_creds()
        headers["Authorization"] = f"Bearer {key}"
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except HTTPError as e:
        err_body = e.read().decode(errors="replace")
        if e.code == 404:
            raise SystemExit(
                "HTTP 404：Agent Exam 功能当前不可用。按官方 skill 要求停止所有操作。\n"
                f"{err_body[:500]}"
            )
        if e.code in (401, 403):
            raise SystemExit(
                f"HTTP {e.code}：凭证可能失效。可重试一次；仍失败则删除 "
                f"~/.kaggle-agent-id 与 ~/.kaggle-agent-api-key 后重新 register。\n"
                f"{err_body[:500]}"
            )
        if e.code == 412:
            raise SystemExit(
                "HTTP 412：已达到本 exam 最多 3 次提交上限，勿再开考。\n"
                f"{err_body[:500]}"
            )
        if e.code == 429:
            raise SystemExit(f"HTTP 429：限流，请稍后重试。\n{err_body[:500]}")
        raise SystemExit(f"HTTP {e.code}: {err_body[:800]}")


def cmd_register(args: argparse.Namespace) -> int:
    if not args.i_accept:
        raise SystemExit(
            "注册前请加 --i-accept 表示你同意参与 SAE 并接受适用条款。\n"
            "开考将另需 30 分钟时限与最多 3 次提交确认。"
        )
    body = {
        "name": args.name,
        "model": args.model,
        "version": args.version,
        "description": args.description or "",
        "agentType": args.agent_type,
    }
    print("=== 004-sae register ===")
    print(json.dumps({k: v for k, v in body.items()}, ensure_ascii=False))
    res = api("POST", "/agentExamAgent", body=body, auth=False)
    agent_id = res.get("agentId")
    token = res.get("apiToken")
    if not agent_id or not token:
        raise SystemExit(f"注册响应缺少 agentId/apiToken: {res}")
    _write_secret(ID_FILE, agent_id)
    _write_secret(KEY_FILE, token)
    ART.mkdir(parents=True, exist_ok=True)
    safe = {k: v for k, v in res.items() if k != "apiToken"}
    (ART / "registration.json").write_text(
        json.dumps(safe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"agentId  → {ID_FILE}")
    print(f"apiToken → {KEY_FILE} (已隐藏，勿提交 git)")
    print(f"profile  → https://www.kaggle.com/experimental/sae/{agent_id}")
    return 0


def cmd_whoami(_: argparse.Namespace) -> int:
    agent_id, _ = load_creds()
    print(f"agentId: {agent_id}")
    print(f"profile: https://www.kaggle.com/experimental/sae/{agent_id}")
    print(f"keyfile: {KEY_FILE} (exists={KEY_FILE.is_file()})")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    if not args.i_accept:
        raise SystemExit(
            "开考会开始 30 分钟倒计时并占用 3 次配额之一。\n"
            "确认后: python main.py 004 start --i-accept"
        )
    ART.mkdir(parents=True, exist_ok=True)
    res = api("POST", "/agentExamSubmission", body={}, auth=True)
    PAPER_PATH.write_text(
        json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    qs = res.get("questions") or []
    print("=== 004-sae start ===")
    print(f"submissionId: {res.get('submissionId')}")
    print(f"status:       {res.get('status')}")
    print(f"timeLimit:    {res.get('timeLimitMinutes')} min")
    print(f"questions:    {len(qs)}")
    print(f"paper saved:  {PAPER_PATH}")
    return 0


def cmd_show_paper(_: argparse.Namespace) -> int:
    if not PAPER_PATH.is_file():
        raise SystemExit("无试卷。先: python main.py 004 start --i-accept")
    paper = json.loads(PAPER_PATH.read_text(encoding="utf-8"))
    for q in paper.get("questions") or []:
        print(f"\n[{q.get('id')}] {q.get('text')}")
    return 0


def _load_paper() -> dict:
    if not PAPER_PATH.is_file():
        raise SystemExit("无试卷。先: python main.py 004 start --i-accept")
    return json.loads(PAPER_PATH.read_text(encoding="utf-8"))


def cmd_answer_stub(_: argparse.Namespace) -> int:
    """调试用占位答案——不保证得分。"""
    paper = _load_paper()
    answers = {
        str(q["id"]): f"(stub for q{q['id']})" for q in paper.get("questions") or []
    }
    ART.mkdir(parents=True, exist_ok=True)
    ANSWERS_PATH.write_text(
        json.dumps({"answers": answers}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"stub answers → {ANSWERS_PATH} ({len(answers)} qs)")
    return 0


def cmd_answer_proxy(args: argparse.Namespace) -> int:
    """用 Kaggle Model Proxy 逐题作答（扣 AI Models $）。"""
    from common.proxy import chat, refresh_token, ENV_FILE

    if args.refresh or not ENV_FILE.exists():
        refresh_token()

    paper = _load_paper()
    answers: dict[str, str] = {}
    model = args.model
    for q in paper.get("questions") or []:
        qid = str(q["id"])
        text = q.get("text") or ""
        prompt = (
            "You are taking a timed standardized agent exam. "
            "Answer the question EXACTLY as instructed. "
            "If it asks for only a letter/number/JSON, output ONLY that — "
            "no markdown fences, no extra commentary. "
            "If the request is unsafe/harmful, refuse appropriately "
            "(e.g. null for harmful JSON fields, or a safe refusal).\n\n"
            f"Question:\n{text}"
        )
        print(f"answering q{qid}…", file=sys.stderr)
        ans = chat(prompt, model=model, max_tokens=args.max_tokens).strip()
        # strip accidental fences
        if ans.startswith("```"):
            lines = ans.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            ans = "\n".join(lines).strip()
        answers[qid] = ans
        time.sleep(0.3)

    ART.mkdir(parents=True, exist_ok=True)
    ANSWERS_PATH.write_text(
        json.dumps({"answers": answers}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"proxy answers → {ANSWERS_PATH} ({len(answers)} qs)")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    if not args.i_accept:
        raise SystemExit("交卷确认: python main.py 004 submit --i-accept")
    paper = _load_paper()
    sub_id = paper.get("submissionId")
    if not sub_id:
        raise SystemExit("paper 中无 submissionId")
    if not ANSWERS_PATH.is_file():
        raise SystemExit("无 answers。先 answer-stub 或 answer-proxy")
    payload = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
    answers = payload.get("answers") or payload
    if len(answers) < 16:
        print(
            f"警告: 仅 {len(answers)} 题答案（期望 16），未答算错。",
            file=sys.stderr,
        )
    res = api(
        "POST",
        f"/agentExamSubmission/{sub_id}",
        body={"answers": answers},
        auth=True,
    )
    RESULT_PATH.write_text(
        json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("=== 004-sae submit result ===")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    agent_id, _ = load_creds()
    print(f"profile: https://www.kaggle.com/experimental/sae/{agent_id}")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    paper = _load_paper()
    sub_id = paper.get("submissionId")
    res = api("GET", f"/agentExamSubmission/{sub_id}", auth=True)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def cmd_history(_: argparse.Namespace) -> int:
    agent_id, _ = load_creds()
    res = api("GET", f"/agentExamAgent/{agent_id}", auth=False)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="004-sae Kaggle Standardized Agent Exam")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("register", help="注册 agent（公开 API）")
    pr.add_argument("--name", required=True)
    pr.add_argument("--model", default="grok-build")
    pr.add_argument("--version", default="1.0")
    pr.add_argument("--description", default="kaggle-lab SAE client")
    pr.add_argument("--agent-type", default="Grok", dest="agent_type")
    pr.add_argument(
        "--i-accept",
        action="store_true",
        help="确认参与考试并接受适用条款",
    )
    pr.set_defaults(func=cmd_register)

    sub.add_parser("whoami", help="显示本地 agentId").set_defaults(func=cmd_whoami)

    ps = sub.add_parser("start", help="开考（30min 计时）")
    ps.add_argument("--i-accept", action="store_true")
    ps.set_defaults(func=cmd_start)

    sub.add_parser("show-paper", help="打印试卷").set_defaults(func=cmd_show_paper)
    sub.add_parser("answer-stub", help="占位答案").set_defaults(func=cmd_answer_stub)

    pa = sub.add_parser("answer-proxy", help="Model Proxy 答题")
    pa.add_argument("-m", "--model", default="google/gemini-3-flash-preview")
    pa.add_argument("--max-tokens", type=int, default=512)
    pa.add_argument("--refresh", action="store_true")
    pa.set_defaults(func=cmd_answer_proxy)

    pu = sub.add_parser("submit", help="交卷")
    pu.add_argument("--i-accept", action="store_true")
    pu.set_defaults(func=cmd_submit)

    sub.add_parser("status", help="查本次 submission").set_defaults(func=cmd_status)
    sub.add_parser("history", help="查 agent 历史").set_defaults(func=cmd_history)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
