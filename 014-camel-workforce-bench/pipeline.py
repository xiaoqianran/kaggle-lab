"""014 staged multi-agent pipeline (Workforce-style).

Stages:
  1. scout   — CAMEL ChatAgent + Kaggle tools (MCP/quota)
  2. debate  — CAMEL RolePlaying (advocate vs critic)
  3. author  — produce structured cases JSON (not free-form Python)
  4. assemble/validate — render task.py + static checks (+ optional local kbench)
  5. publish — optional 005 push/run (requires --i-accept)
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
TEMPLATE = HERE / "templates" / "task_template.py"
WORK_DIR = ART / "workspace"
TASK_OUT = WORK_DIR / "task.py"
SPEC_OUT = ART / "spec.json"
BRIEF_OUT = ART / "scout-brief.md"
DEBATE_OUT = ART / "debate.md"
LOG_OUT = ART / "pipeline-log.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.camel_proxy import (  # noqa: E402
    extract_text,
    make_camel_model,
    make_chat_agent,
)
from common.proxy import ENV_FILE, ensure_kaggle_token, refresh_token  # noqa: E402
from common.usage import log_usage  # noqa: E402

EXP = "014-camel-workforce-bench"

FALLBACK_SPEC = {
    "slug": "kaggle-lab-strict-format",
    "title": "Strict short-form instruction following",
    "rationale": (
        "Offline string-match eval of format obedience: numbers, single words, "
        "minimal JSON keys. Suitable for multi-agent output discipline."
    ),
    "cases": [
        {"prompt": "What is 9*6? Reply with the number only.", "expected": "54"},
        {
            "prompt": "What is the capital of France? Reply with the city name only.",
            "expected": "Paris",
        },
        {
            "prompt": 'Return ONLY this JSON with no markdown fences: {"ok":true,"n":2}',
            "expected": '"ok"',
        },
        {
            "prompt": "Which is larger, 12 or 8? Reply with just the larger number.",
            "expected": "12",
        },
        {
            "prompt": "How many letters are in the word agent? Reply with the number only.",
            "expected": "5",
        },
        {
            "prompt": "Output the lowercase word yes and nothing else.",
            "expected": "yes",
        },
    ],
}


@dataclass
class Case:
    prompt: str
    expected: str


@dataclass
class Spec:
    slug: str
    title: str
    rationale: str
    cases: list[Case] = field(default_factory=list)
    source_theme: str = ""
    model: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "rationale": self.rationale,
            "source_theme": self.source_theme,
            "model": self.model,
            "created_at": self.created_at,
            "cases": [asdict(c) for c in self.cases],
        }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s.startswith("kaggle-lab-"):
        s = "kaggle-lab-" + s
    return s[:60].strip("-") or "kaggle-lab-agent-bench"


def _extract_json_block(text: str) -> Any:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", t)
        if m:
            return json.loads(m.group(0))
        m = re.search(r"\[[\s\S]*\]", t)
        if m:
            return json.loads(m.group(0))
        raise


def _cases_from_obj(obj: dict) -> list[Case]:
    cases: list[Case] = []
    for c in obj.get("cases") or []:
        if not isinstance(c, dict):
            continue
        p = str(c.get("prompt") or "").strip()
        e = str(c.get("expected") or c.get("expect") or "").strip()
        if p and e:
            cases.append(Case(prompt=p, expected=e))
    return cases


def _load_proxy_env(env: dict[str, str]) -> dict[str, str]:
    """Merge Model Proxy env + defaults needed by kaggle-benchmarks."""
    ensure_kaggle_token()
    if not ENV_FILE.exists():
        refresh_token()
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    # kaggle-benchmarks requires these at import time
    env.setdefault("LLM_DEFAULT", "gemini-3.5-flash")
    env.setdefault("LLM_DEFAULT_EVAL", env.get("LLM_DEFAULT", "gemini-3.5-flash"))
    if "MODEL_PROXY_URL" in env and "MODEL_PROXY_API_KEY" in env:
        # some loaders also want OPENAI_* style
        env.setdefault("OPENAI_API_KEY", env["MODEL_PROXY_API_KEY"])
        base = env["MODEL_PROXY_URL"].rstrip("/")
        env.setdefault("OPENAI_BASE_URL", f"{base}/openapi")
    return env


def _build_kaggle_tools():
    from camel.toolkits import FunctionTool
    import importlib.util

    def _mcp(name: str, arguments: dict) -> Any:
        path = ROOT / "007-mcp-harness" / "run.py"
        spec = importlib.util.spec_from_file_location("mcp007", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        r = mod.mcp_request(
            "tools/call", {"name": name, "arguments": arguments}, req_id=60
        )
        return mod._unwrap_text(r)

    def search_kaggle_competitions(query: str, n: int = 5) -> str:
        """Search Kaggle competitions; return compact JSON."""
        try:
            data = _mcp(
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
                        "description": (c.get("description") or "")[:200],
                    }
                )
            return json.dumps({"competitions": comps}, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False)[:3500]

    def get_gpu_quota() -> str:
        """Current GPU/TPU weekly quota text."""
        ensure_kaggle_token()
        try:
            return subprocess.check_output(
                ["kaggle", "quota"], text=True, stderr=subprocess.STDOUT
            ).strip()
        except Exception as e:
            return f"error: {e}"

    def list_proxy_budget_hints() -> str:
        """Cheap model hints for daily $10 budget."""
        return json.dumps(
            {
                "daily_usd": 10,
                "prefer": [
                    "google/gemini-3-flash-preview",
                    "google/gemini-3.5-flash-lite",
                    "openai/gpt-5.4-nano-2026-03-17",
                ],
            }
        )

    return [
        FunctionTool(search_kaggle_competitions),
        FunctionTool(get_gpu_quota),
        FunctionTool(list_proxy_budget_hints),
    ]


def stage_scout(
    *,
    theme: str,
    model: str,
    query: str,
    n: int,
    refresh: bool,
    max_tokens: int,
) -> dict[str, Any]:
    tools = _build_kaggle_tools()
    agent = make_chat_agent(
        (
            "You are Scout, stage-1 researcher in a Kaggle multi-agent lab.\n"
            "Use tools to fetch REAL competitions and GPU quota.\n"
            "Do not invent competition names.\n"
            "Output Chinese Markdown with:\n"
            "## Theme fit\n## Top competitions (from tools)\n"
            "## GPU vs Model-Proxy recommendation\n"
            "## Proposed benchmark angle (instruction-following / format / "
            "tiny reasoning — must be evaluable offline without datasets)\n"
            "## Risks\n"
        ),
        model_type=model,
        tools=tools,
        max_tokens=max_tokens,
        temperature=0.2,
        refresh=refresh,
    )
    prompt = (
        f"主题: {theme}\n"
        f"请搜索 query={query!r} 约 {n} 个竞赛，查配额，"
        "并提议一个适合做成 Kaggle Benchmarks task 的最小评测角度。"
        "注意: task 只能是纯文本 prompt→expected 字符串匹配，不能依赖外部数据集下载。"
    )
    print(f"[scout] theme={theme!r} query={query!r}")
    resp = agent.step(prompt)
    text = extract_text(resp)
    ART.mkdir(parents=True, exist_ok=True)
    BRIEF_OUT.write_text(text + "\n", encoding="utf-8")
    log_usage(experiment=EXP, model=model, extra={"stage": "scout", "chars": len(text)})
    print(f"[scout] saved → {BRIEF_OUT} ({len(text)} chars)")
    return {"brief": text, "path": str(BRIEF_OUT)}


def stage_debate(
    *,
    brief: str,
    model: str,
    rounds: int,
    refresh: bool,
    max_tokens: int,
) -> dict[str, Any]:
    from camel.societies import RolePlaying

    brief_clip = brief[:3500]
    m = make_camel_model(
        model, temperature=0.3, max_tokens=max(max_tokens, 1600), refresh=refresh
    )
    task = (
        "根据 Scout 简报，辩论并最终收敛：\n"
        "1) 选一个具体的 benchmark 主题（instruction-following / JSON-format / "
        "tiny arithmetic 等，可离线字符串评测）；\n"
        "2) 4–6 条 (prompt, expected_substring) 设计原则；\n"
        "3) 推荐 slug（英文 kebab-case，kaggle-lab- 前缀）。\n"
        "Critic 攻击可测性；Advocate 给出可执行最终方案。\n\n"
        f"--- Scout brief ---\n{brief_clip}"
    )
    print(f"[debate] rounds={rounds}")
    session = RolePlaying(
        assistant_role_name="Benchmark Advocate",
        user_role_name="Benchmark Critic",
        task_prompt=task,
        with_task_specify=False,
        model=m,
        output_language="Chinese",
    )
    input_msg = session.init_chat()
    turns: list[dict[str, str]] = []
    final = ""
    for i in range(1, rounds + 1):
        print(f"[debate] round {i}/{rounds}")
        try:
            asst, user = session.step(input_msg)
        except ValueError as e:
            print(f"[debate] step failed ({e}); stop early")
            break
        u, a = extract_text(user), extract_text(asst)
        if not u and not a:
            print("[debate] empty messages; stop early")
            break
        turns.append({"round": str(i), "critic": u, "advocate": a})
        if a:
            final = a
        print(f"  critic:   {(u or '')[:140].replace(chr(10), ' ')}…")
        print(f"  advocate: {(a or '')[:140].replace(chr(10), ' ')}…")
        if getattr(asst, "terminated", False) or getattr(user, "terminated", False):
            break
        if not getattr(asst, "msgs", None):
            break
        input_msg = asst.msg
        time.sleep(0.4)

    if not final and turns:
        final = turns[-1].get("advocate") or turns[-1].get("critic") or ""
    if not final:
        final = (
            "Fallback: strict instruction-following. "
            "Slug: kaggle-lab-strict-format."
        )
        print("[debate] using fallback final")

    ART.mkdir(parents=True, exist_ok=True)
    md = ["# Debate transcript\n", f"## Task\n\n{session.task_prompt}\n"]
    for t in turns:
        md.append(f"\n## Round {t['round']}\n")
        md.append(f"### Critic\n\n{t['critic']}\n")
        md.append(f"### Advocate\n\n{t['advocate']}\n")
    md.append(f"\n## Final (for author)\n\n{final}\n")
    DEBATE_OUT.write_text("\n".join(md), encoding="utf-8")
    log_usage(
        experiment=EXP,
        model=model,
        extra={"stage": "debate", "rounds": len(turns), "chars": len(final)},
    )
    print(f"[debate] saved → {DEBATE_OUT} turns={len(turns)}")
    return {"final": final, "turns": turns, "path": str(DEBATE_OUT)}


AUTHOR_SYSTEM = """You are Author. Output ONLY valid JSON (no markdown) with schema:
{"slug":"kaggle-lab-...","title":"...","rationale":"...","cases":[{"prompt":"...","expected":"..."}]}
Rules:
- Exactly 5 cases
- expected is a SHORT substring
- English prompts; prefer "number only" / "city only" / "JSON only"
- no external data or tools
- slug kebab-case starting with kaggle-lab-
Keep each prompt under 120 characters. Keep expected under 20 characters.
"""


def stage_author(
    *,
    brief: str,
    debate_final: str,
    model: str,
    refresh: bool,
    max_tokens: int,
    theme: str,
) -> Spec:
    agent = make_chat_agent(
        AUTHOR_SYSTEM,
        model_type=model,
        max_tokens=max(max_tokens, 1800),
        temperature=0.05,
        refresh=refresh,
    )
    prompt = (
        f"Theme: {theme}\n"
        f"Scout summary (cut):\n{brief[:1200]}\n\n"
        f"Debate decision (cut):\n{debate_final[:1500]}\n\n"
        "Emit the JSON object now. 5 cases."
    )
    print("[author] generating structured cases…")
    resp = agent.step(prompt)
    raw = extract_text(resp)
    obj: dict | None = None
    try:
        parsed = _extract_json_block(raw)
        if isinstance(parsed, dict):
            obj = parsed
    except Exception as e:
        print(f"[author] parse fail: {e}")

    if obj is None or len(_cases_from_obj(obj)) < 3:
        print("[author] repair pass…")
        repair = make_chat_agent(
            "Convert input into valid JSON only. "
            "Schema {slug,title,rationale,cases:[{prompt,expected}]}. "
            "Need 5 cases. No markdown.",
            model_type=model,
            max_tokens=1800,
            temperature=0,
        )
        seed = raw if raw and len(raw) > 40 else json.dumps(FALLBACK_SPEC)
        raw2 = extract_text(
            repair.step(
                "Repair or complete this into valid JSON with 5 cases:\n" + seed[:4000]
            )
        )
        try:
            parsed = _extract_json_block(raw2)
            if isinstance(parsed, dict):
                obj = parsed
        except Exception as e:
            print(f"[author] repair parse fail: {e}")

    cases = _cases_from_obj(obj) if obj else []
    if len(cases) < 3:
        print("[author] using deterministic FALLBACK_SPEC")
        obj = dict(FALLBACK_SPEC)
        cases = _cases_from_obj(obj)

    slug = _slugify(str((obj or {}).get("slug") or theme))
    spec = Spec(
        slug=slug,
        title=str((obj or {}).get("title") or slug),
        rationale=str((obj or {}).get("rationale") or ""),
        cases=cases[:6],
        source_theme=theme,
        model=model,
        created_at=_now(),
    )
    ART.mkdir(parents=True, exist_ok=True)
    SPEC_OUT.write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log_usage(
        experiment=EXP,
        model=model,
        extra={"stage": "author", "n_cases": len(spec.cases), "slug": spec.slug},
    )
    print(f"[author] slug={spec.slug} cases={len(spec.cases)} → {SPEC_OUT}")
    return spec


def assemble_task_py(spec: Spec) -> Path:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    cases_py = (
        "[\n"
        + ",\n".join(f"    ({c.prompt!r}, {c.expected!r})" for c in spec.cases)
        + "\n]"
    )
    fn = re.sub(r"[^a-z0-9_]", "_", spec.slug.replace("-", "_"))
    if not fn or fn[0].isdigit():
        fn = "task_" + (fn or "bench")
    doc = (spec.rationale or spec.title).replace('"""', "'")
    body = (
        tpl.replace("__SLUG__", spec.slug)
        .replace("__CASES__", cases_py)
        .replace("__FN__", fn)
        .replace("__DOC__", doc[:200])
    )
    ast.parse(body)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    TASK_OUT.write_text(body, encoding="utf-8")
    (WORK_DIR / "spec.json").write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[assemble] → {TASK_OUT}")
    return TASK_OUT


def static_validate(path: Path) -> list[str]:
    problems: list[str] = []
    src = path.read_text(encoding="utf-8")
    try:
        ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    if "kaggle_benchmarks" not in src:
        problems.append("missing kaggle_benchmarks import")
    if "task(name=" not in src and "@kbench.task" not in src:
        problems.append("missing @kbench.task decorator")
    if ".run(" not in src:
        problems.append("missing .run(...) call — silent no-op risk")
    if "assert_" not in src:
        problems.append("missing assertions")
    if re.search(r"CASES\s*=\s*\[\s*\]", src):
        problems.append("empty CASES")
    return problems


def local_kbench_validate(path: Path) -> int:
    env = _load_proxy_env(os.environ.copy())
    # write full .env beside task for dotenv loaders
    lines = [f"{k}={v}" for k, v in env.items() if k.startswith(("MODEL_", "LLM_", "OPENAI_", "KAGGLE_"))]
    (path.parent / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[validate] running {path} …")
    print(f"[validate] LLM_DEFAULT={env.get('LLM_DEFAULT')}")
    r = subprocess.run([sys.executable, str(path)], cwd=str(path.parent), env=env)
    return r.returncode


def publish(
    *,
    slug: str,
    task_path: Path,
    do_push: bool,
    do_run: bool,
    run_model: str,
    wait: bool,
) -> dict[str, Any]:
    ensure_kaggle_token()
    out: dict[str, Any] = {"slug": slug, "pushed": False, "ran": False}
    if do_push:
        cmd = ["kaggle", "b", "t", "push", slug, "-f", str(task_path)]
        if wait:
            cmd.append("--wait")
        print("+", " ".join(cmd))
        r = subprocess.run(cmd, env=os.environ.copy())
        out["push_code"] = r.returncode
        out["pushed"] = r.returncode == 0
        if r.returncode != 0:
            return out
    if do_run:
        bare = run_model.split("/")[-1]
        cmd = ["kaggle", "b", "t", "run", slug, "-m", bare]
        if wait:
            cmd.append("--wait")
        print("+", " ".join(cmd))
        r = subprocess.run(cmd, env=os.environ.copy())
        out["run_code"] = r.returncode
        out["ran"] = r.returncode == 0
    return out


def run_pipeline(
    *,
    theme: str,
    query: str,
    n_comp: int,
    model: str,
    debate_rounds: int,
    refresh: bool,
    max_tokens: int,
    local_validate: bool,
    push: bool,
    run_remote: bool,
    run_model: str,
    i_accept: bool,
) -> dict[str, Any]:
    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    log: dict[str, Any] = {
        "started": _now(),
        "theme": theme,
        "model": model,
        "stages": {},
    }

    scout = stage_scout(
        theme=theme,
        model=model,
        query=query,
        n=n_comp,
        refresh=refresh,
        max_tokens=max_tokens,
    )
    log["stages"]["scout"] = {"path": scout["path"], "chars": len(scout["brief"])}

    debate = stage_debate(
        brief=scout["brief"],
        model=model,
        rounds=debate_rounds,
        refresh=False,
        max_tokens=max_tokens,
    )
    log["stages"]["debate"] = {
        "path": debate["path"],
        "rounds": len(debate["turns"]),
    }

    spec = stage_author(
        brief=scout["brief"],
        debate_final=debate["final"],
        model=model,
        refresh=False,
        max_tokens=max_tokens,
        theme=theme,
    )
    log["stages"]["author"] = {"slug": spec.slug, "cases": len(spec.cases)}

    task_path = assemble_task_py(spec)
    problems = static_validate(task_path)
    log["stages"]["assemble"] = {"path": str(task_path), "problems": problems}
    if problems:
        raise RuntimeError("static validate failed: " + "; ".join(problems))
    print("[assemble] static OK")

    if local_validate:
        code = local_kbench_validate(task_path)
        log["stages"]["local_validate"] = {"exit": code}
        if code != 0:
            print(
                "[validate] local kbench failed — review cases before push",
                file=sys.stderr,
            )
        else:
            print("[validate] local kbench OK")

    if push or run_remote:
        if not i_accept:
            raise SystemExit(
                "publish 需要显式确认:\n"
                "  python main.py 014 run --push --i-accept\n"
                "  python main.py 014 run --push --run-remote --i-accept"
            )
        pub = publish(
            slug=spec.slug,
            task_path=task_path,
            do_push=push,
            do_run=run_remote,
            run_model=run_model,
            wait=True,
        )
        log["stages"]["publish"] = pub

    log["elapsed_sec"] = round(time.time() - t0, 2)
    log["finished"] = _now()
    LOG_OUT.write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[pipeline] done in {log['elapsed_sec']}s → {LOG_OUT}")
    return log


def run_workforce_probe(*, model: str, refresh: bool, max_tokens: int = 2048) -> dict:
    from camel.societies.workforce import Workforce
    from camel.societies.workforce.workforce import WorkforceMode

    print("[workforce] building PIPELINE (experimental)…")
    m = make_camel_model(model, temperature=0.2, max_tokens=max_tokens, refresh=refresh)
    coord = make_chat_agent(
        "You are coordinator. Prefer assigning each task to a matching worker.",
        model_type=model,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    task_agent = make_chat_agent(
        "You track task dependencies briefly.",
        model_type=model,
        max_tokens=512,
        temperature=0.1,
    )
    wf = Workforce(
        "kaggle-lab-014",
        coordinator_agent=coord,
        task_agent=task_agent,
        default_model=m,
        mode=WorkforceMode.PIPELINE,
        use_structured_output_handler=True,
        graceful_shutdown_timeout=20,
        task_timeout_seconds=180,
    )
    researcher = make_chat_agent(
        "You research multi-agent eval ideas. Short Chinese bullets.",
        model_type=model,
        max_tokens=800,
        temperature=0.3,
    )
    writer = make_chat_agent(
        'Write compact JSON: {"idea":"...","why":"..."} only.',
        model_type=model,
        max_tokens=400,
        temperature=0.1,
    )
    wf.add_single_agent_worker("researcher of agent evaluation ideas", researcher)
    wf.add_single_agent_worker("json writer of final idea object", writer)
    wf.pipeline_add(
        "Propose 2 offline-evaluable Kaggle Benchmark ideas for multi-agent "
        "instruction following (no external datasets).",
        task_id="research",
    )
    wf.pipeline_add(
        "Pick the better idea and output JSON {idea, why}.",
        task_id="write",
    )
    wf.pipeline_build()
    pending = list(wf.get_pending_tasks())
    result = wf.process_task(pending[0])
    payload = {
        "state": str(result.state),
        "result": result.result,
        "completed": [
            {"id": t.id, "state": str(t.state), "result": t.result}
            for t in wf.get_completed_tasks()
        ],
    }
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / "workforce-probe.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[workforce] → {path} state={result.state}")
    return payload
