"""User-facing CLI: journeys first, numbered labs still work.

    python -m kaggle_lab
    python -m kaggle_lab list
    python -m kaggle_lab chat "你好"
    python -m kaggle_lab debate --rounds 3
    python main.py 014 run          # unchanged
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from kaggle_lab import __version__
from kaggle_lab.catalog import (
    GROUP_TITLE,
    Journey,
    Lab,
    discover_unregistered_labs,
    discover_unregistered_tracks,
    iter_labs_by_group,
    journeys,
    labs,
    resolve_lab,
    resolve_track,
    tracks,
)
from kaggle_lab.paths import LABS_DIR, REPO_ROOT, TRACKS_DIR, ensure_import_path

# Journey command names that dispatch to a lab. Checked before lab aliases.
_JOURNEY_CMDS = {j.id: j for j in journeys() if j.lab_id}


def _venv_python(exp_dir: Path) -> Path | None:
    if sys.platform == "win32":
        p = exp_dir / ".venv" / "Scripts" / "python.exe"
    else:
        p = exp_dir / ".venv" / "bin" / "python"
    return p if p.is_file() else None


def _maybe_reexec(lab: Lab, rest: list[str]) -> None:
    """Honor per-lab or root .venv, same as the old main.py."""
    exp_dir = lab.dir
    vpy = _venv_python(exp_dir)
    root_vpy = _venv_python(REPO_ROOT)
    target = vpy or root_vpy
    if target is None:
        return
    if Path(sys.executable).resolve() == target.resolve():
        return
    new_argv = [str(target), str(REPO_ROOT / "main.py"), lab.id, *rest]
    os.execv(str(target), new_argv)


def run_lab(lab: Lab, rest: list[str]) -> int:
    if lab.dangerous and rest and rest[0] in {"register", "start", "submit"}:
        if "--i-accept" not in rest:
            print(
                f"[{lab.id}] {lab.danger_note or '默认禁止开考。'}\n"
                "真要动正式考试，必须带 --i-accept，并且用户书面确认。",
                file=sys.stderr,
            )
            return 2
    run_py = lab.run_py
    if not run_py.is_file():
        print(f"experiment script not found: {run_py}", file=sys.stderr)
        return 2
    _maybe_reexec(lab, rest)
    ensure_import_path()
    if str(lab.dir) not in sys.path:
        sys.path.insert(0, str(lab.dir))
    sys.argv = [str(run_py), *rest]
    runpy.run_path(str(run_py), run_name="__main__")
    return 0


def _print_welcome() -> None:
    print(f"kaggle-lab {__version__} — Kaggle 实验台")
    print()
    print("两块，互不打架：")
    print(f"  labs/    Agent / Model Proxy / CAMEL    ({LABS_DIR.relative_to(REPO_ROOT)}/)")
    print(f"  tracks/  可跟练的学习轨                  ({TRACKS_DIR.relative_to(REPO_ROOT)}/)")
    print()
    print("你想做什么？")
    for j in journeys():
        how = f"python -m kaggle_lab {j.id}" if j.lab_id else "python -m kaggle_lab track"
        print(f"  {j.title:<18}  {how}")
    print()
    print("列出全部入口     python -m kaggle_lab list")
    print("旧命令仍然可用   python main.py 014 run")
    print()
    print("SAE 默认不开考。假卷：python -m kaggle_lab run 009 dry-run")


def _print_help() -> None:
    _print_welcome()
    print(
        """
命令
  list [--json]          按「你想做什么」列出 labs / tracks
  run ID [args...]       跑一个 lab（ID、编号或别名）
  track [NAME]           学习轨说明
  gateway                启动双智对谈本地网关
  auth | chat | debate | workforce | scout | quota
                         上面这些是意图入口，等价于对应 lab 的默认动作

示例
  python -m kaggle_lab auth
  python -m kaggle_lab chat "你好"
  python -m kaggle_lab debate --preset socratic --rounds 3
  python -m kaggle_lab workforce
  python -m kaggle_lab track arc-agi-3
  python main.py 015 run --rounds 2
""".rstrip()
    )


def cmd_list(argv: list[str]) -> int:
    as_json = "--json" in argv
    if as_json:
        import json

        payload = {
            "labs": [
                {
                    "id": lab.id,
                    "title": lab.title,
                    "summary": lab.summary,
                    "group": lab.group,
                    "aliases": list(lab.aliases),
                    "uses_proxy": lab.uses_proxy,
                    "dangerous": lab.dangerous,
                    "product": lab.product,
                    "path": str(lab.dir.relative_to(REPO_ROOT)),
                }
                for lab in labs()
            ],
            "tracks": [
                {
                    "id": tr.id,
                    "title": tr.title,
                    "summary": tr.summary,
                    "aliases": list(tr.aliases),
                    "kaggle_users": list(tr.kaggle_users),
                    "path": str(tr.dir.relative_to(REPO_ROOT)),
                }
                for tr in tracks()
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("== 你想做什么 ==")
    for j in journeys():
        target = f"{j.lab_id} {j.cmd}" if j.lab_id else "track"
        print(f"  {j.id:<10} {j.title}  →  {target}")
        print(f"             {j.when}")
    print()
    for group, group_labs in iter_labs_by_group():
        print(f"== {GROUP_TITLE.get(group, group)} ==")
        for lab in group_labs:
            mark = " ⚠" if lab.dangerous else ""
            proxy = "proxy" if lab.uses_proxy else "    "
            print(f"  {lab.id:<28} {proxy}  {lab.title}{mark}")
            print(f"  {'':<28}         {lab.summary}")
        print()
    print("== 学习轨 ==")
    for tr in tracks():
        print(f"  {tr.id:<24} {tr.title}")
        print(f"  {'':<24} {tr.summary}")
    extra_labs = discover_unregistered_labs()
    extra_tracks = discover_unregistered_tracks()
    if extra_labs or extra_tracks:
        print()
        print("== 未登记（有目录，catalog 里还没有）==")
        for p in extra_labs:
            print(f"  lab    {p.relative_to(REPO_ROOT)}")
        for p in extra_tracks:
            print(f"  track  {p.relative_to(REPO_ROOT)}")
        print("  加一条 Lab/Track 到 kaggle_lab/catalog.py 即可被 list / 别名解析。")
    return 0


def cmd_track(argv: list[str]) -> int:
    if not argv:
        print("学习轨（cd 进去读 start_here，再跑脚本 / notebook）\n")
        for tr in tracks():
            start = tr.dir / tr.start_here
            rel = start.relative_to(REPO_ROOT) if start.is_file() else tr.dir.relative_to(REPO_ROOT)
            users = f"  kaggle: {', '.join(tr.kaggle_users)}" if tr.kaggle_users else ""
            print(f"  {tr.id:<22} {tr.title}{users}")
            print(f"  {'':<22} {tr.summary}")
            print(f"  {'':<22} {rel}")
            print()
        print("详情: python -m kaggle_lab track depth-estimation")
        return 0
    tr = resolve_track(argv[0])
    if tr is None:
        known = ", ".join(t.id for t in tracks())
        print(f"unknown track '{argv[0]}'. known: {known}", file=sys.stderr)
        return 2
    start = tr.dir / tr.start_here
    print(f"{tr.title}  [{tr.id}]")
    print(tr.summary)
    if tr.kaggle_users:
        print(f"Kaggle: {', '.join(tr.kaggle_users)}")
    print(f"目录:   {tr.dir}")
    if start.is_file():
        print(f"先读:   {start}")
    print()
    print(f"  cd {tr.dir.relative_to(REPO_ROOT)}")
    return 0


def cmd_gateway(argv: list[str]) -> int:
    lab = resolve_lab("015")
    assert lab is not None
    gateway = lab.dir / "gateway.py"
    if not gateway.is_file():
        print(f"gateway not found: {gateway}", file=sys.stderr)
        return 2
    ensure_import_path()
    sys.argv = [str(gateway), *argv]
    runpy.run_path(str(gateway), run_name="__main__")
    return 0


def journey_argv(j: Journey, argv: list[str]) -> list[str]:
    """Turn a journey invocation into the lab's argv."""
    if not j.cmd:
        return list(argv)
    if j.inject == "always" or not argv or argv[0].startswith("-"):
        return [j.cmd, *argv]
    return list(argv)


def _dispatch_journey(j: Journey, argv: list[str]) -> int:
    if j.lab_id is None:
        return cmd_track(argv)
    lab = resolve_lab(j.lab_id)
    if lab is None:
        print(f"catalog journey '{j.id}' points at missing lab {j.lab_id}", file=sys.stderr)
        return 2
    return run_lab(lab, journey_argv(j, argv))


def main(argv: list[str] | None = None) -> int:
    ensure_import_path()
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0
    if raw[0] in {"-V", "--version", "version"}:
        print(__version__)
        return 0

    cmd = raw[0]
    rest = raw[1:]

    if cmd == "list":
        return cmd_list(rest)
    if cmd == "track":
        return cmd_track(rest)
    if cmd == "gateway":
        return cmd_gateway(rest)
    if cmd == "run":
        if not rest:
            print("usage: kaggle-lab run <lab-id> [args...]", file=sys.stderr)
            return 2
        lab = resolve_lab(rest[0])
        if lab is None:
            print(_unknown_lab(rest[0]), file=sys.stderr)
            return 2
        return run_lab(lab, rest[1:])

    if cmd in _JOURNEY_CMDS:
        return _dispatch_journey(_JOURNEY_CMDS[cmd], rest)

    lab = resolve_lab(cmd)
    if lab is not None:
        return run_lab(lab, rest)

    print(_unknown_lab(cmd), file=sys.stderr)
    return 2


def _unknown_lab(token: str) -> str:
    known = ", ".join(lab.id for lab in labs())
    return (
        f"unknown command or lab '{token}'.\n"
        f"try: python -m kaggle_lab list\n"
        f"labs: {known}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
