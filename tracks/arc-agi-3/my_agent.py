"""ARC-AGI-3 智能体：教程关收割技能 + 因果分类 + 向目标 A* 走近。

官方赛题（不是公开本）真正在考四件事：探索、建模、自己定目标、规划执行。
环境至少 6 关；第 1 关是教程；后面关要组合前面学会的机制；没打完后面关整局按加权完成度封顶。

我们的打法：
1. 教程关用便宜实验认清「会不会走 / 要点哪里 / 按一下会不会赢」
2. 认出角色后，朝「上一关赢过的颜色」或小稀有块 A* 走近，踩上就 ACTION5
3. 走路没用再点；图穷了才问 Gemma（模型内部思考不计步）
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import threading
import time
import traceback
from collections import deque
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

_SUBMISSION_STARTED_AT = time.monotonic()
_LLM_LOCK = threading.Lock()
_LLM_STATE: dict[str, Any] = {
    "started": False,
    "ready": False,
    "failed": False,
    "model": None,
    "processor": None,
    "path": None,
    "error": None,
}

MODEL_CANDIDATES = [
    "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1",
    "/kaggle/input/google/gemma-4/transformers/gemma-4-31b-it/1",
    "/kaggle/input/models/google/gemma-4/transformers/gemma-4-26b-a4b-it/1",
    "/kaggle/input/models/google/gemma-4/transformers/gemma-4-26b-a4b-it/1",
    "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1",
    "/kaggle/input/google/gemma-4/transformers/gemma-4-e2b-it/1",
]

# 官方默认：1上 2下 3左 4右。坐标 (x,y)，y 向下变大。
DEFAULT_DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
SIMPLE_IDS = (1, 2, 3, 4, 5)
ID_TO_NAME = {
    0: "RESET",
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
    5: "ACTION5",
    6: "ACTION6",
    7: "ACTION7",
}

# 人步数未知，只用先验做停手，不进提交分。
REPLAY_RESETS = 3
MAX_CLICKS = 16
CLICK_KEEP = 4
NAV_BIAS = 0.7
DYN_WARMUP = 24
DYN_RATE = 0.45
DYN_MAX_FRAC = 0.08
NAV_REFRESH = 12
BULKY_WALL = 8


def _le(a, b) -> bool:
    return not (a > b)


def _lt(a, b) -> bool:
    return not (a >= b)


def _clamp63(v: int) -> int:
    return int(max(0, min(63, v)))


def _in_bounds(x: int, y: int) -> bool:
    return x >= 0 and _le(x, 63) and y >= 0 and _le(y, 63)


def rhae_level_score(human_actions: float, ai_actions: float) -> float:
    """官方关卡分：平方效率，上限 1.15。"""
    if _le(ai_actions, 0):
        return 0.0
    score = (human_actions / ai_actions) ** 2
    return 1.15 if score > 1.15 else score


def _hard_cap(level_index: int) -> int:
    idx = 0 if _lt(level_index, 0) else int(level_index)
    hard = 700 + 80 * (idx if _le(idx, 6) else 6)
    return 1100 if hard > 1100 else hard


def attempt_exhausted(
    level_spent: int,
    can_still_explore: bool,
    remaining_s: float,
    level_index: int = 0,
    has_plan: bool = False,
) -> bool:
    """这一轮尝试是否已经没油。有可达猎点时不要在硬上限一刀切。"""
    if _le(remaining_s, 0):
        return True
    hard = _hard_cap(level_index)
    if level_spent >= hard and not has_plan:
        return True
    if level_spent >= hard + 250:
        return True
    budget = level_action_budget(level_index)
    if level_spent >= budget and not can_still_explore and not has_plan:
        return True
    return False


def stall_policy(
    level_spent: int,
    can_still_explore: bool,
    remaining_s: float,
    level_index: int = 0,
    has_plan: bool = False,
    level_resets: int = 0,
    max_resets: int = REPLAY_RESETS,
) -> str:
    """卡关时：continue / reset / quit。

    硬上限先 RESET 重打本关（已知墙还在），不要直接退整局。
    官方完成度封顶：退局 = 后面关全 0。
    """
    if _le(remaining_s, 0):
        return "quit"
    if not attempt_exhausted(
        level_spent, can_still_explore, remaining_s, level_index, has_plan
    ):
        return "continue"
    if _lt(level_resets, max_resets):
        return "reset"
    return "quit"


def should_abandon(
    level_spent: int,
    can_still_explore: bool,
    remaining_s: float,
    level_index: int = 0,
    has_plan: bool = False,
) -> bool:
    """这一轮尝试该停。整局退赛请看 stall_policy == quit。"""
    return attempt_exhausted(
        level_spent, can_still_explore, remaining_s, level_index, has_plan
    )


def rhae_game_score(level_scores: list[float], n_levels: int | None = None) -> float:
    """官方整局分：关卡号加权平均，再按「从第 1 关起连续打完几关」封顶。

    5 关权重 1+2+3+4+5=15。只打完前 3 关，封顶 6/15=0.40。
    所以教程关可以慢，后面关必须打完。
    """
    n = n_levels if n_levels is not None else len(level_scores)
    scores = list(level_scores) + [0.0] * max(0, n - len(level_scores))
    scores = scores[:n]
    k = 0
    for s in scores:
        if s > 0:
            k += 1
        else:
            break
    denom = n * (n + 1) / 2.0
    if _le(denom, 0):
        return 0.0
    cap = (k * (k + 1) / 2.0) / denom
    weighted = sum((i + 1) * float(scores[i]) for i in range(n)) / denom
    return cap if _le(cap, weighted) else weighted


def level_action_budget(level_index: int) -> int:
    """第 1 关是教程（权重最小）：把机制学到手就过。后面关权重更大、机制要组合，多给步数。"""
    idx = 0 if _lt(level_index, 0) else int(level_index)
    budget = 160 + 50 * idx
    return 700 if budget > 700 else budget


def classify_genre(move_hits: int, click_hits: int, interact_hits: int) -> str:
    if move_hits > 0 and click_hits > 0:
        return "hybrid"
    if move_hits > 0:
        return "nav"
    if click_hits > 0:
        return "click"
    if interact_hits > 0:
        return "nav"
    return "unknown"


def first_step_toward(start, goal, blocked, dir_map):
    """在已知墙里走一步靠近目标。目标格本身可踩（旗子要走进去）。"""
    if start is None or goal is None:
        return None
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    if sx == gx and sy == gy:
        return None
    prev: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {(sx, sy): None}
    queue = deque([(sx, sy)])
    found = False
    while queue:
        x, y = queue.popleft()
        if x == gx and y == gy:
            found = True
            break
        for sid, (dx, dy) in dir_map.items():
            nx, ny = x + dx, y + dy
            if not _in_bounds(nx, ny) or (nx, ny) in blocked or (nx, ny) in prev:
                continue
            prev[(nx, ny)] = ((x, y), int(sid))
            queue.append((nx, ny))
    if not found:
        return None
    node = (gx, gy)
    sid_out = None
    while prev[node] is not None:
        parent, sid_out = prev[node]
        node = parent
        if node == (sx, sy):
            break
    return sid_out


class SkillSheet:
    """跨关技能纸。官方：后面关要组合前面关学到的机制，不是每关当新游戏。

    只记一个 goal_color 不够：后面关常常是「先点开关，再走到出口」。
    """

    def __init__(self) -> None:
        self.genre = "unknown"
        self.move_hits = 0
        self.click_hits = 0
        self.interact_hits = 0
        self.goal_color: int | None = None
        self.goal_colors: list[int] = []
        self.useful_click_colors: list[int] = []
        self.win_kinds: list[str] = []
        self.wall_colors: list[int] = []

    def note_effect(self, kind: str) -> None:
        if kind == "move":
            self.move_hits += 1
        elif kind == "click":
            self.click_hits += 1
        elif kind == "interact":
            self.interact_hits += 1
        self.genre = classify_genre(self.move_hits, self.click_hits, self.interact_hits)

    def _push_color(self, bucket: list[int], color: int, cap: int = 4) -> None:
        color = int(color)
        if color in bucket:
            bucket.remove(color)
        bucket.insert(0, color)
        del bucket[cap:]

    def note_win_color(self, color: int | None) -> None:
        if color is None:
            return
        self.goal_color = int(color)
        self._push_color(self.goal_colors, int(color))

    def note_win(self, color: int | None, kind: str | None) -> None:
        self.note_win_color(color)
        if not kind:
            return
        if kind in self.win_kinds:
            self.win_kinds.remove(kind)
        self.win_kinds.insert(0, kind)

    def note_click_color(self, color: int | None) -> None:
        if color is None:
            return
        self._push_color(self.useful_click_colors, int(color))

    def prefer_colors(self) -> list[int]:
        out: list[int] = []
        for color in self.goal_colors:
            if color not in out:
                out.append(color)
        if self.goal_color is not None and self.goal_color not in out:
            out.append(int(self.goal_color))
        return out

    def note_wall_color(self, color: int) -> None:
        c = int(color)
        if _le(c, 0) or c in self.wall_colors:
            return
        self.wall_colors.append(c)
        if len(self.wall_colors) > 8:
            self.wall_colors = self.wall_colors[-8:]


def pick_hunt_candidates(
    comps: list[dict[str, Any]],
    agent_pos,
    prefer_colors: list[int] | None,
    limit: int = 6,
    skip_colors: list[int] | set[int] | None = None,
) -> list[tuple[int, int, int]]:
    """只追小色块。赢过的颜色如果铺成地板，追它等于撞墙。返回多个候选，供 A* 试可达性。"""
    prefers = list(prefer_colors or [])
    skip = {int(c) for c in (skip_colors or [])}
    small = [c for c in comps if _le(c["size"], 24) and int(c["color"]) not in skip]

    def _rank(comp: dict[str, Any]) -> tuple[int, int, int, int]:
        pref = 99
        if comp["color"] in prefers:
            pref = prefers.index(comp["color"])
        return (pref, int(comp["size"]), int(comp["y"]), int(comp["x"]))

    small.sort(key=_rank)
    out: list[tuple[int, int, int]] = []
    for comp in small:
        if agent_pos is not None and _le(abs(int(comp["x"]) - agent_pos[0]) + abs(int(comp["y"]) - agent_pos[1]), 0):
            continue
        out.append((int(comp["x"]), int(comp["y"]), int(comp["color"])))
        if len(out) >= limit:
            break
    return out


def pick_hunt_target(comps: list[dict[str, Any]], agent_pos, prefer_color: int | None):
    prefers = [int(prefer_color)] if prefer_color is not None else None
    cands = pick_hunt_candidates(comps, agent_pos, prefers)
    return cands[0] if cands else None


def plan_hunt(pos, candidates, blocked, dir_map):
    """在已知墙里找第一个走得到的猎点。目标格本身可踩，不要当墙。

    返回 ('on', x, y, color) 或 ('step', sid, x, y, color) 或 None。
    """
    if pos is None or not candidates:
        return None
    walls = set(blocked)
    for x, y, color in candidates:
        walls.discard((x, y))
        if pos[0] == x and pos[1] == y:
            return ("on", x, y, color)
        sid = first_step_toward(pos, (x, y), walls, dir_map)
        if sid is not None:
            return ("step", int(sid), x, y, color)
    return None


def planning_walls(comps, blocked, hunt_colors, agent_xy, wall_colors):
    """大色块和已知墙色在规划里当墙。猎点颜色整块可走，避免把出口自己挡住。"""
    walls = set(blocked)
    hunt = {int(c) for c in (hunt_colors or [])}
    known = {int(c) for c in (wall_colors or [])}
    axay = None
    if agent_xy is not None:
        axay = (int(agent_xy[0]), int(agent_xy[1]))
    for comp in comps or []:
        color = int(comp["color"])
        if color in hunt:
            continue
        bulky = (not _lt(comp["size"], BULKY_WALL)) or (color in known and not _lt(comp["size"], 4))
        if not bulky:
            continue
        cells = comp.get("cells") or [(int(comp["x"]), int(comp["y"]))]
        for cell in cells:
            xy = (int(cell[0]), int(cell[1]))
            if axay is not None and xy == axay:
                continue
            walls.add(xy)
    return walls


def next_skill_move(
    pref_plan,
    other_plan,
    level_index: int,
    compose_left: bool,
    has_pref_targets: bool,
) -> str:
    """出口还在但走不到：先点开关，不要先去摸旁边的小色块。"""
    if pref_plan is not None:
        return "hunt-pref"
    if (not _lt(level_index, 1)) and compose_left and has_pref_targets:
        return "compose"
    if other_plan is not None:
        return "hunt-other"
    if (not _lt(level_index, 1)) and compose_left:
        return "compose"
    return "graph"


def pick_compose_click(mem, h: str, painted, bg: int, useful_colors, hunt_colors, comps=None):
    """A* 走不到出口时：点小开关，不要点出口本身，也不要点大墙。"""
    local = mem.untested(h, "c")
    if not local:
        return mem.bfs_step(h, "c")
    useful = list(useful_colors or [])
    avoid = set(hunt_colors or [])
    size_at: dict[tuple[int, int], int] = {}
    for comp in comps or []:
        size_at[(int(comp["x"]), int(comp["y"]))] = int(comp["size"])
    node = mem.nodes.get(h) or {}
    meta = node.get("meta") or {}
    scored: list[tuple[int, int, str]] = []
    for key in local:
        if key not in meta:
            continue
        _sid, x, y = meta[key]
        if not _in_bounds(x, y):
            continue
        color = int(painted[y, x])
        if color == bg:
            continue
        sz = size_at.get((x, y), 32)
        if sz >= 40:
            continue
        if color in avoid:
            pref = 80
        elif color in useful:
            pref = useful.index(color)
        else:
            pref = 40
        scored.append((pref, sz, key))
    scored.sort()
    if scored:
        return scored[0][2]
    return local[0]


def _find_model_path() -> str | None:
    env = os.getenv("GEMMA4_MODEL_PATH")
    if env and os.path.exists(env):
        return env
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def _start_llm_background() -> None:
    with _LLM_LOCK:
        if _LLM_STATE["started"] or _LLM_STATE["failed"]:
            return
        _LLM_STATE["started"] = True
    threading.Thread(target=_load_llm, name="gemma4-load", daemon=True).start()


def _load_llm() -> None:
    path = _find_model_path()
    if path is None:
        _LLM_STATE["failed"] = True
        _LLM_STATE["error"] = "no Gemma-4 weights mounted"
        print("LLM skip:", _LLM_STATE["error"], flush=True)
        return
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

        print("LLM loading", path, flush=True)
        t0 = time.perf_counter()
        processor = AutoProcessor.from_pretrained(path)
        kwargs: dict[str, Any] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        try:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        except Exception as exc:
            print("LLM 4bit failed, trying fp16:", type(exc).__name__, exc, flush=True)
            kwargs.pop("quantization_config", None)
            kwargs["torch_dtype"] = torch.float16
            model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        model.eval()
        _LLM_STATE["processor"] = processor
        _LLM_STATE["model"] = model
        _LLM_STATE["path"] = path
        _LLM_STATE["ready"] = True
        print("LLM ready in", round(time.perf_counter() - t0, 1), "s", flush=True)
    except Exception as exc:
        _LLM_STATE["failed"] = True
        _LLM_STATE["error"] = f"{type(exc).__name__}: {exc}"
        print("LLM load failed:", _LLM_STATE["error"], flush=True)
        traceback.print_exc()


def _avail_ids(latest_frame: FrameData) -> set[int]:
    raw = getattr(latest_frame, "available_actions", None) or []
    out: set[int] = set()
    for item in raw:
        out.add(int(item.value) if hasattr(item, "value") else int(item))
    return out or {1, 2, 3, 4, 5, 6}


def _grid(latest_frame: FrameData) -> np.ndarray:
    frame = np.asarray(latest_frame.frame, dtype=np.int64)
    if frame.ndim == 3:
        frame = frame[-1]
    if frame.shape != (64, 64):
        frame = np.resize(frame, (64, 64))
    return frame


def _bg_color(frame: np.ndarray) -> int:
    return int(np.bincount(frame.ravel(), minlength=16).argmax())


def _widest_run(row: np.ndarray, bg: int) -> int:
    best = run = 0
    for col in range(row.size):
        if row[col] != bg:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def mask_hud(frame: np.ndarray, bg: int) -> np.ndarray:
    """抹掉顶/底很长的分数条，短标记留下。不抹的话计步条每步都制造假新状态。"""
    out = frame.copy()
    height, width = out.shape
    wide = max(8, width // 2)
    active = np.sum(out != bg, axis=1)

    sep = None
    for row in range(0, min(16, height)):
        if active[row] == 0:
            sep = row
            break
    if sep is not None:
        for row in range(0, sep):
            if _widest_run(out[row], bg) >= wide:
                out[row, :] = bg

    sep = None
    for row in range(height - 1, max(height - 17, 0), -1):
        if active[row] == 0:
            sep = row
            break
    if sep is not None:
        for row in range(sep + 1, height):
            if _widest_run(out[row], bg) >= wide:
                out[row, :] = bg
    return out


def hash_frame(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()[:16]


def components(frame: np.ndarray, bg: int) -> list[dict[str, Any]]:
    """四连通同色块。代表点一定在色块里面，给 ACTION6 点。"""
    height, width = frame.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[dict[str, Any]] = []
    for y0 in range(height):
        for x0 in range(width):
            if seen[y0, x0] or frame[y0, x0] == bg:
                continue
            color = int(frame[y0, x0])
            queue = deque([(y0, x0)])
            seen[y0, x0] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if (
                        ny >= 0
                        and _le(ny, height - 1)
                        and nx >= 0
                        and _le(nx, width - 1)
                        and (not seen[ny, nx])
                        and frame[ny, nx] == color
                    ):
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            ys = np.array([p[0] for p in pixels])
            xs = np.array([p[1] for p in pixels])
            k = int(np.argmin(np.abs(ys - ys.mean()) + np.abs(xs - xs.mean())))
            size = len(pixels)
            rarity_size = 1.0 if _le(size, 4) else 0.8 if _le(size, 16) else 0.5 if _le(size, 64) else 0.2
            cells = [(int(p[1]), int(p[0])) for p in pixels]
            comps.append(
                {
                    "color": color,
                    "size": size,
                    "x": int(xs[k]),
                    "y": int(ys[k]),
                    "ssc": rarity_size,
                    "cells": cells,
                }
            )
    return comps


def click_rank(comps: list[dict[str, Any]], frame: np.ndarray) -> list[tuple[float, int, int]]:
    """小、颜色少的块更像按钮，先点。太大的色块多半是墙/地板，跳过。"""
    counts = np.bincount(frame.ravel(), minlength=16)
    total = max(int(frame.size), 1)
    ranked: list[tuple[float, int, int]] = []
    for comp in comps:
        if comp["size"] >= 81:
            continue
        rarity = 1.0 - counts[comp["color"]] / total
        score = 0.5 * rarity + 0.5 * float(comp["ssc"])
        ranked.append((-score, int(comp["x"]), int(comp["y"])))
    ranked.sort()
    return ranked


def locate_agent(frame: np.ndarray, bg: int, color: int | None, size: int | None):
    """已经认出角色颜色后，在当前帧里找它。"""
    if color is None:
        return None
    comps = [c for c in components(frame, bg) if c["color"] == color]
    if size is not None:
        close = [c for c in comps if _le(abs(int(c["size"]) - int(size)), 4)]
        if close:
            comps = close
    if not comps:
        return None
    comps.sort(key=lambda c: (c["size"], c["y"], c["x"]))
    return int(comps[0]["x"]), int(comps[0]["y"])


def detect_translation(prev: np.ndarray, now: np.ndarray, bg: int):
    """两帧之间哪个小色块平移了。用来认角色、认方向。"""
    prev_c = [c for c in components(prev, bg) if _le(c["size"], 24)]
    now_c = [c for c in components(now, bg) if _le(c["size"], 24)]
    best = None
    best_key = None
    for p in prev_c:
        for n in now_c:
            if p["color"] != n["color"]:
                continue
            if abs(int(p["size"]) - int(n["size"])) > 4:
                continue
            dx = int(n["x"]) - int(p["x"])
            dy = int(n["y"]) - int(p["y"])
            dist = abs(dx) + abs(dy)
            if dist == 0 or dist > 4:
                continue
            unique = 0 if sum(1 for c in prev_c if c["color"] == p["color"]) == 1 else 1
            key = (unique, dist, int(n["size"]))
            better = best_key is None
            if not better:
                for ka, kb in zip(key, best_key):
                    if ka != kb:
                        better = _lt(ka, kb)
                        break
            if better:
                best_key = key
                best = (dx, dy, int(n["color"]), int(n["size"]), int(n["x"]), int(n["y"]))
    return best


def _downsample(frame: np.ndarray, size: int = 16) -> str:
    block = 64 // size
    rows = []
    for r in range(size):
        cells = []
        for c in range(size):
            patch = frame[r * block : (r + 1) * block, c * block : (c + 1) * block]
            cells.append(str(int(np.bincount(patch.ravel(), minlength=16).argmax())))
        rows.append("".join(cells))
    return "\n".join(rows)


def _action_from_id(action_id: int, x: int = 32, y: int = 32) -> GameAction:
    # 每次 from_name 拿新对象。公开本踩过坑：改枚举上的旧 ACTION6 坐标会串台。
    name = ID_TO_NAME.get(int(action_id), "ACTION1")
    if hasattr(GameAction, "from_name"):
        action = GameAction.from_name(name)
    elif hasattr(GameAction, "from_id"):
        action = GameAction.from_id(int(action_id))
    else:
        action = getattr(GameAction, name)
    if name == "ACTION6" and hasattr(action, "set_data"):
        action.set_data({"x": _clamp63(x), "y": _clamp63(y)})
    return action


def _parse_llm_line(text: str, avail: set[int]) -> tuple[int, int, int] | None:
    blob = (text or "").upper()
    match = re.search(r"ACTION\s*6[^\d]{0,8}(\d{1,2})[^\d]{1,8}(\d{1,2})", blob)
    if match and 6 in avail:
        return 6, int(match.group(1)), int(match.group(2))
    match = re.search(r"CLICK[^\d]{0,8}(\d{1,2})[^\d]{1,8}(\d{1,2})", blob)
    if match and 6 in avail:
        return 6, int(match.group(1)), int(match.group(2))
    match = re.search(r"ACTION\s*([1-7])", blob)
    if match:
        action_id = int(match.group(1))
        if action_id in avail or action_id == 7:
            return action_id, 32, 32
    if "RESET" in blob:
        return 0, 0, 0
    return None


class GraphMemory:
    """一张图：画面哈希 -> 还没试的动作；试过没变就标死。"""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.succ: dict[str, dict[str, str]] = {}
        self.dir_map = dict(DEFAULT_DIRS)
        self.simple_order = list(SIMPLE_IDS)
        self.agent_color: int | None = None
        self.agent_size: int | None = None
        self.nav_target: tuple[int, int] | None = None
        self.visit: dict[tuple[int, int], int] = {}
        self.flash: dict[tuple[int, int], int] = {}
        self.flash_steps = 0
        self.flash_mask: np.ndarray | None = None
        self.recent = deque(maxlen=12)

    def clear_level(self) -> None:
        # 键位和角色颜色跨关保留；地图每关清空。
        self.nodes.clear()
        self.succ.clear()
        self.flash.clear()
        self.flash_steps = 0
        self.flash_mask = None
        self.recent.clear()
        self.visit.clear()
        self.nav_target = None

    def update_flash(self, prev: np.ndarray | None, now: np.ndarray, agent_color: int | None) -> None:
        """同一格反复闪、又不是角色，才从哈希里抠掉。面积太大就放弃，免得把真墙抹掉。"""
        if prev is None or prev.shape != now.shape:
            return
        self.flash_steps += 1
        changed = np.argwhere(prev != now)
        for y, x in changed:
            self.flash[(int(x), int(y))] = self.flash.get((int(x), int(y)), 0) + 1
        if self.flash_steps >= DYN_WARMUP and self.flash_steps % 8 == 0:
            need = max(3, int(self.flash_steps * DYN_RATE))
            mask = np.zeros_like(now, dtype=bool)
            for (x, y), n in self.flash.items():
                if n >= need and _in_bounds(x, y):
                    if agent_color is None or now[y, x] != agent_color:
                        mask[y, x] = True
            if mask.any() and _lt(float(mask.mean()), DYN_MAX_FRAC):
                self.flash_mask = mask
            else:
                self.flash_mask = None

    def paint(self, frame: np.ndarray, bg: int) -> np.ndarray:
        masked = mask_hud(frame, bg)
        if self.flash_mask is not None:
            masked = masked.copy()
            masked[self.flash_mask] = bg
        return masked

    def refresh_target(self, frame: np.ndarray, bg: int) -> None:
        """找一个最像按钮、又不是自己的小色块，当软导航目标。"""
        ranked = click_rank(components(frame, bg), frame)
        pos = locate_agent(frame, bg, self.agent_color, self.agent_size)
        for _score, x, y in ranked:
            if pos is not None and _le(abs(x - pos[0]) + abs(y - pos[1]), 1):
                continue
            self.nav_target = (x, y)
            return
        self.nav_target = None

    def ensure(self, h: str, frame: np.ndarray, bg: int, avail: set[int]) -> None:
        if h in self.nodes:
            return
        order: list[str] = []
        meta: dict[str, tuple[int, int, int]] = {}
        # 先方向和交互。一上来乱点会把走路关玩死。
        for sid in self.simple_order:
            if sid in avail:
                key = f"s{sid}"
                order.append(key)
                meta[key] = (sid, 32, 32)
        if 6 in avail:
            ranked = click_rank(components(frame, bg), frame)[:MAX_CLICKS]
            keep = ranked[:CLICK_KEEP]
            rest = ranked[CLICK_KEEP:]
            random.shuffle(rest)
            for _score, x, y in keep + rest:
                key = f"c{x}_{y}"
                if key in meta:
                    continue
                order.append(key)
                meta[key] = (6, x, y)
        self.nodes[h] = {"order": order, "tested": set(), "meta": meta}
        self.succ.setdefault(h, {})

    def record(self, prev_h: str | None, key: str | None, now_h: str) -> None:
        if prev_h is None or key is None:
            return
        node = self.nodes.get(prev_h)
        if not node or key not in node["meta"]:
            return
        node["tested"].add(key)
        if now_h != prev_h:
            self.succ.setdefault(prev_h, {})[key] = now_h

    def untested(self, h: str, kind: str | None = None) -> list[str]:
        node = self.nodes.get(h)
        if not node:
            return []
        keys = [k for k in node["order"] if k not in node["tested"]]
        if kind == "s":
            keys = [k for k in keys if k.startswith("s")]
        elif kind == "c":
            keys = [k for k in keys if k.startswith("c")]
        return keys

    def has_untested(self) -> bool:
        return any(self.untested(h) for h in self.nodes)

    def nav_pick(self, local: list[str], painted: np.ndarray | None, bg: int) -> str | None:
        """软偏向目标：只在没踩过的格子上走，全贪心会撞墙来回抖。"""
        if painted is None or self.agent_color is None or self.nav_target is None:
            return None
        if random.random() >= NAV_BIAS:
            return None
        pos = locate_agent(painted, bg, self.agent_color, self.agent_size)
        if pos is None:
            return None
        tx, ty = self.nav_target
        px, py = pos
        best = None
        best_d = 10**9
        for key in local:
            if not key.startswith("s"):
                continue
            sid = int(key[1:])
            if sid not in self.dir_map:
                continue
            dx, dy = self.dir_map[sid]
            nx, ny = px + dx, py + dy
            if self.visit.get((nx, ny), 0) > 0:
                continue
            dist = abs(nx - tx) + abs(ny - ty)
            if _lt(dist, best_d):
                best_d = dist
                best = key
        return best

    def bfs_step(self, start: str, kind: str) -> str | None:
        """沿已知边走一步，朝最近「还有 kind 类没试动作」的点。"""
        prev: dict[str, tuple[str | None, str | None]] = {start: (None, None)}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for key, nxt in self.succ.get(cur, {}).items():
                if nxt in prev:
                    continue
                prev[nxt] = (cur, key)
                if self.untested(nxt, kind):
                    node = nxt
                    while prev[node][0] is not None and prev[node][0] != start:
                        node = prev[node][0]
                    return prev[node][1]
                queue.append(nxt)
        return None

    def pick(self, start: str, painted: np.ndarray | None = None, bg: int = 0, kinds: tuple[str, ...] = ("s", "c")) -> str | None:
        """默认先全图走路再点击。点选关认出来之后反过来，避免在墙上空走。"""
        for kind in kinds:
            local = self.untested(start, kind)
            if local:
                if kind == "s":
                    if "s5" in local:
                        return "s5"
                    nav = self.nav_pick(local, painted, bg)
                    if nav is not None:
                        return nav
                return local[0]
            step = self.bfs_step(start, kind)
            if step is not None:
                return step
        return None


class MyAgent(Agent):
    """教程关收割技能纸；后面关 A* 猎目标；Gemma 只当顾问。"""

    # 官方后面关权重大。6 关预算加起来已超过 1500，1500 会把第 5、6 关直接掐死。
    MAX_ACTIONS = 8000
    GLOBAL_TIME_LIMIT_S = 8 * 60 * 60
    GLOBAL_RESERVE_S = 20 * 60
    LLM_MAX_CALLS = 8
    LLM_MAX_NEW_TOKENS = 48

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        self.started_at = time.monotonic()
        self.g = GraphMemory()
        dirs = [1, 2, 3, 4]
        random.shuffle(dirs)
        # 每个新画面先 ACTION5：走进目标格立刻走开，交互关会永远点不到。
        self.g.simple_order = [5] + dirs
        self.skill = SkillSheet()
        self.blocked: set[tuple[int, int]] = set()
        self.failed_hunt_colors: set[int] = set()
        self.pending_interact = False
        self.hunt_color: int | None = None
        self.last_hunt_on = False
        self.compose_clicks = 0
        self.last_pos: tuple[int, int] | None = None
        self.level = -1
        self.level_spent = 0
        self.level_resets = 0
        self.prev_h: str | None = None
        self.prev_key: str | None = None
        self.prev_paint: np.ndarray | None = None
        self.abandoned = False
        self.llm_calls = 0
        self.no_change = 0
        self.tried_undo = False
        self.ask_llm_once = False
        _start_llm_background()

    @property
    def name(self) -> str:
        return f"{super().name}.skill-hunt"

    def _remaining_global(self) -> float:
        deadline = _SUBMISSION_STARTED_AT + self.GLOBAL_TIME_LIMIT_S - self.GLOBAL_RESERVE_S
        return max(0.0, deadline - time.monotonic())

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        if self.abandoned:
            return True
        if self.action_counter == 0:
            return False
        return _le(self._remaining_global(), 0)

    def _ask_llm(self, frame: np.ndarray, avail: set[int], latest_frame: FrameData):
        if not _LLM_STATE["ready"] or self.llm_calls >= self.LLM_MAX_CALLS:
            return None
        if _le(self._remaining_global(), 30):
            return None
        comps = sorted(components(frame, _bg_color(frame)), key=lambda c: c["size"])[:8]
        obj_txt = "; ".join(
            f"c{c['color']} n={c['size']} xy={c['x']},{c['y']}" for c in comps
        ) or "none"
        levels = getattr(latest_frame, "levels_completed", "?")
        prompt = (
            "Unknown ARC-AGI-3 game. Reply ONE line: ACTION1-5, ACTION6 x y, ACTION7, RESET.\n"
            f"available={sorted(avail)} levels={levels} spent={self.level_spent} nochg={self.no_change}\n"
            f"genre={self.skill.genre} goal_color={self.skill.goal_color} "
            f"walls={self.skill.wall_colors} failed={sorted(self.failed_hunt_colors)}\n"
            f"objects={obj_txt}\n"
            f"grid16=\n{_downsample(frame)}\n"
        )
        try:
            processor = _LLM_STATE["processor"]
            model = _LLM_STATE["model"]
            messages = [
                {"role": "system", "content": "You choose the next game action. No extra words."},
                {"role": "user", "content": prompt},
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = processor(text=text, return_tensors="pt")
            first_param = next(model.parameters())
            inputs = {k: v.to(first_param.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]
            t0 = time.perf_counter()
            with _LLM_LOCK:
                output = model.generate(**inputs, max_new_tokens=self.LLM_MAX_NEW_TOKENS, do_sample=False)
            raw = processor.decode(output[0][input_len:], skip_special_tokens=True)
            self.llm_calls += 1
            parsed = _parse_llm_line(raw, avail)
            print("LLM", self.game_id, "s", round(time.perf_counter() - t0, 2), parsed, flush=True)
            return parsed
        except Exception as exc:
            print("LLM generate failed:", type(exc).__name__, exc, flush=True)
            return None

    def _emit(self, action_id: int, x: int, y: int, key: str, reason: str) -> GameAction:
        if action_id == 0:
            action = GameAction.RESET
            self.prev_h = None
            self.prev_key = None
            self.prev_paint = None
        else:
            action = _action_from_id(action_id, x, y)
            self.prev_key = key
        action.reasoning = reason
        return action

    def _lookup(self, now_h: str, key: str) -> tuple[int, int, int]:
        node = self.g.nodes.get(now_h)
        if node and key in node["meta"]:
            return node["meta"][key]
        for other in self.g.nodes.values():
            if key in other["meta"]:
                return other["meta"][key]
        if key.startswith("s") and key[1:].isdigit():
            return int(key[1:]), 32, 32
        if key.startswith("c") and "_" in key:
            try:
                xs, ys = key[1:].split("_", 1)
                return 6, int(xs), int(ys)
            except ValueError:
                pass
        return 1, 32, 32

    def _replay_level(self, why: str) -> None:
        """重打本关：墙和技能纸留下，地图哈希清掉。不要退整局。"""
        self.level_resets += 1
        self.level_spent = 0
        self.g.clear_level()
        self.pending_interact = False
        self.compose_clicks = 0
        self.last_pos = None
        self.no_change = 0
        self.tried_undo = False
        self.ask_llm_once = True
        self.last_hunt_on = False
        print(
            why,
            self.game_id,
            "n",
            self.level_resets,
            "blocked",
            len(self.blocked),
            "walls",
            self.skill.wall_colors,
            flush=True,
        )

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # 没开局或死了：只能 RESET，否则服务器 400。
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_h = None
            self.prev_key = None
            self.prev_paint = None
            return self._emit(0, 0, 0, "reset", "reset after not-played/game-over")

        frame = _grid(latest_frame)
        bg = _bg_color(frame)
        avail = _avail_ids(latest_frame)
        levels = int(getattr(latest_frame, "levels_completed", getattr(latest_frame, "score", 0)) or 0)

        # 过关了：记下「怎么赢的」；地图清空，键位保留。
        if levels != self.level:
            if levels > self.level and self.level >= 0:
                kind = "unknown"
                win_color = self.hunt_color
                if self.prev_key == "s5":
                    kind = "interact"
                    self.skill.note_effect("interact")
                elif self.prev_key and self.prev_key.startswith("c"):
                    kind = "click"
                    if self.prev_paint is not None:
                        try:
                            xs, ys = self.prev_key[1:].split("_", 1)
                            cx, cy = int(xs), int(ys)
                            if _in_bounds(cx, cy):
                                win_color = int(self.prev_paint[cy, cx])
                                self.skill.note_click_color(win_color)
                        except ValueError:
                            pass
                self.skill.note_win(win_color, kind)
                print(
                    "level-up",
                    self.game_id,
                    "to",
                    levels,
                    "spent",
                    self.level_spent,
                    "genre",
                    self.skill.genre,
                    "win",
                    kind,
                    "goal_color",
                    self.skill.goal_color,
                    "clicks",
                    self.skill.useful_click_colors,
                    flush=True,
                )
            self.g.clear_level()
            self.blocked.clear()
            self.failed_hunt_colors.clear()
            self.pending_interact = False
            self.last_hunt_on = False
            self.compose_clicks = 0
            self.last_pos = None
            self.level = levels
            self.level_spent = 0
            self.level_resets = 0
            self.prev_h = None
            self.prev_key = None
            self.prev_paint = None
            self.no_change = 0
            self.tried_undo = False
            self.ask_llm_once = False

        # 认角色：哪个小色块跟着 ACTION1-4 平移。
        if self.prev_paint is not None and self.prev_key and self.prev_key.startswith("s"):
            trans = detect_translation(self.prev_paint, mask_hud(frame, bg), bg)
            if trans is not None:
                dx, dy, color, size, _x, _y = trans
                self.g.agent_color = color
                self.g.agent_size = size
                sid = int(self.prev_key[1:])
                if sid in DEFAULT_DIRS:
                    self.g.dir_map[sid] = (dx, dy)
                if sid in (1, 2, 3, 4):
                    self.skill.note_effect("move")
                if sid == 5:
                    self.skill.note_effect("interact")

        painted = self.g.paint(frame, bg)
        self.g.update_flash(self.prev_paint, painted, self.g.agent_color)
        painted = self.g.paint(frame, bg)
        now_h = hash_frame(painted)

        # 记下上一步：变了连边，没变标死。
        if self.prev_h is not None and now_h == self.prev_h:
            self.no_change += 1
            if self.prev_key and self.prev_key.startswith("s") and self.last_pos is not None:
                sid = int(self.prev_key[1:])
                if sid in self.g.dir_map:
                    dx, dy = self.g.dir_map[sid]
                    wall = (self.last_pos[0] + dx, self.last_pos[1] + dy)
                    if _in_bounds(wall[0], wall[1]):
                        self.blocked.add(wall)
                        wc = int(painted[wall[1], wall[0]])
                        if wc != bg:
                            self.skill.note_wall_color(wc)
        else:
            self.no_change = 0
            if self.prev_key and self.prev_key.startswith("c"):
                self.skill.note_effect("click")
                if self.prev_paint is not None:
                    try:
                        xs, ys = self.prev_key[1:].split("_", 1)
                        cx, cy = int(xs), int(ys)
                        if _in_bounds(cx, cy):
                            clicked = int(self.prev_paint[cy, cx])
                            if clicked != bg:
                                self.skill.note_click_color(clicked)
                    except ValueError:
                        pass
                # 点开了门：旧撞墙记录作废，否则 A* 仍以为过不去。
                self.blocked.clear()
        self.g.record(self.prev_h, self.prev_key, now_h)
        self.g.ensure(now_h, painted, bg, avail)
        self.g.recent.append(now_h)
        self.prev_h = now_h
        self.prev_paint = painted
        self.level_spent += 1

        if self.last_hunt_on:
            if self.hunt_color is not None:
                self.failed_hunt_colors.add(int(self.hunt_color))
            self.last_hunt_on = False

        pos = locate_agent(painted, bg, self.g.agent_color, self.g.agent_size)
        if pos is not None:
            self.g.visit[pos] = self.g.visit.get(pos, 0) + 1
            self.last_pos = pos
        comps = components(painted, bg)
        prefers = self.skill.prefer_colors()
        pref_set = set(prefers)
        cands = pick_hunt_candidates(comps, pos, prefers, skip_colors=self.failed_hunt_colors)
        pref_cands = [c for c in cands if c[2] in pref_set]
        other_cands = [c for c in cands if c[2] not in pref_set]
        walls = planning_walls(comps, self.blocked, list(pref_set), pos, self.skill.wall_colors)
        pref_plan = plan_hunt(pos, pref_cands, walls, self.g.dir_map) if pos is not None else None
        other_plan = plan_hunt(pos, other_cands, walls, self.g.dir_map) if pos is not None else None
        planned = pref_plan if pref_plan is not None else other_plan
        if planned is not None:
            if planned[0] == "on":
                self.g.nav_target = (planned[1], planned[2])
                self.hunt_color = planned[3]
            else:
                self.g.nav_target = (planned[2], planned[3])
                self.hunt_color = planned[4]
        elif cands:
            self.g.nav_target = (cands[0][0], cands[0][1])
            self.hunt_color = cands[0][2]
        elif self.level_spent % NAV_REFRESH == 1:
            self.g.refresh_target(painted, bg)

        can_explore = self.g.has_untested()
        policy = stall_policy(
            self.level_spent,
            can_explore,
            self._remaining_global(),
            self.level,
            has_plan=planned is not None,
            level_resets=self.level_resets,
        )
        if policy == "quit":
            self.abandoned = True
            print("abandon", self.game_id, "spent", self.level_spent, "level", self.level, flush=True)
            fallback = 1 if 1 in avail else (sorted(avail)[0] if avail else 1)
            return self._emit(fallback, 32, 32, "s1", "rhae-abandon")
        if policy == "reset":
            self._replay_level("stall-reset")
            return self._emit(0, 0, 0, "reset", "stall-reset")

        if self.pending_interact and 5 in avail:
            self.pending_interact = False
            self.last_hunt_on = True
            return self._emit(5, 32, 32, "s5", "hunt-arrive")

        # 还没认出角色：新格子先按一下。认出来之后只在目标上按，避免走路关步数翻倍。
        if self.g.agent_color is None and "s5" in self.g.untested(now_h, "s") and 5 in avail:
            return self._emit(5, 32, 32, "s5", "new-cell-interact")

        # 纯点选且还在教程关：不要 A* 乱走。后面关即使教程是点选，也允许走路组合。
        click_only = (
            self.skill.genre == "click"
            and self.skill.move_hits == 0
            and _le(self.level, 0)
        )
        move = next_skill_move(
            pref_plan if not click_only else None,
            other_plan if not click_only else None,
            self.level,
            6 in avail and _lt(self.compose_clicks, 12) and pos is not None,
            bool(pref_cands),
        )
        if move in ("hunt-pref", "hunt-other"):
            use = pref_plan if move == "hunt-pref" else other_plan
            if use is not None and 5 in avail:
                if use[0] == "on":
                    self.last_hunt_on = True
                    return self._emit(5, 32, 32, "s5", "hunt-on-goal")
                _kind, sid, tx, ty, _color = use
                if sid in avail:
                    dx, dy = self.g.dir_map.get(sid, (0, 0))
                    if pos is not None and pos[0] + dx == tx and pos[1] + dy == ty:
                        self.pending_interact = True
                    return self._emit(sid, 32, 32, f"s{sid}", "hunt-astar")
        if move == "compose":
            click_key = pick_compose_click(
                self.g,
                now_h,
                painted,
                bg,
                self.skill.useful_click_colors,
                prefers,
                comps,
            )
            if click_key is not None:
                self.compose_clicks += 1
                action_id, x, y = self._lookup(now_h, click_key)
                print("compose-click", self.game_id, "xy", x, y, "n", self.compose_clicks, flush=True)
                return self._emit(action_id, x, y, click_key, "compose-click")

        # ABAB 死循环：把正在走的边当已探索，逼它换动作。
        rec = list(self.g.recent)
        if len(rec) >= 6 and rec[-1] == rec[-3] == rec[-5] and rec[-2] == rec[-4] == rec[-6]:
            node = self.g.nodes.get(now_h)
            if node and self.prev_key:
                node["tested"].add(self.prev_key)

        if self.ask_llm_once:
            self.ask_llm_once = False
            asked = self._ask_llm(painted, avail, latest_frame)
            if asked is not None:
                action_id, x, y = asked
                k = f"s{action_id}" if action_id != 6 else f"c{x}_{y}"
                return self._emit(action_id, x, y, k, "llm-after-reset")

        kinds = ("c", "s") if self.skill.genre == "click" else ("s", "c")
        key = self.g.pick(now_h, painted, bg, kinds=kinds)
        if key is None:
            # 图穷了：关卡重打，墙留下，哈希清掉。
            if _lt(self.level_resets, REPLAY_RESETS):
                self._replay_level("stranded-reset")
                return self._emit(0, 0, 0, "reset", "stranded-reset")
            if 7 in avail and not self.tried_undo:
                self.tried_undo = True
                return self._emit(7, 32, 32, "s7", "undo-stranded")
            asked = self._ask_llm(painted, avail, latest_frame)
            if asked is not None:
                action_id, x, y = asked
                k = f"s{action_id}" if action_id != 6 else f"c{x}_{y}"
                return self._emit(action_id, x, y, k, "llm")
            leftover = [a for a in sorted(avail) if a != 0]
            action_id = leftover[0] if leftover else 1
            x = y = 32
            if action_id == 6:
                x, y = random.randint(8, 55), random.randint(8, 55)
            return self._emit(action_id, x, y, "leftover", "leftover")

        action_id, x, y = self._lookup(now_h, key)
        if self.level_spent % 40 == 1:
            print("tick", self.game_id, "spent", self.level_spent, "nodes", len(self.g.nodes), "key", key, flush=True)
        return self._emit(action_id, x, y, key, f"graph {key}")
