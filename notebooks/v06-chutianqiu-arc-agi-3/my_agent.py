"""RHAE-budgeted world-model agent for ARC Prize 2026 / ARC-AGI-3.

Design: notebooks/v06-chutianqiu-arc-agi-3/DESIGN.md
Official scoring is (human_actions / ai_actions)^2 per level. This file
builds an online map from ACTION1-4 translations and ACTION6 clicks, then
spends a hard action budget. Gemma-4 is a rare advisor, not the policy.
No game_id special cases: the Kaggle set is hidden.
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
    "/kaggle/input/google/gemma-4/transformers/gemma-4-26b-a4b-it/1",
    "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1",
    "/kaggle/input/google/gemma-4/transformers/gemma-4-e2b-it/1",
]

# Official WASD mapping: 1=up, 2=down, 3=left, 4=right. (x, y), y grows down.
DEFAULT_DIRS: dict[int, tuple[int, int]] = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}
CARDINAL = (1, 2, 3, 4)
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

EST_HUMAN_ACTIONS = 40
REPLAY_AFTER = 80
ABANDON_AFTER = 160
MAX_LEVEL_RESETS = 1


def _le(left: float, right: float) -> bool:
    return not (left > right)


def _lt(left: float, right: float) -> bool:
    return not (left >= right)


def _clamp63(value: int) -> int:
    return int(max(0, min(63, value)))


def rhae_level_score(human_actions: float, ai_actions: float) -> float:
    if _le(ai_actions, 0):
        return 0.0
    score = (human_actions / ai_actions) ** 2
    return 1.15 if score > 1.15 else score


def should_replay(level_spent: int, level_resets: int, has_map: bool) -> bool:
    return (
        level_spent >= REPLAY_AFTER
        and _lt(level_resets, MAX_LEVEL_RESETS)
        and has_map
    )


def should_abandon(
    level_spent: int,
    has_short_path: bool,
    remaining_s: float,
) -> bool:
    if _le(remaining_s, 0):
        return True
    if level_spent >= ABANDON_AFTER and not has_short_path:
        return True
    return False


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
    thread = threading.Thread(target=_load_llm, name="gemma4-load", daemon=True)
    thread.start()


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


def _mask_hud(frame: np.ndarray, bg: int) -> np.ndarray:
    masked = frame.copy()
    height, width = masked.shape
    wide = max(24, width // 2)

    def widest_run(row: int) -> int:
        best = run = 0
        for col in range(width):
            if masked[row, col] != bg:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best

    for row in list(range(0, 6)) + list(range(height - 6, height)):
        if widest_run(row) >= wide:
            masked[row, :] = bg
    return masked


def _hash_frame(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()[:16]


def _in_bounds(x: int, y: int) -> bool:
    return x >= 0 and _le(x, 63) and y >= 0 and _le(y, 63)


def _components(frame: np.ndarray, bg: int) -> list[dict[str, Any]]:
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
            comps.append(
                {
                    "color": color,
                    "size": len(pixels),
                    "x": int(xs[k]),
                    "y": int(ys[k]),
                    "pixels": pixels,
                }
            )
    return comps


def _static_view(frame: np.ndarray, bg: int, mover_pixels: list[tuple[int, int]] | None) -> np.ndarray:
    static = frame.copy()
    if mover_pixels:
        for py, px in mover_pixels:
            static[py, px] = bg
    return static


def detect_translation(
    prev: np.ndarray, now: np.ndarray, bg: int
) -> tuple[int, int, int, int] | None:
    """Match small blobs between two frames. Overlap-safe (walking onto paint)."""
    prev_c = [c for c in _components(prev, bg) if _le(c["size"], 24)]
    now_c = [c for c in _components(now, bg) if _le(c["size"], 24)]
    if not prev_c or not now_c:
        return None
    best: tuple[int, int, int, int] | None = None
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
            # Prefer unique-color, then shorter steps, then smaller sprites.
            key = (
                0 if sum(1 for c in prev_c if c["color"] == p["color"]) == 1 else 1,
                dist,
                int(n["size"]),
            )
            better = best_key is None
            if not better:
                for ka, kb in zip(key, best_key):
                    if ka != kb:
                        better = _lt(ka, kb)
                        break
            if better:
                best_key = key
                best = (dx, dy, int(n["color"]), int(n["size"]))
    return best


def locate_mover(
    frame: np.ndarray,
    bg: int,
    color: int | None,
    prev_xy: tuple[int, int] | None,
    expected_size: int | None,
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    if color is None:
        return None
    matches = []
    for comp in _components(frame, bg):
        if comp["color"] != color:
            continue
        if expected_size is not None and abs(comp["size"] - expected_size) > max(4, expected_size):
            continue
        if comp["size"] > 24:
            continue
        matches.append(comp)
    if not matches:
        return None
    if prev_xy is not None:
        matches.sort(
            key=lambda c: abs(c["x"] - prev_xy[0]) + abs(c["y"] - prev_xy[1])
        )
    else:
        matches.sort(key=lambda c: c["size"])
    best = matches[0]
    return (int(best["x"]), int(best["y"])), list(best["pixels"])


def click_rank(
    comps: list[dict[str, Any]],
    frame: np.ndarray,
    dead: set[tuple[int, int, str]],
    static_hash: str,
    boost: set[tuple[int, int]],
) -> list[tuple[float, int, int]]:
    counts = np.bincount(frame.ravel(), minlength=16)
    total = max(int(frame.size), 1)
    ranked: list[tuple[float, int, int]] = []
    for comp in comps:
        size = int(comp["size"])
        if _le(size, 0) or size >= 81:
            continue
        x, y = int(comp["x"]), int(comp["y"])
        if (x, y, static_hash) in dead:
            continue
        rarity = 1.0 - counts[int(comp["color"])] / total
        score = 0.55 * rarity + 0.45 * (1.0 / (1.0 + size))
        if (x, y) in boost:
            score += 0.35
        ranked.append((-score, x, y))
    ranked.sort()
    return ranked


def shortest_first_action(
    edges: dict[tuple[int, int, int], tuple[int, int]],
    start: tuple[int, int],
    is_goal,
) -> int | None:
    if is_goal(start):
        return None
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {start: None}
    found: tuple[int, int] | None = None
    while queue:
        pos = queue.popleft()
        if pos != start and is_goal(pos):
            found = pos
            break
        for action_id in CARDINAL:
            nxt = edges.get((pos[0], pos[1], action_id))
            if nxt is None or nxt in parent:
                continue
            parent[nxt] = (pos, action_id)
            queue.append(nxt)
    if found is None:
        return None
    cur = found
    first = None
    while parent[cur] is not None:
        prev, action_id = parent[cur]
        first = action_id
        cur = prev
    return first


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
    name = ID_TO_NAME.get(int(action_id), "ACTION1")
    if hasattr(GameAction, "from_name"):
        action = GameAction.from_name(name)
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


class WorldModel:
    """Online map: mover, walls, clicks, and a cheap mode label."""

    def __init__(self) -> None:
        self.mode = "probe"
        self.dir_map = dict(DEFAULT_DIRS)
        self.dir_votes: dict[int, list[tuple[int, int]]] = {}
        self.mover_color: int | None = None
        self.mover_size: int | None = None
        self.mover_pos: tuple[int, int] | None = None
        self.mover_pixels: list[tuple[int, int]] | None = None
        self.edges: dict[tuple[int, int, int], tuple[int, int]] = {}
        self.tried: set[tuple[int, int, int]] = set()
        self.walls: set[tuple[int, int]] = set()
        self.dead_clicks: set[tuple[int, int, str]] = set()
        self.useful_clicks: list[tuple[int, int]] = []
        self.visit: dict[tuple[int, int], int] = {}
        self.win_cells: list[tuple[int, int]] = []
        self.win_clicks: list[tuple[int, int]] = []
        self.boost: set[tuple[int, int]] = set()
        self.prev_grid: np.ndarray | None = None
        self.prev_static_hash: str | None = None
        self.prev_pos: tuple[int, int] | None = None
        self.pending: tuple[int, int, int, str] | None = None
        self.level = -1
        self.level_spent = 0
        self.level_resets = 0
        self.move_hits = 0
        self.click_hits = 0
        self.interact_hits = 0
        self.probe_n = 0
        self.no_change = 0
        self.seen_hashes: set[str] = set()
        self.replay_target: tuple[int, int] | None = None
        self.last_win_pending: tuple[int, int, int, str] | None = None
        self.want_interact = False

    def clear_layout(self, keep_skills: bool) -> None:
        self.edges.clear()
        self.tried.clear()
        self.walls.clear()
        self.dead_clicks.clear()
        self.useful_clicks.clear()
        self.visit.clear()
        self.boost.clear()
        self.prev_grid = None
        self.prev_static_hash = None
        self.prev_pos = None
        self.pending = None
        self.level_spent = 0
        self.probe_n = 0
        self.no_change = 0
        self.seen_hashes.clear()
        self.replay_target = None
        self.mover_pos = None
        self.mover_pixels = None
        self.want_interact = False
        self.win_cells.clear()
        self.win_clicks.clear()
        if not keep_skills:
            self.mode = "probe"
            self.dir_map = dict(DEFAULT_DIRS)
            self.dir_votes.clear()
            self.mover_color = None
            self.mover_size = None
            self.move_hits = 0
            self.click_hits = 0
            self.interact_hits = 0
            self.level_resets = 0

    def classify(self) -> str:
        nav = self.move_hits >= 2
        point = self.click_hits >= 1
        if nav and point:
            self.mode = "hybrid"
        elif nav:
            self.mode = "nav"
        elif point:
            self.mode = "point"
        elif self.probe_n >= 8:
            self.mode = "unknown"
        else:
            self.mode = "probe"
        return self.mode

    def _lock_dir(self, action_id: int, dx: int, dy: int) -> None:
        votes = self.dir_votes.setdefault(action_id, [])
        votes.append((dx, dy))
        tally: dict[tuple[int, int], int] = {}
        for item in votes:
            tally[item] = tally.get(item, 0) + 1
        self.dir_map[action_id] = max(tally.items(), key=lambda kv: kv[1])[0]

    def observe(self, grid: np.ndarray, bg: int, levels: int) -> None:
        if self.prev_grid is None or self.pending is None:
            return
        action_id, px, py, _key = self.pending
        if levels > self.level and self.level >= 0:
            self.last_win_pending = self.pending
            if action_id == 6:
                self.win_clicks.append((px, py))
            elif self.prev_pos is not None:
                self.win_cells.append(self.prev_pos)
            elif self.mover_pos is not None:
                self.win_cells.append(self.mover_pos)
            self.classify()
            return
        trans = detect_translation(self.prev_grid, grid, bg)
        located = locate_mover(grid, bg, self.mover_color, self.prev_pos, self.mover_size)
        if located:
            self.mover_pos, self.mover_pixels = located
        static = _static_view(grid, bg, self.mover_pixels)
        static_hash = _hash_frame(static)
        changed_static = static_hash != self.prev_static_hash
        frame_hash = _hash_frame(grid)
        if frame_hash in self.seen_hashes and not trans and not changed_static:
            self.no_change += 1
        else:
            self.no_change = 0
        self.seen_hashes.add(frame_hash)

        if action_id in CARDINAL and trans is not None:
            dx, dy, color, size = trans
            self.move_hits += 1
            self.mover_color = color
            self.mover_size = size
            self._lock_dir(action_id, dx, dy)
            now_pos = (self.prev_pos[0] + dx, self.prev_pos[1] + dy) if self.prev_pos else None
            if located:
                now_pos = located[0]
            if self.prev_pos is not None and now_pos is not None:
                self.edges[(self.prev_pos[0], self.prev_pos[1], action_id)] = now_pos
                self.tried.add((self.prev_pos[0], self.prev_pos[1], action_id))
            if now_pos is not None:
                self.mover_pos = now_pos
                self.visit[now_pos] = self.visit.get(now_pos, 0) + 1
            if changed_static:
                self.want_interact = True
        elif action_id in CARDINAL and located and self.prev_pos is not None and located[0] != self.prev_pos:
            now_pos = located[0]
            dx = now_pos[0] - self.prev_pos[0]
            dy = now_pos[1] - self.prev_pos[1]
            self.move_hits += 1
            self._lock_dir(action_id, dx, dy)
            self.edges[(self.prev_pos[0], self.prev_pos[1], action_id)] = now_pos
            self.tried.add((self.prev_pos[0], self.prev_pos[1], action_id))
            self.mover_pos = now_pos
            self.visit[now_pos] = self.visit.get(now_pos, 0) + 1
            if changed_static:
                self.want_interact = True
        elif action_id in CARDINAL and self.prev_pos is not None:
            dx, dy = self.dir_map[action_id]
            wall = (self.prev_pos[0] + dx, self.prev_pos[1] + dy)
            if _in_bounds(wall[0], wall[1]):
                self.walls.add(wall)
            self.tried.add((self.prev_pos[0], self.prev_pos[1], action_id))

        if action_id == 6:
            if changed_static:
                self.click_hits += 1
                self.useful_clicks.append((px, py))
                self.boost.add((px, py))
            elif self.prev_static_hash is not None:
                self.dead_clicks.add((px, py, self.prev_static_hash))

        if action_id == 5 and changed_static:
            self.interact_hits += 1
            if self.prev_pos is not None:
                self.useful_clicks.append(self.prev_pos)

        self.classify()

    def sync(self, grid: np.ndarray, bg: int) -> str:
        located = locate_mover(grid, bg, self.mover_color, self.mover_pos, self.mover_size)
        if located:
            self.mover_pos, self.mover_pixels = located
        static = _static_view(grid, bg, self.mover_pixels)
        self.prev_static_hash = _hash_frame(static)
        self.prev_grid = grid
        self.prev_pos = self.mover_pos
        return self.prev_static_hash

    def has_map(self) -> bool:
        return bool(self.edges) or bool(self.useful_clicks) or bool(self.win_cells) or bool(self.win_clicks)

    def has_short_path(self) -> bool:
        if self.mover_pos is None:
            return bool(self.win_clicks) or bool(self.useful_clicks)
        goals = self.win_cells[-1:] + self.useful_clicks[-3:]
        if not goals:
            return False
        goal_set = set(goals)

        def is_goal(pos: tuple[int, int]) -> bool:
            return pos in goal_set

        return shortest_first_action(self.edges, self.mover_pos, is_goal) is not None

    def interesting_target(self, comps: list[dict[str, Any]]) -> tuple[int, int] | None:
        if self.replay_target is not None:
            return self.replay_target
        if self.win_cells:
            return self.win_cells[-1]
        small = [
            c
            for c in comps
            if _lt(c["size"], 25)
            and (self.mover_color is None or c["color"] != self.mover_color)
        ]
        if not small:
            return None
        if self.mover_pos is None:
            small.sort(key=lambda c: c["size"])
            return int(small[0]["x"]), int(small[0]["y"])
        small.sort(
            key=lambda c: abs(c["x"] - self.mover_pos[0]) + abs(c["y"] - self.mover_pos[1])
        )
        return int(small[0]["x"]), int(small[0]["y"])

    def untried_cardinals(self, pos: tuple[int, int], avail: set[int]) -> list[int]:
        out = []
        for action_id in CARDINAL:
            if action_id not in avail:
                continue
            if (pos[0], pos[1], action_id) in self.tried:
                continue
            dx, dy = self.dir_map[action_id]
            wall = (pos[0] + dx, pos[1] + dy)
            if wall in self.walls:
                continue
            out.append(action_id)
        return out

    def pick_nav(self, comps: list[dict[str, Any]], avail: set[int]) -> tuple[int, int, int, str] | None:
        pos = self.mover_pos
        if pos is None:
            return None
        if self.want_interact and 5 in avail:
            self.want_interact = False
            self.tried.add((pos[0], pos[1], 5))
            return 5, pos[0], pos[1], "step-interact"
        target = self.interesting_target(comps)
        if target is not None and pos == target and 5 in avail and (pos[0], pos[1], 5) not in self.tried:
            self.tried.add((pos[0], pos[1], 5))
            return 5, pos[0], pos[1], "at-target"
        untried = self.untried_cardinals(pos, avail)
        if untried:
            if target is not None:
                untried.sort(
                    key=lambda a: abs(pos[0] + self.dir_map[a][0] - target[0])
                    + abs(pos[1] + self.dir_map[a][1] - target[1])
                )
            action_id = untried[0]
            return action_id, pos[0], pos[1], f"nav{action_id}"

        def node_open(node: tuple[int, int]) -> bool:
            return bool(self.untried_cardinals(node, avail)) or (target is not None and node == target)

        step = shortest_first_action(self.edges, pos, node_open)
        if step is not None and step in avail:
            return step, pos[0], pos[1], f"bfs{step}"

        if 5 in avail and (pos[0], pos[1], 5) not in self.tried:
            self.tried.add((pos[0], pos[1], 5))
            return 5, pos[0], pos[1], "interact"

        return None

    def pick_point(
        self, comps: list[dict[str, Any]], frame: np.ndarray, static_hash: str, avail: set[int]
    ) -> tuple[int, int, int, str] | None:
        if 6 not in avail:
            return None
        ranked = click_rank(comps, frame, self.dead_clicks, static_hash, self.boost)
        for _score, x, y in ranked:
            return 6, x, y, f"click{x}_{y}"
        if self.win_clicks:
            x, y = self.win_clicks[-1]
            if (x, y, static_hash) not in self.dead_clicks:
                return 6, x, y, "winclick"
        for y in range(8, 56, 8):
            for x in range(8, 56, 8):
                if (x, y, static_hash) in self.dead_clicks:
                    continue
                return 6, x, y, f"lat{x}_{y}"
        return None

    def pick_probe(self, comps: list[dict[str, Any]], avail: set[int]) -> tuple[int, int, int, str] | None:
        if _lt(self.probe_n, 4):
            action_id = CARDINAL[self.probe_n]
            if action_id in avail:
                self.probe_n += 1
                return action_id, 32, 32, f"probe{action_id}"
            self.probe_n += 1
        if self.probe_n == 4:
            self.probe_n += 1
            if 5 in avail:
                return 5, 32, 32, "probe5"
        if 6 in avail:
            small = [c for c in comps if _lt(c["size"], 81)]
            small.sort(key=lambda c: c["size"])
            if small and _lt(self.probe_n, 9):
                self.probe_n += 1
                c = small[(self.probe_n - 6) % len(small)]
                return 6, int(c["x"]), int(c["y"]), "probe6"
        self.probe_n += 1
        return None

    def policy(
        self, grid: np.ndarray, bg: int, avail: set[int]
    ) -> tuple[int, int, int, str] | None:
        comps = _components(grid, bg)
        static_hash = self.prev_static_hash or _hash_frame(grid)
        need_probe = self.mode == "probe" or (
            _lt(self.probe_n, 6) and (self.mode == "unknown" or self.mover_pos is None)
        )
        if need_probe:
            probed = self.pick_probe(comps, avail)
            if probed is not None:
                return probed
        if self.mode in ("nav", "hybrid", "probe", "unknown"):
            nav = self.pick_nav(comps, avail)
            if nav is not None:
                return nav
        if self.mode in ("point", "hybrid", "probe", "unknown"):
            point = self.pick_point(comps, grid, static_hash, avail)
            if point is not None:
                return point
        if 7 in avail and self.no_change >= 6:
            return 7, 32, 32, "undo"
        return None


class MyAgent(Agent):
    """World model plus RHAE action budget; Gemma-4 only when stuck."""

    MAX_ACTIONS = 720
    GLOBAL_TIME_LIMIT_S = 8 * 60 * 60
    GLOBAL_RESERVE_S = 20 * 60
    LLM_MAX_CALLS = 8
    LLM_MAX_NEW_TOKENS = 48
    STUCK_BEFORE_LLM = 10

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        self.started_at = time.monotonic()
        self.world = WorldModel()
        self.abandoned = False
        self.llm_calls = 0
        self.consec_resets = 0
        _start_llm_background()

    @property
    def name(self) -> str:
        return f"{super().name}.rhae-world"

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

    def _ask_llm(
        self, frame: np.ndarray, avail: set[int], latest_frame: FrameData
    ) -> tuple[int, int, int] | None:
        if not _LLM_STATE["ready"] or self.llm_calls >= self.LLM_MAX_CALLS:
            return None
        if _le(self._remaining_global(), 30):
            return None
        comps = _components(frame, _bg_color(frame))
        comps = sorted(comps, key=lambda c: c["size"])[:8]
        obj_txt = "; ".join(
            f"c{c['color']} n={c['size']} xy={c['x']},{c['y']}" for c in comps
        ) or "none"
        levels = getattr(latest_frame, "levels_completed", getattr(latest_frame, "score", "?"))
        prompt = (
            "Unknown ARC-AGI-3 game. Reply ONE line: ACTION1-5, ACTION6 x y, ACTION7, RESET.\n"
            f"mode={self.world.mode} pos={self.world.mover_pos} "
            f"available={sorted(avail)} levels={levels} "
            f"spent={self.world.level_spent} nochg={self.world.no_change}\n"
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
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = processor(text=text, return_tensors="pt")
            first_param = next(model.parameters())
            inputs = {k: v.to(first_param.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]
            t0 = time.perf_counter()
            with _LLM_LOCK:
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.LLM_MAX_NEW_TOKENS,
                    do_sample=False,
                )
            elapsed = time.perf_counter() - t0
            raw = processor.decode(output[0][input_len:], skip_special_tokens=True)
            self.llm_calls += 1
            parsed = _parse_llm_line(raw, avail)
            print("LLM", self.game_id, "s", round(elapsed, 2), "->", raw[:80], parsed, flush=True)
            return parsed
        except Exception as exc:
            print("LLM generate failed:", type(exc).__name__, exc, flush=True)
            return None

    def _emit(self, action_id: int, x: int, y: int, reason: str) -> GameAction:
        if action_id == 0:
            action = GameAction.RESET
        else:
            action = _action_from_id(action_id, x, y)
        action.reasoning = reason
        if action_id == 0:
            self.world.pending = None
            self.world.prev_grid = None
        else:
            self.world.pending = (action_id, x, y, reason)
        return action

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.consec_resets += 1
            self.world.pending = None
            self.world.prev_grid = None
            return self._emit(0, 0, 0, "reset after not-played/game-over")

        frame = _grid(latest_frame)
        bg = _bg_color(frame)
        masked = _mask_hud(frame, bg)
        avail = _avail_ids(latest_frame)
        levels = int(getattr(latest_frame, "levels_completed", getattr(latest_frame, "score", 0)) or 0)

        if self.world.prev_grid is not None and self.world.pending is not None:
            self.world.observe(masked, bg, levels)

        if levels != self.world.level:
            keep = self.world.level >= 0 and levels > self.world.level
            if keep:
                print(
                    "level-up",
                    self.game_id,
                    "to",
                    levels,
                    "mode",
                    self.world.mode,
                    "spent",
                    self.world.level_spent,
                    flush=True,
                )
            self.world.clear_layout(keep_skills=keep)
            self.world.level = levels
            self.world.level_resets = 0
            self.consec_resets = 0

        static_hash = self.world.sync(masked, bg)
        self.world.level_spent += 1
        _ = static_hash

        if should_abandon(self.world.level_spent, self.world.has_short_path(), self._remaining_global()):
            self.abandoned = True
            print(
                "abandon",
                self.game_id,
                "mode",
                self.world.mode,
                "spent",
                self.world.level_spent,
                flush=True,
            )
            fallback = 1 if 1 in avail else (sorted(avail)[0] if avail else 1)
            return self._emit(fallback, 32, 32, "rhae-abandon")

        if should_replay(self.world.level_spent, self.world.level_resets, self.world.has_map()):
            self.world.level_resets += 1
            self.world.level_spent = 0
            self.world.replay_target = self.world.interesting_target(_components(masked, bg))
            self.world.pending = None
            self.world.prev_grid = None
            self.world.probe_n = 6
            print("replay-reset", self.game_id, "target", self.world.replay_target, flush=True)
            return self._emit(0, 0, 0, "rhae-replay")

        picked = self.world.policy(masked, bg, avail)
        if picked is None and self.world.no_change >= self.STUCK_BEFORE_LLM:
            asked = self._ask_llm(masked, avail, latest_frame)
            if asked is not None:
                action_id, x, y = asked
                return self._emit(action_id, x, y, "llm")

        if picked is None:
            leftover = [a for a in sorted(avail) if a != 0]
            action_id = leftover[0] if leftover else 1
            x = y = 32
            if action_id == 6:
                x, y = random.randint(8, 55), random.randint(8, 55)
            picked = (action_id, x, y, "leftover")

        if self.world.level_spent % 40 == 1:
            print(
                "tick",
                self.game_id,
                "mode",
                self.world.mode,
                "pos",
                self.world.mover_pos,
                "spent",
                self.world.level_spent,
                "pick",
                picked[3],
                flush=True,
            )
        return self._emit(picked[0], picked[1], picked[2], picked[3])
