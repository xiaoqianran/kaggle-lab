"""Scout-then-Ask agent for ARC Prize 2026 / ARC-AGI-3.

Always-on policy is a cheap novelty scout (hash frames, try untested
simple moves, click small rare-color blobs). Gemma-4 is loaded once and
asked only when the scout is stuck. If the LLM is missing or slow, the
scout still produces a valid Kaggle submission.

This file is the only strategy. Notebook plumbing (wheels / gateway /
dummy parquet) stays the official starter pattern.
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

SIMPLE_IDS = (1, 2, 3, 4, 5, 7)
def _le(left, right) -> bool:
    return not (left > right)


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
        model = None
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


def _hash_frame(masked: np.ndarray) -> str:
    return hashlib.md5(masked.tobytes()).hexdigest()[:16]


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
                    in_bounds = ny >= 0 and _le(ny, height - 1) and nx >= 0 and _le(nx, width - 1)
                    if in_bounds and not seen[ny, nx] and frame[ny, nx] == color:
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
                }
            )
    return comps


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
        action.set_data({"x": int(max(0, min(63, x))), "y": int(max(0, min(63, y)))})
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


class MyAgent(Agent):
    """Novelty scout with sparse Gemma-4 asks."""

    MAX_ACTIONS = 480
    GAME_TIME_LIMIT_S = 18 * 60
    GLOBAL_TIME_LIMIT_S = 8 * 60 * 60
    GLOBAL_RESERVE_S = 15 * 60
    STUCK_BEFORE_LLM = 6
    LLM_MAX_CALLS = 18
    LLM_MAX_NEW_TOKENS = 48
    LLM_TIMEOUT_S = 45
    MAX_CONSEC_RESETS = 3

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        self.started_at = time.monotonic()
        self.seen: set[str] = set()
        self.dead: set[tuple[str, str]] = set()
        self.prev_hash: str | None = None
        self.prev_key: str | None = None
        self.stuck = 0
        self.llm_calls = 0
        self.consec_resets = 0
        self.level = -1
        _start_llm_background()

    @property
    def name(self) -> str:
        return f"{super().name}.scout-gemma4"

    def _remaining_global(self) -> float:
        deadline = _SUBMISSION_STARTED_AT + self.GLOBAL_TIME_LIMIT_S - self.GLOBAL_RESERVE_S
        return max(0.0, deadline - time.monotonic())

    def _remaining_game(self) -> float:
        return max(0.0, min(self.GAME_TIME_LIMIT_S - (time.monotonic() - self.started_at), self._remaining_global()))

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        if self.action_counter == 0:
            return False
        return _le(self._remaining_game(), 0)

    def _reset_level(self) -> None:
        self.seen.clear()
        self.dead.clear()
        self.prev_hash = None
        self.prev_key = None
        self.stuck = 0

    def _record(self, now_hash: str) -> None:
        if self.prev_hash is None or self.prev_key is None:
            return
        if now_hash == self.prev_hash:
            self.dead.add((self.prev_hash, self.prev_key))
            self.stuck += 1
        else:
            self.stuck = 0

    def _candidates(self, frame: np.ndarray, bg: int, avail: set[int], state_hash: str) -> list[tuple[str, int, int, int]]:
        items: list[tuple[float, str, int, int, int]] = []
        for action_id in SIMPLE_IDS:
            if action_id not in avail:
                continue
            key = f"s{action_id}"
            if (state_hash, key) in self.dead:
                continue
            items.append((float(action_id), key, action_id, 32, 32))
        if 6 in avail:
            counts = np.bincount(frame.ravel(), minlength=16)
            total = max(int(frame.size), 1)
            for comp in _components(frame, bg):
                if _le(comp["size"], 0) or comp["size"] >= 81:
                    continue
                key = f"c{comp['x']}_{comp['y']}"
                if (state_hash, key) in self.dead:
                    continue
                rarity = 1.0 - counts[comp["color"]] / total
                score = 0.55 * rarity + 0.45 * (1.0 / (1.0 + comp["size"]))
                items.append((-score, key, 6, comp["x"], comp["y"]))
        items.sort(key=lambda row: row[0])
        return [(key, action_id, x, y) for _, key, action_id, x, y in items]

    def _ask_llm(self, frame: np.ndarray, avail: set[int], latest_frame: FrameData) -> tuple[int, int, int] | None:
        if not _LLM_STATE["ready"] or self.llm_calls >= self.LLM_MAX_CALLS:
            return None
        if _le(self._remaining_game(), 19):
            return None
        comps = _components(frame, _bg_color(frame))
        comps = sorted(comps, key=lambda c: c["size"])[:8]
        obj_txt = "; ".join(f"c{c['color']} n={c['size']} xy={c['x']},{c['y']}" for c in comps) or "none"
        levels = getattr(latest_frame, "levels_completed", getattr(latest_frame, "score", "?"))
        prompt = (
            "Play an unknown ARC-AGI-3 game. Reply with ONE line only:\n"
            "ACTION1 ACTION2 ACTION3 ACTION4 ACTION5 ACTION6 x y ACTION7 RESET\n"
            f"available={sorted(avail)} levels={levels} stuck={self.stuck}\n"
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

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._reset_level()
            self.consec_resets += 1
            action = GameAction.RESET
            action.reasoning = "reset after not-played/game-over"
            return action

        level_now = int(getattr(latest_frame, "levels_completed", getattr(latest_frame, "score", 0)) or 0)
        if level_now != self.level:
            self.level = level_now
            self._reset_level()
            self.consec_resets = 0

        frame = _grid(latest_frame)
        bg = _bg_color(frame)
        masked = _mask_hud(frame, bg)
        state_hash = _hash_frame(masked)
        self.seen.add(state_hash)
        self._record(state_hash)
        avail = _avail_ids(latest_frame)

        if self.stuck >= self.STUCK_BEFORE_LLM:
            asked = self._ask_llm(masked, avail, latest_frame)
            if asked is not None:
                action_id, x, y = asked
                if action_id == 0:
                    self._reset_level()
                    action = GameAction.RESET
                    action.reasoning = "llm reset"
                    self.prev_hash, self.prev_key = state_hash, "reset"
                    return action
                key = f"s{action_id}" if action_id != 6 else f"c{x}_{y}"
                self.prev_hash, self.prev_key = state_hash, key
                action = _action_from_id(action_id, x, y)
                action.reasoning = f"llm {ID_TO_NAME.get(action_id)} {x},{y}"
                return action

        options = self._candidates(masked, bg, avail, state_hash)
        if options:
            key, action_id, x, y = options[0]
            self.prev_hash, self.prev_key = state_hash, key
            self.consec_resets = 0
            action = _action_from_id(action_id, x, y)
            action.reasoning = f"scout {key}"
            return action

        if _le(self.consec_resets, self.MAX_CONSEC_RESETS - 1):
            self._reset_level()
            self.consec_resets += 1
            action = GameAction.RESET
            action.reasoning = "scout exhausted, reset"
            return action

        fallback_id = random.choice(sorted(avail) or [1])
        action = _action_from_id(fallback_id, random.randint(0, 63), random.randint(0, 63))
        action.reasoning = "random leftover"
        return action
