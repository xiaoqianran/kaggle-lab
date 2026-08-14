"""ARC-AGI-3 智能体的假网格单测。

这个仓库没装 arcengine，测试会先塞假模块再 import my_agent。
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


class _Act:
    def __init__(self, name: str, value: int) -> None:
        self.name = name
        self.value = value
        self.reasoning = None
        self.data: dict = {}

    def set_data(self, data: dict) -> None:
        self.data = dict(data)

    def is_simple(self) -> bool:
        return self.name != "ACTION6"

    def is_complex(self) -> bool:
        return self.name == "ACTION6"


class _GameAction:
    RESET = _Act("RESET", 0)
    ACTION1 = _Act("ACTION1", 1)
    ACTION2 = _Act("ACTION2", 2)
    ACTION3 = _Act("ACTION3", 3)
    ACTION4 = _Act("ACTION4", 4)
    ACTION5 = _Act("ACTION5", 5)
    ACTION6 = _Act("ACTION6", 6)
    ACTION7 = _Act("ACTION7", 7)

    @classmethod
    def from_name(cls, name: str) -> _Act:
        template = getattr(cls, name)
        return _Act(template.name, template.value)

    @classmethod
    def from_id(cls, value: int) -> _Act:
        names = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]
        return cls.from_name(names[int(value)])


class _GameState:
    NOT_PLAYED = object()
    GAME_OVER = object()
    WIN = object()
    PLAYING = object()


class _Agent:
    MAX_ACTIONS = 80

    def __init__(self, *args, **kwargs) -> None:
        self.card_id = kwargs.get("card_id", "")
        self.game_id = kwargs.get("game_id", "test")
        self.agent_name = kwargs.get("agent_name", "t")
        self.action_counter = 0
        self.frames = []


def _install_stubs() -> None:
    if "arcengine" in sys.modules and hasattr(sys.modules["arcengine"], "GameAction"):
        return
    arcengine = types.ModuleType("arcengine")
    arcengine.FrameData = object
    arcengine.GameAction = _GameAction
    arcengine.GameState = _GameState
    agents = types.ModuleType("agents")
    agents_agent = types.ModuleType("agents.agent")
    agents_agent.Agent = _Agent
    agents.agent = agents_agent
    sys.modules["arcengine"] = arcengine
    sys.modules["agents"] = agents
    sys.modules["agents.agent"] = agents_agent


_install_stubs()

import my_agent as ma  # noqa: E402

ma._LLM_STATE["started"] = True
ma._LLM_STATE["failed"] = True
ma._LLM_STATE["ready"] = False


class Frame:
    def __init__(self, grid, state, levels, actions=(1, 2, 3, 4, 5, 6, 7)) -> None:
        self.frame = np.asarray(grid, dtype=np.int64)
        self.state = state
        self.levels_completed = levels
        self.available_actions = list(actions)
        self.score = levels
        self.full_reset = False


class ToyNav:
    """Open room: walk onto the goal cell, then ACTION5 to win."""

    def __init__(self) -> None:
        self.x, self.y = 8, 8
        self.goal = (22, 8)
        self.level = 0
        self.state = _GameState.PLAYING
        self.walls = set()
        for i in range(4, 28):
            self.walls.add((4, i))
            self.walls.add((28, i))
            self.walls.add((i, 4))
            self.walls.add((i, 20))

    def grid(self) -> np.ndarray:
        g = np.zeros((64, 64), dtype=np.int64)
        g[0, :] = 5
        for x, y in self.walls:
            g[y, x] = 1
        gx, gy = self.goal
        if (self.x, self.y) != self.goal:
            g[gy, gx] = 3
        g[self.y, self.x] = 2
        return g

    def step(self, action: _Act) -> Frame:
        aid = int(action.value)
        if aid == 0:
            self.x, self.y = 8, 8
            self.level = 0
            self.state = _GameState.PLAYING
            return Frame(self.grid(), self.state, self.level)
        if aid in ma.DEFAULT_DIRS:
            dx, dy = ma.DEFAULT_DIRS[aid]
            nx, ny = self.x + dx, self.y + dy
            if (nx, ny) not in self.walls and ma._in_bounds(nx, ny):
                self.x, self.y = nx, ny
        if aid == 5 and (self.x, self.y) == self.goal:
            self.level += 1
            self.state = _GameState.WIN
        if aid == 6:
            data = getattr(action, "data", {}) or {}
            if (int(data.get("x", -1)), int(data.get("y", -1))) == self.goal:
                self.level += 1
                self.state = _GameState.WIN
        return Frame(self.grid(), self.state, self.level)


class ToyPoint:
    """ACTION1-4 do nothing. Click the small rare blob to win."""

    def __init__(self) -> None:
        self.level = 0
        self.state = _GameState.PLAYING
        self.target = (40, 20)

    def grid(self) -> np.ndarray:
        g = np.zeros((64, 64), dtype=np.int64)
        g[0, :] = 4
        g[10:18, 10:18] = 1
        tx, ty = self.target
        g[ty, tx] = 7
        g[ty, tx + 1] = 7
        return g

    def step(self, action: _Act) -> Frame:
        aid = int(action.value)
        if aid == 6:
            data = getattr(action, "data", {}) or {}
            x, y = int(data.get("x", -1)), int(data.get("y", -1))
            tx, ty = self.target
            if x in (tx, tx + 1) and y == ty:
                self.level += 1
                self.state = _GameState.WIN
        return Frame(self.grid(), self.state, self.level)


def _play(agent: ma.MyAgent, env, first: Frame, limit: int = 180) -> tuple[int, Frame]:
    frame = first
    for _ in range(limit):
        if agent.is_done([], frame):
            break
        action = agent.choose_action([], frame)
        frame = env.step(action)
        agent.action_counter += 1
        if frame.state is _GameState.WIN:
            break
    return agent.action_counter, frame


class RhaeTests(unittest.TestCase):
    def test_quadratic_penalty(self) -> None:
        self.assertEqual(ma.rhae_level_score(10, 10), 1.0)
        self.assertEqual(ma.rhae_level_score(10, 20), 0.25)
        self.assertAlmostEqual(ma.rhae_level_score(10, 100), 0.01)
        self.assertEqual(ma.rhae_level_score(10, 5), 1.15)

    def test_governor_thresholds(self) -> None:
        self.assertTrue(ma.should_abandon(250, False, 10.0))
        self.assertFalse(ma.should_abandon(250, True, 10.0))
        self.assertTrue(ma.should_abandon(900, True, 10.0))
        self.assertTrue(ma.should_abandon(10, True, 0.0))


class VisionTests(unittest.TestCase):
    def test_hud_mask_clears_top_bar(self) -> None:
        frame = np.zeros((64, 64), dtype=np.int64)
        frame[0, :] = 9
        frame[10, 10] = 3
        masked = ma.mask_hud(frame, 0)
        self.assertEqual(int(masked[0, 0]), 0)
        self.assertEqual(int(masked[10, 10]), 3)

    def test_detect_translation_right(self) -> None:
        prev = np.zeros((64, 64), dtype=np.int64)
        now = np.zeros((64, 64), dtype=np.int64)
        prev[8, 8] = 2
        prev[8, 9] = 2
        now[8, 9] = 2
        now[8, 10] = 2
        got = ma.detect_translation(prev, now, 0)
        self.assertIsNotNone(got)
        dx, dy, color, size, _x, _y = got
        self.assertEqual((dx, dy, color, size), (1, 0, 2, 2))

    def test_click_rank_prefers_small_rare(self) -> None:
        frame = np.zeros((64, 64), dtype=np.int64)
        frame[5:20, 5:20] = 1
        frame[40, 40] = 7
        comps = ma.components(frame, 0)
        ranked = ma.click_rank(comps, frame)
        self.assertTrue(ranked)
        self.assertEqual((ranked[0][1], ranked[0][2]), (40, 40))

    def test_graph_marks_self_loop_dead(self) -> None:
        mem = ma.GraphMemory()
        frame = np.zeros((64, 64), dtype=np.int64)
        frame[4, 4] = 3
        h = ma.hash_frame(frame)
        mem.ensure(h, frame, 0, {1, 6})
        mem.record(h, "s1", h)
        self.assertIn("s1", mem.nodes[h]["tested"])
        self.assertEqual(mem.pick(h)[0], "c")

    def test_pick_walks_before_local_clicks(self) -> None:
        """邻格还有没试过的走路时，不要先在本格乱点。"""
        mem = ma.GraphMemory()
        a = np.zeros((64, 64), dtype=np.int64)
        b = np.zeros((64, 64), dtype=np.int64)
        a[4, 4] = 3
        b[4, 5] = 3
        ha, hb = ma.hash_frame(a), ma.hash_frame(b)
        mem.ensure(ha, a, 0, {1, 2, 6})
        mem.ensure(hb, b, 0, {1, 2, 6})
        mem.record(ha, "s1", ha)
        mem.record(ha, "s2", hb)
        self.assertEqual(mem.pick(ha), "s2")


class AgentLoopTests(unittest.TestCase):
    def _agent(self, game_id: str) -> ma.MyAgent:
        return ma.MyAgent(
            card_id="c",
            game_id=game_id,
            agent_name="t",
            ROOT_URL="http://local",
            record=False,
            arc_env=None,
        )

    def test_nav_reaches_goal(self) -> None:
        toy = ToyNav()
        agent = self._agent("nav")
        start = Frame(toy.grid(), _GameState.NOT_PLAYED, 0)
        action = agent.choose_action([], start)
        self.assertEqual(action.name, "RESET")
        agent.action_counter += 1
        playing = Frame(toy.grid(), _GameState.PLAYING, 0)
        steps, end = _play(agent, toy, playing, limit=160)
        self.assertIs(end.state, _GameState.WIN)
        self.assertEqual(end.levels_completed, 1)
        self.assertTrue(ma._le(steps, 140))

    def test_point_clicks_small_blob(self) -> None:
        toy = ToyPoint()
        agent = self._agent("point")
        start = Frame(toy.grid(), _GameState.NOT_PLAYED, 0)
        agent.choose_action([], start)
        agent.action_counter += 1
        playing = Frame(toy.grid(), _GameState.PLAYING, 0)
        steps, end = _play(agent, toy, playing, limit=40)
        self.assertIs(end.state, _GameState.WIN)
        self.assertTrue(ma._le(steps, 20))

    def test_parse_llm_click(self) -> None:
        self.assertEqual(ma._parse_llm_line("ACTION6 40 20", {1, 6}), (6, 40, 20))
        self.assertEqual(ma._parse_llm_line("ACTION3", {1, 2, 3}), (3, 32, 32))


if __name__ == "__main__":
    unittest.main()
