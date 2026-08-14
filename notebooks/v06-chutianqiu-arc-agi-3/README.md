# ARC Prize 2026 — ARC-AGI-3

**换环境 / 下一任 agent 先读 [HANDOFF.md](HANDOFF.md)。** 那里有已交卷分数、禁止事项、怎么再提交。

核心打法：**教程关收割技能纸，后面关 A\* 追赢过的颜色；走不到就先点开关再走。整局 8000 步。** 方案见 [DESIGN.md](DESIGN.md)。公开本对照见 [SURVEY.md](SURVEY.md)。

## 线上（2026-08-14 已核实）

- 公开榜 **0.17**（提交 `55511330`，kernel **v10**）。上一笔 0.06。
- Kernel：https://www.kaggle.com/code/chutianqiu/arc-prize-2026-arc-agi-3-starter
- 竞赛 slug 不改：`arc-prize-2026-arc-agi-3`
- Save and Run All 不打游戏。真打要点 Submit to Competition（必须 `-k` + `-v`，不要交假 parquet）。

## 约束

- 无网；机器：`NvidiaRtxPro6000`
- 模型：`gemma-4-31b-it`（卡住才问）
- 不按游戏名写死；不绑第三方 dataset

## 改代码

```bash
python3 -m unittest test_world_agent.py
python3 build_notebook.py
```

不要手改 ipynb。GitHub 用 MCP 以 `xiaoqianran` 写入；`cursor[bot]` 的 `git push` 对这个仓库 403。`push_files` 内容里不能出现英文小于号。
