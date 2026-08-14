# ARC Prize 2026 — ARC-AGI-3（按官方题目设计）

在官方 starter 的 **Kaggle 竞赛绑定和提交骨架** 上，策略换成自己的 **在线世界模型 + RHAE 预算**（不是抄公开 duck-harness，也不特化 `ls20`/`ft09`/`vc33`）。

完整推理见 [DESIGN.md](DESIGN.md)。实现见 `my_agent.py`。合成格子单测：`python3 -m unittest test_world_agent.py`。

- 竞赛来源不改：`arc-prize-2026-arc-agi-3`
- 无网：`enable_internet=false`
- 加速器：`NvidiaRtxPro6000`
- 模型：`google/gemma-4/transformers/gemma-4-31b-it/1`（卡住才问；加载失败仍交卷）

Kaggle kernel：`chutianqiu/arc-prize-2026-arc-agi-3-starter`

## 官方题目 → 策略

| 官方约束 | 我们怎么做 |
|---|---|
| 无说明书、hidden 集 | 开局探测分类 nav / point / hybrid，禁止 `game_id` 分支 |
| ACTION1-4 语义上/下/左/右；ACTION6 不给可点格 | 用小色块平移建导航图；点击只打连通块并记死点 |
| RHAE `(H/A)^2`，100 步相对 10 步只剩 0.01 | 本关过预算则关卡 RESET 一次按图重打；再超则放弃 |
| 没打完所有关有游戏分上限 | 优先把能过的关过完，不把整次提交耗在一关上 |
| Swarm 并行 + 9 小时 | 不要用 18 分钟墙钟揠死全局；用动作预算。Gemma 加锁稀缺调用 |

改 `my_agent.py` 后运行 `python3 build_notebook.py` 再推 kernel。

## 怎么推到 Kaggle

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
kaggle kernels push -p notebooks/v06-chutianqiu-arc-agi-3 \
  --accelerator NvidiaRtxPro6000 \
  --timeout 32400
kaggle kernels status chutianqiu/arc-prize-2026-arc-agi-3-starter
```

Commit 成功后，打开 kernel → **Submit to Competition** → 选 `submission.parquet`。
Save & Run All 只验证 dummy parquet / 模型路径，**不会**打分。
