# ARC Prize 2026 — ARC-AGI-3 Starter（重写）

在官方 starter 的 **Kaggle 竞赛绑定和提交骨架** 上，把随机 agent 换成自己的 **Scout-then-Ask**：

- 竞赛来源不改：`arc-prize-2026-arc-agi-3`
- 无网提交：`enable_internet=false`
- 加速器：`NvidiaRtxPro6000`（ARC-AGI-3 专属，配额消耗更快）
- 模型：官方 `google/gemma-4/transformers/gemma-4-31b-it/1`（4-bit；加载失败则纯探索器仍能交卷）

Kaggle kernel：`chutianqiu/arc-prize-2026-arc-agi-3-starter`

## 别人在干什么（2026-08 抽样）

| 路线 | 代表 notebook | 模型 / 方法 | 机器 | 实际在做的事 |
|---|---|---|---|---|
| 官方随机 | `inversion/arc3-sample-submission-random-agent` | 无 | CPU | 证明提交管道 |
| Stochastic Goose | `inversion/arc3-sample-submission-stochastic-goose` | 在线 CNN | T4×2 | 学「哪一步会改变画面」 |
| Just Explore / 图搜索 | `pscamillo/arc-prize-2026-arc-agi-3-starter` | 无大模型 | T4 | 状态图 + 点小色块 |
| Duck harness | `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` | 搜索 / harness | T4 / RTX | 目前高分抄得最多 |
| Gemma-4-31B | `ko0kip/arc-agi-3-gemma-4-31b-reflection-agent` | Gemma 4 31B + vLLM | **RTX Pro 6000** | 每步问模型 + 显著点击 |
| Gemma 4 26B NVFP4 | `pranshubahadur/xcaliber-aa3-nvidia-gemma-4-26b-nvfp4` | 量化 26B | RTX Pro 6000 | 自定义 runtime |
| GPT-OSS-120B | `gregkamradt/arc-agi-3-gpt-oss-120b` | 120B + vLLM | RTX Pro 6000 | 官方模板：**不要每步都问**，太慢 |

排行榜头部（约 1.5–2.7）主要是 **duck / 搜索 harness**，不是裸大模型。官方 GPT-OSS 文档也写了：RTX 上 120B 很慢，只在真正需要推理时拉一把。

## 我们怎么参考、怎么写自己的

1. **竞赛接口照抄官方 starter**（唯一安全的交卷方式）  
   `pip` 离线装 `arc-agi` → 写 `MyAgent` → rerun 时等 `gateway:8001` → `python main.py --agent myagent` → commit 时写 dummy `submission.parquet`。
2. **策略不要抄 2000 行 vLLM 笔记本**。那些依赖别人的 wheel dataset，断了就交不了。
3. **能得分的部分先做成探索器**（Goose / Just Explore 的共识：先找会改变画面的动作，点击小而稀有的色块）。
4. **大模型当顾问**：卡住 N 步才问 Gemma-4，输出一行 `ACTION1`…`ACTION6 x y`。
5. **时间预算必须短**：超时拿不到 scorecard，整次 rerun 失败。

本地策略文件：`my_agent.py`。改它，再重新打包 notebook。

## 怎么推到 Kaggle

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
kaggle kernels push -p notebooks/v06-chutianqiu-arc-agi-3 \
  --accelerator NvidiaRtxPro6000 \
  --timeout 32400
kaggle kernels status chutianqiu/arc-prize-2026-arc-agi-3-starter
```

Commit 成功后，打开 kernel → **Submit to Competition** → 选 `submission.parquet`。  
第一次先 Save & Run All，确认 dummy parquet 和模型路径打印正常，再花每日提交次数。
