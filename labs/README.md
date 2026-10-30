# labs — Agent / Proxy 最小实验

编号实验，不是学习轨。每个实验一个目录、一个 `run.py`、自己解析参数：

```bash
python labs/014-camel-workforce-bench/run.py run   # 从仓库根
cd labs/014-camel-workforce-bench && python run.py run
```

本表就是索引（标题、默认动作、备注都在这里，不再有单独的登记表）。

## 地图

| 组 | 实验 | 默认动作 | 备注 |
|----|------|----------|------|
| 先跑 | [001 Model Proxy 聊天](001-model-proxy/) | `auth` | 所有扣 AI $ 的实验从这里通 |
| 先跑 | [003 模型表](003-list-models/) | `list` | 模型表只存这一份 |
| 先跑 | [010 额度看板](010-quota-dashboard/) | `show` | GPU / AI $ 额度 |
| Proxy | [002 单轮 Tool Calling](002-tool-call/) | `run` | |
| Proxy | [006 多模型裁判](006-judge-llm/) | `run` | |
| Proxy | [008 手写 ReAct](008-agent-loop/) | `run` | |
| 平台 | [005 Benchmark 最小闭环](005-benchmark-task/) | `validate` | |
| 平台 | [007 Kaggle MCP](007-mcp-harness/) | `tools` | |
| CAMEL | [011 CAMEL 底座](011-camel-proxy/) | `smoke` | 012–014 的底座 |
| CAMEL | [012 CAMEL RolePlaying](012-camel-roleplay/) | `run` | |
| CAMEL | [013 竞赛侦察简报](013-camel-kaggle-crew/) | `scout` | |
| CAMEL | [014 Workforce 写 Benchmark](014-camel-workforce-bench/) | `run` | publish 必须 `--i-accept` |
| 产品 | [015 双智对谈](015-dual-agent-chat/) | `run` | Web 见目录内 README |
| SAE | [004 SAE 正式客户端](004-sae/) | `whoami` | **默认不开考**，start/submit 强制 `--i-accept` |
| SAE | [009 SAE dry-run](009-sae-better/) | `dry-run` | 假卷，不占考试次数 |

## 加一条实验

1. 复制任意一个现有目录的骨架：`run.py` 用 argparse subcommands。
2. `ROOT = Path(__file__).resolve().parents[2]`，并把 `ROOT` 放进 `sys.path`（或 `from kaggle_lab.paths import ensure_import_path`）。
3. 在上表加一行。

共享能力：`kaggle_lab.proxy` / `kaggle_lab.camel_proxy` / `kaggle_lab.usage`。
