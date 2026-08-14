# labs — Agent / Proxy 最小实验

这里是 **编号实验**，不是学习轨。每条实验一个目录、一个 `run.py`、同一套入口：

```bash
python -m kaggle_lab 014 run
python main.py 015 smoke          # 旧写法，等价
python -m kaggle_lab debate       # 按意图，不必记编号
```

目录册（标题、别名、是否扣 Proxy、是否危险）在 [`kaggle_lab/catalog.py`](../kaggle_lab/catalog.py)。  
`python -m kaggle_lab list` 按「你想做什么」打印，而不是按文件夹堆砌。

## 地图

| 组 | 实验 | 别名 | 默认动作 |
|----|------|------|----------|
| 先跑 | 001 Model Proxy | `001` `proxy` | `auth` |
| 先跑 | 003 模型表 | `003` `models` | `list` |
| 先跑 | 010 额度 | `010` `quota` | `show` |
| Proxy | 002 tool-call / 006 judge / 008 ReAct | | `run` |
| 平台 | 005 Benchmark / 007 MCP | `bench` `mcp` | `validate` / `tools` |
| CAMEL | 011 底座 → 012 roleplay → 013 scout → **014 workforce** | `workforce` | `smoke` / `run` |
| 产品 | **015 双智对谈** | `015` `dual` | `run` · Web 见目录内 README |
| SAE | 004 正式客户端 / 009 dry-run | | **默认不开考** |

## 加一条实验

1. 复制任意一个现有目录的骨架：`run.py` 用 argparse subcommands。
2. `ROOT = Path(__file__).resolve().parents[2]`，并把 `ROOT` 放进 `sys.path`（或 `from kaggle_lab.paths import ensure_import_path`）。
3. 在 `catalog.py` 的 `LABS` 里加一条 `Lab`（id、aliases、title、group、`dangerous` 如需）。
4. 不登记也能被 `list` 扫到，但没有别名、没有意图入口。

共享能力：`kaggle_lab.proxy` / `kaggle_lab.camel_proxy` / `kaggle_lab.usage`。  
`common.*` 是给旧 `from common.proxy import` 的兼容层。
