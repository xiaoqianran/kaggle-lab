# kaggle-lab

Kaggle 实验台：编号 + 主题目录做最小可跑实验。

账号 API：`KAGGLE_API_TOKEN` / `~/.kaggle/access_token`（`KGAT_…`）。

---

## SAE 禁考（必读）

> **SAE** = Standardized Agent Exam。默认 **不开考、不注册、不交卷**（用户要求）。  
> 详见 **[docs/SAE.md](docs/SAE.md)**。允许：`009 dry-run` 假卷。

---

## 结构（Agent / Proxy 主线）

```text
001–010  基础 Proxy / tools / MCP / SAE(禁) / benchmark / judge / quota
011-camel-proxy           CAMEL ↔ Model Proxy
012-camel-roleplay        双智能体 RolePlaying
013-camel-kaggle-crew     CAMEL + 真 Kaggle 信号简报
014-camel-workforce-bench 检索→辩论→写 task→(可选)005 push   ★
015-dual-agent-chat       双智对谈（自由辩论，Model Proxy）
```

## 环境

```bash
source .venv/bin/activate   # Python ≥3.11
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
uv pip install 'camel-ai==0.2.90' && uv pip install 'mcp==1.9.4'
# 005/014 validate 还需: uv pip install -e vendor/kaggle-benchmarks
```

## 最有意义的命令

```bash
# CAMEL 底座
python main.py 011 smoke

# 竞赛侦察简报
python main.py 013 scout

# ★ Workforce 风格流水线：Scout→Debate→Author→本地评测
python main.py 014 run
python main.py 014 show
# 发布到 Kaggle Benchmarks（确认后）
python main.py 014 publish --push --run-remote --i-accept

# 额度
python main.py 010 show

# 双智对谈（自由多轮 A↔B）
python main.py 015 models
python main.py 015 run --preset debate --rounds 3

# Web UI（Pages 演示 / 本地网关 Live）
# https://xiaoqianran.github.io/kaggle-lab/
cd 015-dual-agent-chat/web && npm i && npm run build && cd ../..
python 015-dual-agent-chat/gateway.py
```

## 实验一览

| # | 作用 | 调 Proxy？ |
|---|------|-----------|
| 001–003 | chat / tools / 模型表 | 部分 |
| 004/009 | SAE（禁考 / dry-run） | dry-run |
| 005–006 | Benchmark / Judge | ✅ |
| 007–008 | MCP / 手写 ReAct | 008 ✅ |
| 010 | 额度 | ❌ |
| 011–013 | CAMEL 系列 | ✅ |
| **014** | **多角色写 Benchmark 并可选 push** | ✅ |
| **015** | **双智对谈（自动 A↔B）** | ✅ |

## 约定

- 密钥 / artifacts / logs / vendor 不入库  
- **不自动开 SAE 考**  
- 014 publish 必须 `--i-accept`

仓库：https://github.com/xiaoqianran/kaggle-lab

---

## ARC Prize 2026 / ARC-AGI-3

代码在 [`notebooks/v06-chutianqiu-arc-agi-3/`](notebooks/v06-chutianqiu-arc-agi-3/)。

**换环境先读 [`notebooks/v06-chutianqiu-arc-agi-3/HANDOFF.md`](notebooks/v06-chutianqiu-arc-agi-3/HANDOFF.md)。**

- Kaggle 账号 `chutianqiu`；kernel https://www.kaggle.com/code/chutianqiu/arc-prize-2026-arc-agi-3-starter
- 2026-08-14 已 Submit to Competition：提交 `55511330`，公开榜 **0.17**（此前 0.06）
- Save and Run All 不等于上榜；`cursor[bot]` 对本仓库 `git push` 为 403，用 GitHub MCP 以 `xiaoqianran` 写入
