# kaggle-lab

Kaggle 实验台。打开仓库时先问自己：**我想做什么**，而不是先数 001–015。

两块，互不打架：

| 目录 | 给谁 | 里面是什么 |
|------|------|------------|
| [`labs/`](labs/) | 调模型、写 agent、碰 Kaggle API | 编号最小实验，统一 `run.py` |
| [`tracks/`](tracks/) | 跟一条研究/学习轨 | 分类、检测、深度、Instruct、ARC… |

共享代码在 [`kaggle_lab/`](kaggle_lab/)。目录册（别名、意图入口、危险标记）在 [`catalog.py`](kaggle_lab/catalog.py)。

账号 API：`KAGGLE_API_TOKEN` / `~/.kaggle/access_token`（`KGAT_…`）。

---

## 30 秒上手

```bash
source .venv/bin/activate    # Python ≥3.11
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
python -m kaggle_lab         # 打印「你想做什么」
```

CAMEL 实验（011–014）额外：

```bash
uv pip install 'camel-ai==0.2.90' && uv pip install 'mcp==1.9.4'
# 005 / 014 validate 还需: uv pip install -e vendor/kaggle-benchmarks
```

---

## 你想做什么

| 意图 | 命令 |
|------|------|
| 接通 Model Proxy（临时 key ≈ 1h） | `python -m kaggle_lab auth` |
| 跟模型聊一句 | `python -m kaggle_lab chat "你好"` |
| 看两个智能体对谈 | `python -m kaggle_lab debate --rounds 3` |
| 多角色写一道 Benchmark | `python -m kaggle_lab workforce` |
| 侦察一场竞赛 | `python -m kaggle_lab scout` |
| 看 GPU / AI 额度 | `python -m kaggle_lab quota` |
| 打开一条学习轨 | `python -m kaggle_lab track` |
| 列出全部入口 | `python -m kaggle_lab list` |

旧命令仍然可用：`python main.py 014 run`、`python main.py 015 models`。

Web 对谈（Pages 演示 / 本地 Live）：

- 演示：https://xiaoqianran.github.io/kaggle-lab/
- 本地：`python -m kaggle_lab gateway`（先 `cd labs/015-dual-agent-chat/web && npm i && npm run build`）

---

## 目录怎么读

```text
labs/001-model-proxy     先跑：auth / chat
labs/010-quota-dashboard 额度
labs/011-camel-proxy     CAMEL 底座
labs/014-…-bench         侦察→辩论→写 task→可选 push
labs/015-dual-agent-chat 双智对谈（CLI + Web）★产品

tracks/image-classification
tracks/object-detection
tracks/depth-estimation
tracks/instruct-t4
tracks/diffusion-gemma
tracks/arc-agi-3         换环境先读 HANDOFF.md（公开榜 0.17）

kaggle_lab/              目录册 + CLI + Proxy
data/                    模型表（003 dump）
docs/                    MCP / SAE / Skills
```

加实验：在 `labs/` 放一个带 `run.py` 的目录，再往 `catalog.py` 加一条 `Lab`。没登记的目录也会出现在 `list` 底部。

---

## SAE（默认不开考）

**SAE** = Standardized Agent Exam。默认 **不开考、不注册、不交卷**。  
详见 [docs/SAE.md](docs/SAE.md)。允许：`python -m kaggle_lab run 009 dry-run`。

`004 start/register/submit` 必须带 `--i-accept`，且用户书面确认。

---

## 约定

- 密钥 / artifacts / logs / vendor 不入库
- **不自动开 SAE 考**
- 014 publish 必须 `--i-accept`
- 新代码：`from kaggle_lab.proxy import …`；`common.*` 仍是兼容层

仓库：https://github.com/xiaoqianran/kaggle-lab

架构说明：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
