# kaggle-lab

Kaggle 实验台。两块内容，互不依赖，目录本身就是索引：

| 目录 | 是什么 |
|------|--------|
| [`labs/`](labs/) | 调模型、写 agent、碰 Kaggle API 的编号小实验，每个一个 `run.py`，索引在 [labs/README.md](labs/README.md) |
| [`tracks/`](tracks/) | 按主题的学习/研究轨（分类、检测、深度、Instruct、ARC…），索引在 [tracks/README.md](tracks/README.md) |

共享代码只有 [`kaggle_lab/`](kaggle_lab/) 一个包（proxy / camel_proxy / usage / paths），labs 直接 import 它。没有统一 CLI、没有登记表：**跑实验 = 进目录跑 `run.py`**（或从仓库根用全路径）。

账号 API：`KAGGLE_API_TOKEN` / `~/.kaggle/access_token`（`KGAT_…`）。

---

## 30 秒上手

```bash
source .venv/bin/activate    # Python ≥3.11
export KAGGLE_API_TOKEN=<你的 token>
python labs/001-model-proxy/run.py auth        # 接通 Model Proxy（临时 key ≈ 1h）
python labs/001-model-proxy/run.py chat "你好"  # 跟模型聊一句
```

常用入口：

| 想做什么 | 命令 |
|----------|------|
| 看额度（GPU / AI $） | `python labs/010-quota-dashboard/run.py show` |
| 看两个智能体对谈 | `python labs/015-dual-agent-chat/run.py run` |
| 多角色写一道 Benchmark | `python labs/014-camel-workforce-bench/run.py run` |
| 侦察一场竞赛 | `python labs/013-camel-kaggle-crew/run.py scout` |
| 全部实验 | 看 [labs/README.md](labs/README.md) 的索引表 |
| 学习轨 | 看 [tracks/README.md](tracks/README.md)，cd 进去读该轨 README |

CAMEL 实验（011–014）额外：

```bash
uv pip install 'camel-ai==0.2.90' && uv pip install 'mcp==1.9.4'
# 005 / 014 validate 还需: uv pip install -e vendor/kaggle-benchmarks
```

Web 对谈（015）：

- 演示：https://xiaoqianran.github.io/kaggle-lab/
- 本地：先 `cd labs/015-dual-agent-chat/web && npm i && npm run build`，再 `python labs/015-dual-agent-chat/gateway.py`

---

## SAE（默认不开考）

**SAE** = Standardized Agent Exam。默认 **不开考、不注册、不交卷**。
详见 [docs/SAE.md](docs/SAE.md)。允许：`python labs/009-sae-better/run.py dry-run`。

004 的 `start/register/submit` 在它自己的 `run.py` 里强制 `--i-accept`，且需用户书面确认；014 publish 同样要 `--i-accept`。

---

## 约定

- 密钥 / artifacts / logs / vendor 不入库
- **不自动开 SAE 考**
- 加实验：在 `labs/` 放一个带 `run.py` 的目录，并在 [labs/README.md](labs/README.md) 索引表加一行；加轨同理改 [tracks/README.md](tracks/README.md)
- 共享代码只进 `kaggle_lab/`，import 写 `from kaggle_lab.proxy import …`

仓库：https://github.com/xiaoqianran/kaggle-lab
