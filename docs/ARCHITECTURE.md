# 架构

仓库曾经把两件无关的事摊在根目录：15 个编号实验，和一堆以 Kaggle 用户名命名的 notebook 轨。打开仓库看不出「从哪进、下一步加什么」。

## 按交互拆

用户先选意图，再落到实现：

```text
我想做什么？
    ├─ 调模型 / 写 agent / 看额度     →  labs/ + python -m kaggle_lab <journey>
    ├─ 跟一条研究轨                   →  tracks/<topic>
    └─ 打开网页对谈                   →  apps/ 指向 labs/015 + gateway
```

CLI（`kaggle_lab.cli`）的解析顺序：

1. 内置命令：`list` / `run` / `track` / `gateway` / `help`
2. 意图入口：`auth` `chat` `debate` `workforce` `scout` `quota`
3. lab id / 别名 / 编号前缀（`014`、`workforce`、`015-dual-agent-chat`）

空参数不再默默掉进 `001`，而是打印地图。

## 按可读性拆

| 以前 | 现在 |
|------|------|
| 根目录 15 个 `NNN-*` | `labs/NNN-*` |
| `notebooks/v03-zhengyingxionger@-图像分类` | `tracks/image-classification` |
| 根上三份 `kaggle_ai_models.*` | `data/` |
| README 先喊 SAE 禁考 | README 先问「你想做什么」 |
| 只有编号能跑 | 编号 + 别名 + 意图命令 |

Kaggle 账号仍写在各轨 README / `Track.kaggle_users`，只是不再当文件夹名。

## 按可扩展性拆

单一登记处：[`kaggle_lab/catalog.py`](../kaggle_lab/catalog.py)。

- **Lab**：id、aliases、group、default_cmd、uses_proxy、dangerous、product
- **Track**：id、aliases、start_here、kaggle_users
- **Journey**：用户意图 → lab + 子命令

加实验的契约：

1. `labs/<id>/run.py` 自己解析 argv（lab 内部 CLI 保持独立，避免上帝命令）。
2. 在 `LABS` 加一行。`list` 按 group 展示；没登记的 `run.py` 也会出现在底部，避免「加了文件夹但世界看不见」。
3. 共享能力只进 `kaggle_lab/`（proxy / usage / camel / paths）。`common.*` 是 shim。

路径：不要再写 `Path(__file__).parent.parent` 当仓库根。lab 脚本用 `parents[2]`，或 `kaggle_lab.paths.find_repo_root()`。`.env.model-proxy` 和 `logs/usage.jsonl` 永远在仓库根。

## 刻意没做的

- 没有把 15 个 `run.py` 收成一个包：每个实验仍是可单独读完的小程序。
- 没有改 tracks 里的训练脚本：它们已经用相对本轨的 `ROOT`，搬家即生效。
- 没有自动跑 SAE：危险命令仍要 `--i-accept`。
