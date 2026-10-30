# tracks — 学习 / 研究轨

按 **主题** 命名，不再用 Kaggle 用户名当目录名。账号信息留在各轨 README 里。

用法：看下表选一条轨，cd 进去先读「先读」那篇，再跑脚本 / notebook。

| id | 主题 | 先读 | 曾用名 / Kaggle |
|----|------|------|-----------------|
| [`instruct-t4`](instruct-t4/) | Mini-Instruct · img3d · world models | 本目录 notebook 顶部说明 | `v01-wangran521` |
| [`depth-estimation`](depth-estimation/) | 深度估计 From-Scratch | [README](depth-estimation/README.md) | `v02-shuhuaqaq@-深度估计` |
| [`image-classification`](image-classification/) | 图像分类 | [LEARNING_ROADMAP.md](image-classification/LEARNING_ROADMAP.md) | `v03-zhengyingxionger@-图像分类` |
| [`object-detection`](object-detection/) | 目标检测 | [LEARNING_ROADMAP.md](object-detection/LEARNING_ROADMAP.md) | `v04-xiaoshuhuaer@-目标检测` |
| [`diffusion-gemma`](diffusion-gemma/) | DiffusionGemma T4×2 | [README](diffusion-gemma/README.md) | `v05-yaoyunqqq-…` |
| [`arc-agi-3`](arc-agi-3/) | ARC Prize 2026 | **[HANDOFF.md](arc-agi-3/HANDOFF.md)** | `v06-chutianqiu-arc-agi-3` · 公开榜 **0.17** |

旧路径 `notebooks/v0N-…` 已整体搬家。脚本里的 `Path(__file__).parents[1]` 仍然指向各轨根目录，不用改业务代码。

## 加一条轨

1. `tracks/<topic-slug>/`，ASCII 目录名，主题写进 README。
2. 在上表加一行（主题、先读、曾用名）。
