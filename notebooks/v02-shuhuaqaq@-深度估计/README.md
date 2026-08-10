# v02 · shuhuaqaq@ · 深度估计

本目录 = **深度估计研究学习工程**。  
权威学习蓝图：**[LEARNING_ROADMAP.md](./LEARNING_ROADMAP.md)**（领域地图 · 精选资料 · 最短路径 · 工程结构）。

Kaggle 账号侧用户：`shuhuaqqq`。Notebook 命名：`de-NN-*.ipynb`。

---

## 快速导航

| 文档 | 内容 |
|------|------|
| [LEARNING_ROADMAP.md](./LEARNING_ROADMAP.md) | **主蓝图**（先读这个） |
| [catalog.json](./catalog.json) | 竞赛/数据/模型机器索引 |
| [papers/TEMPLATE.md](./papers/TEMPLATE.md) | 论文笔记模板 |

## 阶段目录

| 目录 | Phase | 主题 |
|------|-------|------|
| `00-map/` | — | 术语、指标、概念卡 |
| `01-geometry/` | P1 | 相机与立体最小实现 |
| `02-supervised-classic/` | P2 | 有监督 baseline |
| `03-self-supervised/` | P3 | Monodepth2 思想与源码 |
| `04-foundation/` | P4 | MiDaS / DPT / DA-V2 |
| `05-metric/` | P5 | Depth Pro / Metric3D / UniDepth |
| `06-frontier/` | P6 | 新论文与失败 case |
| `07-research-lab/` | P7 | 个人假设与实验日志 |
| `scripts/` `tests/` | 全程 | 指标、对齐、单元测试 |
| `vendor/` `data/` `results/` | 全程 | 源码/数据/结果（大文件不入库） |

## Kaggle 深度估计任务速查（调研 2026-08-10）

### 竞赛

| 竞赛 | 状态 | 要点 |
|------|------|------|
| [ETHZ CIL Monocular Depth Estimation 2025](https://www.kaggle.com/competitions/ethz-cil-monocular-depth-estimation-2025) | 已结束 | SI-RMSE；课程向；可作工程闭环 |
| [MDEC](https://jspenmar.github.io/MDEC/)（CVPR workshop） | 社区挑战 | 零样本泛化协议 |

### 高票参考 Notebook（仅参考，不抄终点）

| Notebook | 用途 |
|----------|------|
| [shreydan/monocular-depth-estimation-nyuv2](https://www.kaggle.com/code/shreydan/monocular-depth-estimation-nyuv2) | NYU 监督路径 |
| [amanattheedge/depth-anything-v2-metric-fine-tunning-on-nyu](https://www.kaggle.com/code/amanattheedge/depth-anything-v2-metric-fine-tunning-on-nyu) | DA-V2 度量微调 |
| [cilabeth/monocular-depth-example-notebook](https://www.kaggle.com/code/cilabeth/monocular-depth-example-notebook) | ETHZ 官方示例 |

### Models Hub

`intel/midas` · `artemmmtry/depth-anything-v2` · `keras/depth-anything` · `tensorflow/ar-portrait-depth`

---

## 约定

- 每个知识点：**原理 → 手写 → 源码 → Kaggle → 消融 → 解释**
- 每次实质改动：**阿里规范 commit + push**
- 密钥与大权重：**永不入库**
