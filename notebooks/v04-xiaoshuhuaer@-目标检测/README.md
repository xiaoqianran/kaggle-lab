# v04 · xiaoshuhuaer@ · 目标检测

本目录 = **目标检测（Object Detection）研究学习工程**。

| 文档 | 作用 |
|------|------|
| **[LEARNING_ROADMAP.md](./LEARNING_ROADMAP.md)** | **唯一权威蓝图**：领域地图 · 全球资料筛选 · 最短路径 · 工程结构 |
| [catalog.json](./catalog.json) | 竞赛/数据/模型/课程/论文机器索引 |
| [papers/TEMPLATE.md](./papers/TEMPLATE.md) | 论文笔记 |
| [papers/SOURCE_MAP_TEMPLATE.md](./papers/SOURCE_MAP_TEMPLATE.md) | 源码↔论文对照 |

Kaggle 账号侧用户：`xiaosuhuaer`。Notebook 命名：`det-NN-*.ipynb`，**全部放本目录**。

> 当前阶段：**规划完成，尚未开课教学。** 先读蓝图，确认后再从 P0/`det-00` 进入六段闭环。

---

## 阶段目录

| 目录 | Phase | 主题 | 笔记 |
|------|-------|------|------|
| `00-map/` | P0 | 指标、IoU/mAP、NMS、复现纪律 | [NOTES](./00-map/NOTES.md) |
| `01-foundations/` | P1 | 框、锚、分配 | [NOTES](./01-foundations/NOTES.md) |
| `02-two-stage/` | P2 | Faster R-CNN | [NOTES](./02-two-stage/NOTES.md) |
| `03-one-stage/` | P3 | YOLO / Retina / FCOS | [NOTES](./03-one-stage/NOTES.md) |
| `04-modern-detr/` | P4 | DETR / DINO / RT-DETR | [NOTES](./04-modern-detr/NOTES.md) |
| `05-training-recipe/` | P5 | 配方与消融 | [NOTES](./05-training-recipe/NOTES.md) |
| `06-frontier/` | P6 | 开放词汇 + Kaggle 域 | [NOTES](./06-frontier/NOTES.md) |
| `07-research-lab/` | P7 | 假设与 claim 审查 | [NOTES](./07-research-lab/NOTES.md) |
| `scripts/` `tests/` | 全程 | 指标、可视化、单测 | |
| `vendor/` `data/` `results/` | 全程 | 源码/数据/结果（大文件不入库） | |

## 最短路径（详见蓝图 §3）

```text
P0 尺子 → P1 框/匹配 → P2 两阶段 → P3 YOLO 闭环
       → P4 DETR → P5 消融 → P6 Kaggle 域 → P7 独立假设
```

每个核心点强制：**原理 → 手写 → GitHub 源码 → Kaggle 实验 → 消融 → 解释**。

## Kaggle 任务速查（调研 2026-08-10）

### 进行中

| 竞赛 | 要点 |
|------|------|
| [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) | 医学定位 · P6 |
| [Biohub Cell Tracking](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) | 检测+跟踪 · P6 |
| [Hyperspectral Object Detection 2026](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026) | 高光谱 OD |
| [3LC Multi-Vehicle Detection](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge) | 车辆检测 |
| CS444 YOLO / DETR 课程赛 | 手写向作业赛 |

### 经典教材赛

Global Wheat · Great Barrier Reef · VinBigData CXR · Airbus Ship · Open Images OD

### 主粮数据 / 模型

- 数据：VOC · COCO 子集 · Wheat/Reef · Construction Safety YOLO  
- 模型：`ultralytics/yolo11` · `yolov8` · EfficientDet · OWL-ViT · detectron2/mmdet zoo  

完整字段见 `catalog.json`。

---

## 约定

- 蓝图冲突时以 **LEARNING_ROADMAP.md** 为准  
- 密钥与大权重：**永不入库**  
- 每次实质改动：**阿里规范 pull → commit → push**  
- 研究主指标：固定 COCO-style 协议；Kaggle LB 只作工程闭环  
