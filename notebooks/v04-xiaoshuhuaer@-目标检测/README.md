# v04 · xiaoshuhuaer@ · 目标检测

本目录 = **目标检测（Object Detection）研究学习工程**。  
权威学习蓝图：**[LEARNING_ROADMAP.md](./LEARNING_ROADMAP.md)**（领域地图 · 精选资料 · 最短路径 · 工程结构）。

Kaggle 账号侧用户：`xiaosuhuaer`（API whoami）。Notebook 命名：`det-NN-*.ipynb`。

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
| `00-map/` | — | 术语、IoU/mAP、框表示、NMS |
| `01-foundations/` | P0–P1 | 指标协议、锚框、assign、手写 toy detector |
| `02-two-stage/` | P2 | R-CNN 族 / Faster R-CNN 精读 |
| `03-one-stage/` | P3 | YOLO 族 / RetinaNet / FCOS |
| `04-modern-detr/` | P4 | DETR / Deformable-DETR / DINO |
| `05-training-recipe/` | P5 | Ultralytics / MMDetection 配方与消融 |
| `06-frontier/` | P6 | 开放词汇、小目标、Kaggle 真实域 |
| `07-research-lab/` | P7 | 个人假设与实验日志 |
| `scripts/` `tests/` | 全程 | 指标、可视化、单元测试 |
| `vendor/` `data/` `results/` | 全程 | 源码/数据/结果（大文件不入库） |

## Kaggle 目标检测任务速查（调研 2026-08-10）

### 进行中 / 近期（优先闭环）

| 竞赛 | 状态 | 要点 |
|------|------|------|
| [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) | **进行中** (~2026-10) | 医学定位；P6 真实域 |
| [Biohub Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) | **进行中** (~2026-09) | 细胞检测+跟踪；检测头可练 |
| [Hyperspectral Object Detection Challenge 2026](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026) | 进行中 | 高光谱 OD |
| [3LC Multi-Vehicle Detection](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge) | 进行中 | 多车检测社区赛 |
| [CS444 YOLO / DETR MP](https://www.kaggle.com/competitions/cs-444-sp-26-mp-4-a-yolo-object-detection) | 课程向 | 手写 YOLO / DETR 作业赛 |

### 经典已结束（学习教材）

| 竞赛 | 要点 |
|------|------|
| [Global Wheat Detection](https://www.kaggle.com/competitions/global-wheat-detection) | 密集小目标；YOLO 教材级 |
| [TensorFlow Great Barrier Reef](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef) | 视频帧 + COTS；YOLOv5 高票 notebook |
| [VinBigData Chest X-ray Abnormalities](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection) | 医学多类框；多标注者 |
| [Airbus Ship Detection](https://www.kaggle.com/competitions/airbus-ship-detection) | 船只；常作分割/检测对照 |
| [Open Images Object Detection RVC 2020](https://www.kaggle.com/competitions/open-images-object-detection-rvc-2020) | 大规模开放图像检测 |
| [3D Object Detection for Autonomous Vehicles](https://www.kaggle.com/competitions/3d-object-detection-for-autonomous-vehicles) | 3D 检测入门对照 |

### 数据集（训练主粮）

| 数据集 | 用途 |
|--------|------|
| PASCAL VOC 2007/2012（`zaraks/pascal-voc-2007` 等） | P1–P2 小规模协议 |
| COCO 子集 / YOLO 格式（`malaychand/coco-25-class-object-detection-yolo-datasets` 等） | P3–P5 标准 mAP |
| Global Wheat / Reef 官方数据 | Kaggle 闭环 |
| Construction Site Safety（`snehilsanyal/construction-site-safety-image-dataset-roboflow`） | YOLO 微调练手 |
| SKU110K / BDD100K（可选） | 密集零售 / 驾驶场景 |

### Models Hub（示例）

`ultralytics/yolo11` · `ultralytics/yolov8` · `ultralytics/yolov5` · `keras/yolov8` · `tensorflow/efficientdet` · `google/owl-vit` · 本地 **ultralytics / mmdet / torchvision** 为主

---

## 约定

- 每个知识点：**原理 → 手写 → 源码 → Kaggle → 消融 → 解释**
- 每次实质改动：**阿里规范 pull + commit + push**
- 密钥与大权重：**永不入库**
- 研究主指标用固定 COCO-style mAP 协议；Kaggle LB 只作工程闭环
- 后续 notebook 一律放本目录：`det-00-…`、`det-01-…`
