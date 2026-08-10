# 目标检测 · 研究导向学习蓝图

> **本文件只做规划，不教学。**  
> 目标：最短路径达到「能独立读最新检测论文、判断贡献、复现、设计 baseline/ablation、提出研究假设」的研究能力。  
> 原则：每个主题 **1–3 个最优来源**；顺序按研究能力最短路径，不按年代流水账。  
> 领域锁定：**Object Detection（2D 目标检测）** — 含密集/小目标/开放词汇；实例分割与 3D 检测仅作对照。

最后更新：2026-08-10

---

## 0. 一句话问题定义

| 任务 | 输入 | 输出 | 核心难点 |
|------|------|------|----------|
| **闭集检测** | 图像 \(x\) | \(\{(b_i, c_i)\}\) 框+类 | 定位精度、类别不平衡、尺度变化、NMS/匹配 |
| **密集 / 小目标** | 同上，目标小且密 | 同上 | 感受野、分配策略、高分辨率、遮挡 |
| **开放词汇 / 零样本** | 图像 + 文本类名 | 开放类框 | 视觉-语言对齐、假阳性、领域迁移 |

研究主线默认：**指标与框表示 → 两阶段 → 一阶段/无锚 → DETR 族 → 训练配方与 Kaggle 域 → 开放词汇/研究假设**。

---

## 1. 领域知识地图

### 1.1 分层（前置 / 核心 / 进阶 / 前沿 / 可跳过）

```text
前置 PRE
  ├─ 框表示：xyxy / xywh / cxcywh；归一化 vs 像素
  ├─ IoU / GIoU / DIoU / CIoU；NMS / Soft-NMS
  ├─ 指标：AP@0.5、AP@0.5:0.95、AP_s/m/l、AR；COCO 协议直觉
  ├─ 数据：VOC / COCO 标注格式；train/val 泄漏；增广对框的几何变换
  └─ 工程：Dataset 返回 image+boxes+labels；可视化；固定 seed

核心 CORE（研究能力的骨架）
  ├─ 两阶段：RPN → RoI → 分类+回归；Faster R-CNN 必须读通
  ├─ 一阶段：密集预测、锚框、多尺度 FPN
  ├─ 损失：分类（CE / Focal）+ 定位（L1 / IoU-based）
  ├─ 正负样本分配：IoU 阈值 / ATSS / SimOTA / Hungarian
  └─ 评测协议：固定 COCO-style mAP；可视化 FP/FN 错误模式

进阶 ADV
  ├─ 无锚：FCOS / 中心采样；与锚框范式对照
  ├─ YOLO 现代族：YOLOv5/v8/11 的 head、assign、aug 配方
  ├─ DETR 族：set prediction、Hungarian matching、query
  └─ 训练配方：mosaic/mixup、multi-scale、EMA、AMP、T4 可跑规模

前沿 FRONTIER
  ├─ DINO / RT-DETR 等实时/强 baseline DETR 变体
  ├─ 开放词汇：OWL-ViT、Grounding DINO 族（读 1–2 篇即可）
  ├─ 小目标 / 遮挡 / 长尾类；域适应检测
  └─ Kaggle 医学/高光谱/车辆等真实域约束

可跳过 SKIP（除非课题需要）
  ├─ 穷尽 YOLO 每个小版本 release note 当主路径
  ├─ 完整 3D / 点云检测栈（仅对照）
  ├─ 手写完整 Detectron2 配置宇宙
  └─ 以单一 Kaggle 金牌为唯一目标而无固定 mAP 协议
```

### 1.2 概念关系

```text
标注与框表示 ──▶ IoU 匹配 / 分配 ──▶ 分类损失 + 定位损失
        │
        ▼
Backbone + Neck(FPN) ──▶ Head（密集 / 查询）
        │
两阶段(RPN+RoI) ‖ 一阶段(YOLO/Retina) ‖ DETR(set pred)
        │
后处理：NMS 或 端到端无 NMS
        │
评测：COCO mAP + 错误分析 ──▶ 消融 ──▶ 研究假设
```

### 1.3 能力终点（验收标准）

你能独立完成：

1. 读 2024–2026 检测新论文，5 分钟说清 **问题设定 / 匹配方式 / 损失 / 相对 Faster-RCNN 或 DETR 的增量 / 实验是否支撑 claim**  
2. 在 VOC 或 COCO-子集上训通 **至少两种范式**（例：Faster R-CNN 或 torchvision + YOLO 或 DETR-lite），报告可复现 mAP  
3. 精读 **ultralytics 或 mmdet 的 metrics + head + trainer** 中至少一条主路径，能定位 assign、loss、decode  
4. 做消融：aug × 输入尺寸 × 分配策略 × 损失；解释为何某一配置胜出  
5. 提出可检验假设（例：小目标 AP_s 差来自 stride 过大 vs 分配噪声；DETR 收敛慢来自匹配不稳定 vs 容量不足）

---

## 2. 全球最佳资料筛选（每个主题 ≤3）

### 2.1 前置：IoU / mAP / 框

| # | 来源 | 为何入选 | 用法 |
|---|------|----------|------|
| 1 | **COCO 评测定义** + `pycocotools` / ultralytics metrics | 全球事实标准 mAP | 手写 IoU + 读通 AP 曲线含义 |
| 2 | **CS231n** detection 相关讲义 | 问题定义清晰 | 建立两阶段/一阶段对照 |
| 3 | 本仓库 `det-00`（待写）指标协议 notebook | 固定尺子 | 之后所有实验用同一协议 |

### 2.2 核心：两阶段

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **Faster R-CNN** (Ren et al., 2015) | 检测公共语：RPN + RoI | torchvision / detectron2 / mmdet |
| 2 | **Fast R-CNN** 作桥梁 | 理解 RoI pooling 与多任务损失 | 精读即可 |
| 3 | （对照）R-CNN 原文摘要 | 知道「慢」从哪里来 | 不复现整条 selective search |

### 2.3 核心：一阶段与密集预测

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **RetinaNet + Focal Loss** | 正负不平衡的标准答案 | mmdet / detectron2 |
| 2 | **YOLOv1 思想 + YOLOv8/11 现代实现** | 实时检测主航道 | [ultralytics](https://github.com/ultralytics/ultralytics) **必读** |
| 3 | **FCOS** | 无锚对照 | mmdet |

### 2.4 现代：DETR 族

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **DETR** (Carion et al., ECCV 2020) | set prediction 分水岭 | [facebookresearch/detr](https://github.com/facebookresearch/detr) |
| 2 | **Deformable DETR** 或 **DINO** | 收敛与精度的现实改进 | 读 1 篇 + 跟官方 demo |
| 3 | CS444 DETR Kaggle 作业赛（可选） | 短周期可提交 | 见 catalog |

### 2.5 配方与工程

| # | 来源 | 为何入选 |
|---|------|----------|
| 1 | Ultralytics docs + 官方 notebook | Kaggle T4 最省事闭环 |
| 2 | MMDetection 文档（配置系统） | 研究向消融更清晰 |
| 3 | Global Wheat / Great Barrier Reef 高票 writeup | 真实竞赛数据管道 |

---

## 3. 最短学习路径（P0–P7）

| Phase | 目录 | 目标 | 产出 |
|-------|------|------|------|
| **P0** | `00-map/` | IoU、NMS、mAP 尺子统一 | `det-00-iou-map-protocol.ipynb` |
| **P1** | `01-foundations/` | 框编解码、锚框、简易 assign | `det-01-boxes-anchors-nms.ipynb` |
| **P2** | `02-two-stage/` | Faster R-CNN 源码地图 + 小数据跑通 | `det-02-faster-rcnn-source-map.ipynb` |
| **P3** | `03-one-stage/` | YOLO 微调 mini 集 | `det-03-yolo-train-mini.ipynb` |
| **P4** | `04-modern-detr/` | DETR-lite 或官方 demo + 匹配可视化 | `det-04-detr-lite.ipynb` |
| **P5** | `05-training-recipe/` | 同一数据上 aug/size 消融 | `det-05-ultralytics-recipe-ablation.ipynb` |
| **P6** | `06-frontier/` | Kaggle 域 baseline（Wheat/Reef 或进行中赛） | `det-06-kaggle-*-baseline.ipynb` |
| **P7** | `07-research-lab/` | 一条可证伪假设 + 实验日志 | `det-07-research-hypothesis.ipynb` |

**时间感（兼职）：** P0–P1 约 3–5 天；P2–P3 约 1–2 周；P4–P5 约 1–2 周；P6–P7 持续迭代。

---

## 4. Kaggle 使用策略

1. **主粮数据**：VOC / COCO 子集练协议；再上 Wheat/Reef 或进行中赛。  
2. **进行中赛**：RSNA Knee / Biohub Cell / 社区车辆与高光谱 — 作 P6，不抢 P0–P3 时间。  
3. **Models Hub**：优先 `ultralytics/yolo11` 与 `yolov8` 权重做迁移。  
4. **Notebook 命名**：`det-NN-slug.ipynb`，全部保存在本目录。  
5. **密钥**：仅 `KAGGLE_API_TOKEN` / `~/.kaggle/access_token`，永不入库。

---

## 5. 工程约定

```text
v04-xiaoshuhuaer@-目标检测/
  LEARNING_ROADMAP.md   # 本文件
  catalog.json          # 机器索引
  README.md
  det-*.ipynb           # 学习 notebook
  00-map/ … 07-research-lab/
  scripts/ tests/ papers/ results/
  data/ vendor/         # gitignore
```

- 六段闭环：**原理 → 手写 → 源码 → 实验 → 消融 → 解释**  
- 结果 JSON 进 `results/`（小文件）；权重不入库  
- 论文笔记用 `papers/TEMPLATE.md`  
- 每次实质改动：阿里规范 commit + push  

---

## 6. 当前状态

| 项 | 状态 |
|----|------|
| 目录脚手架 | ✅ 2026-08-10 |
| Kaggle 竞赛/数据/模型调研 | ✅ catalog.json |
| P0–P7 notebook | ⏳ 待写 |
| 首次可跑 baseline | ⏳ 建议 det-03 YOLO mini |
