# 目标检测 · 研究导向学习蓝图

> **本文件只做规划，不教学。**  
> 目标：最短路径达到「能独立读最新检测论文、判断贡献、复现、设计 baseline/ablation、提出研究假设」的研究能力。  
> 原则：每个主题 **1–3 个最优来源**；顺序按研究能力最短路径，不按教材章节/编年史。  
> 领域锁定：**2D Object Detection（目标检测）** — 含密集/小目标/开放词汇；实例分割、跟踪、3D 检测仅作对照边界。  
> 工程落点：本目录 `notebooks/v04-xiaoshuhuaer@-目标检测/` · Kaggle 用户 `xiaosuhuaer`

最后更新：2026-08-10（全球资料交叉筛选 v2）

---

## 0. 一句话问题定义

| 任务 | 输入 | 输出 | 核心难点（研究常盯这里） |
|------|------|------|--------------------------|
| **闭集检测** | 图像 \(x\) | \(\{(b_i,c_i)\}_i\) 框+类 | 定位精度、正负不平衡、多尺度、匹配/NMS |
| **密集 / 小目标** | 同上，目标小且密 | 同上 | stride/FPN、分配噪声、高分辨率、遮挡 |
| **开放词汇** | 图像 + 文本类名/短语 | 开放类框 | 对齐、假阳性、域移、评测协议漂移 |

**默认研究主线：**  
指标与框协议 → 两阶段公共语 → 一阶段/无锚 → DETR set prediction → 工业配方与源码 → Kaggle 真实域 → 开放词汇与可证伪假设。

---

## 1. 领域知识地图

### 1.1 分层：前置 / 核心 / 进阶 / 前沿 / 可跳过

```text
前置 PRE（不会就无法“研究”，只能“调包”）
  ├─ 框表示：xyxy / xywh / cxcywh；像素 vs 归一化；坐标变换与增广联动
  ├─ 重叠：IoU / GIoU / DIoU / CIoU（匹配 vs 损失 两种用法）
  ├─ 后处理：NMS / Soft-NMS；分数阈值；max-det
  ├─ 指标：AP50、AP@0.5:0.95、AP_s/m/l、AR；COCO 协议直觉
  ├─ 数据：VOC XML / COCO JSON / YOLO txt；泄漏、类不均、标注噪声
  └─ 工程：Dataset 契约、可视化 GT/Pred、固定 seed、AMP、T4 显存边界

核心 CORE（检测研究者的公共语，必须到“能讲清失败模式”）
  ├─ 两阶段 DNA：RPN → RoI Align → cls+reg；Faster R-CNN 必通
  ├─ 一阶段 DNA：密集预测、锚框、多尺度 FPN、decode
  ├─ 损失：分类（CE / Focal）+ 定位（L1 / Smooth-L1 / IoU-based）
  ├─ 分配（label assignment）：IoU 阈值 / ATSS / SimOTA / Hungarian
  ├─ 特征：Backbone + Neck(FPN/PAN) + Head 三段解耦读源码
  └─ 评测纪律：固定 COCO-style 尺子；错误类型（定位/分类/重复/漏检）

进阶 ADV（能做消融、能读懂 2019–2023 主流改动）
  ├─ 无锚：FCOS / 中心采样；与锚框范式对照
  ├─ YOLO 现代族：v5→v8→11 的 head、assign、aug、工程默认
  ├─ DETR 族：object queries、Hungarian matching、no NMS
  ├─ 训练配方：mosaic/mixup、multi-scale、EMA、LR schedule、EMA 对 AP 的影响
  └─ 工具链：ultralytics（快闭环）‖ mmdet/detectron2（研究配置）

前沿 FRONTIER（2023–2026 论文主战场）
  ├─ 实时端到端：RT-DETR / RT-DETRv2 与 YOLO 对照
  ├─ 强 DETR baseline：DINO / Deformable DETR 系
  ├─ 开放词汇与指代：OWL-ViT、Grounding DINO 族（精读 1–2）
  ├─ 小目标 / 遮挡 / 长尾 / 域适应检测
  └─ 真实域：医学框、高光谱、车辆、视频帧（Kaggle 进行中赛）

可跳过 SKIP（除非课题强制）
  ├─ 穷尽 YOLO 每个小版本 changelog 当主路径
  ├─ 手写完整 Detectron2/mmdet 配置宇宙（会改配置即可）
  ├─ 完整 3D/点云/BEV 检测栈（只读 1 篇对照边界）
  ├─ 2014 前传统检测（HOG+SVM）超过 1 讲的深度
  ├─ 以单一 Kaggle 金牌为唯一目标、无固定 mAP 协议
  └─ 堆叠二流“从零实现 XXX 全集”视频课
```

### 1.2 概念关系（研究导航图）

```text
标注格式 ──几何增广──▶ 框张量
        │
        ▼
Backbone ──▶ Neck(FPN) ──▶ Head
        │                    │
        │         ┌──────────┴──────────┐
        │         ▼                     ▼
        │   密集/锚/无锚预测      Query/set prediction
        │         │                     │
        │    IoU/ATSS/SimOTA      Hungarian matching
        │         │                     │
        │    NMS 后处理            （常无 NMS）
        │         └──────────┬──────────┘
        ▼                    ▼
   分类损失 + 定位损失     COCO mAP + 错误分析
        │
        ▼
   消融（aug × size × assign × loss）──▶ 研究假设
```

### 1.3 能力终点（验收标准）

你能独立完成：

1. **读新论文（5–10 分钟口述）：** 问题设定 / 匹配方式 / 损失 / 相对 Faster R-CNN 或 DETR 的增量 / 实验是否支撑 claim  
2. **两条范式可训：** 在 VOC 或 COCO 子集上至少训通 **(A) 两阶段或 torchvision/detectron2 Faster R-CNN** 与 **(B) YOLO 或 DETR-lite** 之一对，报告可复现 mAP  
3. **源码地图：** 在 ultralytics **或** mmdet **或** detectron2 中锁定 metrics + head + assign + decode 的真实文件路径  
4. **消融：** ≥2 可控变量（例：输入尺寸 × mosaic 开关；assign 策略 × 损失），解释胜出原因  
5. **研究假设：** 可证伪、可脚本化、阴性结果也写清楚（例：AP_s 差来自 stride 过大 vs 正样本分配噪声）

---

## 2. 全球最佳资料筛选（每个主题 ≤3）

> 交叉比较来源：Stanford / MIT / CMU / Berkeley / PKU 公开课、Meta/Google/百度/Ultralytics/OpenMMLab 官方代码、COCO 协议、经典与 2023–2025 代表作。  
> **未列入 ≠ 差，而是边际收益低或与主路径重复。**

### 2.1 高校课程：怎么用（不堆课）

| 课 | 角色 | 取舍 |
|----|------|------|
| **Stanford CS231n**（[cs231n.stanford.edu](https://cs231n.stanford.edu/) · [notes](https://cs231n.github.io/) · 2025 Lec9 Detection） | **主课入口** | 建立检测问题定义与 R-CNN→YOLO→DETR 叙事；**作业以 CNN 为主，检测靠讲+自建实验** |
| **CMU 16-824 Visual Learning**（[visual-learning.cs.cmu.edu](https://visual-learning.cs.cmu.edu/)） | **论文阅读节奏** | 取 recognition/detection 相关 paper list 与读论文纪律；不整课搬家 |
| **MIT 6.8300/6.869 Advances in CV**（[OCW](https://ocw.mit.edu/courses/6-8300-advances-in-computer-vision-spring-2025/)） | 几何/表征边界 | **检测主路径不深入**；需要几何/多视图时再开 |
| **Berkeley CS280**（[cs280-berkeley.github.io](http://cs280-berkeley.github.io/)） | 现代检测/分割对照 | 取 “Modern Detection” 相关讲；可选 |
| **PKU Intro2CV**（[pku-epic schedule](https://pku-epic.github.io/Intro2CV_2025/schedule/)） | 中文术语 + 2D detector 提纲 | **不作为主路径**；对照 SSD/RCNN/YOLO 一讲即可 |
| **THU Advanced CV**（[thu-acv.github.io](https://thu-acv.github.io/)） | 前沿补充 | 有公开讲义再局部引用；默认不依赖 |

**明确不优先当主路径：** 商业“百日 YOLO 打卡”、只调 ultralytics CLI 不读论文/指标的课、过时 TF1/Darknet 教程全集。

### 2.2 前置：指标、框、协议

| # | 来源 | 为何入选 | 用法 |
|---|------|----------|------|
| 1 | **COCO Detection Evaluation** + `pycocotools` | 全球对话语言：AP@0.5:0.95 | 手写 IoU/NMS + 对照官方 AP |
| 2 | **Ultralytics metrics** 源码（`ultralytics/utils/metrics.py`） | 工业界可跑实现，与 COCO 对齐的细节 | SOURCE_MAP 必做 |
| 3 | **CS231n** Detection 讲（2025 Lec9 或历年） | 问题定义清晰、路径总览 | 1 次精读，不反复刷 |

### 2.3 核心：两阶段

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **Ren et al. Faster R-CNN** (NeurIPS 2015) | 检测公共语：RPN + 共享特征 | 论文原文 **必读** |
| 2 | **Meta Detectron2**（[facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)） | 工业级两阶段/配置真源 | `modeling/proposal_generator` · `roi_heads` · Model Zoo Faster R-CNN |
| 3 | **torchvision detection**（`torchvision.models.detection`） | 更短路径跑通 Faster R-CNN | 与 Detectron2 对照差异列表 |

**不优先：** 手写完整 selective search；复现 2014 R-CNN 全流程。

### 2.4 核心：一阶段、无锚、密集预测

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **Lin et al. RetinaNet + Focal Loss** (ICCV 2017) | 正负不平衡的标准答案 | mmdet / detectron2 RetinaNet |
| 2 | **Ultralytics YOLO**（YOLOv8/YOLO11 + docs）· [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 实时检测主航道、Kaggle T4 最省事闭环 | **精读** `nn/tasks.py` · `engine/trainer.py` · `utils/metrics.py` · loss/assign 相关 |
| 3 | **Tian et al. FCOS** (ICCV 2019) | 无锚对照，理解 center sampling | mmdet FCOS head |

**YOLOv1 原文：** 只取“统一回归思想”半页笔记，实现以现代 YOLO 为准。

### 2.5 核心：DETR 族（端到端 set prediction）

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **Carion et al. DETR** (ECCV 2020) · [facebookresearch/detr](https://github.com/facebookresearch/detr) | set prediction + Hungarian 分水岭 | 官方 repo + matching 可视化 |
| 2 | **Zhu et al. Deformable DETR** 或 **Zhang et al. DINO**（2022，二选一深读） | 解决收敛/精度的现实路径 | 官方实现 + 与 DETR 对照表 |
| 3 | **Zhao et al. RT-DETR** (CVPR 2024) · [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | 实时端到端 vs YOLO 的当代对照 | 推理 benchmark（T4）+ 论文表 |

### 2.6 进阶：研究向工具链与配方

| # | 来源 | 为何入选 | 用法 |
|---|------|----------|------|
| 1 | **OpenMMLab MMDetection**（[mmdetection](https://github.com/open-mmlab/mmdetection) · [docs](https://mmdetection.readthedocs.io/)） | 配置化消融、模型动物园最全 | 学 config / head / assigner；**不必全装精通** |
| 2 | **Detectron2** 训练与数据注册教程 | 两阶段与研究默认基线 | 自定义 Dataset 注册一次 |
| 3 | **Kaggle 高票 writeup 各 1 个：** Great Barrier Reef YOLOv5（awsaf49）· Ultralytics 官方 notebook | 真实数据管线与提交格式 | **只抄管线，不抄“神秘涨点”** |

### 2.7 前沿：开放词汇与定位

| # | 来源 | 为何入选 | 代码 |
|---|------|----------|------|
| 1 | **OWL-ViT**（Google，开放词汇检测清晰设定）· Hub `google/owl-vit` | 文本查询 → 框 | 官方/HF 推理 + 失败 case |
| 2 | **Grounding DINO**（或同族 2023–2024 代表，**择一**） | 短语定位 SOTA 路线 | 官方 demo；与闭集 YOLO 对照 |
| 3 | （可选）**PaliGemma / 多模态定位** 一篇 | VLM 与检测边界 | 只读设定与评测，不深陷训练 |

### 2.8 数据与基准（研究对话语言）

| # | 基准 | 角色 |
|---|------|------|
| 1 | **COCO 2017**（或 25-class / 子集） | 论文默认尺子；子集须在报告声明 |
| 2 | **PASCAL VOC 07+12** | 小数据快速迭代、两阶段入门 |
| 3 | **Kaggle 域**（见 `catalog.json`）：Global Wheat、Great Barrier Reef、VinBigData、进行中 RSNA Knee / Biohub Cell / 车辆与高光谱 | 脏数据与提交闭环；**主研究指标仍固定 COCO-style** |

### 2.9 关键论文阅读栈（按此顺序读**原文**）

| 序 | 论文 | 读法 |
|----|------|------|
| 1 | **Faster R-CNN** 2015 | 精读：RPN、多任务损失、共享卷积 |
| 2 | **Focal Loss / RetinaNet** 2017 | 精读：不平衡与 α/γ 消融 |
| 3 | **FCOS** 2019 或 YOLO 现代技术报告 **择一** | 对照锚 vs 无锚 |
| 4 | **DETR** 2020 | 精读：matching cost、no NMS、慢收敛 claim |
| 5 | **DINO** 2022 **或** **Deformable DETR** | 看“如何修 DETR” |
| 6 | **RT-DETR** 2024 | 看实时端到端 vs YOLO 表 |
| 7 | **OWL-ViT 或 Grounding DINO 择一** | 开放词汇设定 |
| 8 | 当周 arXiv 检测 1 篇 | 训练“判断贡献 / 找不支持 claim 的实验” |

每篇产出：`papers/<id>/` 下笔记（用 `TEMPLATE.md`）+ `SOURCE_MAP.md`（若有官方代码）。

---

## 3. 最优学习顺序（最短路径 → 研究能力）

> 顺序逻辑：**先能量化对错 → 框与匹配公共语 → 两阶段 DNA → 一阶段可训闭环 → DETR 范式跳跃 → 工业配方消融 → 真实域 → 独立假设**。  
> **不是** 从 R-CNN 2014 线性扫到 2026 每一个 YOLO 小版本。

| Phase | 名称 | 周期感 | 出口能力 | 绑定实验 / 目录 |
|-------|------|--------|----------|-----------------|
| **P0** | 指标与复现纪律 | 2–4 天 | 手写 IoU/NMS；解释 AP50 vs AP；固定 seed；识破“假 SOTA” | `00-map/` · `det-00` |
| **P1** | 框、锚、分配直觉 | 3–5 天 | 编解码框；IoU 匹配；可视化正负样本 | `01-foundations/` · `det-01` |
| **P2** | 两阶段 Faster R-CNN | 1–2 周 | 讲清 RPN/RoI；VOC 或子集可训；Detectron2 或 torchvision SOURCE_MAP | `02-two-stage/` · `det-02` |
| **P3** | 一阶段 YOLO 闭环 | 1–2 周 | 自有/公开 YOLO 数据训通；读透 loss+metrics 路径 | `03-one-stage/` · `det-03` |
| **P4** | DETR 范式 | 1–2 周 | Hungarian 匹配可视化；DETR 或 RT-DETR 推理+小规模实验 | `04-modern-detr/` · `det-04` |
| **P5** | 训练配方消融 | 1–2 周 | 同数据上 aug/size/assign 消融表；mmdet 或 ultralytics 配置可改 | `05-training-recipe/` · `det-05` |
| **P6** | Kaggle 真实域 | 1–2 周 | Wheat/Reef **或** 进行中赛上可辩护 baseline + 错误分析 | `06-frontier/` · `det-06` |
| **P7** | 独立研究 | 持续 | 新论文 claim 审查；可证伪假设；阴性结果 | `07-research-lab/` · `det-07` / `det-rNN-*` |

### 3.1 每个核心知识点的六段绑定（强制模板）

所有 `det-NN` 必须填满：

| 段 | 含义 | 交付物 |
|----|------|--------|
| **原理** | 公式/假设/失效条件 | 阶段 `NOTES.md` 或论文笔记半页 |
| **手写实现** | 最小可运行（不抄整库） | notebook 核心 cell 或 `scripts/` |
| **GitHub 源码** | 指定文件/函数精读 | `papers/*/SOURCE_MAP.md` |
| **Kaggle 实验** | 可 GPU 复现 | `.ipynb` + seed + 环境记录 |
| **消融** | ≥2 可控变量 | 表：变量 → mAP/AP_s → 解释 |
| **结果解释** | 成功/失败归因 | FP/FN 可视化 + 文字 |

### 3.2 阶段门禁（不会就回退）

| 若你… | 回退到 |
|--------|--------|
| 说不清 AP 与 IoU 阈值关系 | P0 |
| 改增广后框错位、不会同步变换 | P1 |
| 只会 `yolo train` 说不清 loss 各项 | P3 源码段 |
| 把 DETR 当“更大 YOLO” | P4 原理+matching |
| Kaggle 只刷分无固定 val 协议 | P0 + P6 协议重做 |
| 读新论文只会复述 abstract | P7 claim 审查模板 |

### 3.3 持续测试机制（后续教学轮使用，此处只定义）

| 类型 | 示例 |
|------|------|
| **解释** | 为何 focal loss 的 γ 增大会更关注难例？ |
| **预测** | 关掉 mosaic，AP_s 与 AP_l 谁更可能掉？ |
| **Debug** | mAP=0 的 10 个检查清单（类别映射、xyxy 顺序、conf 阈值…） |
| **实现** | 不调用库函数写 batch IoU |
| **研究** | 给一篇 2025 检测论文，标出“实验未支撑的 claim” |

---

## 4. GitHub / Kaggle 学习工程结构

本仓库路径（已创建）：

```text
notebooks/v04-xiaoshuhuaer@-目标检测/
├── LEARNING_ROADMAP.md          # 本蓝图（权威）
├── README.md                    # 人类导航 + 竞赛速查
├── catalog.json                 # 机器索引：竞赛/数据/模型/论文/阶段
│
├── 00-map/                      # P0：术语、IoU/mAP、NMS
├── 01-foundations/              # P1：框、锚、分配
├── 02-two-stage/                # P2：Faster R-CNN
├── 03-one-stage/                # P3：YOLO / Retina / FCOS
├── 04-modern-detr/              # P4：DETR / DINO / RT-DETR
├── 05-training-recipe/          # P5：配方与消融
├── 06-frontier/                 # P6：开放词汇 + Kaggle 域
├── 07-research-lab/             # P7：假设、claim 审查、实验日志
│
├── papers/                      # 笔记 + SOURCE_MAP（不存大 PDF）
│   └── TEMPLATE.md
├── scripts/                     # metrics、viz、download、协议
├── tests/                       # IoU/NMS/编解码单元测试（防自欺）
├── data/                        # 本地缓存（gitignore）
├── vendor/                      # clone 官方源码（gitignore 权重）
└── results/                     # 小 JSON/图；大 ckpt 不入库
```

### 4.1 Notebook 命名

```text
det-00-iou-map-protocol.ipynb
det-01-boxes-anchors-nms.ipynb
det-02-faster-rcnn-source-map.ipynb
det-03-yolo-train-mini.ipynb
det-04-detr-matching-lite.ipynb
det-05-recipe-ablation.ipynb
det-06-kaggle-domain-baseline.ipynb
det-07-research-hypothesis.ipynb
det-rNN-<hypothesis-slug>.ipynb    # 研究假设实验
```

全部保存在本目录（Kaggle 同步时同名即可）。

### 4.2 源码阅读约定（强制）

| 库 | 必读入口（实现时锁定到文件级） |
|----|--------------------------------|
| **ultralytics** | `utils/metrics.py`、loss/assign 路径、`engine/trainer.py`、`nn/tasks.py` |
| **detectron2** | `proposal_generator`、`roi_heads`、data 注册、COCO evaluator |
| **mmdet** | `models/dense_heads/*`、assigner/sampler、config 继承链 |
| **detr 官方** | matcher（Hungarian）、transformer decoder、loss |
| **RT-DETR 官方** | 与 YOLO 对照的 decode / 查询选择 |
| **torchvision detection** | Faster R-CNN `forward`、anchor、postprocess |

阅读产出：`papers/<id>/SOURCE_MAP.md`：

```text
| Path | Symbol | Paper eq/table |
|------|--------|----------------|
| ...  | ...    | ...            |
```

### 4.3 Kaggle 使用策略

| 用途 | 做法 |
|------|------|
| GPU 训练/推理 | Notebook + 固定 seed；记录 git commit / 权重版本 |
| 主粮数据 | VOC / COCO 子集练协议 → Wheat/Reef 或进行中赛 |
| 进行中赛 | RSNA Knee、Biohub Cell、车辆/高光谱等作 **P6**，不抢 P0–P3 |
| Models Hub | `ultralytics/yolo11` · `yolov8` · EfficientDet · OWL-ViT |
| 高票 kernel | 只借鉴数据管线与可视化，**终点是可辩护消融** |
| 密钥 | `KAGGLE_API_TOKEN` / `~/.kaggle/access_token`，**永不入库** |

### 4.4 实验日志最小字段

```yaml
exp_id: det-03-a1
commit: <git sha>
data: voc07_trainval_split_v1
model: yolo11n
recipe: {imgsz: 640, aug: mosaic, epochs: 50, seed: 42}
metrics: {mAP50: …, mAP: …, AP_s: …, AP_m: …, AP_l: …}
ablation: {var: value}
protocol: coco_style_v0
claim: "…"
result_interpretation: "…"
error_modes: [localization, duplicate, missed_small, …]
```

### 4.5 vendor / 数据纪律

```bash
# 示例：官方源码进 vendor（权重 gitignore）
git clone --depth 1 https://github.com/ultralytics/ultralytics vendor/ultralytics
git clone --depth 1 https://github.com/facebookresearch/detectron2 vendor/detectron2
git clone --depth 1 https://github.com/facebookresearch/detr vendor/detr
```

- `data/`、`*.pt`、`*.pth`、大数据 zip：**不入库**  
- 结果 JSON/小图：可进 `results/`  
- 每次实质改动：**阿里规范** `pull --rebase` → `commit` → `push`

---

## 5. 与相邻领域边界

| 领域 | 关系 | 本轨策略 |
|------|------|----------|
| 图像分类（v03） | backbone 预训练来源 | 需要时借分类轨；不回头重学全部分类 |
| 实例分割 | 检测 + mask 头 | Mask R-CNN 一章对照即可 |
| 跟踪 | 检测 + 时序关联 | Biohub 等赛可延伸，非主线 |
| 深度估计（v02） | 几何另一轨 | 不混线 |
| 3D / BEV 检测 | 另一问题设定 | 仅 1 篇对照 |

---

## 6. 当前状态与下一步（仍不教学）

| 项 | 状态 |
|----|------|
| 领域地图 + 资料筛选 + 顺序 + 工程结构 | ✅ 本文件 v2 |
| `catalog.json` 竞赛/数据/模型索引 | ✅（随调研更新） |
| 阶段目录脚手架 | ✅ `00`–`07` |
| P0–P7 notebook 与 tests | ⏳ 下一阶段再实现 |
| vendor 官方源码 clone | ⏳ 进入 P2/P3 时按需 |

**你确认本蓝图后，下一动作（需你说开始）才是：** 从 P0 起按六段闭环开写 `det-00`，并持续用 §3.3 测试缺口。
