# 目标检测 · 研究导向学习蓝图

> **本文件只做规划，不教学。**  
> 目标：最短路径达到「能独立读最新检测论文、判断贡献、复现、设计 baseline/ablation、提出研究假设」的研究能力。  
> 原则：每个主题 **1–3 个最优来源**；顺序按研究能力最短路径，不按教材章节/编年史。  
> 领域锁定：**2D Object Detection（目标检测）** — 含密集/小目标/开放词汇；实例分割、跟踪、3D 检测仅作对照边界。  
> 工程落点：本目录 `tracks/object-detection/` · Kaggle 用户 `xiaosuhuaer`

最后更新：2026-08-10（**P0–P7 已执行验收**，见 `PROGRESS.md`）

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

| # | 标准 | 本工程证据 |
|---|------|------------|
| 1 | 读新论文口述 claim 结构 | `papers/CLAIM_REVIEWS.md` + 模板 |
| 2 | 两条范式可训 | P2 TwoStageLite + P3 CenterNetLite；P4 DETR-lite |
| 3 | 源码地图 | `papers/*/SOURCE_MAP.md` |
| 4 | 消融 ≥2 变量 | P5 recipes；P0 conf；P4 queries |
| 5 | 可证伪假设 | P7 H1–H3 全部 resolved |

---

## 2. 全球最佳资料筛选（每个主题 ≤3）

> 详见历史 v2 全文；执行期未改筛选原则。主路径：CS231n · Faster R-CNN · RetinaNet · Ultralytics/mmdet · DETR · RT-DETR · COCO eval。

### 2.8 高校课程（摘要）

| 课 | 角色 |
|----|------|
| Stanford CS231n | 主课入口 |
| CMU 16-824 | 读论文纪律 |
| MIT 6.8300 | 几何边界（非主线） |
| Berkeley CS280 / PKU Intro2CV | 可选对照 |

### 2.9 关键论文栈

Faster R-CNN → RetinaNet → FCOS/YOLO 现代 → DETR → DINO/Deformable → RT-DETR → OWL-ViT/Grounding DINO 择一

---

## 3. 最优学习顺序与执行映射

| Phase | 目录 | 脚本 | Notebook | 结果 |
|-------|------|------|----------|------|
| P0 | `00-map/` | `run_p0_protocol.py` | `det-00` | `results/p0_protocol` |
| P1 | `01-foundations/` | `run_p1_boxes.py` | `det-01` | `results/p1_boxes` |
| P2 | `02-two-stage/` | `run_p2_two_stage.py` | `det-02` | `results/p2_two_stage` |
| P3 | `03-one-stage/` | `run_p3_yolo_lite.py` | `det-03` | `results/p3_yolo_lite` |
| P4 | `04-modern-detr/` | `run_p4_detr.py` | `det-04` | `results/p4_detr` |
| P5 | `05-training-recipe/` | `run_p5_recipe_ablation.py` | `det-05` | `results/p5_recipe` |
| P6 | `06-frontier/` | `run_p6_domain.py` | `det-06` | `results/p6_domain` |
| P7 | `07-research-lab/` | `run_p7_hypothesis.py` | `det-07` | `results/p7_hypothesis` |

### 3.1 六段绑定（强制）

原理 → 手写 → GitHub 源码 → Kaggle/本地实验 → 消融 → 解释 — 每阶段 `NOTES.md` + JSON findings。

---

## 4. GitHub / Kaggle 学习工程结构

```text
tracks/object-detection/
├── LEARNING_ROADMAP.md · PROGRESS.md · README.md · catalog.json
├── det-00 … det-07.ipynb
├── 00-map/ … 07-research-lab/
├── scripts/   # boxes metrics assignment data_synth models run_p*
├── tests/
├── papers/    # SOURCE_MAP + CLAIM_REVIEWS
└── results/   # p0…p7 JSON
```

### 4.5 环境

```bash
# 本沙箱已验证：
source /workspace/.venv-det/bin/activate  # torch CPU + numpy + scipy
cd tracks/object-detection && ./scripts/run_all.sh
```

---

## 5. 路线修正记录

见 `PROGRESS.md`「路线修正」。核心：CPU 下用可学习合成检测数据保证闭环；概念与官方库 SOURCE_MAP 对齐；真 COCO/Wheat 留给 Kaggle GPU 增强项。

---

## 6. 当前状态

| 项 | 状态 |
|----|------|
| 领域地图 + 资料筛选 + 顺序 | ✅ |
| P0–P7 可运行实验 | ✅ |
| 单元测试 | ✅ |
| notebook 入口 | ✅ |
| 远程仓库 | 随 commit 推送 |
