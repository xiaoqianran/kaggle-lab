# Object Detection From Scratch

> 类似 **LLM from scratch**：一条可跑通的实验链，而不是概念清单。  
> 每一步固定五段：**概念 → 最小实现 → 真实输入 → 可观察输出 → 与上一步对比**。  
> 目标：用眼睛看到「这项技术到底修了什么病」。

配套：研究向蓝图 [LEARNING_ROADMAP.md](./LEARNING_ROADMAP.md) · 进度 [PROGRESS.md](./PROGRESS.md)

---

## 0. 全领域实验地图（一页总览）

```text
从零实现                      经典方法                    关键突破                 现代系统                前沿
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
FS00 框/IoU/NMS/mAP 协议
FS01 穷举滑窗 + 模板打分
FS02 图像金字塔多尺度
FS03 手工特征（HOG 直觉）        ← Dalal 2005 思想
FS04 候选区域 vs 穷举            ← Selective Search 思想
FS05 R-CNN：裁剪 + CNN 分类      ← Girshick 2014
FS06 Fast R-CNN：共享卷积+RoI    ← 2015
FS07 Faster R-CNN：RPN           ← 2015 两阶段闭合
FS08 单阶段密集头 YOLO/SSD 味    ← Redmon / Liu
FS09 Focal Loss 救不平衡         ← RetinaNet 2017
FS10 FPN 多尺度特征              ← Lin 2017
FS11 无锚 FCOS 中心分配          ← Tian 2019
FS12 DETR set prediction         ← Carion 2020
FS13 现代实时：YOLO 配方/RT-DETR ← 2023–2024
FS14 开放词汇检测                ← OWL-ViT / Grounding
FS15 研究闭环：消融·假设·claim   ← 你的 H*
```

**主线一句话：**  
穷举窗口 → 候选区域 → 共享特征两阶段 → 密集单阶段 → 多尺度/损失修病 → 端到端集合预测 → 实时与开放词汇。

**刻意不并入主线（旁支）：** 完整 3D/BEV 检测、多目标跟踪全栈、实例分割 Mask 全流程（需要时从 FS07 开分支）。

---

## 1. 步骤规格（每步五段）

图例：

| 标记 | 含义 |
|------|------|
| ✅ 已有可跑代码 | 本仓库已实现 |
| 🔗 映射研究轨 | 与 `run_p*` / `det-*` 共用 |
| 👁 必须看数/图 | 验收以对比表或可视化为主 |

---

### FS00 · 检测协议：框、IoU、NMS、mAP

| 段 | 内容 |
|----|------|
| **概念** | xyxy；IoU；NMS；AP50 vs AP@0.5:0.95；conf 阈值是协议一部分 |
| **最小实现** | `scripts/metrics.py` `boxes.py` |
| **真实输入** | 合成 GT + 刻意噪声/漏检/误检 pred |
| **可观察输出** | 表：perfect / loc_noise / miss / fp 的 AP 断崖 |
| **对比** | （起点） |
| **状态** | ✅ `run_p0_protocol.py` · `det-00` · `results/p0_protocol` |
| **你应感知到** | 方法对比前必须锁尺子；定位噪声专杀高 IoU AP |

---

### FS01 · 从零检测：穷举滑窗

| 段 | 内容 |
|----|------|
| **概念** | 检测=对每个窗口做分类；尺度/步长决定召回与速度 |
| **最小实现** | `scripts/fs01_sliding_window.py` 颜色模板打分 |
| **真实输入** | 合成色块图 64×64 |
| **可观察输出** | 每图窗口数、NMS 前后框数、AP；单图 best IoU |
| **对比 FS00** | 从「评指标」→「真的从图像搜物体」 |
| **状态** | ✅ `results/fs01_sliding_window` |
| **你应感知到** | 穷举极慢、重复框爆炸、固定窗卡尺度——三大古典病 |

---

### FS02 · 图像金字塔：尺度搜索

| 段 | 内容 |
|----|------|
| **概念** | 物体尺度变化 → 对图像做多分辨率再滑窗 |
| **最小实现** | `scripts/fs02_image_pyramid.py` |
| **真实输入** | 同合成，但 min/max 尺寸跨度加大 |
| **可观察输出** | single-scale vs pyramid 的 AP 与窗口代价比 |
| **对比 FS01** | 召回尺度更全，但计算 ≈ ×n_scales |
| **状态** | ✅ `results/fs02_image_pyramid` |
| **你应感知到** | 金字塔是「用算力买尺度」；不是优雅解 |

---

### FS03 · 手工特征直觉（HOG 思想）

| 段 | 内容 |
|----|------|
| **概念** | 梯度方向直方图抗光照；线性 SVM 时代的标准表征 |
| **最小实现** | 用 **通道梯度能量** 替代纯 RGB 均值作窗口打分（轻量 HOG 味） |
| **真实输入** | 合成图 + 轻微亮度扰动 |
| **可观察输出** | 亮度扰动下 color-template vs gradient-score 的 AP 稳定性 |
| **对比 FS01** | 同样滑窗，表征决定是否「怕光照」 |
| **状态** | 🔗 思想嵌入 FS01 对照实验可扩；主线可先读 NOTES 后进 FS05 |
| **你应感知到** | 深度学习前，**特征工程** 就是检测上限 |

---

### FS04 · 候选区域：不要扫全图

| 段 | 内容 |
|----|------|
| **概念** | Selective Search / EdgeBoxes：先猜「可能有物体的框」 |
| **最小实现** | 在 FS05 中用 **GT 抖动 + 随机背景框** 模拟好坏提案混合 |
| **真实输入** | 同合成 |
| **可观察输出** | 提案数 ≪ 滑窗数；漏提案 ⇒ 召回硬顶 |
| **对比 FS01–02** | 搜索空间从像素网格 → 稀疏框集合 |
| **状态** | ✅ 集成于 `fs05_rcnn_crops.py` |
| **你应感知到** | 提案质量=命根；R-CNN 系列上半辈子都在修这个 |

---

### FS05 · R-CNN：提案裁剪 + CNN 分类

| 段 | 内容 |
|----|------|
| **概念** | 每个提案 crop → resize → CNN → 类分数；再 NMS |
| **最小实现** | `scripts/fs05_rcnn_crops.py` TinyCropNet |
| **真实输入** | 合成图提案 |
| **可观察输出** | R-CNN-lite AP vs 滑窗 AP；CNN forward 次数对比叙事 |
| **对比 FS01** | 深度网络只打在提案上，不再穷举每个窗做深度特征 |
| **状态** | ✅ `results/fs05_rcnn_crops` |
| **你应感知到** | 2014 的革命是 **CNN 特征**，瓶颈是 **重复前向 + 外置提案** |

---

### FS06 · Fast R-CNN：共享卷积 + RoI

| 段 | 内容 |
|----|------|
| **概念** | 整图一次 backbone；RoIPool/RoIAlign 取区域特征 |
| **最小实现** | 教学说明 + 与 TwoStageLite 对照（网格中心近似 RoI） |
| **真实输入** | 合成 |
| **可观察输出** | 同样 AP 目标下，前向从「#提案次」降到「1 次 backbone」的复杂度叙事 |
| **对比 FS05** | 速度数量级改善的来源是 **共享计算** |
| **状态** | 🔗 `models.TwoStageLite` + `papers/FasterRCNN/SOURCE_MAP.md` |
| **你应感知到** | Fast 的「快」几乎全是系统结构，不是更深 |

---

### FS07 · Faster R-CNN：RPN 学会提案

| 段 | 内容 |
|----|------|
| **概念** | Region Proposal Network；端到端两阶段 |
| **最小实现** | `run_p2_two_stage.py`（RPN 式 obj 图 + cls/reg head） |
| **真实输入** | SynthDet |
| **可观察输出** | untrained AP50=0 → trained ≈0.64；loss 分项 |
| **对比 FS05** | 提案不再外挂启发式，**网络自己学 WHERE** |
| **状态** | ✅ `results/p2_two_stage` · `det-02` |
| **你应感知到** | 两阶段 DNA：先找物体性，再认类与精修框 |

---

### FS08 · 单阶段密集预测（YOLO/SSD 味）

| 段 | 内容 |
|----|------|
| **概念** | 网格直接回归框；一次前向全图 |
| **最小实现** | `run_p3_yolo_lite.py` CenterNetLite |
| **真实输入** | SynthDet |
| **可观察输出** | AP50≈0.81；conf 消融表 |
| **对比 FS07** | 去掉显式 RPN 阶段；更快迭代，平衡靠后续 FS09–10 |
| **状态** | ✅ `results/p3_yolo_lite` · `det-03` |
| **你应感知到** | 单阶段「快」的代价是 **正负样本极端不平衡**（下一刀 FS09） |

---

### FS09 · Focal Loss：救密集头的梯度

| 段 | 内容 |
|----|------|
| **概念** | 易分负样本淹没梯度；\((1-p_t)^\gamma\) 重加权 |
| **最小实现** | `scripts/fs09_focal_loss.py` CE vs Focal objectness |
| **真实输入** | 同 SynthDet；打印 pos_fraction |
| **可观察输出** | 正样本占比 ~1%；CE vs Focal 的 AP/曲线 |
| **对比 FS08** | 同结构，只改损失 → 一阶段重新能打两阶段 |
| **状态** | ✅ `results/fs09_focal_loss` |
| **你应感知到** | RetinaNet 的贡献首先是 **损失问题定义** |

---

### FS10 · FPN：多尺度特征

| 段 | 内容 |
|----|------|
| **概念** | 大物体看深层、小物体看浅层；自顶向下融合 |
| **最小实现** | `scripts/fs10_fpn_multiscale.py` 双 stride head |
| **真实输入** | 尺寸跨度大的 SynthDet |
| **可观察输出** | single stride vs dual：AP / AP_s / AP_l |
| **对比 FS08** | 尺度问题从「图像金字塔」内化到「特征金字塔」 |
| **状态** | ✅ `results/fs10_fpn_multiscale` |
| **你应感知到** | 现代检测标配 Neck；小目标经常是 stride 路由问题 |

---

### FS11 · 无锚：中心分配（FCOS 味）

| 段 | 内容 |
|----|------|
| **概念** | 不预设 anchor box；落在物体中心的点负责回归 ltrb |
| **最小实现** | CenterNetLite 即无锚中心分配（已是主实现） |
| **真实输入** | SynthDet |
| **可观察输出** | 与「人为 anchor 匹配」叙事对照（`assignment.assign_iou`） |
| **对比 FS07 锚** | 少一套超参（scales/ratios）；匹配更直观 |
| **状态** | ✅ 主路径即此；锚对照见 `run_p1_boxes.py` |
| **你应感知到** | 锚是归纳偏置；无锚把病转到 **中心采样与范围** |

---

### FS12 · DETR：集合预测与 Hungarian

| 段 | 内容 |
|----|------|
| **概念** | 固定 queries；二分图匹配；可无 NMS |
| **最小实现** | `run_p4_detr.py` DETRLite |
| **真实输入** | SynthDet |
| **可观察输出** | 匹配 pred↔gt 索引；loss 下降；query 数消融 |
| **对比 FS08** | 不再「网格上的密集候选 + NMS」，而是 **集合** |
| **状态** | ✅ `results/p4_detr` · `det-04` |
| **你应感知到** | DETR 慢收敛是特征不是 bug；后续 DINO/RT-DETR 修的是这个 |

---

### FS13 · 现代实时系统（YOLO 配方 / RT-DETR）

| 段 | 内容 |
|----|------|
| **概念** | 强增广、EMA、标签分配进化、实时端到端 |
| **最小实现** | `run_p5_recipe_ablation.py`（配方）+ SOURCE_MAP 指向 Ultralytics/RT-DETR |
| **真实输入** | SynthDet；Kaggle 上换 VOC/Wheat |
| **可观察输出** | longer / flip / lr 消融表；best recipe |
| **对比 FS08–12** | 结构之上，**训练系统** 决定能否工业可用 |
| **状态** | ✅ `results/p5_recipe`；真 YOLO11 权重 → Kaggle 增强 |
| **你应感知到** | 2023+ 差距常在数据与配方，不在再堆 3 个 block |

---

### FS14 · 开放词汇与真实域

| 段 | 内容 |
|----|------|
| **概念** | 文本类名查询；域偏移；Kaggle 脏数据 |
| **最小实现** | `run_p6_domain.py` 域偏移 + catalog 竞赛 playbook |
| **真实输入** | clean train vs dark/small OOD；竞赛索引 |
| **可观察输出** | ID vs OOD AP；error modes（miss/fp/cls） |
| **对比 FS13** | 闭集合成满分 ≠ 真实域；开放类是另一问题设定 |
| **状态** | ✅ `results/p6_domain` · `det-06` |
| **你应感知到** | 产品检测的主敌经常是 **域与长尾**，不是 COCO 再涨 0.3 AP |

---

### FS15 · 研究闭环

| 段 | 内容 |
|----|------|
| **概念** | 可证伪假设；claim 审查；阴性结果 |
| **最小实现** | `run_p7_hypothesis.py` · `papers/CLAIM_REVIEWS.md` |
| **真实输入** | H1 尺度、H2 指标敏感、H3 matching cost |
| **可观察输出** | accept/reject 表 |
| **对比 全程** | 从「会跑模型」→「会判定贡献」 |
| **状态** | ✅ `results/p7_hypothesis` · `det-07` |
| **你应感知到** | 研究能力 = 协议 + 对照 + 诚实解释 |

---

## 2. 推荐执行顺序（严格串行）

```text
FS00 → FS01 → FS02 → FS04 → FS05 → FS06 → FS07
                                              ↓
                         FS08 → FS09 → FS10 → FS11 → FS12
                                              ↓
                                    FS13 → FS14 → FS15
```

**最短「能讲清现代检测」路径（时间紧）：**  
`FS00 → FS01 → FS05 → FS07 → FS08 → FS09 → FS12 → FS13 → FS15`  
（FS02/04/06/10/11/14 读地图+结果 JSON 补感知）

---

## 3. 统一记分板（做下一步强制填一行）

| FS | 方法 | 搜索方式 | 共享 backbone | 典型病 | 速度感 | 一眼失败 |
|----|------|----------|---------------|--------|--------|----------|
| 01 | 滑窗模板 | 穷举 | 否 | 慢/尺度 | 极慢 | 固定窗、误检刷屏 |
| 05 | R-CNN | 提案 | 否（每 crop） | 提案漏检 | 慢 | 裁剪错位 |
| 07 | Faster | RPN | 是 | 锚超参 | 中 | 小目标 |
| 08 | 密集单阶段 | 网格 | 是 | 不平衡 | 快 | 背景分数 |
| 09 | +Focal | 网格 | 是 | （缓解） | 快 | 仍尺度 |
| 10 | +FPN | 多层 | 是 | （缓解） | 中快 | 极小目标 |
| 12 | DETR | queries | 是 | 收敛慢 | 中 | 小数据不稳 |
| 13 | YOLO 配方 | 网格+工程 | 是 | 域偏移 | 很快 | 域外 |
| 14 | 域/开词 | 不定 | 是 | 标注/文本 | 不定 | 长尾类 |

---

## 4. 与仓库文件映射

| FS | 代码 / 结果 |
|----|-------------|
| 00 | `run_p0_protocol.py` → `results/p0_protocol` |
| 01 | `fs01_sliding_window.py` → `results/fs01_sliding_window` |
| 02 | `fs02_image_pyramid.py` → `results/fs02_image_pyramid` |
| 04–05 | `fs05_rcnn_crops.py` → `results/fs05_rcnn_crops` |
| 06–07 | `run_p2_two_stage.py` · `papers/FasterRCNN/SOURCE_MAP.md` |
| 08 | `run_p3_yolo_lite.py` · `papers/YOLO_Ultralytics/SOURCE_MAP.md` |
| 09 | `fs09_focal_loss.py` → `results/fs09_focal_loss` |
| 10 | `fs10_fpn_multiscale.py` → `results/fs10_fpn_multiscale` |
| 11 | `models.CenterNetLite` + `run_p1_boxes.py` 锚对照 |
| 12 | `run_p4_detr.py` · `papers/DETR/SOURCE_MAP.md` |
| 13 | `run_p5_recipe_ablation.py` |
| 14 | `run_p6_domain.py` · `catalog.json` |
| 15 | `run_p7_hypothesis.py` · `papers/CLAIM_REVIEWS.md` |

Notebook：研究轨 `det-*` 与 from-scratch 共用脚本；可选 `fs-XX-*.ipynb` 薄封装。  
一键：`./scripts/run_fs_chain.sh`

---

## 5. 验收：什么叫这条 from-scratch 完成

你能 **不看笔记** 演示：

1. 滑窗为何必然慢，并指出 NMS 前后框数变化；  
2. R-CNN「提案+分类」相对滑窗省了什么、死在什么上；  
3. Faster 的 RPN 与 YOLO 密集头差在哪；  
4. 展示 pos_fraction 并解释 Focal 修什么；  
5. 画出/说出 DETR matching 与 NMS 范式对立；  
6. 用一张消融表说明配方或域偏移；  
7. 对一篇检测论文做 5 分钟 claim 审查。

---

## 6. 与 P0–P7 研究轨的关系

| 轨 | 目的 |
|----|------|
| **FROM_SCRATCH (FS\*)** | 历史因果链与「修病」感知 |
| **LEARNING_ROADMAP (P\*)** | 研究能力与协议纪律 |

两者共享 `scripts/` 与 `results/`；不要当成两套互斥课程——**先 FS 建立直觉，用 P 轨固化研究肌肉**。

本文件是 **from-scratch 实验地图的唯一权威索引**；数字以 `results/**/results.json` 为准。
