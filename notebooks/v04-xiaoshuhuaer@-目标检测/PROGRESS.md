# PROGRESS · 目标检测学习工程

最后更新：2026-08-10  
状态：**P0–P7 验收通过（可复现脚本 + 测试 + 实验 JSON + notebook）**

## 总览

| Phase | 状态 | 验收证据 |
|-------|------|----------|
| P0 指标与协议 | ✅ | `tests/test_metrics.py` + `results/p0_protocol` |
| P1 框/锚/分配 | ✅ | `tests/test_boxes.py` + `results/p1_boxes` |
| P2 两阶段 | ✅ | `results/p2_two_stage` AP50 0.00→**0.64** |
| P3 单阶段 | ✅ | `results/p3_yolo_lite` AP50→**0.81** |
| P4 DETR | ✅ | `results/p4_detr` loss↓ + matching + query 消融 |
| P5 配方消融 | ✅ | `results/p5_recipe` 5 recipes，best=`flip_longer` AP50**0.79** |
| P6 域偏移 | ✅ | `results/p6_domain` ID AP50 0.79 vs OOD **0.30** |
| P7 假设 | ✅ | `results/p7_hypothesis` H1–H3 **全部 accepted** |

## 关键实验结果

| 实验 | 结论 |
|------|------|
| P0 perfect vs loc_noise | 定位噪声：AP50 0.99→0.78，AP 0.99→**0.22**（高 IoU 更敏感） |
| P0 conf thr | 协议必须固定 conf；否则模型对比无效 |
| P1 pos_thr | 0.5→0.7 正样本锚均值 4.2→1.9 |
| P2 two-stage lite | 可训，AP50≈0.64（合成色块） |
| P3 dense lite | 同设定 AP50≈0.81，收敛快于 DETR |
| P4 DETR-lite | 匹配可运行；收敛慢（预期） |
| P5 recipes | longer + flip 最优；low_lr 欠拟合 |
| P6 domain | 域偏移主因 miss；小 FT 可部分恢复（见 JSON） |
| P7 H1 | 小目标 val 上 train-small ≫ train-large |
| P7 H2 | loc noise 下 AP 降幅 ≥ AP50 |
| P7 H3 | Hungarian class cost 提升匹配类别一致性 |

## 产物清单

### 代码
- `scripts/boxes.py` `metrics.py` `assignment.py` `data_synth.py` `models.py`
- `scripts/run_p{0..7}_*.py` `scripts/run_all.sh`
- `tests/test_{metrics,boxes,assignment}.py`

### Notebooks
- `det-00` … `det-07`（调用脚本）

### 文档
- `LEARNING_ROADMAP.md` · 各 Phase `NOTES.md`
- `papers/FasterRCNN/SOURCE_MAP.md` · `YOLO_Ultralytics` · `DETR`
- `papers/CLAIM_REVIEWS.md`

### 实验 JSON
- `results/p0_protocol` … `results/p7_hypothesis`

## 路线修正（已执行）

| 修正 | 原因 |
|------|------|
| 使用 **合成色块检测数据** 而非完整 COCO/Wheat 下载 | 本环境 CPU、无多 GB 数据集缓存；保证「能训练收敛 + 协议正确」的验收 |
| P2 为 **TwoStageLite**（RPN 式 obj + head）非完整 Faster R-CNN+RoIAlign | CPU 预算；概念与 SOURCE_MAP 对齐 Detectron2/torchvision |
| P3 为 **CenterNetLite**（中心分配+ltrb）映射 YOLO/FCOS 范式 | 同左；Ultralytics 全量训练放 Kaggle GPU 后续 |
| P6 用 **可控域偏移** 模拟 Kaggle 真实域 | 不阻塞于竞赛数据下载；playbook 写入结果 JSON |
| DETR AP 绝对值偏低 | 已知慢收敛；验收用 loss 下降 + matching 正确 + query 消融 |

## 一键复验

```bash
# 建议：source /workspace/.venv-det/bin/activate  # 含 torch
cd notebooks/v04-xiaoshuhuaer@-目标检测
./scripts/run_all.sh
# 或逐步：
python tests/test_metrics.py && python tests/test_boxes.py && python tests/test_assignment.py
python scripts/run_p0_protocol.py
python scripts/run_p1_boxes.py
python scripts/run_p2_two_stage.py
python scripts/run_p3_yolo_lite.py
python scripts/run_p4_detr.py
python scripts/run_p5_recipe_ablation.py
python scripts/run_p6_domain.py
python scripts/run_p7_hypothesis.py
```

## 当前任务

无阻塞。可选增强（**非**停止条件）：
1. Kaggle GPU：VOC/COCO 子集或 Global Wheat 重跑 P3/P6  
2. vendor clone detectron2 / ultralytics 做真源码步进  
3. 当周 arXiv 检测论文 claim 卡  

## 问题日志

| 问题 | 处理 |
|------|------|
| 系统 Python 无 torch/numpy | 使用 `/workspace/.venv-det`（torch CPU） |
| P4 初版匹配 cost 很大 | 未训权重正常；训练后 loss 从 1.77→0.71 |
| P0 false_pos 仍高 AP | 高分 GT 仍排前；结论写入「排序与阈值共同决定 precision」 |
