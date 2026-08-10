# PROGRESS · 深度估计学习工程

最后更新：2026-08-10  
状态：**P0–P7 验收通过（可复现脚本 + 文档 + 实验 JSON）**

## 总览

| Phase | 状态 | 验收 |
|-------|------|------|
| P0 指标与对齐 | ✅ | `tests/test_metrics.py` + `results/p0_alignment` |
| P1 几何 | ✅ | `tests/test_geometry.py` |
| P2 有监督 | ✅ | silog AbsRel≈0.014 ≪ untrained 0.47 |
| P3 自监督 | ✅ | bad/good photo ratio≈460；loss 下降 |
| P4 基础模型协议 | ✅ | LS ≪ none on disparity |
| P5 度量协议 | ✅ | oracle vs misuse；焦距耦合 |
| P6 研究协议 | ✅ | `papers/CLAIM_REVIEWS.md` |
| P7 假设实验 | ✅ | `results/p7_hypothesis` 可重复检验 H1 |

## 已完成产物

### 代码
- `scripts/metrics.py` `geometry.py` `models.py` `photometric.py` `synth_data.py`
- `scripts/run_p{0,2,3,4,5,7}_*.py`
- `tests/test_metrics.py` `tests/test_geometry.py`

### Notebooks
- `de-00` … `de-07`（调用上述脚本）

### 文档
- 各 Phase `*/NOTES.md`
- `papers/Monodepth2/SOURCE_MAP.md`
- `papers/CLAIM_REVIEWS.md`
- `LEARNING_ROADMAP.md`（蓝图）

### 实验 JSON
- `results/p0_alignment/results.json`
- `results/p2_supervised/results.json`
- `results/p3_photometric/results.json`
- `results/p4_foundation/results.json`
- `results/p5_metric/results.json`
- `results/p7_hypothesis/results.json`

## 路线修正（已记录）

| 修正 | 原因 |
|------|------|
| P2/P7 使用 **easy synthetic**（颜色相关深度）而非完整 NYU | 本环境 CPU、无 NYU 全量；保证“能训练收敛”的验收，而非虚假 SOTA |
| P4/P5 以 **协议实验** 为主，真权重可选 | 多 GB foundation 权重未缓存；MiDaS hub 可选 |
| SI-RMSE 加 `max(var,0)` | 纯尺度时浮点方差为负导致 nan |

## 当前任务

无阻塞。可选增强（非停止条件）：
1. Kaggle GPU：换 NYU/KITTI 真数据重跑 P2  
2. 挂 Depth Anything V2 / Depth Pro 权重重跑 P4/P5  
3. 新开 arXiv 当周论文 claim 卡  

## 如何一键复验

```bash
cd notebooks/v02-shuhuaqaq@-深度估计
python tests/test_metrics.py
python tests/test_geometry.py
python scripts/run_p0_experiment.py
python scripts/run_p2_train.py
python scripts/run_p3_photometric.py
python scripts/run_p4_foundation_protocol.py
python scripts/run_p5_metric_protocol.py
python scripts/run_p7_hypothesis.py
```

## 问题日志

| 问题 | 处理 |
|------|------|
| P2 初版不收敛 | sigmoid→sigmoid·max_depth；easy 数据 |
| P3 初版 good>bad | 重建几何正确的 fronto-plane 立体对 |
| SI-RMSE nan | 数值保护 |

## From Scratch 轨（2026-08-10 增补）

权威地图：`FROM_SCRATCH.md`（FS00–FS14）

| FS | 状态 | 入口 |
|----|------|------|
| FS00 | ✅ | `run_p0_experiment.py` |
| FS01 | ✅ | `scripts/fs01_pointcloud_vis.py` |
| FS02 | ✅ | `scripts/fs02_block_match_stereo.py` |
| FS03 | ✅ | `scripts/fs03_cost_volume.py` |
| FS04 | ✅ | `scripts/fs04_tiny_regressor.py` |
| FS05 | ✅ | SILog 消融 in `run_p2_train.py` |
| FS06 | ✅ | `scripts/fs06_skip_ablation.py` |
| FS07–08 | ✅ | photometric + Monodepth2 SOURCE_MAP |
| FS09 | ✅ | `run_p4_foundation_protocol.py` |
| FS10–13 | 🔜 真权重推理 | 协议已备 |
| FS14 | ✅ | claim reviews + P7 |

