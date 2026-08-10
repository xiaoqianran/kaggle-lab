# v02 · shuhuaqaq@ · 深度估计

**状态：P0–P7 已跑通**（见 [PROGRESS.md](./PROGRESS.md)）。  
蓝图：[LEARNING_ROADMAP.md](./LEARNING_ROADMAP.md)

## 一键复验

```bash
python tests/test_metrics.py
python tests/test_geometry.py
python scripts/run_p0_experiment.py
python scripts/run_p2_train.py
python scripts/run_p3_photometric.py
python scripts/run_p4_foundation_protocol.py
python scripts/run_p5_metric_protocol.py
python scripts/run_p7_hypothesis.py
```

## Notebooks

| 文件 | Phase |
|------|-------|
| `de-00-metrics-and-alignment.ipynb` | P0 |
| `de-01-pinhole-and-stereo-depth.ipynb` | P1 |
| `de-02-nyu-unet-supervised.ipynb` | P2 |
| `de-03-monodepth2-source-and-mini.ipynb` | P3 |
| `de-04-dav2-midas-zero-shot.ipynb` | P4 |
| `de-05-metric-sota-protocol.ipynb` | P5 |
| `de-06-research-claim-review.ipynb` | P6 |
| `de-07-hypothesis-edge-vs-capacity.ipynb` | P7 |

## 目录

| 路径 | 内容 |
|------|------|
| `00-map/`…`07-research-lab/` | 阶段 NOTES |
| `scripts/` | 指标/几何/训练/协议实验 |
| `tests/` | 单元测试 |
| `papers/` | SOURCE_MAP + claim 审查 |
| `results/` | 实验 JSON |
| `vendor/` | 官方源码 clone（gitignore） |

## 关键结果速览

| 实验 | 结论 |
|------|------|
| P0 | 尺度→median；仿射→LS；逆深度不可硬评 |
| P2 | silog AbsRel ≈0.014（easy synth） |
| P3 | 错误深度光度误差 ≈460× |
| P4 | disparity+LS AbsRel 0.07 vs none 0.94 |
| P5 | metric 必须 align=none；median 掩盖尺度偏 |
| P7 | H1 可脚本化检验（见 JSON） |

## 约定

- 六段闭环：原理 → 手写 → 源码 → 实验 → 消融 → 解释  
- 每次实质改动：阿里规范 commit + push  
- 大权重/数据不入库  
