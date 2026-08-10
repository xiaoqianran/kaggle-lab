# v04 · xiaoshuhuaer@ · 目标检测

**From-Scratch FS00–FS15 已全部验收** · 见 [PROGRESS.md](./PROGRESS.md) · [FROM_SCRATCH.md](./FROM_SCRATCH.md)

## 一键复验

```bash
source /workspace/.venv-det/bin/activate
cd notebooks/v04-xiaoshuhuaer@-目标检测
python scripts/run_fs_all.py
```

## 能力阶梯

```text
FS00 协议 → FS01 滑窗 → FS02 金字塔 → FS03 特征 → FS04 提案
 → FS05 R-CNN → FS06 共享计算 → FS07 Faster → FS08 单阶段
 → FS09 Focal → FS10 多尺度 → FS11 无锚 → FS12 DETR
 → FS13 配方 → FS14 域偏移 → FS15 研究假设
```

## 关键对比（实测）

| 对比 | 结果 |
|------|------|
| 滑窗 vs R-CNN-lite | AP50 0.37 → **0.85** |
| 提案数 vs 窗数 | ~22 vs ~169 |
| 共享 vs 500 裁剪前向 | **~10×** 墙钟 |
| TwoStage / Dense | AP50 **0.65 / 0.81** |
| 域偏移 | ID 0.79 → OOD 0.30 → FT 0.92 |
| H1–H3 | 全部 accepted |

每步结果：`results/fsXX_*/`（`results.json` + PNG）。

研究轨 P0–P7：`./scripts/run_all.sh` · 蓝图 `LEARNING_ROADMAP.md`
