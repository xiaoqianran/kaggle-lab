# v04 · xiaoshuhuaer@ · 目标检测

**状态：P0–P7 研究轨 + From-Scratch 实验地图已落地**（见 [PROGRESS.md](./PROGRESS.md)）。

| 文档 | 作用 |
|------|------|
| **[FROM_SCRATCH.md](./FROM_SCRATCH.md)** | **从零→经典→突破→现代→前沿** 全链路实验地图（主入口） |
| [LEARNING_ROADMAP.md](./LEARNING_ROADMAP.md) | 研究向蓝图与能力验收 |
| [PROGRESS.md](./PROGRESS.md) | 进度与数字 |
| [catalog.json](./catalog.json) | Kaggle 索引 |

Kaggle 用户：`xiaosuhuaer`。

## 一键复验

```bash
source /workspace/.venv-det/bin/activate
cd notebooks/v04-xiaoshuhuaer@-目标检测
./scripts/run_fs_chain.sh    # From-Scratch 全链
# 或仅研究轨：
./scripts/run_all.sh
```

## From-Scratch 速览

```text
FS00 协议 → FS01 滑窗 → FS02 金字塔 → FS05 R-CNN
    → FS07 Faster → FS08 单阶段 → FS09 Focal → FS10 多尺度
    → FS12 DETR → FS13 配方 → FS14 域 → FS15 假设
```

## Notebooks（研究轨）

| 文件 | Phase |
|------|-------|
| `det-00` … `det-07` | P0–P7 |

## 关键数字（摘录）

| 实验 | 结果 |
|------|------|
| FS01 滑窗 | ~169 窗/图；AP50≈0.34 |
| FS05 R-CNN-lite vs 滑窗 | AP50 **0.85 vs 0.37** |
| P2/FS07 TwoStage | AP50 → **0.64** |
| P3/FS08 Dense | AP50 → **0.81** |
| FS09 Focalγ1 vs CE | AP50 **0.71 vs 0.67**（γ=2 默认可更差→超参课） |
| P6 域偏移 | ID 0.79 → OOD **0.30** |
| P7 | H1–H3 accepted |

## 约定

- 六段/五段闭环；大文件不入库  
- 实质改动：阿里规范 pull → commit → push  
