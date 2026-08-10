# v04 · xiaoshuhuaer@ · 目标检测

**状态：P0–P7 已跑通**（见 [PROGRESS.md](./PROGRESS.md)）。  
蓝图：[LEARNING_ROADMAP.md](./LEARNING_ROADMAP.md)

Kaggle 用户：`xiaosuhuaer`。Notebook：`det-NN-*.ipynb`。

## 一键复验

```bash
source /workspace/.venv-det/bin/activate   # 或自备 torch
cd notebooks/v04-xiaoshuhuaer@-目标检测
./scripts/run_all.sh
```

## Notebooks

| 文件 | Phase |
|------|-------|
| `det-00-iou-map-protocol.ipynb` | P0 |
| `det-01-boxes-anchors-nms.ipynb` | P1 |
| `det-02-faster-rcnn-source-map.ipynb` | P2 |
| `det-03-yolo-train-mini.ipynb` | P3 |
| `det-04-detr-matching-lite.ipynb` | P4 |
| `det-05-recipe-ablation.ipynb` | P5 |
| `det-06-kaggle-domain-baseline.ipynb` | P6 |
| `det-07-research-hypothesis.ipynb` | P7 |

## 目录

| 路径 | 内容 |
|------|------|
| `00-map/`…`07-research-lab/` | 阶段 NOTES |
| `scripts/` | 指标/框/模型/训练协议 |
| `tests/` | 单元测试 |
| `papers/` | SOURCE_MAP + claim 审查 |
| `results/` | 实验 JSON |
| `catalog.json` | Kaggle 竞赛/数据/模型索引 |

## 关键结果速览

| 实验 | 结论 |
|------|------|
| P0 | 定位噪声：AP 远比 AP50 敏感；conf 必须写入协议 |
| P2 | TwoStageLite AP50 ≈0.64（synth） |
| P3 | Dense lite AP50 ≈0.81 |
| P4 | DETR matching 正确；收敛慢 |
| P5 | `flip_longer` 最优 AP50≈0.79 |
| P6 | ID→OOD AP50 0.79→0.30（miss 为主） |
| P7 | H1–H3 均 accepted |

## 约定

- 六段闭环：原理 → 手写 → 源码 → 实验 → 消融 → 解释  
- 每次实质改动：阿里规范 pull + commit + push  
- 大权重/数据不入库  
