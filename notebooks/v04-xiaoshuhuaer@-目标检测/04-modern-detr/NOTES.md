# 04-modern-detr · P4 DETR 族

**出口：** 讲清 set prediction 与 Hungarian matching；DETR 或 RT-DETR 推理/小实验；不与 YOLO 概念混为一谈。

## 绑定来源（≤3）

1. **DETR** 原文 + facebookresearch/detr  
2. **DINO** 或 **Deformable DETR** 择一  
3. **RT-DETR**（实时端到端对照 YOLO）

## 六段清单（det-04）

- [ ] 原理：bipartite matching cost  
- [ ] 手写：小规模 Hungarian 匹配 demo  
- [ ] 源码：matcher + decoder  
- [ ] 实验：官方权重推理 + 匹配可视化  
- [ ] 消融：query 数或训练 epoch（小规模）  
- [ ] 解释：为何收敛慢 / 如何被后续工作修  

## 门禁

把 DETR 说成“带 attention 的 YOLO” → 重读原理。

## 执行状态

- ✅ 已完成（见 `PROGRESS.md` + `results/`）
