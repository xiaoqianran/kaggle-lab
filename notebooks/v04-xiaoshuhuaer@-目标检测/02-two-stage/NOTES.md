# 02-two-stage · P2 Faster R-CNN

**出口：** 讲清 RPN → RoI Align → cls+reg；VOC/子集可训；Detectron2 或 torchvision SOURCE_MAP。

## 绑定来源（≤3）

1. **Faster R-CNN** 原文  
2. **Detectron2** proposal_generator + roi_heads  
3. torchvision detection Faster R-CNN（短路径对照）

## 六段清单（det-02）

- [ ] 原理：RPN 目标与多任务损失  
- [ ] 手写：RoI 特征池化直觉（可简化）  
- [ ] 源码：Detectron2 或 torchvision 文件级地图  
- [ ] Kaggle/本地：VOC 微调 baseline  
- [ ] 消融：backbone 或训练 epoch  
- [ ] 解释：定位误差 vs 分类误差  

## 不做什么

不复现 2014 selective search 全流程。

## 执行状态

- ✅ 已完成（见 `PROGRESS.md` + `results/`）
