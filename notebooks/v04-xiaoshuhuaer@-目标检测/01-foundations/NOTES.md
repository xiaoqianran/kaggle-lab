# 01-foundations · P1 框、锚、分配

**出口：** 框编解码正确；IoU 匹配；增广后框同步变换；可视化正负样本。

## 核心问题

- 网络输出如何变成框？  
- 哪些预测算正样本（assignment）？  
- 几何增广如何作用于框？

## 绑定来源（≤3）

1. Faster R-CNN 中 anchor / 匹配叙述（为 P2 铺路）  
2. YOLO 文档中的 box loss / 格式说明  
3. 自建 toy：固定特征图上的锚与 GT 匹配可视化

## 六段清单（det-01）

- [ ] 原理：anchor 与预测偏移  
- [ ] 手写：encode/decode + 匹配  
- [ ] 源码：任选 ultralytics 或 torchvision 一处 box coder  
- [ ] Kaggle：可视化 20 张图 GT  
- [ ] 消融：匹配 IoU 阈值  
- [ ] 解释：漏匹配 vs 错匹配的误差模式  

## 门禁

改 random flip 后框仍正确；否则不进 P2。

## 执行状态

- ✅ 已完成（见 `PROGRESS.md` + `results/`）
