# 03-one-stage · P3 YOLO / 密集预测

**出口：** YOLO 数据格式训通；读透 loss + metrics；能改 conf/imgsz 并解释 AP 变化。

## 绑定来源（≤3）

1. RetinaNet + Focal Loss（不平衡）  
2. Ultralytics YOLOv8/11 源码 + docs  
3. FCOS（无锚对照，扫读）

## 六段清单（det-03）

- [ ] 原理：密集头 + 分配 + decode  
- [ ] 手写：grid 上的解码玩具  
- [ ] 源码：ultralytics metrics/loss/trainer  
- [ ] Kaggle：mini 集 train+val mAP  
- [ ] 消融：imgsz 或 mosaic  
- [ ] 解释：AP_s 变化归因  

## 门禁

只会 CLI、说不清 loss 分项 → 回源码段。
