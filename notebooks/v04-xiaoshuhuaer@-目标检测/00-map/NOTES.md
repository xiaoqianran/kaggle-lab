# 00-map · P0 指标与复现纪律

**出口：** 会算 IoU/NMS；能解释 AP50 vs AP@0.5:0.95；固定 seed；识破假 SOTA。

## 必会术语

| 术语 | 一句话 |
|------|--------|
| bbox 表示 | `xyxy` / `xywh` / `cxcywh`；像素 vs 归一化 |
| IoU 族 | 匹配阈值 vs 损失函数 两种用法勿混 |
| NMS | 按分数抑制重叠；Soft-NMS / DETR 无 NMS |
| mAP | COCO 默认 AP@0.5:0.95；另报 AP50、AP_s/m/l |
| 协议 | 同一 val、同一 max-det、同一坐标约定 |

## 绑定来源（≤3）

1. COCO eval 定义 + pycocotools  
2. `ultralytics/utils/metrics.py`  
3. CS231n Detection 讲（总览）

## 六段清单（det-00）

- [ ] 原理：AP 曲线如何汇总  
- [ ] 手写：batch IoU + NMS  
- [ ] 源码：metrics 路径 SOURCE_MAP  
- [ ] Kaggle：小样本上跑通同一协议  
- [ ] 消融：IoU 阈值或 conf 对 AP 的影响  
- [ ] 解释：分数阈值如何同时动 precision/recall  

## 门禁失败 → 停在 P0

说不清「提高 conf 通常怎样影响 AP」则不得进入 P1。

## 执行状态

- ✅ 已完成（见 `PROGRESS.md` + `results/`）
