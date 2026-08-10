# 00-map · 术语与尺子

## 必会术语

- **bbox**：`xyxy` / `xywh` / `cxcywh`；像素 vs 归一化
- **IoU**：交并比；GIoU / DIoU / CIoU 用于损失与匹配
- **NMS**：按分数抑制重叠框；Soft-NMS / 无 NMS（DETR）
- **mAP**：COCO 默认 AP@0.5:0.95；另报 AP50、AP_s/m/l
- **assign**：GT 与预测/锚的正负样本匹配

## P0 验收

- [ ] 手写 IoU 与 NMS，单测通过
- [ ] 理解 AP 曲线：precision-recall 如何汇总成 AP
- [ ] 同一预测用固定协议算出 mAP（后续实验不换尺子）

## 下一步

→ `01-foundations/` 框编解码与锚框
