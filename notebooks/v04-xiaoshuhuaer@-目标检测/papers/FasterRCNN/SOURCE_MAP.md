# SOURCE_MAP · Faster R-CNN concepts → real code

| Concept (paper) | TwoStageLite (this repo) | Detectron2 | torchvision |
|-----------------|--------------------------|------------|-------------|
| Shared backbone | `TinyBackbone` | `build_resnet_fpn_backbone` | `fasterrcnn_resnet50_fpn` body |
| RPN objectness | `self.rpn` 1-ch map | `proposal_generator/rpn.py` | `RPNHead` |
| RoI feature | *simplified: same grid cell* | `roi_heads` + RoIAlign | `MultiScaleRoIAlign` |
| Box regressor | `head` last 4 ch | `box_head` / `box_predictor` | `FastRCNNPredictor` |
| Assign pos/neg | center cell (lite) / `assignment.assign_iou` | `Matcher` + `subsample_labels` | `box_ops` / anchors |
| NMS | `metrics.nms_xyxy` | `batched_nms` | `nms` |

## Read order
1. Faster R-CNN paper §§3.1–3.2 (RPN + loss)
2. Detectron2 `RPN` forward + losses
3. This `TwoStageLite` training loop

## Open questions
- Full anchor-based RPN with multi-scale anchors not trained here (CPU budget).
- RoIAlign omitted; pedagogical center assign only.
