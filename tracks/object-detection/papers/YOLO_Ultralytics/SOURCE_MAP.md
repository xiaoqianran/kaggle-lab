# SOURCE_MAP · YOLO modern / Ultralytics

| Concept | CenterNetLite | Ultralytics (typical paths) |
|---------|---------------|-----------------------------|
| Backbone+Neck+Head | `TinyBackbone`+`head` | `nn/tasks.py` DetectionModel |
| Dense decode | `decode_centernet` | head decode in `ops` / predictor |
| Assign | center cell (lite) | TaskAligned / SimOTA family in loss |
| Loss obj/cls/box | `centernet_loss` | `utils/loss.py` (version-dependent) |
| Metrics mAP | `metrics.coco_style_ap` | `utils/metrics.py` DetMetrics |
| Trainer loop | this script | `engine/trainer.py` |

## Read order
1. RetinaNet Focal Loss (imbalance idea)
2. Ultralytics `utils/metrics.py` AP accumulation
3. This dense train loop

## Note
Full Ultralytics not vendored in CI; SOURCE_MAP is the contract for Kaggle GPU follow-up.
