# Claim reviews (P7)

## Self-study claims tested

See `results/p7_hypothesis/results.json` for H1–H3 accept/reject.

## External paper review template application

| Paper | Claim (example) | Supported? | Notes |
|-------|-----------------|------------|-------|
| DETR | set prediction removes NMS | Method yes; speed claim needs hardware table | Matching is essential |
| YOLO family | real-time SOTA | Depends on hardware + TRT; compare AP & FPS | Don't cite FPS without device |
| RT-DETR | DETRs beat YOLOs real-time | Check COCO AP + T4 FPS from paper table | Reproduce protocol |

## Practice

For each new arXiv detection paper this week:
1. Fill papers/TEMPLATE.md
2. Mark unsupported claims
3. Design one mini hypothesis like H1–H3
