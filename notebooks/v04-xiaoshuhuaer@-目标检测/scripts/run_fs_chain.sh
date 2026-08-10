#!/bin/sh
# Object Detection From-Scratch chain (FS00–FS15 mapped scripts)
set -eu
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

echo "=== tests ==="
$PY tests/test_metrics.py
$PY tests/test_boxes.py
$PY tests/test_assignment.py

echo "=== FS00 protocol ==="
$PY scripts/run_p0_protocol.py

echo "=== FS01 sliding window ==="
$PY scripts/fs01_sliding_window.py

echo "=== FS02 pyramid ==="
$PY scripts/fs02_image_pyramid.py

echo "=== FS05 R-CNN crops ==="
$PY scripts/fs05_rcnn_crops.py

echo "=== FS07 two-stage ==="
$PY scripts/run_p2_two_stage.py

echo "=== FS08 dense one-stage ==="
$PY scripts/run_p3_yolo_lite.py

echo "=== FS09 focal ==="
$PY scripts/fs09_focal_loss.py

echo "=== FS10 multi-scale ==="
$PY scripts/fs10_fpn_multiscale.py

echo "=== FS11/P1 anchors (contrast) ==="
$PY scripts/run_p1_boxes.py

echo "=== FS12 DETR ==="
$PY scripts/run_p4_detr.py

echo "=== FS13 recipe ==="
$PY scripts/run_p5_recipe_ablation.py

echo "=== FS14 domain ==="
$PY scripts/run_p6_domain.py

echo "=== FS15 hypothesis ==="
$PY scripts/run_p7_hypothesis.py

echo "ALL FS CHAIN OK"
