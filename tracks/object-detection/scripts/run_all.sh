#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
$PY tests/test_metrics.py
$PY tests/test_boxes.py
$PY tests/test_assignment.py
$PY scripts/run_p0_protocol.py
$PY scripts/run_p1_boxes.py
$PY scripts/run_p2_two_stage.py
$PY scripts/run_p3_yolo_lite.py
$PY scripts/run_p4_detr.py
$PY scripts/run_p5_recipe_ablation.py
$PY scripts/run_p6_domain.py
$PY scripts/run_p7_hypothesis.py
echo "ALL P0-P7 OK"
