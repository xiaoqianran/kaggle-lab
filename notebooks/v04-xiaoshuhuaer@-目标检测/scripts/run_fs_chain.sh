#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
$PY scripts/run_fs_all.py
