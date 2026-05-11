#!/bin/bash

# Visualize YOLO labels with Vietnamese class names
# Supports both single dataset and nested groups mode

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISUALIZE_PY="$SCRIPT_DIR/visualize_yolo_labels.py"
DATA_YAML="/home/ubuntu/ducanh/CarPartSegmentatonTrainingData/data.yaml"

# ── Groups mode: visualize duplicates_conflict/groups/<hash>/images/train/ ──
# python "$VISUALIZE_PY" \
#   --groups-dir "/home/ubuntu/ducanh/duplicates_conflict/groups" \
#   --data-yaml  "$DATA_YAML" \
#   --output-dir "/home/ubuntu/ducanh/normalize-carpart-annotations/visualizations/groups"

# ── Single dataset mode (comment out groups-dir block above to use this) ──
python "$VISUALIZE_PY" \
  --image-path "/home/ubuntu/ducanh/CarPartSegmentatonTrainingData/images/train/image1663565826244884_2022100401098_THVO_SAU.jpg" \
  --labels-dir "/home/ubuntu/ducanh/CarPartSegmentatonTrainingData/labels/train" \
  --data-yaml  "$DATA_YAML" \
  --output-dir "/home/ubuntu/ducanh/normalize-carpart-annotations/visualizations/"