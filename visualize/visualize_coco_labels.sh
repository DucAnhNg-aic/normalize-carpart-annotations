#!/bin/bash

# Visualize COCO labels with Vietnamese class names

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISUALIZE_PY="$SCRIPT_DIR/visualize_coco_labels.py"

# Define paths
COCO_DIR="/home/ubuntu/ducanh/CarPartSegmentatonTrainingDataCoCo"
IMAGES_DIR="$COCO_DIR/images/val" # Or train
COCO_JSON="$COCO_DIR/annotations/instances_val.json"
OUTPUT_DIR="/home/ubuntu/ducanh/normalize-carpart-annotations/visualize/visualizations_coco"

# Run visualization script
python3 "$VISUALIZE_PY" \
  --images-dir "$IMAGES_DIR" \
  --coco-json "$COCO_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --image-path "/home/ubuntu/ducanh/CarPartSegmentatonTrainingDataCoCo/images/val/image1641349599151850_2022012600220_THVO_TPHAI_canh_cua.jpg"
