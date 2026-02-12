#!/bin/bash

# Visualize YOLO labels with Vietnamese class names
python3 visualize_yolo_labels.py \
  --images-dir /home/dev/ducanhng/Datasets/20260213/YOLO_segmentation/images/train \
  --labels-dir /home/dev/ducanhng/Datasets/20260213/YOLO_segmentation/labels/train \
  --data-yaml /home/dev/ducanhng/Datasets/20260213/YOLO_segmentation/data.yaml \
  --output-dir /home/dev/ducanhng/normalize-carpart-annotations/visualizations \
  --limit 10
