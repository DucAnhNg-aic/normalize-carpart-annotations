#!/bin/bash

# Visualize YOLO labels with Vietnamese class names
python /home/a4000/ducanh/normalize-carpart-annotations/visualize/visualize_yolo_labels.py \
  --image-path "/home/a4000/ducanh/Dataset-new/images/train" \
  --labels-dir "/home/a4000/ducanh/Dataset-new/labels/train" \
  --data-yaml "/home/a4000/ducanh/Dataset/data.yaml" \
  --output-dir "/home/a4000/ducanh/normalize-carpart-annotations/visualizations" \
  --limit 20
