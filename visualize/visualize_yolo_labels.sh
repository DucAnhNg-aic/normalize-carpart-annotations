#!/bin/bash

# Visualize YOLO labels with Vietnamese class names
# python /home/ubuntu/ducanh/normalize-carpart-annotations/visualize/visualize_yolo_labels.py \
#   --image-path "/home/ubuntu/ducanh/New-Data/export_2026-03-26T07_05_46.361Z/images/train/riLW9Raw4YANDZDxx7OME.jpg" \
#   --labels-dir "/home/ubuntu/ducanh/New-Data/export_2026-03-26T07_05_46.361Z/labels/train" \
#   --data-yaml "/home/ubuntu/ducanh/New-Data/export_2026-03-26T07_05_46.361Z/data.yaml" \
#   --output-dir "/home/ubuntu/ducanh/normalize-carpart-annotations/visualizations/new" \
#   --limit 20

python /home/ubuntu/ducanh/normalize-carpart-annotations/visualize/visualize_yolo_labels.py \
  --image-path "/home/ubuntu/ducanh/New-Data/accent-2024/images/train/" \
  --labels-dir "/home/ubuntu/ducanh/New-Data/accent-2024/labels/train" \
  --data-yaml "/home/ubuntu/ducanh/New-Data/accent-2024/data.yaml" \
  --output-dir "/home/ubuntu/ducanh/normalize-carpart-annotations/visualizations/" \
  --limit 15