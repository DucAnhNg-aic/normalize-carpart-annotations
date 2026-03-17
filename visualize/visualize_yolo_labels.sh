#!/bin/bash

# Visualize YOLO labels with Vietnamese class names
python /home/a4000/ducanh/normalize-carpart-annotations/visualize/visualize_yolo_labels.py \
  --images-dir "/home/a4000/Data/ducanhng/CV/Dataset/20260213/raw/Data Collection 6/OPES Truck 10:2023 Part 1/images/train" \
  --labels-dir "/home/a4000/Data/ducanhng/CV/Dataset/20260213/raw/Data Collection 6/OPES Truck 10:2023 Part 1/labels/train" \
  --data-yaml "/home/a4000/Data/ducanhng/CV/Dataset/20260213/YOLO_segmentation/data.yaml" \
  --output-dir "/home/a4000/Data/ducanhng/CV/normalize-carpart-annotations/visualizations" \
  --limit 1
