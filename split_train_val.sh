#!/bin/bash

# Split train/val datasets based on val.txt
# This moves images and labels from images/train and labels/train 
# to images/val and labels/val respectively.

python3 split_train_val.py \
  --val-txt /home/dev/ducanhng/Datasets/val.txt \
  --data-dir /home/dev/ducanhng/Datasets/20260213/YOLO_segmentation
