#!/bin/bash

# Split train/val datasets based on val.txt
# This moves images and labels from images/train and labels/train 
# to images/val and labels/val respectively.

python3 /home/a4000/ducanh/normalize-carpart-annotations/split/split_train_val.py \
  --val-txt /home/a4000/Data/ducanhng/CV/Dataset/val.txt \
  --data-dir /home/a4000/Data/ducanhng/CV/Dataset/20260213/YOLO_segmentation
