#!/bin/bash

# Split train/val datasets based on val.txt
# This moves images and labels from images/train and labels/train 
# to images/val and labels/val respectively.

python3 /home/ubuntu/ducanh/normalize-carpart-annotations/split/split_train_val.py \
  --val-txt /home/ubuntu/ducanh/CarPartSegmentationTrainingDataYOLO/val.txt \
  --data-dir /home/ubuntu/ducanh/CarPartSegmentationTrainingDataYOLO
