#!/bin/bash

# Configuration: Use arguments if provided, else defaults
DATA_PATH="${1:-/home/ubuntu/ducanh/CarPartSegmentationTrainingDataYOLO}"
OUTPUT_PATH="${2:-/home/ubuntu/ducanh/Data_Null}"

echo "Starting cleaning process..."
echo "Source dataset: $DATA_PATH"
echo "Output path:    $OUTPUT_PATH"
echo "-----------------------------------"

python3 /home/ubuntu/ducanh/normalize-carpart-annotations/clean_data/clean_null_data.py \
    --data-path "$DATA_PATH" \
    --output-path "$OUTPUT_PATH"

echo "-----------------------------------"
echo "Done."
