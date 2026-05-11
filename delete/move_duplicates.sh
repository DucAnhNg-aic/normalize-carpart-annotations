#!/bin/bash

# Configuration: Use arguments if provided, else defaults
OLD_PATH="${1:-/home/ubuntu/ducanh/CarPartSegmentatonTrainingData}"
NEW_PATH="${2:-/home/ubuntu/ducanh/New-Data}"
MOVE_TO="${3:-/home/ubuntu/ducanh/Data_Backup_Duplicates}"

# Create a timestamped backup directory (optional but safer)
# MOVE_TO="/home/ubuntu/ducanh/Data_Backup_Duplicates/$(date +%Y%m%d_%H%M%S)"

echo "Moving duplicates from: $OLD_PATH"
echo "If they exist in:      $NEW_PATH"
echo "Moving to:             $MOVE_TO"
echo "-----------------------------------"

python3 /home/ubuntu/ducanh/normalize-carpart-annotations/delete/move_duplicates.py \
    --old-path "$OLD_PATH" \
    --new-path "$NEW_PATH" \
    --move-to "$MOVE_TO"

echo "Done."
