#!/bin/bash

# Cấu hình mặc định sẽ được ghi đè nếu có tham số truyền vào
OLD_DEFAULT="/home/ubuntu/ducanh/CarPartSegmentationTrainingDataYOLO"
NEW_DEFAULT="/home/ubuntu/ducanh/New-Data"


YES_FLAG=""
if [[ "$1" == "--yes" ]]; then
    YES_FLAG="--yes"
    shift
fi

OLD_PATH="${1:-$OLD_DEFAULT}"
NEW_PATH="${2:-$NEW_DEFAULT}"

echo "[*] Syncing:"
echo "    Target: $OLD_PATH"
echo "    Source: $NEW_PATH"
echo "    Mode:   ${YES_FLAG:-DRY-RUN}"
echo "--------------------------------------"

python3 /home/ubuntu/ducanh/normalize-carpart-annotations/sync/sync_yolo.py --old "$OLD_PATH" --new "$NEW_PATH" $YES_FLAG
