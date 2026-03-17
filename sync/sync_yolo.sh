#!/bin/bash

# Cấu hình mặc định
OLD_PATH="/home/a4000/ducanh/Dataset"
NEW_PATH="/home/a4000/ducanh/Dataset-new/VF5"

# Cách dùng:
# 1. Chạy thử: ./sync_yolo.sh
# 2. Chạy thật: ./sync_yolo.sh --yes
# 3. Chạy thật với folder khác: ./sync_yolo.sh --yes /đường/dẫn/mới

YES_FLAG=""
if [[ "$1" == "--yes" ]]; then
    YES_FLAG="--yes"
    shift
fi

if [ ! -z "$1" ]; then
    NEW_PATH=$1
fi

echo "[*] Syncing:"
echo "    Target: $OLD_PATH"
echo "    Source: $NEW_PATH"
echo "    Mode:   ${YES_FLAG:-DRY-RUN}"
echo "--------------------------------------"

python3 /home/a4000/ducanh/normalize-carpart-annotations/sync/sync_yolo.py --old "$OLD_PATH" --new "$NEW_PATH" $YES_FLAG
