#!/bin/bash

# Cấu hình mặc định
OLD_PATH="/home/ubuntu/ducanh/Data"
NEW_PATH="/home/ubuntu/ducanh/New-Data/accent-2024"

# Nếu bạn truyền tham số, nó sẽ coi đó là NEW_PATH mới
if [ ! -z "$1" ]; then
    NEW_PATH=$1
fi

echo "[*] Comparing:"
echo "    Old: $OLD_PATH"
echo "    New: $NEW_PATH"
echo "--------------------------------------"

python "$(dirname "$0")/compare_yolo.py" "$OLD_PATH" "$NEW_PATH"
