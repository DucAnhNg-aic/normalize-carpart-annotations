#!/bin/bash

# Cấu hình mặc định
OLD_PATH="/home/a4000/ducanh/Dataset"
NEW_PATH="/home/a4000/ducanh/Dataset-new/VF5"

# Nếu bạn truyền tham số, nó sẽ coi đó là NEW_PATH mới
if [ ! -z "$1" ]; then
    NEW_PATH=$1
fi

echo "[*] Comparing:"
echo "    Old: $OLD_PATH"
echo "    New: $NEW_PATH"
echo "--------------------------------------"

python3 /home/a4000/ducanh/normalize-carpart-annotations/compare/compare_yolo.py "$OLD_PATH" "$NEW_PATH"
