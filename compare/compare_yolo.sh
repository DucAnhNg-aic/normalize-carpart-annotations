#!/bin/bash

# Cấu hình: Ưu tiên tham số truyền vào, nếu không có thì dùng mặc định
OLD_PATH="${1:-/home/ubuntu/ducanh/CarPartSegmentatonTrainingData}"
NEW_PATH="${2:-/home/ubuntu/ducanh/New-Data/ALL}"

echo "[*] Comparing:"
echo "    Old: $OLD_PATH"
echo "    New: $NEW_PATH"
echo "--------------------------------------"

python "$(dirname "$0")/compare_yolo.py" "$OLD_PATH" "$NEW_PATH"
