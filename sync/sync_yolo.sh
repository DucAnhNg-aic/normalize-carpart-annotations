#!/bin/bash

# Cấu hình mặc định sẽ được ghi đè nếu có tham số truyền vào
OLD_DEFAULT="/home/ubuntu/ducanh/TRAIN_DATA/CarPartSegTrainYOLO"
NEW_DEFAULT="/home/ubuntu/ducanh/New-Data"

print_options() {
    cat <<'EOF'
Cách dùng: bash sync/sync_yolo.sh [--yes] [--no-rename] [OLD_PATH] [NEW_PATH]
  (không cờ)    Chạy thử (dry-run), không copy gì
  --yes         Copy thật. File trùng tên nhưng khác nội dung sẽ được đổi tên
                <timestamp>_<tên> và ghi vào sync_manifest.json (giữ bản cũ)
  --no-rename   Ghi đè thẳng lên file cũ trùng tên (MẤT bản cũ), dùng kèm --yes
  -h, --help    Chỉ in hướng dẫn này
EOF
}

YES_FLAG=""
NO_RENAME_FLAG=""

# Parse flags
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --yes)
            YES_FLAG="--yes"
            shift
            ;;
        --no-rename)
            NO_RENAME_FLAG="--no-rename"
            shift
            ;;
        --rename-diff)
            # Rename is now the default; flag kept so old commands still work
            shift
            ;;
        -h|--help)
            print_options
            exit 0
            ;;
        -*)
            echo "[!] Unknown option: $1"
            print_options
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

OLD_PATH="${1:-$OLD_DEFAULT}"
NEW_PATH="${2:-$NEW_DEFAULT}"

print_options
echo "--------------------------------------"
echo "[*] Syncing:"
echo "    Target: $OLD_PATH"
echo "    Source: $NEW_PATH"
echo "    Mode:   ${YES_FLAG:-DRY-RUN}"
if [[ -n "$NO_RENAME_FLAG" ]]; then
    echo "    Option: --no-rename (OVERWRITE conflicting files)"
else
    echo "    Option: rename conflicting files + update manifest (default)"
fi
echo "--------------------------------------"

python3 /home/ubuntu/ducanh/normalize-carpart-annotations/sync/sync_yolo.py --old "$OLD_PATH" --new "$NEW_PATH" $YES_FLAG $NO_RENAME_FLAG
