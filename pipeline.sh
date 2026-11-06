# Pipeline wrapper
# Default paths (can be overridden via positional arguments: bash pipeline.sh [OLD_PATH] [NEW_PATH])
OLD_PATH="${1:-/home/ubuntu/ducanh/TRAIN_DATA/CarPartSegTrainYOLO}"
NEW_PATH="${2:-/home/ubuntu/ducanh/New-Data}"

echo "PIPELINE: Using paths"
echo "  OLD_PATH = $OLD_PATH"
echo "  NEW_PATH = $NEW_PATH"
echo "----------------------"

# Download images (operates on NEW_PATH)
echo "PIPELINE: Downloading images..."
(cd "$NEW_PATH" && python3 download_images.py)

# Normalize annotations (operates on NEW_PATH)
python3 normalize/normalize_annotations_yolo.py --root-dir "$NEW_PATH" --reference-yaml "$OLD_PATH/data.yaml"


# Compare
bash compare/compare_yolo.sh "$OLD_PATH" "$NEW_PATH"

# Sync
# bash sync/sync_yolo.sh "$OLD_PATH" "$NEW_PATH"

# Remove duplicates
# bash delete/move_duplicates.sh "$OLD_PATH" "$NEW_PATH"