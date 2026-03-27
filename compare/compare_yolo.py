import os
import hashlib
import argparse
from pathlib import Path

TOLERANCE = 1e-2

def get_image_hash(filepath):
    """Compute MD5 hash of image file bytes for pixel-level comparison."""
    try:
        h = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        print(f"Error hashing {filepath}: {e}")
        return None

def labels_are_equal(path1, path2):
    """
    Compare two YOLO label .txt files numerically.
    Sorts boxes by (class_id, first_coord) then compares each float with TOLERANCE.
    Returns True if numerically identical, False otherwise.
    """
    def parse_file(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                rows = []
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    rows.append([int(parts[0])] + [float(x) for x in parts[1:]])
            rows.sort(key=lambda r: (r[0], r[1] if len(r) > 1 else 0))
            return rows
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

    r1, r2 = parse_file(path1), parse_file(path2)
    if r1 is None or r2 is None or len(r1) != len(r2):
        return False
    for b1, b2 in zip(r1, r2):
        if b1[0] != b2[0] or len(b1) != len(b2):
            return False
        if any(abs(c1 - c2) > TOLERANCE for c1, c2 in zip(b1[1:], b2[1:])):
            return False
    return True

def scan_dataset(root_path):
    """
    Scans a directory for image-label pairs.
    Handles standard YOLO structure (images/ and labels/ subdirs) or flat structure.
    """
    root = Path(root_path)
    img_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.heic', '.HEIC', '.webp'}
    results = {} # stem -> {'img': Path, 'lbl': Path}

    # Check for images/ and labels/ subdirectories
    img_dir = root / "images"
    lbl_dir = root / "labels"

    if img_dir.exists() and lbl_dir.exists():
        # Scoped scan (handles train/val subfolders automatically via rglob)
        for img_path in img_dir.rglob('*'):
            if img_path.suffix in img_extensions:
                rel_path = img_path.relative_to(img_dir)
                # Corresponding label path should mirror the image structure
                lbl_path = lbl_dir / rel_path.with_suffix('.txt')
                results[str(rel_path.with_suffix(''))] = {
                    'img': img_path,
                    'lbl': lbl_path if lbl_path.exists() else None
                }
    else:
        # Flat scan in the root directory
        for img_path in root.rglob('*'):
            if img_path.suffix in img_extensions:
                # Avoid scanning inside 'labels' or 'images' if we are doing a flat scan of root
                if "images" in img_path.parts or "labels" in img_path.parts:
                    continue
                lbl_path = img_path.with_suffix('.txt')
                results[img_path.stem] = {
                    'img': img_path,
                    'lbl': lbl_path if lbl_path.exists() else None
                }
    
    return results

def main():
    # --- HARDCODE PATHS HERE FOR CONVENIENCE ---
    OLD_PATH = "/home/a4000/ducanh/Dataset" # Ví dụ: "/home/a4000/ducanh/Dataset/old"
    NEW_PATH = "/home/a4000/ducanh/Dataset-new/VF3/sua-nhan" # Ví dụ: "/home/a4000/ducanh/Dataset/new"
    # -------------------------------------------

    parser = argparse.ArgumentParser(description="Compare two YOLO datasets (Old vs New)")
    parser.add_argument("old", nargs='?', default=OLD_PATH, help="Path to the OLD dataset folder")
    parser.add_argument("new", nargs='?', default=NEW_PATH, help="Path to the NEW dataset folder")
    parser.add_argument("--output", "-o", help="Optional output file for the report")
    args = parser.parse_args()

    # Check if paths are provided either via hardcode or CLI
    if not args.old or not args.new:
        print("[!] Error: Vui lòng điền đường dẫn vào OLD_PATH/NEW_PATH trong code hoặc truyền qua tham số.")
        print("Usage: python compare_yolo.py <old_folder> <new_folder>")
        return

    print(f"[*] Scanning OLD folder: {args.old}")
    old_data = scan_dataset(args.old)
    print(f"[*] Scanning NEW folder: {args.new}")
    new_data = scan_dataset(args.new)

    old_keys = set(old_data.keys())
    new_keys = set(new_data.keys())

    # 1. New images (in new but not in old)
    added = new_keys - old_keys
    
    # 2. Existing images (in both)
    common = old_keys & new_keys
    
    # 3. Removed images (in old but not in new)
    removed = old_keys - new_keys

    same_labels = []
    diff_labels = []
    missing_lbl_in_one = []
    diff_images = []  # Same name, different pixel content

    for key in common:
        old_lbl = old_data[key]['lbl']
        new_lbl = new_data[key]['lbl']

        # Compare image content
        old_img = old_data[key]['img']
        new_img = new_data[key]['img']
        if get_image_hash(str(old_img)) != get_image_hash(str(new_img)):
            diff_images.append(key)

        # Compare labels
        if old_lbl and new_lbl:
            if labels_are_equal(str(old_lbl), str(new_lbl)):
                same_labels.append(key)
            else:
                diff_labels.append(key)
        else:
            missing_lbl_in_one.append(key)

    # Reporting
    report_title = "YOLO DATASET COMPARISON REPORT"
    summary = [
        "="*50,
        report_title,
        "="*50,
        f"Old Folder: {args.old}",
        f"New Folder: {args.new}",
        "-"*50,
        f"SUMMARY:",
        f"  - Images in NEW folder: {len(new_keys)}",
        f"  - New images (to be added): {len(added)}",
        f"  - Existing images (in old): {len(common)}",
        "-"*50,
        f"IMAGE CONTENT COMPARISON (for existing images):",
        f"  - Identical images:   {len(common) - len(diff_images)}",
        f"  - Different images:   {len(diff_images)}  ← same name, different pixels",
        "-"*50,
        f"LABEL COMPARISON (for existing images):",
        f"  - Identical labels:   {len(same_labels)}",
        f"  - Different labels:   {len(diff_labels)}",
        f"  - Missing label file: {len(missing_lbl_in_one)}",
        "="*50,
    ]

    # File-only detail (Contains ALL items)
    file_detail = []
    if diff_images:
        file_detail.append("\n[~] ALL images with DIFFERENT pixel content (same name, changed image):")
        for item in sorted(diff_images):
            file_detail.append(f"  - {item}")

    if diff_labels:
        file_detail.append("\n[!] ALL images with DIFFERENT labels (Updates):")
        for item in sorted(diff_labels):
            file_detail.append(f"  - {item}")
    
    if added:
        file_detail.append("\n[+] ALL NEW images (Additions):")
        for item in sorted(added):
            file_detail.append(f"  - {item}")

    # Terminal-only preview (Contains first 20 items)
    terminal_preview = []
    if diff_images:
        terminal_preview.append("\n[~] Images with DIFFERENT pixel content (First 20):")
        for item in sorted(diff_images)[:20]:
            terminal_preview.append(f"  - {item}")
        if len(diff_images) > 20:
            terminal_preview.append(f"  ... and {len(diff_images) - 20} more")

    if diff_labels:
        terminal_preview.append("\n[!] Images with DIFFERENT labels (First 20):")
        for item in sorted(diff_labels)[:20]:
            terminal_preview.append(f"  - {item}")
        if len(diff_labels) > 20:
            terminal_preview.append(f"  ... and {len(diff_labels) - 20} more")

    if added:
        terminal_preview.append("\n[+] NEW images added (First 20):")
        for item in sorted(added)[:20]:
            terminal_preview.append(f"  - {item}")
        if len(added) > 20:
            terminal_preview.append(f"  ... and {len(added) - 20} more")

    # Print summary and preview to terminal
    print("\n".join(summary))
    print("\n".join(terminal_preview))

    # Save EVERYTHING to file
    output_path = args.output if args.output else "comparison_report.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary) + "\n")
        f.write("\n".join(file_detail))
    
    print(f"\n[v] Báo cáo chi tiết đã được lưu vào: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
