import os
import hashlib
import argparse
from pathlib import Path

def get_label_hash(filepath):
    """
    Compute a hash of the YOLO label file content.
    Normalizes by stripping whitespace and sorting lines to ignore order.
    """
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Normalize: strip, ignore empty lines, and sort
        normalized = sorted([line.strip() for line in lines if line.strip()])
        content = "\n".join(normalized)
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

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

    for key in common:
        old_lbl = old_data[key]['lbl']
        new_lbl = new_data[key]['lbl']

        if old_lbl and new_lbl:
            if get_label_hash(old_lbl) == get_label_hash(new_lbl):
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
        f"LABEL COMPARISON (for existing images):",
        f"  - Identical labels:   {len(same_labels)}",
        f"  - Different labels:   {len(diff_labels)}",
        f"  - Missing label file: {len(missing_lbl_in_one)}",
        "="*50,
    ]

    # File-only detail (Contains ALL items)
    file_detail = []
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
