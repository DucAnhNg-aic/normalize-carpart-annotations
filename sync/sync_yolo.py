import os
import shutil
import hashlib
import argparse
from pathlib import Path

TOLERANCE = 1e-2

def get_file_hash(filepath):
    """Compute MD5 hash of a file's raw content."""
    if not os.path.exists(filepath):
        return None
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None

def labels_are_equal(path1, path2):
    """
    Compare two YOLO label .txt files numerically.
    Sorts boxes by (class_id, first_coord) then compares each float with TOLERANCE.
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

def is_identical(src_img, tgt_img, src_lbl, tgt_lbl):
    """Check if both image and label are already identical at target."""
    if not tgt_img.exists():
        return False
    
    # Check Image
    if get_file_hash(src_img) != get_file_hash(tgt_img):
        return False
    
    # Check Label
    if src_lbl and src_lbl.exists():
        if not tgt_lbl.exists():
            return False
        if not labels_are_equal(str(src_lbl), str(tgt_lbl)):
            return False
    elif tgt_lbl.exists():
        return False
        
    return True

def sync_dataset(old_path, new_path, dry_run=True):
    old_root = Path(old_path)
    new_root = Path(new_path)
    
    if not old_root.exists():
        print(f"[!] Error: Thư mục OLD không tồn tại: {old_path}")
        return
    if not new_root.exists():
        print(f"[!] Error: Thư mục NEW không tồn tại: {new_path}")
        return

    img_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.heic', '.HEIC', '.webp'}
    
    print(f"[*] {'[DRY RUN] ' if dry_run else ''}Syncing from {new_path} to {old_path}...")
    
    count_new = 0
    count_update = 0
    count_skip = 0
    
    new_img_dir = new_root / "images"
    new_lbl_dir = new_root / "labels"
    
    if new_img_dir.exists() and new_lbl_dir.exists():
        for img_path in new_img_dir.rglob('*'):
            if img_path.suffix in img_extensions:
                rel_path = img_path.relative_to(new_img_dir)
                lbl_rel_path = rel_path.with_suffix('.txt')
                
                target_img = old_root / "images" / rel_path
                target_lbl = old_root / "labels" / lbl_rel_path
                src_lbl = new_lbl_dir / lbl_rel_path
                
                if not target_img.exists():
                    status = "[NEW]"
                    count_new += 1
                elif is_identical(img_path, target_img, src_lbl, target_lbl):
                    status = "[SKIP]"
                    count_skip += 1
                else:
                    status = "[UPDATE]"
                    count_update += 1
                
                if status != "[SKIP]" and not dry_run:
                    target_img.parent.mkdir(parents=True, exist_ok=True)
                    target_lbl.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, target_img)
                    if src_lbl.exists():
                        shutil.copy2(src_lbl, target_lbl)
                
                # Logging logic
                if dry_run or status != "[SKIP]":
                    if count_new + count_update + count_skip <= 15:
                        print(f"  {status} {rel_path}")
                    elif not dry_run and (count_new + count_update) % 50 == 0:
                        print(f"  Processed {count_new + count_update} changes...")

    else:
        print("[!] Dataset structure not standard (images/labels not found). Skipping complex sync.")

    print("\n" + "="*40)
    print(f"SUMMARY ({'DRY RUN' if dry_run else 'EXECUTION'}):")
    print(f"  - Files to Add:    {count_new}")
    print(f"  - Files to Update: {count_update}")
    print(f"  - Files Skipped:  {count_skip}")
    print(f"  - Total processed: {count_new + count_update + count_skip}")
    print("="*40)
    
    if dry_run:
        print("\n[TIP] Đây là lệnh chạy thử. Để thực hiện thay đổi thật, hãy dùng thêm tham số --yes")
    else:
        print("\n[v] Đồng bộ hoàn tất!")
    
    if dry_run:
        print("\n[TIP] Đây là lệnh chạy thử. Để thực hiện thay đổi thật, hãy dùng thêm tham số --yes")
    else:
        print("\n[v] Đồng bộ hoàn tất!")

def main():
    # --- HARDCODE PATHS HERE ---
    OLD_PATH = "/home/a4000/ducanh/Dataset"
    NEW_PATH = "/home/a4000/ducanh/Dataset-new/VF3"
    # ---------------------------

    parser = argparse.ArgumentParser(description="Sync NEW YOLO subset results into OLD dataset")
    parser.add_argument("--old", default=OLD_PATH, help="Path to OLD dataset")
    parser.add_argument("--new", default=NEW_PATH, help="Path to NEW dataset")
    parser.add_argument("--yes", action="store_true", help="Perform actual copy (otherwise dry-run)")
    
    args = parser.parse_args()

    sync_dataset(args.old, args.new, dry_run=not args.yes)

if __name__ == "__main__":
    main()
