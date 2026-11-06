import os
import shutil
import hashlib
import argparse
import json
from pathlib import Path

TOLERANCE = 1e-2
MANIFEST_FILENAME = "sync_manifest.json"

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

def load_manifest(old_root: Path) -> dict:
    """
    Load sync_manifest.json from old_root.
    Returns a dict: { "rel/path/to/src.jpg": "rel/path/to/dest.jpg" }
    Keys & values are relative to the images/ directory.
    """
    manifest_path = old_root / MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Warning: Could not read manifest ({e}). Starting fresh.")
    return {}

def save_manifest(old_root: Path, manifest: dict, dry_run: bool):
    """Persist manifest to disk (skipped in dry-run)."""
    if dry_run:
        return
    manifest_path = old_root / MANIFEST_FILENAME
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

def sync_dataset(old_path, new_path, dry_run=True, rename_diff=True):
    import datetime
    old_root = Path(old_path)
    new_root = Path(new_path)
    
    if not old_root.exists():
        print(f"[!] Error: Thư mục OLD không tồn tại: {old_path}")
        return
    if not new_root.exists():
        print(f"[!] Error: Thư mục NEW không tồn tại: {new_path}")
        return

    img_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.heic', '.HEIC', '.webp'}
    timestamp_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_"
    
    # Load manifest (source_rel -> dest_rel, both relative to images/ dir)
    manifest = load_manifest(old_root)
    manifest_updated = False
    
    print(f"[*] {'[DRY RUN] ' if dry_run else ''}Syncing from {new_path} to {old_path}...")
    if rename_diff:
        print(f"[*] Mode: rename conflicting files + update manifest (prefix: '{timestamp_prefix}')")
    else:
        print("[!] Mode: --no-rename -> conflicting files in OLD will be OVERWRITTEN")
    if manifest:
        print(f"[*] Loaded manifest: {len(manifest)} existing source→dest mappings.")
    
    count_new = 0
    count_update = 0
    count_rename = 0
    count_skip = 0
    
    new_img_dir = new_root / "images"
    new_lbl_dir = new_root / "labels"
    
    if new_img_dir.exists() and new_lbl_dir.exists():
        for img_path in new_img_dir.rglob('*'):
            if img_path.suffix not in img_extensions:
                continue
                
            rel_path = img_path.relative_to(new_img_dir)
            rel_key = str(rel_path)
            lbl_rel_path = rel_path.with_suffix('.txt')
            src_lbl = new_lbl_dir / lbl_rel_path
            
            # Resolve destination: prefer manifest mapping if it exists
            if rel_key in manifest:
                mapped_rel = Path(manifest[rel_key])
                target_img = old_root / "images" / mapped_rel
                target_lbl = old_root / "labels" / mapped_rel.with_suffix('.txt')
            else:
                target_img = old_root / "images" / rel_path
                target_lbl = old_root / "labels" / lbl_rel_path
                
            if not target_img.exists():
                status = "[NEW]"
                count_new += 1
            elif is_identical(img_path, target_img, src_lbl, target_lbl):
                status = "[SKIP]"
                count_skip += 1
            else:
                if rename_diff and rel_key not in manifest:
                    status = "[RENAME]"
                    new_name = f"{timestamp_prefix}{rel_path.name}"
                    new_lbl_name = f"{timestamp_prefix}{lbl_rel_path.name}"
                    target_img = target_img.parent / new_name
                    target_lbl = target_lbl.parent / new_lbl_name
                    mapped_rel = rel_path.parent / new_name
                    manifest[rel_key] = str(mapped_rel)
                    manifest_updated = True
                    count_rename += 1
                else:
                    status = "[UPDATE]"
                    count_update += 1
            
            if status != "[SKIP]" and not dry_run:
                target_img.parent.mkdir(parents=True, exist_ok=True)
                target_lbl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, target_img)
                if src_lbl.exists():
                    shutil.copy2(src_lbl, target_lbl)
            
            if dry_run or status != "[SKIP]":
                total_changes = count_new + count_update + count_rename
                if total_changes + count_skip <= 15:
                    if status == "[RENAME]":
                        print(f"  {status} {rel_path} -> {mapped_rel}")
                    elif status == "[UPDATE]" and rel_key in manifest:
                        print(f"  {status} {rel_path} -> {manifest[rel_key]}")
                    else:
                        print(f"  {status} {rel_path}")
                elif not dry_run and total_changes % 50 == 0:
                    print(f"  Processed {total_changes} changes...")

    else:
        print("[!] Dataset structure not standard (images/labels not found). Skipping complex sync.")

    if manifest_updated:
        save_manifest(old_root, manifest, dry_run)
        if not dry_run:
            print(f"[*] Manifest saved: {old_root / MANIFEST_FILENAME}")

    print("\n" + "="*40)
    print(f"SUMMARY ({'DRY RUN' if dry_run else 'EXECUTION'}):")
    print(f"  - Files to Add:    {count_new}")
    if rename_diff:
        print(f"  - Files to Rename: {count_rename}")
    print(f"  - Files to Update: {count_update}")
    print(f"  - Files Skipped:  {count_skip}")
    print(f"  - Total processed: {count_new + count_update + count_rename + count_skip}")
    print("="*40)
    
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
    parser.add_argument("--no-rename", action="store_true", help="Overwrite conflicting images/labels in OLD instead of renaming them (default: rename with timestamp prefix and track in sync_manifest.json)")
    parser.add_argument("--rename-diff", action="store_true", help=argparse.SUPPRESS)  # kept for backward compatibility; renaming is now the default
    
    args = parser.parse_args()

    sync_dataset(args.old, args.new, dry_run=not args.yes, rename_diff=not args.no_rename)

if __name__ == "__main__":
    main()
