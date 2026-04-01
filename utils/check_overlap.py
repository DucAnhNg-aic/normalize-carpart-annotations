import os
import hashlib
from pathlib import Path

def get_image_info(directory):
    """Returns a dict mapping filename to its MD5 hash for all images in the directory."""
    path = Path(directory)
    image_extensions = ('.jpg', '.png', '.jpeg', '.bmp')
    info = {}
    
    # Pre-count for progress
    all_files = []
    for ext in image_extensions:
        all_files.extend(list(path.rglob(f"*{ext}")))
    
    total = len(all_files)
    print(f"  - Found {total} files. Starting hash calculation...")
    
    for i, img_path in enumerate(all_files):
        try:
            with open(img_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            info[img_path.name] = {
                'hash': file_hash,
                'full_path': img_path
            }
            if (i + 1) % 500 == 0:
                print(f"    Processed {i + 1}/{total} ({(i + 1)/total*100:.1f}%)")
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
                
    return info

import shutil

def move_duplicates(overlap_hashes, train_info, val_info, backup_root):
    print(f"\n--- MOVING DUPLICATES TO {backup_root} ---")
    backup_path = Path(backup_root)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    img_backup = backup_path / "images" / "train"
    lbl_backup = backup_path / "labels" / "train"
    img_backup.mkdir(parents=True, exist_ok=True)
    lbl_backup.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for h in overlap_hashes:
        train_img_path = train_info[h]['full_path']
        train_name = train_img_path.name
        
        # 1. Move Image
        dst_img = img_backup / train_name
        shutil.move(str(train_img_path), str(dst_img))
        
        # 2. Try move Label
        # Determine source label path
        # images/train -> labels/train
        src_lbl = Path(str(train_img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        if src_lbl.exists():
            dst_lbl = lbl_backup / src_lbl.name
            shutil.move(str(src_lbl), str(dst_lbl))
            print(f"  - Moved: {train_name} (+label)")
        else:
            print(f"  - Moved: {train_name} (no label found)")
        
        count += 1
    
    print(f"\nSuccessfully moved {count} duplicate image-label pairs from TRAIN to backup.")

def check_overlap(train_dir, val_dir, move_out=False, backup_dir=None):
    print(f"Scanning Train: {train_dir}")
    train_info_by_name = get_image_info(train_dir)
    print(f"Scanning Val:   {val_dir}")
    val_info_by_name = get_image_info(val_dir)
    
    # Restructure into hash-indexed dicts for easy lookup
    train_info_by_hash = {data['hash']: {'full_path': data['full_path'], 'name': name} 
                          for name, data in train_info_by_name.items()}
    val_info_by_hash = {data['hash']: {'full_path': data['full_path'], 'name': name} 
                        for name, data in val_info_by_name.items()}
    
    train_filenames = set(train_info_by_name.keys())
    val_filenames = set(val_info_by_name.keys())
    
    # 1. Overlap by filename
    name_overlap = train_filenames.intersection(val_filenames)
    
    # 2. Overlap by content (hash)
    hash_overlap_values = set(train_info_by_hash.keys()).intersection(set(val_info_by_hash.keys()))
    
    print("\n" + "="*50)
    print("OVERLAP REPORT")
    print("="*50)
    print(f"Train images: {len(train_info_by_name)}")
    print(f"Val images:   {len(val_info_by_name)}")
    print("-" * 50)
    
    if name_overlap:
        print(f"[!] Found {len(name_overlap)} images with SAME FILENAME in both train and val:")
        for name in sorted(list(name_overlap))[:10]:
            print(f"  - {name}")
    else:
        print("[v] No duplicate filenames found.")
        
    print("-" * 50)
    
    if hash_overlap_values:
        print(f"[!] Found {len(hash_overlap_values)} images with SAME CONTENT (Hash) in both train and val:")
        for h in hash_overlap_values:
            train_name = train_info_by_hash[h]['name']
            val_name = val_info_by_hash[h]['name']
            print(f"  - Hash {h[:8]}...: '{train_name}' (train) == '{val_name}' (val)")
            
        if move_out and backup_dir:
            move_duplicates(hash_overlap_values, train_info_by_hash, val_info_by_hash, backup_dir)
    else:
        print("[v] No duplicate content (hashes) found.")
    print("="*50)

if __name__ == "__main__":
    train_images = "/home/ubuntu/ducanh/Data/images/train"
    val_images = "/home/ubuntu/ducanh/Data/images/val"
    backup_folder = "/home/ubuntu/ducanh/Data_Backup_Val_Overlap"
    
    # Run with move_out=True to actually perform the move
    check_overlap(train_images, val_images, move_out=True, backup_dir=backup_folder)
