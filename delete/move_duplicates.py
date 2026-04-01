import os
import shutil
import glob
from pathlib import Path
import argparse

def get_file_list(root_dir):
    """Scan all image files in all subdirectories of root_dir."""
    root = Path(root_dir)
    image_extensions = ('.jpg', '.png', '.jpeg', '.bmp')
    file_names = set()
    
    # Only collect the stem (filename without extension) to match images and labels
    for ext in image_extensions:
        for file in root.rglob(f"*{ext}"):
            file_names.add(file.stem)
            
    return file_names

def move_duplicates(old_path, new_path, move_to):
    """Find images in old_path that exist in new_path and move them out."""
    print(f"Scanning NEW_PATH: {new_path}")
    new_stems = get_file_list(new_path)
    print(f"Found {len(new_stems)} images in NEW_PATH.")

    if not os.path.exists(move_to):
        os.makedirs(move_to)
        os.makedirs(os.path.join(move_to, "images"))
        os.makedirs(os.path.join(move_to, "labels"))
        print(f"Created backup directory: {move_to}")

    old_root = Path(old_path)
    count = 0
    
    print(f"Checking OLD_PATH: {old_path}...")
    
    # Iterate through images first to decide what to move
    image_extensions = ('.jpg', '.png', '.jpeg', '.bmp')
    for ext in image_extensions:
        for image_file in old_root.rglob(f"*{ext}"):
            stem = image_file.stem
            
            if stem in new_stems:
                # This file exists in NEW_PATH, so move it from OLD_PATH
                count += 1
                
                # Determine subdirectory relative to OLD_PATH (e.g. images/train)
                rel_path = image_file.parent.relative_to(old_root)
                target_img_dir = Path(move_to) / rel_path
                target_img_dir.mkdir(parents=True, exist_ok=True)
                
                # Move Image
                target_img_file = target_img_dir / image_file.name
                shutil.move(str(image_file), str(target_img_file))
                
                # Try to move corresponding label (.txt)
                # Labels usually follow the same subfolder structure but in 'labels' dir
                # E.g. OLD_PATH/images/train/1.jpg -> OLD_PATH/labels/train/1.txt
                
                # Strategy: If it is under an 'images' subfolder, replace with 'labels'
                if "images" in rel_path.parts:
                    new_rel_parts = list(rel_path.parts)
                    idx = new_rel_parts.index("images")
                    new_rel_parts[idx] = "labels"
                    label_rel_path = Path(*new_rel_parts)
                    
                    label_file = old_root / label_rel_path / (stem + ".txt")
                    if label_file.exists():
                        target_label_dir = Path(move_to) / label_rel_path
                        target_label_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(label_file), str(target_label_dir / (stem + ".txt")))
                        # print(f"Moved duplicate: {stem} (Image & Label)")
                else:
                    # Generic lookup for .txt if not in standard images/labels structure
                    label_file = image_file.with_suffix(".txt")
                    if label_file.exists():
                        shutil.move(str(label_file), str(target_img_dir / (stem + ".txt")))
                
                if count % 100 == 0:
                    print(f"Moved {count} duplicates...")

    print(f"\nCOMPLETED. Moved a total of {count} duplicates to {move_to}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move duplicate images from OLD_PATH to another location if they exist in NEW_PATH.")
    parser.add_argument("--old-path", default="/home/ubuntu/ducanh/Data", help="Path to the original directory to clean.")
    parser.add_argument("--new-path", default="/home/ubuntu/ducanh/New-Data/ALL/1", help="Path to the newer directory containing duplicates.")
    parser.add_argument("--move-to", default="/home/ubuntu/ducanh/Data_Backup_Duplicates", help="Directory where duplicates will be moved for safety.")
    
    args = parser.parse_args()
    move_duplicates(args.old_path, args.new_path, args.move_to)
