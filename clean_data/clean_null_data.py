import os
import shutil
from pathlib import Path
import argparse

def is_label_empty(label_path):
    """Check if a label file is empty (size 0 or only contains whitespace)."""
    if not label_path.exists():
        return True
    if label_path.stat().st_size == 0:
        return True
    try:
        with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
            return len(content) == 0
    except Exception:
        return False

def find_corresponding_label(image_file, root_path):
    """
    Find the label (.txt) file corresponding to an image file.
    Supports YOLO structures (images/split/name.ext -> labels/split/name.txt)
    and flat structures.
    """
    stem = image_file.stem
    rel_path = image_file.parent.relative_to(root_path)
    
    # 1. Try YOLO-style structure by replacing 'images' directory with 'labels'
    if "images" in rel_path.parts:
        new_parts = list(rel_path.parts)
        idx = new_parts.index("images")
        new_parts[idx] = "labels"
        label_rel_path = Path(*new_parts)
        label_file = root_path / label_rel_path / (stem + ".txt")
        if label_file.exists():
            return label_file
            
    # 2. Try the same folder as the image
    label_file = image_file.with_suffix(".txt")
    if label_file.exists():
        return label_file
        
    # 3. If standard structure but label is in 'labels' parallel folder (without subfolders)
    label_file = root_path / "labels" / (stem + ".txt")
    if label_file.exists():
        return label_file
        
    return None

def find_corresponding_image(label_file, root_path):
    """
    Find the image file corresponding to a label file.
    Checks typical image extensions (.jpg, .jpeg, .png, .bmp, .webp).
    """
    stem = label_file.stem
    rel_path = label_file.parent.relative_to(root_path)
    image_extensions = ('.jpg', '.png', '.jpeg', '.bmp', '.webp', '.JPG', '.PNG', '.JPEG')
    
    # 1. Try YOLO-style structure by replacing 'labels' directory with 'images'
    if "labels" in rel_path.parts:
        new_parts = list(rel_path.parts)
        idx = new_parts.index("labels")
        new_parts[idx] = "images"
        image_rel_path = Path(*new_parts)
        for ext in image_extensions:
            img_file = root_path / image_rel_path / (stem + ext)
            if img_file.exists():
                return img_file
                
    # 2. Try the same folder as the label
    for ext in image_extensions:
        img_file = label_file.with_suffix(ext)
        if img_file.exists():
            return img_file
            
    # 3. Try standard parallel images folder
    for ext in image_extensions:
        img_file = root_path / "images" / (stem + ext)
        if img_file.exists():
            return img_file
            
    return None

def move_file(src_file, root_path, target_sub_root):
    """Move a file to target_sub_root while preserving its relative path structure."""
    rel_path = src_file.parent.relative_to(root_path)
    dest_dir = target_sub_root / rel_path
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / src_file.name
    shutil.move(str(src_file), str(dest_file))
    return dest_file

def clean_dataset(data_path, output_path):
    root_path = Path(data_path).resolve()
    output_root = Path(output_path).resolve()
    
    unpaired_root = output_root / "unpaired"
    empty_root = output_root / "empty"
    
    print(f"Scanning dataset at: {root_path}")
    print(f"Output directory:     {output_root}\n")
    
    # 1. Gather all files
    image_extensions = ('.jpg', '.png', '.jpeg', '.bmp', '.webp', '.JPG', '.PNG', '.JPEG')
    
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(root_path.rglob(f"*{ext}")))
    all_images = sorted(list(set(all_images)))
    
    all_labels_raw = sorted(list(root_path.rglob("*.txt")))
    all_labels = []
    for f in all_labels_raw:
        # Ignore text files directly under root_path (like val.txt, train.txt)
        if f.parent == root_path:
            continue
        all_labels.append(f)
    
    print(f"Found {len(all_images)} total images.")
    print(f"Found {len(all_labels)} total labels (.txt files).\n")
    
    # Track sets for processed/moved files
    moved_images = set()
    moved_labels = set()
    
    unpaired_images_count = 0
    unpaired_labels_count = 0
    empty_labels_count = 0
    
    # Step 1: Process empty labels and their corresponding images
    # We do this first so if an empty label has a valid image, both are moved to 'empty'.
    print("Checking for empty labels...")
    for label_file in all_labels:
        if label_file in moved_labels:
            continue
            
        if is_label_empty(label_file):
            # Find its corresponding image
            image_file = find_corresponding_image(label_file, root_path)
            
            # Move the label to empty_root
            move_file(label_file, root_path, empty_root)
            moved_labels.add(label_file)
            
            if image_file and image_file.exists() and image_file not in moved_images:
                move_file(image_file, root_path, empty_root)
                moved_images.add(image_file)
                print(f"Moved empty label and its image: {label_file.name} & {image_file.name}")
            else:
                print(f"Moved empty label (no matching image found): {label_file.name}")
                
            empty_labels_count += 1

    # Step 2: Find unpaired images (images that don't have matching label files)
    print("\nChecking for unpaired images (images without labels)...")
    for image_file in all_images:
        if image_file in moved_images:
            continue
            
        label_file = find_corresponding_label(image_file, root_path)
        if not label_file or not label_file.exists():
            move_file(image_file, root_path, unpaired_root)
            moved_images.add(image_file)
            print(f"Moved unpaired image (no label): {image_file.name}")
            unpaired_images_count += 1

    # Step 3: Find unpaired labels (labels that don't have matching image files)
    print("\nChecking for unpaired labels (labels without images)...")
    for label_file in all_labels:
        if label_file in moved_labels:
            continue
            
        image_file = find_corresponding_image(label_file, root_path)
        if not image_file or not image_file.exists():
            move_file(label_file, root_path, unpaired_root)
            moved_labels.add(label_file)
            print(f"Moved unpaired label (no image): {label_file.name}")
            unpaired_labels_count += 1

    # Summary
    print("\n============================================================")
    print("CLEANING SUMMARY")
    print("============================================================")
    print(f"Empty labels (and their images) moved: {empty_labels_count}")
    print(f"Unpaired images moved:                 {unpaired_images_count}")
    print(f"Unpaired labels moved:                 {unpaired_labels_count}")
    print(f"Total files moved:                     {len(moved_images) + len(moved_labels)}")
    print(f"Output saved to:                       {output_root}")
    print("============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean dataset by moving unpaired and empty files.")
    parser.add_argument("--data-path", default="/home/ubuntu/ducanh/CarPartSegmentationTrainingDataYOLO", help="Path to YOLO dataset directory.")
    parser.add_argument("--output-path", default="/home/ubuntu/ducanh/Data_Null", help="Path to move clean files.")
    
    args = parser.parse_args()
    clean_dataset(args.data_path, args.output_path)
