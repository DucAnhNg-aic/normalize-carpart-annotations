#!/usr/bin/env python3
"""
Split train/val datasets based on a text file containing validation image names.

This script:
1. Reads a val.txt file containing image filenames
2. Moves those images from images/train to images/val
3. Moves corresponding labels from labels/train to labels/val
"""

import argparse
from pathlib import Path
import shutil

def split_train_val(val_txt_path, data_dir, dry_run=False):
    """
    Split train/val datasets based on val.txt
    
    Args:
        val_txt_path: Path to val.txt containing validation image names
        data_dir: Root directory containing images/ and labels/ folders
        dry_run: If True, only print what would be done without moving files
    """
    val_txt_path = Path(val_txt_path)
    data_dir = Path(data_dir)
    
    # Define directories
    train_images_dir = data_dir / 'images' / 'train'
    val_images_dir = data_dir / 'images' / 'val'
    train_labels_dir = data_dir / 'labels' / 'train'
    val_labels_dir = data_dir / 'labels' / 'val'
    
    # Create val directories if they don't exist
    if not dry_run:
        val_images_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Read validation image names
    with open(val_txt_path, 'r', encoding='utf-8') as f:
        val_image_names = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(val_image_names)} validation images in {val_txt_path}")
    print(f"Data directory: {data_dir}")
    
    if dry_run:
        print("\n=== DRY RUN MODE - No files will be moved ===\n")
    
    # Statistics
    images_moved = 0
    images_not_found = 0
    labels_moved = 0
    labels_not_found = 0
    
    # Process each validation image
    for image_name in val_image_names:
        # Source and destination for image
        image_src = train_images_dir / image_name
        image_dst = val_images_dir / image_name
        
        # Get corresponding label filename (replace extension with .txt)
        label_name = Path(image_name).stem + '.txt'
        label_src = train_labels_dir / label_name
        label_dst = val_labels_dir / label_name
        
        # Move image
        if image_src.exists():
            if not dry_run:
                shutil.move(str(image_src), str(image_dst))
            images_moved += 1
            if dry_run:
                print(f"Would move image: {image_name}")
        else:
            images_not_found += 1
            print(f"Warning: Image not found: {image_name}")
        
        # Move label
        if label_src.exists():
            if not dry_run:
                shutil.move(str(label_src), str(label_dst))
            labels_moved += 1
            if dry_run and image_src.exists():
                print(f"Would move label: {label_name}")
        else:
            labels_not_found += 1
            if image_src.exists():  # Only warn if image exists
                print(f"Warning: Label not found: {label_name}")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Images moved:      {images_moved}")
    print(f"Images not found:  {images_not_found}")
    print(f"Labels moved:      {labels_moved}")
    print(f"Labels not found:  {labels_not_found}")
    
    if dry_run:
        print("\n*** This was a DRY RUN - no files were moved ***")
        print("Run without --dry-run to apply changes")
    else:
        print(f"\n✅ Successfully split dataset!")
        print(f"Train images: {train_images_dir}")
        print(f"Val images:   {val_images_dir}")
        print(f"Train labels: {train_labels_dir}")
        print(f"Val labels:   {val_labels_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Split train/val datasets based on val.txt file"
    )
    parser.add_argument(
        '--val-txt',
        type=Path,
        required=True,
        help='Path to val.txt file containing validation image names'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Root directory containing images/ and labels/ folders'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without moving files'
    )
    args = parser.parse_args()
    
    split_train_val(args.val_txt, args.data_dir, args.dry_run)

if __name__ == '__main__':
    main()
