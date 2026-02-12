#!/usr/bin/env python3
"""
Batch download images from all datasets and copy corresponding labels.

This script:
1. Finds all images.json files in the raw datasets directory
2. Downloads all images to a single output directory
3. Copies corresponding labels from labels/train to output labels directory
4. Supports parallel downloading and progress tracking
"""

import json
import os
import shutil
import urllib.request
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import sys
import time
import argparse

# Thread-safe counter for progress
class ProgressCounter:
    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.downloaded = 0
        self.skipped = 0
        self.errors = 0
        self.error_details = []
        self.lock = threading.Lock()
    
    def increment(self, status, filename=None, error=None):
        with self.lock:
            self.completed += 1
            if status == 'success':
                self.downloaded += 1
            elif status == 'skipped':
                self.skipped += 1
            elif status == 'error':
                self.errors += 1
                if filename and error:
                    self.error_details.append((filename, error))
            return self.completed

def update_progress_bar(counter, dataset_name=""):
    """Update progress bar in terminal"""
    percentage = int((counter.completed / counter.total) * 100) if counter.total > 0 else 0
    bar_length = 40
    filled_length = int((counter.completed / counter.total) * bar_length) if counter.total > 0 else 0
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    # ANSI color codes
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    BLUE = '\033[36m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    progress_text = f"{BOLD}{BLUE}Progress:{RESET} [{bar}] {percentage}%"
    stats = f"{GREEN}✓ {counter.downloaded}{RESET} | {YELLOW}⊘ {counter.skipped}{RESET} | {RED}✗ {counter.errors}{RESET}"
    count = f"{counter.completed}/{counter.total}"
    
    # Clear line and print progress
    dataset_info = f" | {dataset_name[:30]}" if dataset_name else ""
    sys.stdout.write(f'\r\033[K{progress_text} {stats} ({count}){dataset_info}')
    sys.stdout.flush()

def get_extension_from_url(url):
    """Extract file extension from URL"""
    path = urlparse(url).path
    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg']:
        if ext in path.lower():
            return ext
    return '.jpg'

def download_single_image(item, output_dir, counter, dataset_name=""):
    """Download a single image with retry mechanism"""
    name = item['name']
    url = item['url']
    
    # Detect file extension from URL
    ext = get_extension_from_url(url)
    
    # Add extension if not present
    if not any(name.lower().endswith(e) for e in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg']):
        filename = f"{name}{ext}"
    else:
        filename = name
    
    filepath = os.path.join(output_dir, filename)
    
    # Check for name conflict and add prefix if needed
    if os.path.exists(filepath):
        # Clean dataset name for use as prefix
        clean_prefix = "".join([c if c.isalnum() or c in "._-" else "_" for c in dataset_name])
        filename = f"{clean_prefix}_{filename}"
        filepath = os.path.join(output_dir, filename)
        
        # If even the prefixed name exists, we skip it (assuming it's already downloaded)
        if os.path.exists(filepath):
            counter.increment('skipped')
            update_progress_bar(counter, dataset_name)
            return filename
    
    # Retry mechanism with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(request, timeout=30) as response:
                with open(filepath, 'wb') as f:
                    chunk_size = 8192
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
            
            counter.increment('success')
            update_progress_bar(counter, dataset_name)
            return filename
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                if os.path.exists(filepath):
                    os.remove(filepath)
                counter.increment('error', filename, str(e))
                update_progress_bar(counter, dataset_name)
                return None

def copy_label_file(image_filename, original_item_name, dataset_dir, output_labels_dir):
    """Copy corresponding label file for an image, handling potential renaming"""
    # Use the original name from images.json to find the source label
    original_label_name = Path(original_item_name).stem + '.txt'
    
    # The destination should match the potentially prefixed image name
    dest_label_name = Path(image_filename).stem + '.txt'
    
    # Look for label in labels/train directory
    label_source = dataset_dir / 'labels' / 'train' / original_label_name
    
    if label_source.exists():
        label_dest = output_labels_dir / dest_label_name
        try:
            shutil.copy2(label_source, label_dest)
            return True
        except Exception as e:
            print(f"\nWarning: Failed to copy label {dest_label_name}: {e}")
            return False
    return False

def find_all_images_json(root_dir):
    """Find all images.json files in the raw datasets directory"""
    root_path = Path(root_dir)
    images_json_files = list(root_path.rglob("images.json"))
    images_json_files.sort()
    return images_json_files

def process_dataset(images_json_path, output_images_dir, output_labels_dir, max_workers=10):
    """Process a single dataset: download images and copy labels"""
    dataset_dir = images_json_path.parent
    dataset_name = dataset_dir.name
    
    # Read images.json
    with open(images_json_path, 'r', encoding='utf-8') as f:
        images = json.load(f)
    
    total = len(images)
    counter = ProgressCounter(total)
    
    print(f"\n📦 Processing: {dataset_name} ({total} images)")
    update_progress_bar(counter, dataset_name)
    
    downloaded_images = []
    
    # Download images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks, storing the original item name for label mapping
        futures = {executor.submit(download_single_image, item, output_images_dir, counter, dataset_name): item['name'] 
                   for item in images}
        
        for future in as_completed(futures):
            original_name = futures[future]
            try:
                actual_filename = future.result()
                if actual_filename:
                    downloaded_images.append((actual_filename, original_name))
            except Exception as e:
                pass
    
    # Copy labels for downloaded images
    labels_copied = 0
    for actual_filename, original_name in downloaded_images:
        if copy_label_file(actual_filename, original_name, dataset_dir, output_labels_dir):
            labels_copied += 1
    
    print(f"\n   Labels copied: {labels_copied}/{len(downloaded_images)}")
    
    return {
        'dataset': dataset_name,
        'total': total,
        'downloaded': counter.downloaded,
        'skipped': counter.skipped,
        'errors': counter.errors,
        'labels_copied': labels_copied
    }

def main():
    parser = argparse.ArgumentParser(description="Batch download images and copy labels from all datasets")
    parser.add_argument(
        '--raw-dir',
        type=Path,
        default=Path("/home/dev/ducanhng/Datasets/20260213/raw"),
        help='Root directory containing raw datasets'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("/home/dev/ducanhng/Datasets/20260213/merged_output"),
        help='Output directory for images and labels'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=50,
        help='Number of parallel download workers (default: 50)'
    )
    parser.add_argument(
        '--dataset-filter',
        type=str,
        help='Filter datasets by name (substring match)'
    )
    args = parser.parse_args()
    
    # Create output directories
    output_images_dir = args.output_dir / 'images' / 'train'
    output_labels_dir = args.output_dir / 'labels' / 'train'
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images.json files
    images_json_files = find_all_images_json(args.raw_dir)
    
    # Filter by dataset name if specified
    if args.dataset_filter:
        images_json_files = [f for f in images_json_files if args.dataset_filter in str(f)]
    
    # Print header
    BLUE = '\033[36m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    print(f"{BOLD}{BLUE}{'━' * 60}{RESET}")
    print(f"{BOLD}📥 Batch Image Download & Label Copy Script{RESET}")
    print(f"{BOLD}{BLUE}{'━' * 60}{RESET}")
    print(f"Raw datasets directory: {BOLD}{args.raw_dir}{RESET}")
    print(f"Output directory: {BOLD}{args.output_dir}{RESET}")
    print(f"Total datasets found: {BOLD}{len(images_json_files)}{RESET}")
    print(f"Parallel workers: {BOLD}{args.max_workers}{RESET}")
    if args.dataset_filter:
        print(f"Filter: {BOLD}{args.dataset_filter}{RESET}")
    print(f"{BOLD}{BLUE}{'━' * 60}{RESET}")
    
    start_time = time.time()
    all_stats = []
    
    # Process each dataset
    for i, images_json_path in enumerate(images_json_files, 1):
        print(f"\n[{i}/{len(images_json_files)}]", end=" ")
        stats = process_dataset(images_json_path, output_images_dir, output_labels_dir, args.max_workers)
        all_stats.append(stats)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    print(f"\n\n{BOLD}{BLUE}{'━' * 60}{RESET}")
    print(f"{BOLD}✅ Batch Processing Complete!{RESET}")
    print(f"{BOLD}{BLUE}{'━' * 60}{RESET}")
    print(f"📊 {BOLD}Overall Statistics:{RESET}")
    
    total_images = sum(s['total'] for s in all_stats)
    total_downloaded = sum(s['downloaded'] for s in all_stats)
    total_skipped = sum(s['skipped'] for s in all_stats)
    total_errors = sum(s['errors'] for s in all_stats)
    total_labels = sum(s['labels_copied'] for s in all_stats)
    
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    
    print(f"   Datasets processed: {len(all_stats)}")
    print(f"   Total images:       {total_images}")
    print(f"   {GREEN}✓ Downloaded:       {total_downloaded}{RESET}")
    print(f"   {YELLOW}⊘ Skipped:          {total_skipped}{RESET}")
    print(f"   {RED}✗ Errors:           {total_errors}{RESET}")
    print(f"   📄 Labels copied:    {total_labels}")
    print(f"\n⏱️  {BOLD}Performance:{RESET}")
    print(f"   Duration:   {duration:.2f}s")
    avg_speed = total_downloaded / duration if duration > 0 else 0
    print(f"   Avg Speed:  {avg_speed:.2f} images/s")
    print(f"\n📁 {BOLD}Output:{RESET}")
    print(f"   Images: {output_images_dir}")
    print(f"   Labels: {output_labels_dir}")
    print(f"{BOLD}{BLUE}{'━' * 60}{RESET}\n")

if __name__ == '__main__':
    main()
