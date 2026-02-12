#!/usr/bin/env python3
"""
Script to normalize car part annotations across all annotation files.
Reads all annotations.json files, extracts categories, and normalizes them
to match the standard categories.json format.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set
from tqdm import tqdm


def load_standard_categories(categories_file: str) -> Dict[str, Dict]:
    """Load the standard categories mapping."""
    with open(categories_file, 'r', encoding='utf-8') as f:
        categories = json.load(f)
    
    # Create mapping from category name to category info
    name_to_category = {}
    for cat in categories:
        # Normalize the name by stripping whitespace
        normalized_name = cat['name'].strip()
        name_to_category[normalized_name] = cat
    
    return name_to_category


def find_all_annotation_files(base_dir: str) -> List[Path]:
    """Find all annotations.json files in the dataset."""
    base_path = Path(base_dir)
    annotation_files = list(base_path.rglob('annotations.json'))
    return annotation_files


def normalize_annotation_file(
    annotation_file: Path,
    standard_categories: Dict[str, Dict],
    dry_run: bool = False
) -> Dict:
    """
    Normalize a single annotation file.
    
    Args:
        annotation_file: Path to the annotation file
        standard_categories: Standard category mapping
        dry_run: If True, don't write changes
        
    Returns:
        Dictionary with statistics about the normalization
    """
    stats = {
        'file': str(annotation_file),
        'categories_found': 0,
        'categories_normalized': 0,
        'annotations_updated': 0,
        'unmapped_categories': []
    }
    
    # Load the annotation file
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'categories' not in data:
        return stats
    
    stats['categories_found'] = len(data['categories'])
    
    # Create mapping from old category ID to new category ID
    old_to_new_id = {}
    
    # Build mapping for existing categories
    for old_cat in data['categories']:
        old_id = old_cat['id']
        old_name = old_cat['name'].strip()
        
        # Try to find matching standard category
        if old_name in standard_categories:
            std_cat = standard_categories[old_name]
            new_id = std_cat['id']
            old_to_new_id[old_id] = new_id
            stats['categories_normalized'] += 1
        else:
            # Category not found in standard mapping
            stats['unmapped_categories'].append({
                'id': old_id,
                'name': old_name
            })
            # Keep the old ID mapping
            old_to_new_id[old_id] = old_id
    
    # ALWAYS use the complete set of standard categories, sorted by ID
    new_categories = sorted(
        standard_categories.values(),
        key=lambda x: x['id']
    )
    
    # Update annotations with new category IDs
    if 'annotations' in data:
        for ann in data['annotations']:
            if 'category_id' in ann:
                old_cat_id = ann['category_id']
                if old_cat_id in old_to_new_id:
                    ann['category_id'] = old_to_new_id[old_cat_id]
                    stats['annotations_updated'] += 1
    
    # Replace categories with normalized ones
    data['categories'] = new_categories
    
    # Write back if not dry run
    if not dry_run:
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Normalize car part annotations to standard categories'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='/home/dev/ducanhng/Datasets/CarPartSegmentation_20260121',
        help='Path to the dataset directory'
    )
    parser.add_argument(
        '--categories-file',
        type=str,
        default='/home/dev/ducanhng/normalize-carpart-annotations/categories.json',
        help='Path to the standard categories.json file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without making changes (preview mode)'
    )
    
    args = parser.parse_args()
    
    print("Loading standard categories...")
    standard_categories = load_standard_categories(args.categories_file)
    print(f"Loaded {len(standard_categories)} standard categories")
    
    print("\nFinding annotation files...")
    annotation_files = find_all_annotation_files(args.dataset_dir)
    print(f"Found {len(annotation_files)} annotation files")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No changes will be made\n")
    
    # Process all files
    all_stats = []
    unmapped_categories_set = set()
    
    print("\nProcessing annotation files...")
    for ann_file in tqdm(annotation_files, desc="Normalizing"):
        stats = normalize_annotation_file(
            ann_file,
            standard_categories,
            dry_run=args.dry_run
        )
        all_stats.append(stats)
        
        # Collect unmapped categories
        for unmapped in stats['unmapped_categories']:
            unmapped_categories_set.add(unmapped['name'])
    
    # Print summary
    print("\n" + "="*60)
    print("NORMALIZATION SUMMARY")
    print("="*60)
    
    total_categories = sum(s['categories_found'] for s in all_stats)
    total_normalized = sum(s['categories_normalized'] for s in all_stats)
    total_annotations = sum(s['annotations_updated'] for s in all_stats)
    
    print(f"Files processed: {len(all_stats)}")
    print(f"Total categories found: {total_categories}")
    print(f"Categories normalized: {total_normalized}")
    print(f"Annotations updated: {total_annotations}")
    
    if unmapped_categories_set:
        print(f"\n⚠️  Warning: {len(unmapped_categories_set)} unmapped categories found:")
        for cat_name in sorted(unmapped_categories_set):
            print(f"  - {cat_name}")
    else:
        print("\n✓ All categories successfully mapped!")
    
    if args.dry_run:
        print("\n⚠️  This was a DRY RUN. Run without --dry-run to apply changes.")
    else:
        print("\n✓ Normalization complete!")


if __name__ == '__main__':
    main()
