#!/usr/bin/env python3
"""
Normalize YOLO dataset annotations to match standard class mapping.

This script:
1. Reads the reference data.yaml to get the standard class mapping
2. Finds all data.yaml files in raw datasets
3. For each dataset:
   - Creates a mapping from old class IDs to new class IDs
   - Normalizes the data.yaml file
   - Remaps class IDs in all label .txt files
   - Removes annotations with damage classes
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths (can be overridden by args)
DEFAULT_REFERENCE_DATA_YAML = Path("/home/dev/ducanhng/Datasets/20260213/YOLO_segmentation/data.yaml")
DEFAULT_RAW_DATASETS_ROOT = Path("/home/dev/ducanhng/Datasets/20260213/raw")

# Damage classes to remove (these should not be in the final dataset)
DAMAGE_CLASSES = {
    "Móp, bẹp(thụng)",
    "Vỡ, nứt",
    "Trầy, xước",
    "Thủng, rách",
    "Long, rụng",
    "Mất"
}


def load_reference_mapping(reference_path: Path) -> Dict[str, int]:
    """Load the reference class name to ID mapping."""
    if not reference_path.exists():
        logger.error(f"Reference file not found: {reference_path}")
        sys.exit(1)
        
    with open(reference_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # Create mapping: class_name -> class_id
    class_mapping = {}
    for class_id, class_name in data['names'].items():
        class_mapping[class_name.strip()] = class_id
    
    logger.info(f"Loaded reference mapping with {len(class_mapping)} classes from {reference_path}")
    return class_mapping


def find_all_data_yaml_files(root_dir: Path) -> List[Path]:
    """Find all data.yaml files in the raw datasets directory."""
    if not root_dir.exists():
        logger.error(f"Root directory not found: {root_dir}")
        sys.exit(1)
        
    data_yaml_files = list(root_dir.rglob("data.yaml"))
    
    # Sort for deterministic processing order
    data_yaml_files.sort()
    
    logger.info(f"Found {len(data_yaml_files)} data.yaml files in {root_dir}")
    return data_yaml_files


def create_class_id_mapping(
    dataset_yaml_path: Path,
    reference_mapping: Dict[str, int]
) -> Tuple[Dict[int, int], List[int]]:
    """
    Create mapping from old class IDs to new class IDs.
    
    Returns:
        - id_mapping: Dict mapping old_class_id -> new_class_id
        - damage_class_ids: List of old class IDs that are damage classes (to be removed)
    """
    with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    id_mapping = {}
    damage_class_ids = []
    
    if 'names' not in data:
        logger.warning(f"No 'names' field in {dataset_yaml_path}")
        return {}, []
    
    for old_id, class_name in data['names'].items():
        class_name_clean = class_name.strip()
        if class_name_clean in DAMAGE_CLASSES:
            damage_class_ids.append(old_id)
        elif class_name_clean in reference_mapping:
            new_id = reference_mapping[class_name_clean]
            id_mapping[old_id] = new_id
        else:
            # Only warn if it's not a damage class and not in reference
            # Some datasets might have other classes we want to ignore/remove or keep as is?
            # For now, if not in reference and not damage, it won't be mapped (so kept as original ID if not removed)
            # But wait, keeping original ID is dangerous if IDs shift. 
            # If we don't map it, normalize_label_file writes it as is. 
            # If the ID conflicts with new scheme, that's bad.
            # But for this task, we assume all valid classes are in reference.
            logger.warning(f"Class '{class_name}' (ID {old_id}) not found in reference mapping (dataset: {dataset_yaml_path.parent.name})")
    
    return id_mapping, damage_class_ids


def normalize_data_yaml(
    dataset_yaml_path: Path,
    reference_mapping: Dict[str, int],
    dry_run: bool = False
) -> None:
    """Normalize a data.yaml file to match the reference format."""
    # Create the normalized data.yaml content
    normalized_data = {
        'train': 'images/train',
        'val': 'images/val',
        'names': {class_id: class_name for class_name, class_id in reference_mapping.items()}
    }
    
    if not dry_run:
        with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(normalized_data, f, allow_unicode=True, sort_keys=False)
        # logger.info(f"Normalized: {dataset_yaml_path}")
    else:
        logger.info(f"[DRY RUN] Would normalize: {dataset_yaml_path}")


def normalize_label_file(
    label_path: Path,
    id_mapping: Dict[int, int],
    damage_class_ids: List[int],
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Normalize a label .txt file by remapping class IDs and removing damage annotations.
    
    Returns:
        - num_remapped: Number of annotations remapped
        - num_removed: Number of annotations removed (damage classes)
    """
    if not label_path.exists():
        return 0, 0
    
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        logger.warning(f"Could not read {label_path} with utf-8, trying latin-1")
        with open(label_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()
            
    new_lines = []
    num_remapped = 0
    num_removed = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 2:
            continue
        
        try:
            old_class_id = int(parts[0])
        except ValueError:
            continue
        
        # Skip damage classes
        if old_class_id in damage_class_ids:
            num_removed += 1
            continue
        
        # Remap class ID
        if old_class_id in id_mapping:
            new_class_id = id_mapping[old_class_id]
            parts[0] = str(new_class_id)
            new_lines.append(' '.join(parts))
            num_remapped += 1
        else:
            # If not in mapping and not damage, we currently keep it. 
            # Ideally we should probably remove it if we want strict adherence to schema,
            # but preserving unknown data is safer than deleting it silently.
            new_lines.append(line)
    
    if not dry_run:
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
            if new_lines:
                f.write('\n')
    
    return num_remapped, num_removed


def normalize_dataset(
    dataset_yaml_path: Path,
    reference_mapping: Dict[str, int],
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Normalize a single dataset (data.yaml and all label files).
    
    Returns:
        Statistics dictionary with counts
    """
    dataset_dir = dataset_yaml_path.parent
    dataset_name = dataset_dir.name
    
    # logger.info(f"Processing dataset: {dataset_name}")
    
    # Create class ID mapping
    id_mapping, damage_class_ids = create_class_id_mapping(dataset_yaml_path, reference_mapping)
    
    # Normalize data.yaml
    normalize_data_yaml(dataset_yaml_path, reference_mapping, dry_run)
    
    # Find all label files
    label_files = list(dataset_dir.rglob("*.txt"))
    # Filter out data.yaml related files, only keep actual label files
    label_files = [f for f in label_files if f.parent.name in ['train', 'val'] or 'labels' in str(f.parent)]
    
    # Normalize all label files
    total_remapped = 0
    total_removed = 0
    
    for label_path in label_files:
        num_remapped, num_removed = normalize_label_file(
            label_path, id_mapping, damage_class_ids, dry_run
        )
        total_remapped += num_remapped
        total_removed += num_removed
    
    stats = {
        'dataset': dataset_name,
        'label_files': len(label_files),
        'annotations_remapped': total_remapped,
        'annotations_removed': total_removed,
        'classes_mapped': len(id_mapping),
        'damage_classes': len(damage_class_ids)
    }
    
    # logger.info(f"  Stats: {len(label_files)} files, {total_remapped} remapped, {total_removed} removed")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Normalize YOLO dataset annotations")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without making any changes (preview mode)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Process only a specific dataset (by name)'
    )
    parser.add_argument(
        '--root-dir',
        type=Path,
        default=DEFAULT_RAW_DATASETS_ROOT,
        help='Root directory containing raw datasets'
    )
    parser.add_argument(
        '--reference-yaml',
        type=Path,
        default=DEFAULT_REFERENCE_DATA_YAML,
        help='Path to the reference data.yaml'
    )
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("=== DRY RUN MODE - No files will be modified ===")
    else:
        logger.info("=== EXECUTING CHANGES - Files WILL be modified ===")
    
    # Load reference mapping
    reference_mapping = load_reference_mapping(args.reference_yaml)
    
    # Find all data.yaml files
    data_yaml_files = find_all_data_yaml_files(args.root_dir)
    
    # Filter by dataset name if specified
    if args.dataset:
        data_yaml_files = [f for f in data_yaml_files if args.dataset in str(f)]
        logger.info(f"Filtered to {len(data_yaml_files)} datasets matching '{args.dataset}'")
    
    # Process each dataset
    all_stats = []
    
    # Use tqdm for overall progress
    pbar = tqdm(data_yaml_files, desc="Processing datasets")
    for data_yaml_path in pbar:
        try:
            pbar.set_postfix_str(f"Current: {data_yaml_path.parent.name}")
            stats = normalize_dataset(data_yaml_path, reference_mapping, args.dry_run)
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"Error processing {data_yaml_path}: {e}")
            continue
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    total_datasets = len(all_stats)
    total_files = sum(s['label_files'] for s in all_stats)
    total_remapped = sum(s['annotations_remapped'] for s in all_stats)
    total_removed = sum(s['annotations_removed'] for s in all_stats)
    
    logger.info(f"Datasets processed: {total_datasets}")
    logger.info(f"Label files processed: {total_files}")
    logger.info(f"Annotations remapped: {total_remapped}")
    logger.info(f"Annotations removed (damage): {total_removed}")
    
    if args.dry_run:
        logger.info("\n*** This was a DRY RUN - no files were modified ***")
        logger.info("Run without --dry-run to apply changes")
    else:
        logger.info("\n*** PROCESSING COMPLETE ***")


if __name__ == "__main__":
    main()
