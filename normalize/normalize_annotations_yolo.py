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
DEFAULT_REFERENCE_DATA_YAML = Path("/home/ubuntu/ducanh/TRAIN_DATA/CarPartSegTrainYOLO/data.yaml")
DEFAULT_RAW_DATASETS_ROOT = Path("/home/ubuntu/ducanh/New-Data")

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


def load_dataset_class_names(dataset_yaml_path: Path) -> Dict[int, str]:
    """Load a dataset's class IDs and names without changing its file."""
    with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    names_raw = data.get('names', {})
    if isinstance(names_raw, list):
        return {i: str(name).strip() for i, name in enumerate(names_raw)}
    if isinstance(names_raw, dict):
        return {int(class_id): str(name).strip() for class_id, name in names_raw.items()}
    return {}


def find_label_files(dataset_dir: Path) -> List[Path]:
    """Return annotation files only, excluding unrelated text files."""
    return [
        path for path in dataset_dir.rglob("*.txt")
        if path.parent.name in ['train', 'val'] or 'labels' in str(path.parent)
    ]


def get_classes_used_in_labels(dataset_yaml_path: Path) -> Tuple[set, set, int]:
    """Resolve class names actually used by annotation files, not all YAML names."""
    class_names = load_dataset_class_names(dataset_yaml_path)
    used_ids = set()
    for label_path in find_label_files(dataset_yaml_path.parent):
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                try:
                    used_ids.add(int(parts[0]))
                except ValueError:
                    logger.warning(f"Invalid class ID in {label_path}: {line.strip()}")

    unknown_ids = used_ids - set(class_names)
    return {class_names[class_id] for class_id in used_ids if class_id in class_names}, unknown_ids, len(used_ids)


def report_class_differences(
    data_yaml_files: List[Path], reference_yaml_path: Path, reference_mapping: Dict[str, int]
) -> bool:
    """Compare classes used in NEW/OLD labels; return True for unsafe NEW classes."""
    old_used_names, old_unknown_ids, old_used_id_count = get_classes_used_in_labels(reference_yaml_path)
    if old_unknown_ids:
        logger.warning(f"[CLASS CHECK] Unknown IDs in OLD labels: {sorted(old_unknown_ids)}")
    logger.info(
        f"[CLASS CHECK] OLD labels use {len(old_used_names)} class names "
        f"({old_used_id_count} class IDs)"
    )

    has_unmapped_new_classes = False
    for dataset_yaml_path in data_yaml_files:
        new_used_names, new_unknown_ids, new_used_id_count = get_classes_used_in_labels(dataset_yaml_path)
        new_only = sorted(new_used_names - old_used_names)
        old_only = sorted(old_used_names - new_used_names)
        # Damage classes are dropped intentionally, so they are not unsafe.
        unmapped_new = sorted(new_used_names - set(reference_mapping) - DAMAGE_CLASSES)
        has_unmapped_new_classes |= bool(unmapped_new or new_unknown_ids)

        logger.info(
            f"[CLASS CHECK] {dataset_yaml_path}: NEW labels use {len(new_used_names)} "
            f"class names ({new_used_id_count} class IDs)"
        )
        logger.warning(f"  Used only in NEW labels ({len(new_only)}): {new_only or 'none'}")
        logger.warning(f"  Used only in OLD labels ({len(old_only)}): {old_only or 'none'}")
        if new_unknown_ids:
            logger.error(f"  Class IDs used in NEW labels but missing from its data.yaml: {sorted(new_unknown_ids)}")
        if unmapped_new:
            logger.error(f"  Classes used in NEW labels but missing from OLD schema: {unmapped_new}")

    return has_unmapped_new_classes


def create_class_id_mapping(
    dataset_yaml_path: Path,
    reference_mapping: Dict[str, int]
) -> Tuple[Dict[int, int], List[int], Dict[int, str]]:
    """
    Create mapping from old class IDs to new class IDs.
    
    Returns:
        - id_mapping: Dict mapping old_class_id -> new_class_id
        - damage_class_ids: List of old class IDs that are damage classes (to be removed)
        - old_names_dict: Dict mapping old_class_id -> class_name
    """
    with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    id_mapping = {}
    damage_class_ids = []
    
    names_raw = data.get('names', {})
    if isinstance(names_raw, list):
        old_names_dict = {i: name for i, name in enumerate(names_raw)}
    elif isinstance(names_raw, dict):
        old_names_dict = {int(k): str(v) for k, v in names_raw.items()}
    else:
        old_names_dict = {}
    
    if not old_names_dict:
        logger.warning(f"No valid 'names' field in {dataset_yaml_path}")
        return {}, [], {}
    
    for old_id, class_name in old_names_dict.items():
        class_name_clean = class_name.strip()
        if class_name_clean in DAMAGE_CLASSES:
            damage_class_ids.append(old_id)
        elif class_name_clean in reference_mapping:
            new_id = reference_mapping[class_name_clean]
            id_mapping[old_id] = new_id
        else:
            logger.error(f"Class '{class_name}' (ID {old_id}) not found in reference mapping (dataset: {dataset_yaml_path.parent.name})")
    
    return id_mapping, damage_class_ids, old_names_dict


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
    unmapped_ids_found: set,
    old_names_dict: Dict[int, str] = None,
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
    
    if old_names_dict is None:
        old_names_dict = {}
        
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
        line_str = line.strip()
        if not line_str:
            continue
        
        parts = line_str.split()
        if len(parts) < 2:
            continue
        
        try:
            old_class_id = int(parts[0])
        except ValueError:
            continue
        
        # Skip damage classes
        if old_class_id in damage_class_ids:
            num_removed += 1
            class_name = old_names_dict.get(old_class_id, "Unknown")
            logger.warning(
                f"[REMOVED ANNOTATION] File: {label_path} | Class ID: {old_class_id} ('{class_name}') | Reason: Damage class | Line: '{line_str}'"
            )
            continue
        
        # Remap class ID
        if old_class_id in id_mapping:
            new_class_id = id_mapping[old_class_id]
            parts[0] = str(new_class_id)
            new_lines.append(' '.join(parts))
            num_remapped += 1
        else:
            # If not in mapping and not damage, we remove it to avoid keeping
            # class IDs that don't conform to the reference schema.
            num_removed += 1
            unmapped_ids_found.add(old_class_id)
            class_name = old_names_dict.get(old_class_id, "Unknown")
            logger.warning(
                f"[REMOVED ANNOTATION] File: {label_path} | Class ID: {old_class_id} ('{class_name}') | Reason: Unmapped class (not in reference schema) | Line: '{line_str}'"
            )
    
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
) -> Tuple[Dict[str, int], set]:
    """
    Normalize a single dataset (data.yaml and all label files).
    
    Returns:
        Statistics dictionary with counts
    """
    dataset_dir = dataset_yaml_path.parent
    dataset_name = dataset_dir.name
    
    # logger.info(f"Processing dataset: {dataset_name}")
    
    # Create class ID mapping
    id_mapping, damage_class_ids, old_names_dict = create_class_id_mapping(dataset_yaml_path, reference_mapping)
    
    # Normalize data.yaml
    normalize_data_yaml(dataset_yaml_path, reference_mapping, dry_run)
    
    # Find all label files
    label_files = find_label_files(dataset_dir)
    
    # Normalize all label files
    total_remapped = 0
    total_removed = 0
    unmapped_ids_found = set()
    
    for label_path in label_files:
        num_remapped, num_removed = normalize_label_file(
            label_path, id_mapping, damage_class_ids, unmapped_ids_found, old_names_dict, dry_run
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
    
    return stats, unmapped_ids_found


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
    parser.add_argument(
        '--acknowledge-class-diff',
        action='store_true',
        help='Allow execution when NEW and reference classes differ'
    )
    args = parser.parse_args()

    # Load reference mapping
    reference_mapping = load_reference_mapping(args.reference_yaml)
    
    # Find all data.yaml files
    data_yaml_files = find_all_data_yaml_files(args.root_dir)
    
    # Filter by dataset name if specified
    if args.dataset:
        data_yaml_files = [f for f in data_yaml_files if args.dataset in str(f)]
        logger.info(f"Filtered to {len(data_yaml_files)} datasets matching '{args.dataset}'")

    has_unmapped_new_classes = report_class_differences(
        data_yaml_files, args.reference_yaml, reference_mapping
    )
    if has_unmapped_new_classes and not args.dry_run and not args.acknowledge_class_diff:
        logger.error(
            "Unsafe NEW label classes found. No files were modified. Review the class check "
            "above, then rerun with --acknowledge-class-diff to continue."
        )
        sys.exit(2)

    if args.dry_run:
        logger.info("=== DRY RUN MODE - No files will be modified ===")
    else:
        logger.info("=== EXECUTING CHANGES - Files WILL be modified ===")
    
    # Process each dataset
    all_stats = []
    global_unmapped_ids = set()
    
    # Use tqdm for overall progress
    pbar = tqdm(data_yaml_files, desc="Processing datasets")
    for data_yaml_path in pbar:
        try:
            pbar.set_postfix_str(f"Current: {data_yaml_path.parent.name}")
            stats, unmapped_ids = normalize_dataset(data_yaml_path, reference_mapping, args.dry_run)
            all_stats.append(stats)
            global_unmapped_ids.update(unmapped_ids)
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
    logger.info(f"Annotations removed: {total_removed}")
    
    if global_unmapped_ids:
        logger.error(f"UNMAPPED CLASS IDs FOUND AND REMOVED: {sorted(list(global_unmapped_ids))}")
        logger.error("These IDs were in the label files but not mapped to the reference schema.")
    
    if args.dry_run:
        logger.info("\n*** This was a DRY RUN - no files were modified ***")
        logger.info("Run without --dry-run to apply changes")
    else:
        logger.info("\n*** PROCESSING COMPLETE ***")


if __name__ == "__main__":
    main()
