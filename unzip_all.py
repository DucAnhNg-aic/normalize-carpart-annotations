#!/usr/bin/env python3
"""
Script để giải nén tất cả file zip trong thư mục CarPartSegmentation_20260121
Cấu trúc sau khi giải nén: Folder1/tên-gốc-folder-cần-giải-nén/các-file-bên-trong
"""

import os
import zipfile
from pathlib import Path
from typing import List
import logging

# Đường dẫn thư mục chứa các file zip
ROOT_DATASET_DIR = "/home/dev/ducanhng/Datasets/20260213/raw"

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_all_zip_files(root_dir: str) -> List[Path]:
    """
    Tìm tất cả file zip trong thư mục và các thư mục con
    
    Args:
        root_dir: Thư mục gốc để tìm kiếm
        
    Returns:
        Danh sách các đường dẫn đến file zip
    """
    root_path = Path(root_dir)
    zip_files = list(root_path.rglob("*.zip"))
    logger.info(f"Tìm thấy {len(zip_files)} file zip")
    return zip_files


def extract_zip_file(zip_path: Path, root_dir: Path) -> bool:
    """
    Giải nén một file zip với cấu trúc: Folder1/tên-zip-file/nội-dung
    
    Args:
        zip_path: Đường dẫn đến file zip
        root_dir: Thư mục gốc (để tính toán đường dẫn tương đối)
        
    Returns:
        True nếu giải nén thành công, False nếu có lỗi
    """
    try:
        # Lấy thư mục cha của file zip (ví dụ: DataCollection, DataCollection1, ...)
        parent_folder = zip_path.parent.name
        
        # Lấy tên file zip không có extension (ví dụ: export_2026-01-21T10_47_46.743Z)
        zip_name = zip_path.stem
        
        # Tạo đường dẫn đích: parent_folder/zip_name/
        extract_dir = zip_path.parent / zip_name
        
        # Kiểm tra xem đã giải nén chưa
        if extract_dir.exists():
            logger.warning(f"Thư mục {extract_dir} đã tồn tại, bỏ qua file {zip_path.name}")
            return True
        
        # Tạo thư mục đích
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        # Giải nén file
        logger.info(f"Đang giải nén {zip_path.relative_to(root_dir)} -> {extract_dir.relative_to(root_dir)}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        logger.info(f"✓ Giải nén thành công: {zip_path.name}")
        return True
        
    except zipfile.BadZipFile:
        logger.error(f"✗ File zip bị lỗi: {zip_path}")
        return False
    except Exception as e:
        logger.error(f"✗ Lỗi khi giải nén {zip_path}: {str(e)}")
        return False


def main():
    """Hàm chính để thực thi script"""
    # Kiểm tra thư mục có tồn tại không
    if not os.path.exists(ROOT_DATASET_DIR):
        logger.error(f"Thư mục {ROOT_DATASET_DIR} không tồn tại!")
        return
    
    root_path = Path(ROOT_DATASET_DIR)
    
    # Tìm tất cả file zip
    logger.info(f"Đang tìm kiếm file zip trong {ROOT_DATASET_DIR}...")
    zip_files = find_all_zip_files(ROOT_DATASET_DIR)
    
    if not zip_files:
        logger.warning("Không tìm thấy file zip nào!")
        return
    
    # Giải nén từng file
    logger.info(f"Bắt đầu giải nén {len(zip_files)} file...")
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, zip_file in enumerate(zip_files, 1):
        logger.info(f"\n[{i}/{len(zip_files)}] Xử lý: {zip_file.relative_to(root_path)}")
        
        extract_dir = zip_file.parent / zip_file.stem
        if extract_dir.exists():
            skipped_count += 1
            continue
            
        if extract_zip_file(zip_file, root_path):
            success_count += 1
        else:
            failed_count += 1
    
    # Tổng kết
    logger.info("\n" + "="*60)
    logger.info("TỔNG KẾT:")
    logger.info(f"  Tổng số file zip: {len(zip_files)}")
    logger.info(f"  ✓ Giải nén thành công: {success_count}")
    logger.info(f"  ⊘ Đã tồn tại (bỏ qua): {skipped_count}")
    logger.info(f"  ✗ Thất bại: {failed_count}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
