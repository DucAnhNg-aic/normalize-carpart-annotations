import os
import subprocess
import argparse
from pathlib import Path

def run_all_downloads(root_path):
    root = Path(root_path).expanduser().resolve()
    if not root.exists():
        print(f"[!] Error: Thư mục không tồn tại: {root}")
        return

    print(f"[*] Đang tìm kiếm 'download_images.py' trong: {root}")
    
    # Tìm tất cả file download_images.py
    download_scripts = list(root.rglob("download_images.py"))
    
    if not download_scripts:
        print("[!] Không tìm thấy script download nào.")
        return

    print(f"[*] Tìm thấy {len(download_scripts)} script. Bắt đầu thực thi...\n")
    
    for script in download_scripts:
        script_dir = script.parent
        print(f"===> Đang chạy tại: {script_dir}")
        
        try:
            # Chạy script bằng python, quan trọng là set cwd (thư mục làm việc) 
            # để script tìm thấy images.json của nó
            result = subprocess.run(
                ["python", "download_images.py"], 
                cwd=script_dir,
                text=True
            )
            
            if result.returncode == 0:
                print(f"[v] Hoàn thành: {script_dir.name}\n")
            else:
                print(f"[x] Lỗi khi chạy tại: {script_dir.name} (Exit code: {result.returncode})\n")
                
        except Exception as e:
            print(f"[!] Gặp lỗi hệ thống khi chạy {script}: {e}\n")

    print("="*40)
    print("[*] Tất cả tiến trình đã kết thúc.")

def main():
    parser = argparse.ArgumentParser(description="Tự động chạy tất cả các script download_images.py trong thư mục")
    parser.add_argument("path", nargs="?", default=".", help="Đường dẫn thư mục gốc để quét (mặc định: current dir)")
    args = parser.parse_args()
    
    run_all_downloads(args.path)

if __name__ == "__main__":
    main()
