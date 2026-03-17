#!/bin/bash

# Nhập folder path vào đây
FOLDER_PATH="/home/a4000/ducanh/Dataset-new/VF5"

python3 /home/a4000/ducanh/normalize-carpart-annotations/downloader/run_all_downloads.py "$FOLDER_PATH"
