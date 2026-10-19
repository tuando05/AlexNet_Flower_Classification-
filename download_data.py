#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI tải và lọc bộ dữ liệu Kaggle 3 loại hoa (Daisy, Lily, Rose).

Toàn bộ logic nằm trong `src.data.downloader` — script này chỉ lo phần
đối số dòng lệnh và thiết lập môi trường chạy.

Cách dùng:
    python download_data.py                # tải nếu chưa có dữ liệu
    python download_data.py --force        # xóa dữ liệu cũ rồi tải lại
    python download_data.py --target-dir D # lưu vào thư mục khác
"""

import argparse
import os
import sys

# Ensure root directory and local .venv site-packages are in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Auto-include local virtual environment packages if present
venv_site_packages = os.path.join(BASE_DIR, ".venv", "Lib", "site-packages")
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

import config
from src.data.downloader import KAGGLE_DATASET_HANDLE, download_and_filter_dataset
from src.utils.console import setup_console
from src.utils.logger import setup_logging


def main():
    setup_console()
    setup_logging()

    parser = argparse.ArgumentParser(description="Tải và lọc dữ liệu Kaggle 3 loại hoa")
    parser.add_argument("--force", action="store_true", help="Xóa dữ liệu cũ và tải lại")
    parser.add_argument("--target-dir", type=str, default=config.DATA_DIR, help="Thư mục đích lưu dữ liệu")
    args = parser.parse_args()

    print("=" * 65)
    print("KỊCH BẢN TẢI & TRÍCH XUẤT BỘ DỮ LIỆU HOA (KAGGLE V1)")
    print(f"Dataset: {KAGGLE_DATASET_HANDLE}")
    print("=" * 65)

    success = download_and_filter_dataset(target_base_dir=args.target_dir, force=args.force)
    if not success:
        print("[THẤT BẠI] Không thể chuẩn bị dữ liệu. Xem hướng dẫn ở thông báo phía trên.")
        sys.exit(1)


if __name__ == '__main__':
    main()
