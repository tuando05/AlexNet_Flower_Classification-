#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script tự động tải và lọc bộ dữ liệu Kaggle 3 loại hoa (Daisy, Lily, Rose)
Dataset Kaggle gốc: utkarshsaxenadn/flower-classification-5-classes-roselilyetc (Version 1)
"""

import os
import sys
import argparse
import shutil

# Ensure root directory and local .venv site-packages are in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Auto-include local virtual environment packages if present
venv_site_packages = os.path.join(BASE_DIR, ".venv", "Lib", "site-packages")
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import config
KAGGLE_DATASET_HANDLE = "utkarshsaxenadn/flower-classification-5-classes-roselilyetc"

def download_and_filter_dataset(target_base_dir=config.DATA_DIR, force=False):
    """
    Tải bộ dữ liệu từ Kaggle thông qua kagglehub và chỉ lọc 3 loại hoa cần thiết.
    """
    print("=" * 65)
    print("KICH BAN TAI & TRICH XUAT BO DU LIEU HOA (KAGGLE V1)")
    print("=" * 65)

    # Kiểm tra xem dữ liệu đã tồn tại chưa
    has_existing_data = False
    if os.path.exists(target_base_dir):
        for split in ['train', 'val', 'test']:
            for cls in config.TARGET_CLASSES:
                p = os.path.join(target_base_dir, split, cls)
                if os.path.exists(p) and len(os.listdir(p)) > 0:
                    has_existing_data = True
                    break

    if has_existing_data and not force:
        print(f"[INFO] Du lieu da ton tai tai: {target_base_dir}")
        print("       Su dung co '--force' neu ban muon xoa va tai lai tu dau.")
        print("       Vi du: python download_data.py --force")
        print("-" * 65)
        return

    if force and os.path.exists(target_base_dir):
        print(f"[CLEAN] Dang xoa du lieu cu tai: '{target_base_dir}'...")
        shutil.rmtree(target_base_dir)

    try:
        import kagglehub
        print(f"[DOWNLOAD] Dang tai bo du lieu '{KAGGLE_DATASET_HANDLE}' tu Kaggle...")
        download_path = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
        print(f"[OK] Da tai ve thu muc cache Kaggle: {download_path}")

        # Tìm thư mục chứa v1 (Flower Classification -> Training Data, Validation Data, Testing Data)
        v1_root = None
        for root, dirs, files in os.walk(download_path):
            if 'Training Data' in dirs and 'Validation Data' in dirs:
                v1_root = root
                break

        if not v1_root:
            print("[ERROR] Khong tim thay cau truc thu muc 'Training Data' trong bo du lieu Kaggle.")
            return

        split_map = {
            'Training Data': 'train',
            'Validation Data': 'val',
            'Testing Data': 'test'
        }

        class_map = {
            'Daisy': 'daisy',
            'Lily': 'lily',
            'Rose': 'rose'
        }

        print("\n[EXTRACT] Dang loc va trich xuat 3 loai hoa (Daisy, Lily, Rose)...")
        stats = {}

        for src_split, dst_split in split_map.items():
            stats[dst_split] = {}
            for src_cls, dst_cls in class_map.items():
                src_dir = os.path.join(v1_root, src_split, src_cls)
                dst_dir = os.path.join(target_base_dir, dst_split, dst_cls)
                os.makedirs(dst_dir, exist_ok=True)

                if os.path.exists(src_dir):
                    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    for img in images:
                        shutil.copy2(os.path.join(src_dir, img), os.path.join(dst_dir, img))
                    stats[dst_split][dst_cls] = len(images)
                    print(f"  -> [{dst_split.upper()}] {dst_cls.capitalize()}: {len(images)} anh")

        print("\n" + "=" * 65)
        print("HOAN THANH TAI VA TRICH XUAT DU LIEU!")
        print(f"Vi tri luu: {target_base_dir}")
        print("=" * 65)

    except ModuleNotFoundError:
        print("[ERROR] Khong tim thay thu vien 'kagglehub'.")
        print("        Vui long cai dat bang lenh: pip install kagglehub")
    except Exception as e:
        print(f"[ERROR] Co loi xay ra trong qua trinh tai du lieu: {e}")

def main():
    parser = argparse.ArgumentParser(description="Tai va loc du lieu Kaggle 3 loai hoa")
    parser.add_argument("--force", action="store_true", help="Xoa du lieu cu va tai lai")
    parser.add_argument("--target-dir", type=str, default=config.DATA_DIR, help="Thu muc dich luu du lieu")
    args = parser.parse_args()

    download_and_filter_dataset(target_base_dir=args.target_dir, force=args.force)

if __name__ == '__main__':
    main()
