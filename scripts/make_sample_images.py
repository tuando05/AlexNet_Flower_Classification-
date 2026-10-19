#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tạo 3 ảnh mẫu thật cho web app: samples/{daisy,lily,rose}.jpg

Ảnh được lấy từ tập TEST của `flower_data_3classes/` (mỗi lớp 1 ảnh), thu nhỏ
về tối đa 640px và lưu JPEG để dung lượng repo nhỏ.

Trước đây UI vẽ "ảnh mẫu" bằng canvas (gradient + emoji) nên kết quả dự đoán
hoàn toàn vô nghĩa; script này thay bằng ảnh hoa thật.

Cách dùng:
    python scripts/make_sample_images.py              # tạo nếu chưa có
    python scripts/make_sample_images.py --force      # ghi đè
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from src.data.downloader import list_images
from src.utils.console import setup_console
from src.utils.logger import setup_logging

setup_console()
setup_logging()

import logging

from PIL import Image

logger = logging.getLogger('scripts.make_sample_images')

MAX_SIDE = 640
JPEG_QUALITY = 85
SOURCE_SPLIT = 'test'


def make_sample_images(force=False):
    """
    Tạo ảnh mẫu cho từng lớp trong `config.TARGET_CLASSES`.

    Returns:
        list[str]: Đường dẫn các file đã tạo (hoặc đã có sẵn).
    """
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    created = []

    for cls in config.TARGET_CLASSES:
        dst_path = os.path.join(config.SAMPLE_DIR, f"{cls}.jpg")
        if os.path.exists(dst_path) and not force:
            logger.info("[BỎ QUA] Đã có: %s", dst_path)
            created.append(dst_path)
            continue

        src_dir = os.path.join(config.DATA_DIR, SOURCE_SPLIT, cls)
        images = sorted(list_images(src_dir))
        if not images:
            logger.warning(
                "Không có ảnh cho lớp '%s' trong '%s'. Hãy chạy: python download_data.py",
                cls, src_dir,
            )
            continue

        src_path = os.path.join(src_dir, images[0])
        image = Image.open(src_path)
        image = image.convert('RGB')
        image.thumbnail((MAX_SIDE, MAX_SIDE))
        image.save(dst_path, 'JPEG', quality=JPEG_QUALITY, optimize=True)

        logger.info("[OK] %s -> %s (%d bytes)", src_path, dst_path, os.path.getsize(dst_path))
        created.append(dst_path)

    if len(created) == len(config.TARGET_CLASSES):
        logger.info("Đã sẵn sàng %d ảnh mẫu trong '%s'.", len(created), config.SAMPLE_DIR)
    else:
        logger.warning(
            "Chỉ có %d/%d ảnh mẫu. Web app sẽ ẩn phần ảnh mẫu nếu thiếu.",
            len(created), len(config.TARGET_CLASSES),
        )

    return created


def main():
    parser = argparse.ArgumentParser(description="Tạo ảnh mẫu thật cho web app")
    parser.add_argument("--force", action="store_true", help="Ghi đè ảnh mẫu đã có")
    args = parser.parse_args()

    make_sample_images(force=args.force)


if __name__ == '__main__':
    main()
