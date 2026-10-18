import os
import random
import shutil

import config
from src.data.downloader import (
    IMAGE_EXTENSIONS,
    SPLIT_MAP,
    has_existing_data,
    list_images,
    download_and_filter_dataset,
)

# Tỉ lệ chia dữ liệu khi người dùng truyền `source_path` thủ công
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


def _find_class_dirs(source_path, target_classes):
    """
    Tìm thư mục của từng lớp trong `source_path` bằng cách so khớp CHÍNH XÁC tên lớp.

    Bỏ qua các thư mục split (`train`/`val`/`test`) để tránh lấy nhầm dữ liệu
    đã được chia sẵn của chính dự án.

    Returns:
        dict: {tên lớp: đường dẫn thư mục}
    """
    split_names = set(SPLIT_MAP.values())
    class_mapping = {}

    for root, dirs, _ in os.walk(source_path):
        if os.path.basename(root).lower() in split_names:
            continue
        for d in dirs:
            key = d.lower()
            if key in target_classes and key not in class_mapping:
                class_mapping[key] = os.path.join(root, d)

    return class_mapping


def _split_dataset(source_path, target_base_dir, target_classes):
    """
    Chia dữ liệu từ `source_path` theo tỉ lệ 80/10/10 và copy vào target_base_dir.

    Raises:
        RuntimeError: nếu không tìm thấy thư mục nào cho các lớp mục tiêu.
    """
    class_mapping = _find_class_dirs(source_path, target_classes)
    if not class_mapping:
        raise RuntimeError(
            f"Không tìm thấy thư mục lớp nào ({', '.join(target_classes)}) trong '{source_path}'."
        )

    for cls_name, cls_path in class_mapping.items():
        images = [
            os.path.join(cls_path, f) for f in list_images(cls_path)
        ]
        random.seed(config.SEED)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        split_images = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:],
        }

        for split, files in split_images.items():
            dst_dir = os.path.join(target_base_dir, split, cls_name)
            os.makedirs(dst_dir, exist_ok=True)
            for img_path in files:
                shutil.copy2(img_path, os.path.join(dst_dir, os.path.basename(img_path)))

        print(
            f"  -> {cls_name}: {len(split_images['train'])} train / "
            f"{len(split_images['val'])} val / {len(split_images['test'])} test"
        )


def prepare_flower_dataset(source_path=None, target_base_dir=config.DATA_DIR):
    """
    Chuẩn bị dataset dạng train/val/test cho 3 lớp hoa (daisy, lily, rose).

    Thứ tự ưu tiên:
      1. Nếu truyền `source_path` hợp lệ: chia 80/10/10 rồi copy vào target.
      2. Nếu target đã có ảnh thật: dùng luôn, không làm gì thêm.
      3. Tải bộ dữ liệu Kaggle v1 qua `src.data.downloader`.

    Returns:
        bool: True khi dataset đã sẵn sàng để huấn luyện.

    Raises:
        RuntimeError: khi KHÔNG có dữ liệu thật và cũng không tải được từ Kaggle.
            (Trước đây hàm này tự sinh ảnh random và vẫn "thành công",
            khiến mô hình bị huấn luyện trên nhiễu mà không ai phát hiện.)
    """
    target_classes = config.TARGET_CLASSES

    # Bước tạo sẵn cấu trúc thư mục cho 3 split và 3 lớp
    for split in SPLIT_MAP.values():
        for cls in target_classes:
            os.makedirs(os.path.join(target_base_dir, split, cls), exist_ok=True)

    if source_path:
        if not os.path.isdir(source_path):
            raise FileNotFoundError(f"Thư mục nguồn không tồn tại: '{source_path}'")
        print(f"[PREPARE] Chia dữ liệu từ '{source_path}' vào '{target_base_dir}'...")
        _split_dataset(source_path, target_base_dir, target_classes)
        return True

    if has_existing_data(target_base_dir):
        print(f"[INFO] Dữ liệu đã sẵn sàng tại: {target_base_dir}")
        return True

    print("[PREPARE] Dataset trống, đang tải bộ dữ liệu Kaggle v1...")
    if download_and_filter_dataset(target_base_dir=target_base_dir):
        return True

    raise RuntimeError(
        "Không lấy được dữ liệu hoa thật để huấn luyện.\n"
        "  Cách 1: cài 'kagglehub' rồi chạy: python download_data.py\n"
        "  Cách 2: tự chuẩn bị ảnh rồi gọi prepare_flower_dataset(source_path=...)\n"
        "Dự án CHỦ ĐÍCH không còn sinh ảnh random vì làm vậy mô hình sẽ học trên nhiễu."
    )
