"""
Kịch bản tải & lọc bộ dữ liệu Kaggle 3 loại hoa (Daisy, Lily, Rose).

Đây là NGUỒN DUY NHẤT chứa logic tải dữ liệu Kaggle của dự án.
`download_data.py` chỉ là CLI wrapper gọi hàm `download_and_filter_dataset()`.
"""

import logging
import os
import shutil

import config

logger = logging.getLogger(__name__)

# Dataset Kaggle gốc (Version 1)
KAGGLE_DATASET_HANDLE = "utkarshsaxenadn/flower-classification-5-classes-roselilyetc"

# Ánh xạ tên thư mục gốc trong Kaggle -> tên split chuẩn của dự án
SPLIT_MAP = {
    'Training Data': 'train',
    'Validation Data': 'val',
    'Testing Data': 'test',
}

# Ánh xạ tên lớp trong Kaggle -> tên lớp chuẩn của dự án
CLASS_MAP = {
    'Daisy': 'daisy',
    'Lily': 'lily',
    'Rose': 'rose',
}

# Định dạng ảnh được chấp nhận
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def list_images(directory):
    """Trả về danh sách tên file ảnh hợp lệ trong thư mục."""
    if not os.path.isdir(directory):
        return []
    return [
        f for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(directory, f))
    ]


def has_existing_data(target_base_dir=config.DATA_DIR):
    """
    Kiểm tra dataset đã có ảnh thật cho 3 lớp hay chưa.

    Chỉ cần ít nhất 1 split có ảnh được coi là đã có dữ liệu để tránh việc
    script tự động tải lại và ghi đè lên dữ liệu người dùng.
    """
    for split in SPLIT_MAP.values():
        for cls in config.TARGET_CLASSES:
            if list_images(os.path.join(target_base_dir, split, cls)):
                return True
    return False


def _find_v1_root(download_path):
    """
    Tìm thư mục gốc của bản dataset v1 trong cache Kaggle.

    Cấu trúc v1 chứa đồng thời 'Training Data' và 'Validation Data'.
    """
    for root, dirs, _ in os.walk(download_path):
        if 'Training Data' in dirs and 'Validation Data' in dirs:
            return root
    return None


def download_and_filter_dataset(target_base_dir=config.DATA_DIR, force=False):
    """
    Tải bộ dữ liệu Kaggle và chỉ trích xuất 3 loại hoa cần thiết.

    Args:
        target_base_dir: Thư mục đích (mặc định `config.DATA_DIR`).
        force: Xóa dữ liệu cũ và tải lại từ đầu.

    Returns:
        bool: True nếu dataset sẵn sàng sử dụng, False nếu tải thất bại.
    """
    if has_existing_data(target_base_dir) and not force:
        logger.info("Dữ liệu đã tồn tại tại: %s", target_base_dir)
        logger.info("Sử dụng cờ '--force' nếu bạn muốn xóa và tải lại từ đầu.")
        return True

    if force and os.path.isdir(target_base_dir):
        logger.warning("Đang xóa dữ liệu cũ tại: '%s'...", target_base_dir)
        shutil.rmtree(target_base_dir)

    try:
        import kagglehub
    except ModuleNotFoundError:
        logger.error("Không tìm thấy thư viện 'kagglehub'.")
        logger.error("Vui lòng cài đặt bằng lệnh: pip install kagglehub")
        return False

    try:
        logger.info("Đang tải bộ dữ liệu '%s' từ Kaggle...", KAGGLE_DATASET_HANDLE)
        download_path = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
        logger.info("Đã tải về thư mục cache Kaggle: %s", download_path)
    except Exception as e:
        logger.error("Không thể tải dữ liệu từ Kaggle: %s", e)
        return False

    v1_root = _find_v1_root(download_path)
    if not v1_root:
        logger.error("Không tìm thấy cấu trúc thư mục 'Training Data' trong bộ dữ liệu Kaggle.")
        return False

    logger.info("Đang lọc và trích xuất 3 loại hoa (Daisy, Lily, Rose)...")
    for src_split, dst_split in SPLIT_MAP.items():
        for src_cls, dst_cls in CLASS_MAP.items():
            src_dir = os.path.join(v1_root, src_split, src_cls)
            images = list_images(src_dir)
            if not images:
                logger.warning("Không tìm thấy ảnh cho %s/%s, bỏ qua.", src_split, src_cls)
                continue

            dst_dir = os.path.join(target_base_dir, dst_split, dst_cls)
            os.makedirs(dst_dir, exist_ok=True)
            for img in images:
                shutil.copy2(os.path.join(src_dir, img), os.path.join(dst_dir, img))
            logger.info("  [%s] %s: %d ảnh", dst_split.upper(), dst_cls.capitalize(), len(images))

    if not has_existing_data(target_base_dir):
        logger.error("Trích xuất dữ liệu thất bại, không có ảnh nào được ghi ra.")
        return False

    logger.info("Hoàn thành tải và trích xuất dữ liệu! Vị trí lưu: %s", target_base_dir)
    return True
