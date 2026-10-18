import os

from torch.utils.data import DataLoader
import torchvision

import config
from src.data.downloader import list_images
from src.data.transforms import get_transforms
from src.data.dataset import prepare_flower_dataset

SPLITS = ('train', 'val', 'test')


def _ensure_split_ready(data_dir, split):
    """
    Kiểm tra một split có đủ dữ liệu cho cả 3 lớp hay không.

    Raises:
        RuntimeError: nếu split thiếu thư mục lớp hoặc lớp không có ảnh nào.
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise RuntimeError(f"Thiếu thư mục split '{split}': {split_dir}")

    empty_classes = [
        cls for cls in config.TARGET_CLASSES
        if not list_images(os.path.join(split_dir, cls))
    ]
    if empty_classes:
        raise RuntimeError(
            f"Split '{split}' không có ảnh cho các lớp: {', '.join(empty_classes)}.\n"
            f"  Hãy chạy: python download_data.py --force"
        )


def create_dataloaders(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE):
    """
    Creates PyTorch DataLoaders for train, val, and test splits.

    Raises:
        RuntimeError: nếu dataset chưa sẵn sàng hoặc thiếu ảnh ở bất kỳ split nào.
    """
    train_transforms, val_test_transforms = get_transforms()

    # Đảm bảo dataset tồn tại (tải/split nếu cần, raise nếu không thể)
    prepare_flower_dataset(target_base_dir=data_dir)

    # Kiểm tra cả 3 split trước khi dựng ImageFolder
    for split in SPLITS:
        _ensure_split_ready(data_dir, split)

    dirs = {split: os.path.join(data_dir, split) for split in SPLITS}

    train_dataset = torchvision.datasets.ImageFolder(root=dirs['train'], transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(root=dirs['val'], transform=val_test_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=dirs['test'], transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset.classes
