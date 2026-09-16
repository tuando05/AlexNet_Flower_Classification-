import os
import torch
from torch.utils.data import DataLoader
import torchvision

import config
from src.data.transforms import get_transforms
from src.data.dataset import prepare_flower_dataset

def create_dataloaders(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE):
    """
    Creates PyTorch DataLoaders for train, val, and test splits.
    """
    train_transforms, val_test_transforms = get_transforms()

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Ensure dataset exists
    if not os.path.exists(train_dir):
        prepare_flower_dataset(target_base_dir=data_dir)

    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=val_test_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset.classes
