import os
import shutil
import random
from PIL import Image
import numpy as np

import config

KAGGLE_DATASET_HANDLE = "utkarshsaxenadn/flower-classification-5-classes-roselilyetc"

def download_and_extract_kaggle_v1(target_base_dir=config.DATA_DIR):
    """
    Downloads flower classification dataset (v1) from Kaggle via kagglehub,
    and extracts only the 3 target classes: daisy, lily, rose.
    """
    try:
        import kagglehub
        print(f"Downloading Kaggle dataset '{KAGGLE_DATASET_HANDLE}' via kagglehub...")
        download_path = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
        print(f"Downloaded Kaggle dataset to: {download_path}")

        # Search for v1 'Flower Classification' inner folder containing Training Data, Validation Data, Testing Data
        v1_root = None
        for root, dirs, files in os.walk(download_path):
            if 'Training Data' in dirs and 'Validation Data' in dirs:
                v1_root = root
                break

        if not v1_root:
            print("Warning: Could not locate 'Training Data' in Kaggle dataset structure.")
            return False

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

        for src_split, dst_split in split_map.items():
            for src_cls, dst_cls in class_map.items():
                src_dir = os.path.join(v1_root, src_split, src_cls)
                dst_dir = os.path.join(target_base_dir, dst_split, dst_cls)
                os.makedirs(dst_dir, exist_ok=True)

                if os.path.exists(src_dir):
                    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    for img in images:
                        shutil.copy2(os.path.join(src_dir, img), os.path.join(dst_dir, img))
                    print(f"  -> Extracted {len(images)} images for {dst_split}/{dst_cls}")
        print("Successfully imported Kaggle dataset v1 for 3 target classes!")
        return True
    except Exception as e:
        print(f"Error downloading Kaggle dataset via kagglehub: {e}")
        return False


def prepare_flower_dataset(source_path=None, target_base_dir=config.DATA_DIR):
    """
    Organizes the flower dataset into train/val/test splits for target classes (rose, daisy, lily).
    If source_path is not available and dataset is empty, downloads Kaggle v1 dataset.
    """
    target_classes = config.TARGET_CLASSES

    for split in ['train', 'val', 'test']:
        for cls in target_classes:
            os.makedirs(os.path.join(target_base_dir, split, cls), exist_ok=True)

    if source_path and os.path.exists(source_path):
        all_dirs = []
        for root, dirs, files in os.walk(source_path):
            for d in dirs:
                all_dirs.append((d.lower(), os.path.join(root, d)))

        class_mapping = {}
        for target in target_classes:
            for dir_name, dir_path in all_dirs:
                if target in dir_name or (target == 'daisy' and 'cuc' in dir_name) or (target == 'lily' and ('ly' in dir_name or 'tulip' in dir_name)):
                    class_mapping[target] = dir_path
                    break

        for cls_name, cls_path in class_mapping.items():
            images = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            random.seed(42)
            random.shuffle(images)

            n_total = len(images)
            n_train = int(n_total * 0.8)
            n_val = int(n_total * 0.1)

            train_imgs = images[:n_train]
            val_imgs = images[n_train:n_train + n_val]
            test_imgs = images[n_train + n_val:]

            for img_p in train_imgs:
                shutil.copy(img_p, os.path.join(target_base_dir, 'train', cls_name, os.path.basename(img_p)))
            for img_p in val_imgs:
                shutil.copy(img_p, os.path.join(target_base_dir, 'val', cls_name, os.path.basename(img_p)))
            for img_p in test_imgs:
                shutil.copy(img_p, os.path.join(target_base_dir, 'test', cls_name, os.path.basename(img_p)))
        print("Dataset preparation complete.")
    else:
        # Check if dataset directory already has files
        has_files = False
        for split in ['train', 'val', 'test']:
            for cls in target_classes:
                p = os.path.join(target_base_dir, split, cls)
                if os.path.exists(p) and len(os.listdir(p)) > 0:
                    has_files = True
                    break

        if not has_files:
            print("Dataset empty. Attempting to download Kaggle dataset v1...")
            success = download_and_extract_kaggle_v1(target_base_dir=target_base_dir)
            if not success:
                print("Creating sample synthetic data for testing pipeline...")
                for split in ['train', 'val', 'test']:
                    n_samples = 40 if split == 'train' else 10
                    for cls in target_classes:
                        cls_dir = os.path.join(target_base_dir, split, cls)
                        for i in range(n_samples):
                            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                            img = Image.fromarray(img_array)
                            img.save(os.path.join(cls_dir, f"sample_{cls}_{i}.jpg"))
                print("Sample synthetic data created successfully.")
