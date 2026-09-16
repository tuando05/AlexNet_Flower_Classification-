import os
import shutil
import random
from PIL import Image
import numpy as np

import config

def prepare_flower_dataset(source_path=None, target_base_dir=config.DATA_DIR):
    """
    Organizes the flower dataset into train/val/test splits for target classes (rose, daisy, lily).
    If source_path is not available, creates sample mock data for pipeline testing.
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
            print("Creating sample synthetic data for testing pipeline...")
            for split in ['train', 'val', 'test']:
                n_samples = 40 if split == 'train' else 10
                for cls in target_classes:
                    cls_dir = os.path.join(target_base_dir, split, cls)
                    for i in range(n_samples):
                        # Generate colorful dummy flower images
                        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                        img = Image.fromarray(img_array)
                        img.save(os.path.join(cls_dir, f"sample_{cls}_{i}.jpg"))
            print("Sample synthetic data created successfully.")
