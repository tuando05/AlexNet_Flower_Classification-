"""Kiểm thử rào chắn dữ liệu của create_dataloaders."""

import os
import shutil

import pytest

import config
from src.data.dataloader import _ensure_split_ready
from tests._helpers import make_images


def _build_full_dataset(root, splits=('train', 'val', 'test')):
    for split in splits:
        for cls in config.TARGET_CLASSES:
            make_images(os.path.join(root, split, cls), 'img_0.jpg')


def test_passes_when_all_splits_have_all_classes(tmp_path):
    _build_full_dataset(str(tmp_path))

    for split in ('train', 'val', 'test'):
        _ensure_split_ready(str(tmp_path), split)  # không được raise


@pytest.mark.parametrize("missing_split", ['val', 'test'])
def test_raises_when_split_directory_missing(tmp_path, missing_split):
    splits = tuple(s for s in ('train', 'val', 'test') if s != missing_split)
    _build_full_dataset(str(tmp_path), splits=splits)

    with pytest.raises(RuntimeError, match=f"Thiếu thư mục split '{missing_split}'"):
        _ensure_split_ready(str(tmp_path), missing_split)


def test_raises_when_a_class_is_empty(tmp_path):
    _build_full_dataset(str(tmp_path))
    # Xoá ảnh của một lớp trong split train
    for name in os.listdir(os.path.join(str(tmp_path), 'train', 'lily')):
        os.remove(os.path.join(str(tmp_path), 'train', 'lily', name))

    with pytest.raises(RuntimeError, match="lily"):
        _ensure_split_ready(str(tmp_path), 'train')


def test_raises_when_class_directory_does_not_exist(tmp_path):
    _build_full_dataset(str(tmp_path))
    shutil.rmtree(os.path.join(str(tmp_path), 'test', 'rose'))

    with pytest.raises(RuntimeError, match="rose"):
        _ensure_split_ready(str(tmp_path), 'test')


def test_error_message_suggests_fix(tmp_path):
    _build_full_dataset(str(tmp_path), splits=('train',))

    with pytest.raises(RuntimeError, match="download_data.py"):
        _ensure_split_ready(str(tmp_path), 'val')
