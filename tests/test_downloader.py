"""Kiểm thử logic tải & kiểm tra dữ liệu (không cần mạng)."""

import os

import pytest

import config
from src.data import downloader
from src.data.downloader import (
    CLASS_MAP,
    SPLIT_MAP,
    _find_v1_root,
    download_and_filter_dataset,
    has_existing_data,
    list_images,
)
from tests._helpers import make_images


def test_list_images_filters_extensions_and_dirs(tmp_path):
    make_images(tmp_path, 'a.jpg', 'b.JPEG', 'c.png', 'd.bmp')
    make_images(tmp_path, 'e.gif', 'f.txt')
    make_images(os.path.join(tmp_path, 'thu_muc_ten_nhu_anh.jpg'))

    found = sorted(list_images(str(tmp_path)))

    assert found == ['a.jpg', 'b.JPEG', 'c.png', 'd.bmp']


def test_list_images_returns_empty_for_missing_dir(tmp_path):
    assert list_images(str(tmp_path / 'khong_ton_tai')) == []


def test_has_existing_data_false_when_empty(tmp_path):
    assert has_existing_data(str(tmp_path)) is False


def test_has_existing_data_false_when_only_non_image_files(tmp_path):
    for split in SPLIT_MAP.values():
        for cls in config.TARGET_CLASSES:
            make_images(os.path.join(str(tmp_path), split, cls), 'note.txt')

    assert has_existing_data(str(tmp_path)) is False


@pytest.mark.parametrize("split,cls", [('train', 'daisy'), ('val', 'lily'), ('test', 'rose')])
def test_has_existing_data_true_when_any_class_has_image(tmp_path, split, cls):
    make_images(os.path.join(str(tmp_path), split, cls), 'a.jpg')

    assert has_existing_data(str(tmp_path)) is True


def test_find_v1_root_detects_kaggle_structure(tmp_path):
    v1 = tmp_path / 'Flower Classification'
    (v1 / 'Training Data').mkdir(parents=True)
    (v1 / 'Validation Data').mkdir()
    (tmp_path / 'khac').mkdir()

    assert _find_v1_root(str(tmp_path)) == str(v1)


def test_find_v1_root_returns_none_for_unknown_structure(tmp_path):
    (tmp_path / 'Training Data').mkdir()

    assert _find_v1_root(str(tmp_path)) is None


def test_download_and_filter_returns_true_without_network_when_data_exists(tmp_path):
    """
    Nếu dữ liệu đã có sẵn, hàm phải trả True NGAY và không hề import/gọi kagglehub.

    Điều này đảm bảo chạy lại script không ghi đè dữ liệu người dùng.
    """
    make_images(os.path.join(str(tmp_path), 'train', 'daisy'), 'a.jpg')

    assert download_and_filter_dataset(target_base_dir=str(tmp_path)) is True


def test_download_and_filter_force_removes_old_data(tmp_path, monkeypatch):
    make_images(os.path.join(str(tmp_path), 'train', 'daisy'), 'a.jpg')
    monkeypatch.setattr(downloader, 'has_existing_data', lambda *a, **k: False)

    # Không có kagglehub trong môi trường test thì hàm phải trả False thay vì crash
    import importlib.util
    if importlib.util.find_spec('kagglehub') is not None:
        pytest.skip("kagglehub có sẵn -> bỏ qua test nhánh lỗi")

    assert download_and_filter_dataset(target_base_dir=str(tmp_path), force=True) is False
    assert not os.path.exists(os.path.join(str(tmp_path), 'train'))


def test_class_and_split_maps_match_project_conventions():
    assert set(SPLIT_MAP.values()) == {'train', 'val', 'test'}
    assert set(CLASS_MAP.values()) == set(config.TARGET_CLASSES)
    assert all(src.lower().startswith(dst[:3]) for src, dst in CLASS_MAP.items())
