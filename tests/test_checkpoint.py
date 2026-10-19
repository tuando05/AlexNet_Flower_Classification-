"""Kiểm thử việc load checkpoint (tương thích cũ/mới, thứ tự lớp, weights_only)."""

import torch
import pytest

import config
from src.models import builder
from src.models.builder import load_checkpoint_data, load_trained_model
from tests._helpers import TinyModel


@pytest.fixture
def patched_builder(monkeypatch):
    """Thay build_model bằng bản nhẹ, ghi lại tham số được truyền vào."""
    calls = {}

    def fake_build_model(use_pretrained=False, num_classes=3, device='cpu'):
        calls['use_pretrained'] = use_pretrained
        calls['num_classes'] = num_classes
        return TinyModel(num_classes)

    monkeypatch.setattr(builder, 'build_model', fake_build_model)
    return calls


def _save(tmp_path, name, payload):
    path = tmp_path / name
    torch.save(payload, path)
    return str(path)


def test_load_checkpoint_data_new_format(tmp_path):
    model = TinyModel(3)
    path = _save(tmp_path, 'new.pth', {
        'model_state_dict': model.state_dict(),
        'epoch': 13,
        'val_acc': 0.966,
        'classes': ['daisy', 'lily', 'rose'],
        'arch': 'alexnet',
        'use_pretrained': True,
    })

    state_dict, metadata = load_checkpoint_data(path, device='cpu')

    assert 'fc.weight' in state_dict
    assert metadata['epoch'] == 13
    assert metadata['classes'] == ['daisy', 'lily', 'rose']
    assert 'model_state_dict' not in metadata


def test_load_checkpoint_data_legacy_format(tmp_path):
    """Checkpoint cũ chỉ chứa state_dict thuần vẫn phải load được."""
    path = _save(tmp_path, 'legacy.pth', TinyModel(3).state_dict())

    state_dict, metadata = load_checkpoint_data(path, device='cpu')

    assert 'fc.weight' in state_dict
    assert metadata == {}


def test_load_checkpoint_data_passes_weights_only_false(tmp_path, monkeypatch):
    """Đảm bảo không vỡ khi torch >= 2.6 đổi mặc định weights_only sang True."""
    captured = {}
    original_load = torch.load

    def spy_load(*args, **kwargs):
        captured.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', spy_load)
    path = _save(tmp_path, 'spy.pth', TinyModel(3).state_dict())

    load_checkpoint_data(path, device='cpu')

    assert captured.get('weights_only') is False


def test_load_trained_model_takes_classes_from_checkpoint(tmp_path, patched_builder):
    path = _save(tmp_path, 'ckpt.pth', {
        'model_state_dict': TinyModel(3).state_dict(),
        'classes': ['rose', 'daisy', 'lily'],
        'use_pretrained': False,
    })

    model, metadata, class_names = load_trained_model(path, device='cpu', default_classes=['daisy', 'lily', 'rose'])

    assert class_names == ['rose', 'daisy', 'lily'], "phải ưu tiên thứ tự lớp trong checkpoint"
    assert patched_builder['num_classes'] == 3
    assert isinstance(model, TinyModel)
    assert model.training is False, "model phải ở chế độ eval"


def test_load_trained_model_falls_back_to_default_classes(tmp_path, patched_builder):
    path = _save(tmp_path, 'legacy.pth', TinyModel(3).state_dict())

    _, _, class_names = load_trained_model(path, device='cpu', default_classes=['a', 'b', 'c'])

    assert class_names == ['a', 'b', 'c']


def test_load_trained_model_falls_back_to_config_classes(tmp_path, patched_builder):
    path = _save(tmp_path, 'legacy.pth', TinyModel(len(config.TARGET_CLASSES)).state_dict())

    _, _, class_names = load_trained_model(path, device='cpu')

    assert class_names == list(config.TARGET_CLASSES)


@pytest.mark.parametrize("checkpoint_value,expected", [(True, True), (False, False)])
def test_load_trained_model_honours_use_pretrained_from_checkpoint(
    tmp_path, patched_builder, monkeypatch, checkpoint_value, expected
):
    """
    Kiến trúc phải lấy từ checkpoint, không lấy từ config.

    Đây là lỗi đã sửa: đổi config.USE_PRETRAINED sau khi train sẽ khiến
    state_dict không khớp kiến trúc.
    """
    monkeypatch.setattr(config, 'USE_PRETRAINED', not checkpoint_value)
    path = _save(tmp_path, 'ckpt.pth', {
        'model_state_dict': TinyModel(3).state_dict(),
        'classes': ['daisy', 'lily', 'rose'],
        'use_pretrained': checkpoint_value,
    })

    load_trained_model(path, device='cpu')

    assert patched_builder['use_pretrained'] is expected


def test_load_trained_model_missing_metadata_uses_config_default(tmp_path, patched_builder, monkeypatch):
    monkeypatch.setattr(config, 'USE_PRETRAINED', False)
    path = _save(tmp_path, 'legacy.pth', TinyModel(3).state_dict())

    load_trained_model(path, device='cpu', default_classes=['daisy', 'lily', 'rose'])

    assert patched_builder['use_pretrained'] is False


def test_load_trained_model_raises_on_unmatched_state_dict(tmp_path, patched_builder):
    path = _save(tmp_path, 'bad.pth', {
        'model_state_dict': TinyModel(2).state_dict(),
        'classes': ['daisy', 'lily', 'rose'],
    })

    with pytest.raises(RuntimeError):
        load_trained_model(path, device='cpu')


def test_load_trained_model_raises_when_file_missing(patched_builder):
    with pytest.raises(FileNotFoundError):
        load_trained_model('khong_ton_tai.pth', device='cpu')
