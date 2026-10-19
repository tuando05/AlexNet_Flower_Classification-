"""Kiểm thử kiến trúc AlexNet và lớp builder."""

import pytest
import torch

import config
from src.models.alexnet import AlexNetFromScratch, get_alexnet_pretrained
from src.models.builder import build_model


def test_scratch_forward_shape():
    model = AlexNetFromScratch(num_classes=3)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 3)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("num_classes", [2, 3, 7])
def test_build_model_scratch_num_classes(num_classes):
    model = build_model(use_pretrained=False, num_classes=num_classes, device='cpu')
    assert isinstance(model, AlexNetFromScratch)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, num_classes)


def test_build_model_moves_to_device():
    model = build_model(use_pretrained=False, num_classes=3, device='cpu')
    assert next(model.parameters()).device.type == 'cpu'


def test_pretrained_classifier_head_replaced():
    """
    Kiểm tra AlexNet pre-trained có thay lớp cuối đúng số lớp.

    Bỏ qua nếu không tải được weights ImageNet (môi trường không có mạng).
    """
    try:
        model = get_alexnet_pretrained(num_classes=5)
    except Exception as e:  # pragma: no cover - phụ thuộc mạng
        pytest.skip(f"Không tải được weights ImageNet: {e}")

    assert model.classifier[-1].out_features == 5
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 5)


def test_config_defaults_are_consistent():
    assert config.NUM_CLASSES == len(config.TARGET_CLASSES)
    assert set(config.CLASS_NAMES_VI) == set(config.TARGET_CLASSES)
    assert config.EARLY_STOPPING_PATIENCE >= 0
