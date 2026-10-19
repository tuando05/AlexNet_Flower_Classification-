"""Kiểm thử pipeline biến đổi ảnh (transforms)."""

import numpy as np
import pytest
import torch
from PIL import Image

import config
from src.data.transforms import get_transforms


@pytest.fixture
def sample_image():
    rng = np.random.default_rng(config.SEED)
    array = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
    return Image.fromarray(array, mode='RGB')


def test_returns_two_transform_pipelines():
    train_tf, val_tf = get_transforms()
    assert train_tf is not None and val_tf is not None
    assert train_tf is not val_tf


def test_val_transform_output_shape(sample_image):
    _, val_tf = get_transforms()
    out = val_tf(sample_image)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert out.dtype == torch.float32


def test_train_transform_output_shape(sample_image):
    train_tf, _ = get_transforms()
    out = train_tf(sample_image)
    assert out.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)


def test_image_size_matches_alexnet_expectation():
    # AlexNet (torchvision) yêu cầu ảnh đầu vào tối thiểu 224x224
    assert config.IMAGE_SIZE == 224


def test_normalization_is_applied():
    """
    Với ảnh đồng nhất một màu, sau Resize + CenterCrop giá trị từng kênh phải
    đúng bằng công thức (pixel/255 - mean) / std.
    """
    _, val_tf = get_transforms()

    pixel_value = 128
    image = Image.new('RGB', (300, 300), (pixel_value, pixel_value, pixel_value))
    out = val_tf(image)

    expected = (pixel_value / 255.0 - np.array(config.NORMALIZE_MEAN)) / np.array(config.NORMALIZE_STD)
    for channel in range(3):
        assert out[channel].mean().item() == pytest.approx(expected[channel], abs=1e-4)


def test_normalized_values_are_not_in_raw_pixel_range(sample_image):
    """Đảm bảo đã normalize thật (không phải chỉ ToTensor trả về [0, 1])."""
    _, val_tf = get_transforms()
    out = val_tf(sample_image)
    assert out.min().item() < 0.0 or out.max().item() > 1.5


def test_train_transform_uses_random_augmentation(sample_image):
    """Hai lần gọi train transform trên cùng ảnh phải khác nhau (do augmentation)."""
    torch.manual_seed(config.SEED)
    train_tf, _ = get_transforms()
    first = train_tf(sample_image)
    second = train_tf(sample_image)
    assert not torch.equal(first, second)


def test_val_transform_is_deterministic(sample_image):
    _, val_tf = get_transforms()
    assert torch.equal(val_tf(sample_image), val_tf(sample_image))


def test_transform_handles_non_square_image():
    _, val_tf = get_transforms()
    image = Image.new('RGB', (640, 400), (10, 200, 30))
    assert val_tf(image).shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
