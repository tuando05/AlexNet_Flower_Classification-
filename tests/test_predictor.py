"""Kiểm thử FlowerPredictor (inference engine)."""

import numpy as np
import pytest
import torch
from PIL import Image

import config
from src.models import builder
from src.pipelines import predictor as predictor_module
from src.pipelines.predictor import FlowerPredictor
from tests._helpers import TinyModel


@pytest.fixture
def tiny_model_builder(monkeypatch):
    """Dùng TinyModel thay AlexNet để test không cần checkpoint thật."""

    def fake_build_model(use_pretrained=False, num_classes=3, device='cpu'):
        return TinyModel(num_classes).to(device)

    # Patch cả 2 nơi: builder (dùng bởi load_trained_model) và predictor
    # (dùng cho nhánh chưa có checkpoint).
    monkeypatch.setattr(builder, 'build_model', fake_build_model)
    monkeypatch.setattr(predictor_module, 'build_model', fake_build_model)


@pytest.fixture
def random_image():
    rng = np.random.default_rng(config.SEED)
    return Image.fromarray(rng.integers(0, 256, (240, 260, 3), dtype=np.uint8), mode='RGB')


def _make_checkpoint(tmp_path, classes, name='ckpt.pth'):
    path = tmp_path / name
    torch.save(
        {
            'model_state_dict': TinyModel(len(classes)).state_dict(),
            'classes': classes,
            'use_pretrained': False,
            'epoch': 13,
            'val_acc': 0.966,
        },
        path,
    )
    return str(path)


def test_predict_returns_expected_structure(tmp_path, tiny_model_builder, random_image):
    path = _make_checkpoint(tmp_path, ['daisy', 'lily', 'rose'])

    result = FlowerPredictor(model_path=path, device='cpu').predict_image(random_image)

    assert set(result) == {
        'prediction', 'prediction_vi', 'confidence',
        'confidence_percentage', 'model_trained', 'probabilities',
    }
    assert result['model_trained'] is True
    assert result['prediction'] in ['daisy', 'lily', 'rose']
    assert 0.0 <= result['confidence'] <= 1.0


def test_probabilities_cover_all_classes_and_sum_to_100(tmp_path, tiny_model_builder, random_image):
    path = _make_checkpoint(tmp_path, ['daisy', 'lily', 'rose'])

    result = FlowerPredictor(model_path=path, device='cpu').predict_image(random_image)
    probabilities = result['probabilities']

    assert list(probabilities) == ['daisy', 'lily', 'rose']
    assert sum(p['percentage'] for p in probabilities.values()) == pytest.approx(100.0, abs=0.05)
    assert sum(p['probability'] for p in probabilities.values()) == pytest.approx(1.0, abs=0.001)


def test_prediction_matches_highest_probability(tmp_path, tiny_model_builder, random_image):
    path = _make_checkpoint(tmp_path, ['daisy', 'lily', 'rose'])

    result = FlowerPredictor(model_path=path, device='cpu').predict_image(random_image)

    best = max(result['probabilities'].items(), key=lambda kv: kv[1]['percentage'])
    assert result['prediction'] == best[0]
    assert result['confidence_percentage'] == pytest.approx(best[1]['percentage'], abs=0.01)


def test_uses_class_order_from_checkpoint(tmp_path, tiny_model_builder, random_image):
    """Thứ tự lớp phải theo checkpoint, không phải theo config."""
    path = _make_checkpoint(tmp_path, ['rose', 'daisy', 'lily'])

    predictor = FlowerPredictor(model_path=path, device='cpu')

    assert predictor.class_names == ['rose', 'daisy', 'lily']
    assert list(predictor.predict_image(random_image)['probabilities']) == ['rose', 'daisy', 'lily']


def test_vietnamese_label_is_attached(tmp_path, tiny_model_builder, random_image):
    path = _make_checkpoint(tmp_path, ['daisy', 'lily', 'rose'])

    result = FlowerPredictor(model_path=path, device='cpu').predict_image(random_image)

    assert result['prediction_vi'] == config.CLASS_NAMES_VI[result['prediction']]


def test_accepts_file_path(tmp_path, tiny_model_builder, random_image):
    path = _make_checkpoint(tmp_path, ['daisy', 'lily', 'rose'])
    image_path = tmp_path / 'flower.jpg'
    random_image.save(image_path)

    result = FlowerPredictor(model_path=path, device='cpu').predict_image(str(image_path))

    assert result['prediction'] in ['daisy', 'lily', 'rose']


def test_missing_checkpoint_flags_untrained_and_still_predicts(tmp_path, tiny_model_builder, random_image):
    predictor = FlowerPredictor(model_path=str(tmp_path / 'khong_ton_tai.pth'), device='cpu')

    assert predictor.is_trained is False
    result = predictor.predict_image(random_image)
    assert result['model_trained'] is False
    assert len(result['probabilities']) == len(config.TARGET_CLASSES)


def test_invalid_input_type_raises(tmp_path, tiny_model_builder):
    path = _make_checkpoint(tmp_path, ['daisy', 'lily', 'rose'])
    predictor = FlowerPredictor(model_path=path, device='cpu')

    with pytest.raises(ValueError):
        predictor.predict_image(12345)


def test_grayscale_and_rgba_inputs_are_converted(tmp_path, tiny_model_builder):
    path = _make_checkpoint(tmp_path, ['daisy', 'lily', 'rose'])
    predictor = FlowerPredictor(model_path=path, device='cpu')

    gray = Image.new('L', (224, 224), 90)
    rgba = Image.new('RGBA', (224, 224), (10, 20, 30, 128))

    assert predictor.predict_image(gray)['prediction'] in ['daisy', 'lily', 'rose']
    assert predictor.predict_image(rgba)['prediction'] in ['daisy', 'lily', 'rose']
