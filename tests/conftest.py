"""
Cấu hình chung cho pytest.

- Đưa thư mục gốc dự án vào `sys.path`.
- Cung cấp fixture dùng chung cho nhiều file test (tránh định nghĩa lặp ở
  `test_checkpoint.py` và `test_predictor.py`).
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import builder
from src.pipelines import predictor as predictor_module
from tests._helpers import TinyModel


@pytest.fixture
def fake_model_builder(monkeypatch):
    """
    Thay `build_model` bằng `TinyModel` để test không phải dựng AlexNet thật.

    Patch ở CẢ HAI nơi dùng `build_model`:
      - `src.models.builder`  -> dùng bởi `load_trained_model`
      - `src.pipelines.predictor` -> dùng cho nhánh chưa có checkpoint

    Returns:
        dict: các tham số mà `build_model` đã được gọi với
              (`use_pretrained`, `num_classes`, `device`).
    """
    calls = {}

    def fake_build_model(use_pretrained=False, num_classes=3, device='cpu'):
        calls['use_pretrained'] = use_pretrained
        calls['num_classes'] = num_classes
        calls['device'] = device
        return TinyModel(num_classes).to(device)

    monkeypatch.setattr(builder, 'build_model', fake_build_model)
    monkeypatch.setattr(predictor_module, 'build_model', fake_build_model)
    return calls
