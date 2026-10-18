"""
High-level exports của dự án.

Dùng lazy import (PEP 562) để `from src import build_model` vẫn hoạt động
mà không nạp sẵn toàn bộ module con (gây RuntimeWarning của runpy khi
chạy `python -m src.pipelines.trainer`).
"""

import importlib

__all__ = [
    'FlowerPredictor',
    'train_model',
    'evaluate_model',
    'build_model',
    'create_dataloaders',
]

_MAP = {
    'FlowerPredictor': ('src.pipelines.predictor', 'FlowerPredictor'),
    'train_model': ('src.pipelines.trainer', 'train_model'),
    'evaluate_model': ('src.pipelines.evaluator', 'evaluate_model'),
    'build_model': ('src.models.builder', 'build_model'),
    'create_dataloaders': ('src.data.dataloader', 'create_dataloaders'),
}


def __getattr__(name):
    if name in _MAP:
        module_name, attr = _MAP[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
