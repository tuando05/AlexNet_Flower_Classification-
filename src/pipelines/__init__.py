"""
Export các pipeline cấp cao.

Dùng lazy import (PEP 562) để tránh việc module bị nạp sẵn khi chạy
`python -m src.pipelines.<module>`, vốn gây RuntimeWarning từ runpy.
"""

import importlib

__all__ = ['train_model', 'evaluate_model', 'FlowerPredictor']

_MAP = {
    'train_model': ('src.pipelines.trainer', 'train_model'),
    'evaluate_model': ('src.pipelines.evaluator', 'evaluate_model'),
    'FlowerPredictor': ('src.pipelines.predictor', 'FlowerPredictor'),
}


def __getattr__(name):
    if name in _MAP:
        module_name, attr = _MAP[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
