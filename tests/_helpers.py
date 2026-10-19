"""
Tiện ích dùng chung cho test.

`TinyModel` thay thế AlexNet trong test để không phải load checkpoint ~240MB,
nhờ đó bộ test chạy nhanh và không phụ thuộc GPU/checkpoint thật.
"""

import torch.nn as nn


class TinyModel(nn.Module):
    """Mô hình tối giản: global average pooling + 1 lớp Linear."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x):
        pooled = x.mean(dim=(2, 3))   # (N, 3, H, W) -> (N, 3)
        return self.fc(pooled)
