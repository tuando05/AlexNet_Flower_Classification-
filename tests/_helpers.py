"""
Tiện ích dùng chung cho test.

- `TinyModel`: thay thế AlexNet trong test để không phải load checkpoint ~240MB,
  nhờ đó bộ test chạy nhanh và không phụ thuộc GPU/checkpoint thật.
- `make_images`: tạo file rỗng giả lập ảnh, dùng bởi test downloader/dataloader.

Hai tiện ích này trước đây bị định nghĩa lặp lại ở từng file test.
"""

import os

import torch.nn as nn


class TinyModel(nn.Module):
    """Mô hình tối giản: global average pooling + 1 lớp Linear."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x):
        pooled = x.mean(dim=(2, 3))   # (N, 3, H, W) -> (N, 3)
        return self.fc(pooled)


def make_images(directory, *names):
    """
    Tạo các file rỗng với tên cho trước trong `directory` (giả lập ảnh).

    Nội dung file không quan trọng vì test chỉ kiểm tra sự tồn tại và phần mở
    rộng tên file, không đọc dữ liệu ảnh thật.
    """
    os.makedirs(directory, exist_ok=True)
    for name in names:
        with open(os.path.join(directory, name), 'wb') as f:
            f.write(b'x')
