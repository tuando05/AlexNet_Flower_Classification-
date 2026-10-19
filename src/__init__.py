"""
Package gốc của dự án.

Chủ đích để trống (chỉ có docstring): trước đây file này re-export
`FlowerPredictor`, `train_model`, `build_model`... bằng lazy import, nhưng
KHÔNG nơi nào trong dự án dùng `from src import ...` — mọi module đều import
trực tiếp (`from src.data.dataloader import create_dataloaders`).

Giữ `__init__.py` rỗng còn tránh được việc nạp sẵn các module nặng (torch,
torchvision) mỗi khi chỉ cần chạm vào package `src`.
"""
