import torch

import config
from src.models.alexnet import AlexNetFromScratch, get_alexnet_pretrained

# Key không phải metadata mô tả model -> không trả về trong `metadata`.
# `optimizer_state_dict` (trạng thái Adam) nặng gấp ~3 lần trọng số; nếu để lọt
# vào `metadata` thì nó bị giữ trong RAM suốt vòng đời của FlowerPredictor
# (app web giữ metadata của model) dù không bao giờ được dùng lại — dự án
# không hỗ trợ resume training.
_NON_METADATA_KEYS = ('model_state_dict', 'optimizer_state_dict')


def build_model(use_pretrained=False, num_classes=3, device='cpu'):
    """
    Build and return AlexNet model on specified device.
    """
    if use_pretrained:
        model = get_alexnet_pretrained(num_classes=num_classes)
    else:
        model = AlexNetFromScratch(num_classes=num_classes)
    return model.to(device)


def load_checkpoint_data(model_path, device='cpu'):
    """
    Đọc file checkpoint và trả về `(state_dict, metadata)`.

    Hỗ trợ 2 định dạng để tương thích ngược:
      - Checkpoint mới: dict chứa 'model_state_dict' + metadata (epoch, val_acc, classes, ...)
      - Checkpoint cũ: chính là state_dict thuần.

    `weights_only=False` được chỉ định TƯỜNG MINH vì checkpoint chứa cả metadata
    Python (list/dict) và để hành vi không đổi khi torch >= 2.6 đổi mặc định sang `True`.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        metadata = {k: v for k, v in checkpoint.items() if k not in _NON_METADATA_KEYS}
    else:
        state_dict = checkpoint
        metadata = {}

    return state_dict, metadata


def load_trained_model(model_path, device='cpu', default_classes=None):
    """
    Load mô hình đã huấn luyện kèm metadata và danh sách lớp.

    Kiến trúc và thứ tự lớp được lấy TỪ CHECKPOINT (không phải từ config) để
    tránh lỗi khi `config.USE_PRETRAINED` thay đổi sau lúc huấn luyện.

    Args:
        model_path: Đường dẫn file .pth
        device: Thiết bị để load model
        default_classes: Danh sách lớp dùng khi checkpoint cũ không lưu 'classes'

    Returns:
        tuple: (model đã ở chế độ eval, metadata, class_names)
    """
    state_dict, metadata = load_checkpoint_data(model_path, device)

    class_names = list(metadata.get('classes') or default_classes or config.TARGET_CLASSES)
    use_pretrained = bool(metadata.get('use_pretrained', config.USE_PRETRAINED))

    model = build_model(
        use_pretrained=use_pretrained,
        num_classes=len(class_names),
        device=device,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model, metadata, class_names
