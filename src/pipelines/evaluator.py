import os
import sys

# Ensure root directory is in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils.console import setup_console

setup_console()

import matplotlib
matplotlib.use('Agg')  # Script chỉ lưu hình, không cần cửa sổ hiển thị

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import config
from src.models.builder import load_trained_model
from src.data.dataloader import create_dataloaders


def evaluate_model():
    """
    Evaluates saved AlexNet model on the test dataset and prints evaluation metrics.

    Returns:
        dict | None: {'accuracy': float, 'macro_f1': float, 'class_names': list}
                     hoặc None nếu chưa có checkpoint.
    """
    if not os.path.exists(config.MODEL_PATH):
        print(f"Error: Model checkpoint not found at {config.MODEL_PATH}. Please train the model first.")
        return None

    _, _, test_loader, loader_classes = create_dataloaders()

    model, metadata, class_names = load_trained_model(
        model_path=config.MODEL_PATH,
        device=config.DEVICE,
        default_classes=loader_classes,
    )

    if metadata:
        print(
            f"Checkpoint info: epoch={metadata.get('epoch')} | "
            f"val_acc={metadata.get('val_acc')} | arch={metadata.get('arch')} | "
            f"use_pretrained={metadata.get('use_pretrained')}"
        )

    if list(class_names) != list(loader_classes):
        print(
            "[WARN] Thứ tự lớp trong checkpoint khác thứ tự của ImageFolder "
            f"({class_names} vs {loader_classes}) -> kết quả đánh giá có thể bị sai lệch."
        )

    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating model on test dataset...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=loader_classes)

    print("\n================ BÁO CÁO ĐÁNH GIÁ TẬP TEST ================")
    print(report)

    accuracy = float((np.array(all_preds) == np.array(all_labels)).mean())
    macro_f1 = float(f1_score(all_labels, all_preds, average='macro'))
    print(f"Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=loader_classes, yticklabels=loader_classes)
    plt.title('Confusion Matrix - Tập Test')
    plt.xlabel('Nhãn Dự Đoán (Predicted)')
    plt.ylabel('Nhãn Thực Thể (True)')
    plt.tight_layout()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Lưu Confusion Matrix thành công tại: {save_path}")
    plt.close()

    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'class_names': list(loader_classes)}


if __name__ == '__main__':
    evaluate_model()
