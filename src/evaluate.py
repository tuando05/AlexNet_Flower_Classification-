import os
import sys

# Ensure root directory is in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import config
from src.model import build_model
from src.dataset import create_dataloaders

def evaluate_model():
    """
    Evaluates saved AlexNet model on the test dataset and prints evaluation metrics.
    """
    if not os.path.exists(config.MODEL_PATH):
        print(f"Error: Model checkpoint not found at {config.MODEL_PATH}. Please train the model first.")
        return

    _, _, test_loader, class_names = create_dataloaders()
    model = build_model(use_pretrained=config.USE_PRETRAINED, num_classes=len(class_names), device=config.DEVICE)

    checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

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

    print("\n================ BÁO CÁO ĐÁNH GIÁ TẬP TEST ================")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Tập Test')
    plt.xlabel('Nhãn Dự Đoán (Predicted)')
    plt.ylabel('Nhãn Thực Thể (True)')
    plt.tight_layout()
    
    save_path = os.path.join(config.BASE_DIR, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Lưu Confusion Matrix thành công tại: {save_path}")
    plt.close()

if __name__ == '__main__':
    evaluate_model()
