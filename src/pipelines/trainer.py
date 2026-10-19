import json
import logging
import os
import sys
import time
from datetime import datetime

# Ensure root directory is in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils.console import setup_console
from src.utils.logger import setup_logging
from src.utils.plotting import configure_matplotlib_backend

configure_matplotlib_backend()
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import config
from src.models.builder import build_model
from src.data.dataloader import create_dataloaders
from src.utils.seed import set_seed

logger = logging.getLogger('src.pipelines.trainer')


def save_training_artifacts(history, class_names):
    """
    Lưu lịch sử huấn luyện (JSON) và biểu đồ loss/accuracy của quá trình train.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    history_path = os.path.join(config.OUTPUT_DIR, "history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    logger.info("Đã lưu lịch sử huấn luyện: %s", history_path)

    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, history['train_loss'], marker='o', label='Train Loss')
    ax_loss.plot(epochs, history['val_loss'], marker='o', label='Val Loss')
    ax_loss.set_title('Loss theo Epoch')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, [a * 100 for a in history['train_acc']], marker='o', label='Train Acc')
    ax_acc.plot(epochs, [a * 100 for a in history['val_acc']], marker='o', label='Val Acc')
    ax_acc.set_title('Accuracy theo Epoch')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.suptitle(f'Quá trình huấn luyện AlexNet - Lớp: {", ".join(class_names)}')
    fig.tight_layout()

    curves_path = os.path.join(config.OUTPUT_DIR, "training_curves.png")
    fig.savefig(curves_path, dpi=120)
    plt.close(fig)
    logger.info("Đã lưu biểu đồ huấn luyện: %s", curves_path)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15,
                device=config.DEVICE, early_stopping_patience=config.EARLY_STOPPING_PATIENCE):
    """
    Trains PyTorch AlexNet model and saves best checkpoint based on validation accuracy.

    Args:
        early_stopping_patience: Số epoch liên tiếp không cải thiện val_acc thì dừng sớm.
            Đặt None hoặc 0 để tắt early stopping.

    Returns:
        dict: history gồm train_loss, train_acc, val_loss, val_acc theo từng epoch.
    """
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    logger.info("Bắt đầu huấn luyện trên thiết bị: %s", device)
    logger.info("Epochs: %d | Early stopping patience: %s", num_epochs, early_stopping_patience)
    logger.info("-" * 60)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = (running_corrects.double() / len(train_loader.dataset)).item()

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_val_loss += loss.item() * inputs.size(0)
                running_val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = (running_val_corrects.double() / len(val_loader.dataset)).item()

        if scheduler:
            scheduler.step(epoch_val_loss)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        elapsed = time.time() - epoch_start
        logger.info(
            "Epoch %02d/%02d | Train Loss: %.4f | Train Acc: %.2f%% | "
            "Val Loss: %.4f | Val Acc: %.2f%% | Time: %.1fs",
            epoch + 1, num_epochs,
            epoch_train_loss, epoch_train_acc * 100,
            epoch_val_loss, epoch_val_acc * 100,
            elapsed,
        )

        # Chỉ lưu khi thực sự tốt hơn epoch trước (tránh ghi đè liên tục)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'classes': list(train_loader.dataset.classes),
                'arch': 'alexnet',
                'use_pretrained': config.USE_PRETRAINED,
                'num_classes': len(train_loader.dataset.classes),
                'image_size': config.IMAGE_SIZE,
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'trained_at': datetime.now().isoformat(timespec='seconds'),
            }, config.MODEL_PATH)
            logger.info(
                "  --> Đã lưu checkpoint tốt nhất (Val Acc: %.2f%%) vào %s",
                best_val_acc * 100, config.MODEL_PATH,
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                "  --> Không cải thiện (%d/%s epoch liên tiếp)",
                epochs_without_improvement, early_stopping_patience or '-',
            )

        # Early stopping
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            logger.info("  --> EARLY STOPPING tại epoch %d (best epoch: %d)", epoch + 1, best_epoch)
            break

    logger.info("-" * 60)
    logger.info("Huấn luyện xong! Val Acc tốt nhất: %.2f%% (epoch %d)", best_val_acc * 100, best_epoch)
    return history


def main():
    setup_console()
    setup_logging()

    set_seed(config.SEED)

    # create_dataloaders() tự chuẩn bị dataset và raise lỗi nếu thiếu dữ liệu
    train_loader, val_loader, _, class_names = create_dataloaders()
    logger.info("Các lớp của dataset: %s", class_names)

    model = build_model(
        use_pretrained=config.USE_PRETRAINED,
        num_classes=len(class_names),
        device=config.DEVICE
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.NUM_EPOCHS,
        device=config.DEVICE
    )

    save_training_artifacts(history, class_names)


if __name__ == '__main__':
    main()
