import os
import sys

# Ensure root directory is in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.optim as optim

import config
from src.models.builder import build_model
from src.data.dataloader import create_dataloaders
from src.data.dataset import prepare_flower_dataset

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15, device=config.DEVICE):
    """
    Trains PyTorch AlexNet model and saves best checkpoint based on validation accuracy.
    """
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    print(f"Starting training on device: {device}")
    print("-" * 50)

    for epoch in range(num_epochs):
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

        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'classes': config.TARGET_CLASSES
            }, config.MODEL_PATH)
            print(f"  --> Saved best model checkpoint (Val Acc: {best_val_acc*100:.2f}%) to {config.MODEL_PATH}")

    print("-" * 50)
    print(f"Training completed! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    return history


def main():
    prepare_flower_dataset()
    train_loader, val_loader, _, class_names = create_dataloaders()
    print(f"Loaded dataset classes: {class_names}")

    model = build_model(
        use_pretrained=config.USE_PRETRAINED,
        num_classes=len(class_names),
        device=config.DEVICE
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.NUM_EPOCHS,
        device=config.DEVICE
    )

if __name__ == '__main__':
    main()
