import os
import torch

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data & Model Paths
DATA_DIR = os.path.join(BASE_DIR, "flower_data_3classes")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_alexnet_flowers.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SAMPLE_DIR = os.path.join(BASE_DIR, "samples")

# Dataset Configuration
TARGET_CLASSES = ['daisy', 'lily', 'rose']
CLASS_NAMES_VI = {
    'daisy': 'Hoa Cúc (Daisy)',
    'lily': 'Hoa Ly (Lily)',
    'rose': 'Hoa Hồng (Rose)'
}

# Hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = len(TARGET_CLASSES)
NUM_EPOCHS = 15
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4

# Model Selection
# Set USE_PRETRAINED = True for Transfer Learning, False for AlexNet from Scratch
USE_PRETRAINED = False

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet Normalization Stats
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
