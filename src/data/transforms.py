import torchvision.transforms as transforms
import config

def get_transforms():
    """
    Returns image transformation pipelines for training and validation/testing.
    """
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])

    return train_transforms, val_test_transforms
