from src.models.alexnet import AlexNetFromScratch, get_alexnet_pretrained

def build_model(use_pretrained=False, num_classes=3, device='cpu'):
    """
    Build and return AlexNet model on specified device.
    """
    if use_pretrained:
        model = get_alexnet_pretrained(num_classes=num_classes)
    else:
        model = AlexNetFromScratch(num_classes=num_classes)
    return model.to(device)
