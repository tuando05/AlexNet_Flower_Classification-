import os
import sys

# Ensure root directory is in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
import torch.nn.functional as F
from PIL import Image

import config
from src.models.builder import build_model
from src.data.transforms import get_transforms

class FlowerPredictor:
    """
    Inference helper class for flower image classification.
    """
    def __init__(self, model_path=config.MODEL_PATH, device=config.DEVICE):
        self.device = device
        self.class_names = config.TARGET_CLASSES
        self.class_names_vi = config.CLASS_NAMES_VI
        _, self.transform = get_transforms()

        self.model = build_model(
            use_pretrained=config.USE_PRETRAINED,
            num_classes=len(self.class_names),
            device=self.device
        )

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded trained model checkpoint from: {model_path}")
        else:
            print(f"Warning: Checkpoint '{model_path}' not found! Model will use un-trained random weights.")

        self.model.eval()

    def predict_image(self, image_input):
        """
        Accepts PIL Image object, file path, or bytes, and returns top prediction & confidence scores.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise ValueError("image_input must be a file path string or PIL Image object.")

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            top_prob, top_catid = torch.topk(probabilities, 1)

        predicted_class = self.class_names[top_catid.item()]
        predicted_class_vi = self.class_names_vi.get(predicted_class, predicted_class)
        confidence = float(top_prob.item())

        all_probabilities = {
            self.class_names[i]: {
                'name_en': self.class_names[i],
                'name_vi': self.class_names_vi.get(self.class_names[i], self.class_names[i]),
                'probability': round(float(probabilities[i].item()), 4),
                'percentage': round(float(probabilities[i].item()) * 100, 2)
            }
            for i in range(len(self.class_names))
        }

        return {
            'prediction': predicted_class,
            'prediction_vi': predicted_class_vi,
            'confidence': confidence,
            'confidence_percentage': round(confidence * 100, 2),
            'probabilities': all_probabilities
        }


if __name__ == '__main__':
    predictor = FlowerPredictor()
    import numpy as np
    dummy_img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
    result = predictor.predict_image(dummy_img)
    print("Test Prediction Output:", result)
