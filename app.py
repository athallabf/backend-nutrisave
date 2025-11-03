import io
import os
import json
from PIL import Image

import torch
import timm
from flask import Flask, request, jsonify
from torchvision import transforms


MODEL_PATH = "best_fruit_veg_model.pth"

IMAGE_SIZE = 256

MODEL_NAME = 'convnext_tiny.in12k_ft_in1k'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)


def create_model(num_classes, pretrained=False):
    """Creates a ConvNeXt model from the timm library."""
    # We set pretrained=False because we are loading our own fine-tuned weights.
    model = timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model

def load_model(model_path):
    """Loads the trained model, its weights, and class names."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    class_names = checkpoint.get('class_names')
    if not class_names:
        raise KeyError("Could not find 'class_names' in the model checkpoint.")
    
    num_classes = len(class_names)
    
    model = create_model(num_classes=num_classes)
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    
    model.eval()
    
    model = model.to(DEVICE)
    
    print(f"Model loaded successfully and moved to {DEVICE}.")
    print(f"Found {num_classes} classes.")
    
    return model, class_names

try:
    model, class_names = load_model(MODEL_PATH)
except Exception as e:
    print(f"CRITICAL ERROR: Could not load the model.")
    print(f"Details: {e}")
    model = None
    class_names = None


def prepare_image(image_bytes):
    """
    Prepares a raw image bytes for model prediction.
    - Resizes the image to the target size.
    - Converts it to a PyTorch Tensor.
    - Normalizes pixel values using ImageNet stats.
    - Adds a batch dimension.
    """

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0) # Add batch dimension
    return image_tensor



@app.route("/classify", methods=["POST"])
def classify_image():
    """
    Receives an image, passes it to the model, and returns
    the predicted label as JSON.
    """
    if not model or not class_names:
        return jsonify({"error": "Server error: Model is not loaded."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No 'image' file provided in the request."}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No image file selected."}), 400

    try:
        image_bytes = image_file.read()
        processed_image = prepare_image(image_bytes).to(DEVICE)
        
        with torch.no_grad(): 
            output = model(processed_image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()

        predicted_label = class_names[predicted_idx]

        response = {
            "label": predicted_label
        }
        
        return jsonify(response)

    except Exception as e:
        print(f"ERROR during classification: {e}")
        return jsonify({"error": f"An error occurred during classification: {e}"}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
