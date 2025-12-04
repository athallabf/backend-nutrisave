import io
import os
from PIL import Image
from datetime import datetime, timedelta
from google import genai
from google.genai import types
from dotenv import load_dotenv

import torch
import timm
from flask import Flask, request, jsonify
from torchvision import transforms

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_PATH = "best_fruit_veg_model.pth"
IMAGE_SIZE = 256
MODEL_NAME = 'convnext_tiny.in12k_ft_in1k'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)

def create_model(num_classes, pretrained=False):
    model = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)
    return model

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint.get('class_names')
    if not class_names:
        raise KeyError("Could not find 'class_names' in checkpoint")
    num_classes = len(class_names)
    model = create_model(num_classes=num_classes)
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('module.', ''): v for k,v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    print(f"Model loaded on {DEVICE} with {num_classes} classes.")
    return model, class_names

try:
    model, class_names = load_model(MODEL_PATH)
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None
    class_names = None

def prepare_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

def estimate_condition(image_bytes):
    """Estimate condition based on average brightness + color variance for more variation"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    gray = image.convert("L")
    brightness = sum(gray.getdata()) / (image.width * image.height)
    
    pixels = list(image.getdata())
    avg_r = sum([p[0] for p in pixels])/len(pixels)
    avg_g = sum([p[1] for p in pixels])/len(pixels)
    avg_b = sum([p[2] for p in pixels])/len(pixels)
    variance = (abs(avg_r-avg_g) + abs(avg_r-avg_b) + abs(avg_g-avg_b))/3

    if brightness > 150 and variance > 20:
        return "fresh"
    elif brightness > 120:
        return "ripe"
    elif brightness > 80:
        return "overripe"
    else:
        return "rotten"
    
def get_tips_from_gemini(image_bytes, fruit_name):
    condition = estimate_condition(image_bytes)
    prompt_text = (
        f"Berdasarkan gambar {fruit_name} dengan kondisi '{condition}', "
        "berikan 3 tips singkat terkait PENYIMPANAN saja. "
        "Jangan berikan tips konsumsi, cara makan, atau hal-hal yang tidak relevan "
        "seperti menjaga agar buah tidak terbentur. "
        "Jawab dalam 3 poin bernomor (1, 2, 3) dengan kalimat langsung."
    )

    try:
        img_format = Image.open(io.BytesIO(image_bytes)).format.lower()
        mime_type = f"image/{img_format if img_format in ['jpeg','png'] else 'jpeg'}"
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt_text, image_part]
        )

        raw_text = response.text.strip()

        tips = []
        for line in raw_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() and line[1] == '.' or line[0] in ['-', '*']):
                tip = line[line.index('.')+1:].strip() if '.' in line else line[1:].strip()
                tips.append(tip)

        return tips 
    except Exception as e:
        print(f"Gemini tips error: {e}")
        return "Unable to generate tips."

def get_expiry_from_gemini(image_bytes, fruit_name):
    today = datetime.today().strftime("%Y-%m-%d")
    condition = estimate_condition(image_bytes)
    prompt_text = (
        f"Today is {today}. Based on the image and the detected fruit '{fruit_name}' "
        f"in '{condition}' condition, estimate the expiry date strictly using the rules below:\n\n"
        "- fresh: 5–7 days from today\n"
        "- ripe: 3–5 days from today\n"
        "- overripe: 1–2 days from today\n"
        "- rotten: already expired, return today's date\n\n"
        "Use ONLY the ranges above. Choose one reasonable date inside the correct range.\n"
        "Output only the date in YYYY-MM-DD format with no explanation."
    )
    try:
        img_format = Image.open(io.BytesIO(image_bytes)).format.lower()
        mime_type = f"image/{img_format if img_format in ['jpeg','png'] else 'jpeg'}"
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt_text, image_part]
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

@app.route("/classify", methods=["POST"])
def classify_image():
    if not model or not class_names:
        return jsonify({"error": "Model not loaded"}), 500
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    try:
        image_bytes = image_file.read()
        processed = prepare_image(image_bytes).to(DEVICE)
        with torch.no_grad():
            output = model(processed)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            pred_idx = torch.argmax(probs).item()
        predicted_label = class_names[pred_idx]

        expiry_date = get_expiry_from_gemini(image_bytes, predicted_label)
        tips = get_tips_from_gemini(image_bytes, predicted_label)

        return jsonify({
            "label": predicted_label,
            "expiryDate": expiry_date,
            "tips": tips

        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
