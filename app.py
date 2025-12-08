import io
import os
import json
from PIL import Image
from flask_cors import CORS
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
CORS(app)

items = []
daily_funfact = None
daily_funfact_date = None
item_counter = 1

def create_model(num_classes, pretrained=False):
    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)

def load_model(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        class_names = checkpoint.get('class_names')
        if not class_names: return None, None
        
        num_classes = len(class_names)
        model = create_model(num_classes=num_classes)
        
        state_dict = checkpoint['model_state_dict']
        state_dict = {k.replace('module.', ''): v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        model.to(DEVICE)
        print(f"Model loaded on {DEVICE}")
        return model, class_names
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None

model, class_names = load_model(MODEL_PATH)

def translate_label_with_gemini(label_en):
    try:
        prompt = (
            f"Translate this fruit/vegetable name into Indonesian. "
            f"Only return the word without explanation.\n\n{label_en}"
        )
        
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )

        translated = response.text.strip().lower()
        return translated

    except Exception as e:
        print("Translate Error:", e)
        return label_en  # fallback

def prepare_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

def analyze_with_gemini(image_bytes, fruit_name):
    today = datetime.today().strftime("%Y-%m-%d")
    
    prompt_text = (
        f"Analyze this image of a '{fruit_name}'. Today is {today}.\n"
        "Return a JSON object with exactly these 3 keys:\n"
        "1. 'condition': strictly one of ['fresh', 'ripe', 'overripe', 'rotten'] based on visual appearance.\n"
        "2. 'expiryDate': estimate date in 'YYYY-MM-DD' format based on condition.\n"
        "3. 'tips': an array of 3 useful tips related to the storage of the fruit/vegetable in Bahasa Indonesia.\n"
        "If rotten, give disposal tips. If not rotten, give storage tips in Bahasa Indonesia.\n"
        "Example output format:\n"
        "{\"condition\": \"fresh\", \"expiryDate\": \"2023-10-10\", \"tips\": [\"Tip 1\", \"Tip 2\", \"Tip 3\"]}"
    )

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img_format = img.format.lower() if img.format else 'jpeg'
        mime_type = f"image/{img_format if img_format in ['jpeg','png'] else 'jpeg'}"

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_text(text=prompt_text),
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json" # Paksa output JSON
            )
        )

        result = json.loads(response.text)
        
        if result.get("condition") not in ['fresh', 'ripe', 'overripe', 'rotten']:
            result["condition"] = "ripe" 

        return result

    except Exception as e:
        print(f"Gemini Error: {e}")
        return {
            "condition": "ripe",
            "expiryDate": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
            "tips": ["Simpan di tempat sejuk", "Cek berkala", "Segera konsumsi"]
        }


def generate_funfact():
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Berikan 1 fakta unik dalam bentuk satu kalimat singkat tentang buah/sayur"],
        )
        return response.text.strip()
    except:
        return "Sayuran hijau kaya akan zat besi!"

@app.route("/funfact-today", methods=["GET"])
def funfact_today():
    global daily_funfact, daily_funfact_date
    today = datetime.today().date()
    if daily_funfact is None or daily_funfact_date != today:
        daily_funfact = generate_funfact()
        daily_funfact_date = today
    return jsonify({"date": str(today), "funfact": daily_funfact})

@app.route("/classify", methods=["POST"])
def classify_image():
    global item_counter

    if not model or not class_names:
        return jsonify({"error": "Model not loaded"}), 500
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    try:
        processed = prepare_image(image_bytes).to(DEVICE)
        with torch.no_grad():
            output = model(processed)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            pred_idx = torch.argmax(probs).item()
        
        predicted_label = translate_label_with_gemini(class_names[pred_idx])

        ai_analysis = analyze_with_gemini(image_bytes, predicted_label)

        item = {
            "id": str(item_counter), 
            "label": predicted_label,
            "expiryDate": ai_analysis["expiryDate"],
            "condition": ai_analysis["condition"], 
            "tips": ai_analysis["tips"],
            "createdAt": datetime.now().isoformat()
        }
        
        items.append(item)
        item_counter += 1

        return jsonify(item)

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/items", methods=["GET"])
def get_items():
    return jsonify(items)

@app.route("/suggest-recipes", methods=["POST"])
def suggest_recipes():
    data = request.json
    ingredients = data.get("ingredients", [])

    if not ingredients:
        return jsonify({"error": "No ingredients provided"}), 400

    ingredients_str = ", ".join(ingredients)
    prompt_text = (
        f"Saya punya bahan-bahan berikut: {ingredients_str}. "
        "Buatkan 3 saran resep masakan Indonesia sederhana yang bisa dibuat menggunakan bahan utama tersebut. "
        "Prioritaskan bahan yang mungkin cepat busuk. "
        "Berikan output HANYA dalam format JSON valid (tanpa markdown ```json) dengan struktur: "
        "[{'title': 'Nama Masakan', 'missingIngredients': ['Bahan lain yg perlu dibeli'], 'steps': ['Langkah 1', 'Langkah 2']}]"
    )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt_text],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        recipes = json.loads(response.text)
        return jsonify(recipes)

    except Exception as e:
        print("Recipe Gen Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)