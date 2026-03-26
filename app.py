import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask_cors import CORS
from io import BytesIO
import gdown
import os

app = Flask(__name__)
CORS(app)

# ==================== MODEL SETUP ====================

CELL_ID = "1EfOd-sju9QZ6FAPxuwxwQfFQoC6ALvGk"
CANCER_ID = "1WoTesKl-Fn8OVHG6d9qCUhw2S5wE5B7h"

cell_path = "cell_model.h5"
cancer_path = "cancer_model.h5"

# Download models
def download_model(file_id, output):
    if not os.path.exists(output):
        print(f"Downloading {output}...")
        gdown.download(id=file_id, output=output, quiet=False, fuzzy=True)
        print(f"{output} downloaded successfully.")

download_model(CELL_ID, cell_path)
download_model(CANCER_ID, cancer_path)

# ==================== LOAD MODELS ====================

try:
    cell_model = load_model(cell_path, compile=False)
    cancer_model = load_model(cancer_path, compile=False)
    print("Models loaded successfully ✅")
except Exception as e:
    print("Model loading failed:", e)
    cell_model = None
    cancer_model = None

# ==================== CLASSES ====================

cell_classes = [
    'basophil','eosinophil','erythroblast','ig',
    'lymphocyte','monocyte','neutrophil','platelet'
]

cancer_classes = [
    'benign','early_pre_b','pre_b','pro_b'
]

# ==================== ROUTES ====================

@app.route('/')
def home():
    return "BloodLens API is running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    if cell_model is None or cancer_model is None:
        return jsonify({"error": "Models not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "Empty file"}), 400

    try:
        img = image.load_img(BytesIO(file.read()), target_size=(224,224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        cell_pred = cell_model.predict(img_array)
        cancer_pred = cancer_model.predict(img_array)

        cell_label = cell_classes[np.argmax(cell_pred)]
        cancer_label = cancer_classes[np.argmax(cancer_pred)]

        cell_conf = float(np.max(cell_pred)) * 100
        cancer_conf = float(np.max(cancer_pred)) * 100

        return jsonify({
            "cell": cell_label,
            "cancer": cancer_label,
            "cell_conf": round(cell_conf, 2),
            "cancer_conf": round(cancer_conf, 2)
        })

    except Exception as e:
        print("Prediction ERROR:", str(e))
        return jsonify({"error": "Prediction failed"}), 500

# ==================== RUN ====================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
