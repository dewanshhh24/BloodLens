import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np
from flask_cors import CORS
from io import BytesIO
import gdown
import os

app = Flask(__name__)
CORS(app)

# ==================== MODEL FILE IDs ====================

CELL_JSON_ID = "1qeUVGsun8JBAzHcisMZREs7biwwK60J-"
CELL_WEIGHTS_ID = "1swz-wxFoW4zb5yvPpleoNlv7oX5qcUKo"

CANCER_JSON_ID = "1XvM3D2Jw1_ohOKAck1qazIGN226Sw6f3"
CANCER_WEIGHTS_ID = "105wA4UCS8nUJkXHNIXvEjgQoUN807WHZ"

# ==================== FILE PATHS ====================

cell_json_path = "cell_model.json"
cell_weights_path = "cell_weights.weights.h5"

cancer_json_path = "cancer_model.json"
cancer_weights_path = "cancer_weights.weights.h5"

# ==================== DOWNLOAD FUNCTION ====================

def download_file(file_id, output):
    if not os.path.exists(output):
        print(f"Downloading {output}...")
        try:
            gdown.download(id=file_id, output=output, quiet=False, fuzzy=True)
            print(f"{output} downloaded successfully.")
        except Exception as e:
            print(f"Download failed: {e}")

# Download all files
download_file(CELL_JSON_ID, cell_json_path)
download_file(CELL_WEIGHTS_ID, cell_weights_path)

download_file(CANCER_JSON_ID, cancer_json_path)
download_file(CANCER_WEIGHTS_ID, cancer_weights_path)

# ==================== LOAD MODELS ====================


from tensorflow.keras.models import model_from_json
import json

def load_model_from_files(json_path, weights_path):
    try:
        with open(json_path, "r") as f:
            config = json.load(f)

        # 🔥 FIX INPUT LAYER CONFIG SAFELY
        for layer in config['config']['layers']:
            if layer['class_name'] == 'InputLayer':
                layer_config = layer['config']

                # Fix batch_shape → batch_input_shape
                if 'batch_shape' in layer_config:
                    layer_config['batch_input_shape'] = layer_config.pop('batch_shape')

                # Remove unsupported key
                if 'optional' in layer_config:
                    layer_config.pop('optional')

        # Convert back to JSON string
        model_json = json.dumps(config)

        model = model_from_json(model_json)
        model.load_weights(weights_path)

        print(f"Loaded model from {json_path} ✅")
        return model

    except Exception as e:
        print("Model loading failed:", e)
        return None

cell_model = load_model_from_files(cell_json_path, cell_weights_path)
cancer_model = load_model_from_files(cancer_json_path, cancer_weights_path)

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

        # Predictions
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
