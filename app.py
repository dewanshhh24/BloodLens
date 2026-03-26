from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask_cors import CORS
from io import BytesIO


app = Flask(__name__)
CORS(app)


# Load models
cell_model = load_model('cell_model.keras')
cancer_model = load_model('cancer_model.keras')

cell_classes = ['basophil','eosinophil','erythroblast','ig','lymphocyte','monocyte','neutrophil','platelet']
cancer_classes = ['benign','early_pre_b','pre_b','pro_b']

@app.route('/predict', methods=['POST'])
def predict():
    # ✅ Safety check
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    # ✅ Extra safety (empty file)
    if file.filename == '':
        return jsonify({"error": "Empty file"}), 400

    try:
        # Process image
        img = image.load_img(BytesIO(file.read()), target_size=(224,224))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predictions
        cell_pred = cell_model.predict(img_array)
        cancer_pred = cancer_model.predict(img_array)

        cell_label = cell_classes[np.argmax(cell_pred)]
        cancer_label = cancer_classes[np.argmax(cancer_pred)]

        cell_conf = float(np.max(cell_pred))*100
        cancer_conf = float(np.max(cancer_pred))*100

        return jsonify({
            "cell": cell_label,
            "cancer": cancer_label,
            "cell_conf": round(cell_conf, 2),
            "cancer_conf": round(cancer_conf, 2)
        })

    except Exception as e:
        print("ERROR:", str(e))  
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)