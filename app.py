import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load models
cell_model = load_model("cell_model.h5", compile=False)
cancer_model = load_model("cancer_model.h5", compile=False)

cell_classes = ['basophil','eosinophil','erythroblast','ig',
                'lymphocyte','monocyte','neutrophil','platelet']

cancer_classes = ['benign','early_pre_b','pre_b','pro_b']

def predict(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    cell_pred = cell_model.predict(img)
    cancer_pred = cancer_model.predict(img)

    return {
        "Cell Type": cell_classes[np.argmax(cell_pred)],
        "Cancer Type": cancer_classes[np.argmax(cancer_pred)]
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="BloodLens"
)

demo.launch()
