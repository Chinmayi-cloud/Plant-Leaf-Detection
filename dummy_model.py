import joblib
import numpy as np
from PIL import Image
import os

model_path = os.path.join(os.path.dirname(__file__), 'plant_disease_multiclass_model.pkl')
label_path = os.path.join(os.path.dirname(__file__), 'label_dict.pkl')

model = joblib.load(model_path)
label_dict = joblib.load(label_path)
label_inverse = {v: k for k, v in label_dict.items()}

def predict_leaf_disease(image_path):
    try:
        image = Image.open(image_path).resize((128, 128))
        image = np.array(image)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        image = image.flatten().reshape(1, -1)
        prediction = model.predict(image)
        predicted_label = label_inverse[prediction[0]]
        return f"Predicted Disease: {predicted_label}"
    except Exception as e:
        return f"Error: {str(e)}"
