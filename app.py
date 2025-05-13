from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model and label dictionary
model = joblib.load('model/plant_disease_multiclass_model.pkl')
label_dict = joblib.load('model/label_dict.pkl')

def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    # Assuming you did not convert to RGB during training
    hist = cv2.calcHist([image], [0, 1, 2], None,
                        [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_filename = None

    if request.method == 'POST':
        file = request.files.get('leaf_image')
        if file:
            image_filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            file.save(filepath)

            features = extract_features(filepath)
            prediction = model.predict([features])[0]
            label = label_dict[prediction]

            print("Predicted inde:", prediction)
            print("Predicted labe:", label)

            # More reliable check for healthy label
            if label.lower().endswith('___healthy'):
                result = "Health Leaf ✅"
            else:
                result = "Diseased Leaf ❌"

    return render_template('index.html', result=result, image_filename=image_filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
