import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Initialize Flask app
app = Flask(__name__)

# Load model and class indices
model_path = './model/eye_disease_model.h5'
class_indices_path = './model/class_indices.json'

model = load_model(model_path)
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Reverse the mapping to get class labels
class_labels = {v: k for k, v in class_indices.items()}

# Function to preprocess the image and make predictions
def predict_eye_disease(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    return class_labels[predicted_class_index]

# Home page with upload functionality
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file!"}), 400

    if file:
        file_path = os.path.join('./uploads', file.filename)
        file.save(file_path)  # Save the file

        # Make prediction
        predicted_label = predict_eye_disease(file_path)
        os.remove(file_path)  # Clean up uploaded file after prediction
        return jsonify({"prediction": predicted_label})

if __name__ == '__main__':
    # Create 'uploads' folder if not exists
    if not os.path.exists('./uploads'):
        os.makedirs('./uploads')

    app.run(debug=True)
