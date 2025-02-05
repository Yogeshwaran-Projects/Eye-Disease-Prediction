import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Path to the saved model and class indices
model_path = './model/eye_disease_model.h5'
class_indices_path = './model/class_indices.json'  # Ensure this file exists and was saved during training

# Load the saved model
model = load_model(model_path)

# Load the saved class indices
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Reverse the mapping to get class labels
class_labels = {v: k for k, v in class_indices.items()}

# Function to preprocess the image and make predictions
def predict_eye_disease(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(256, 256))  # Resize to match input shape
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image (specific to model)

    # Make prediction
    prediction = model.predict(img_array)
    
    # Get class index with maximum prediction probability
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Get class label from the saved class labels
    predicted_class_label = class_labels[predicted_class_index]
    
    return predicted_class_label

# Example: Predict using an image
image_path = '/Users/yogeshdevil/Desktop/eyescheck/ocular-disease/data/dataset/diabetic_retinopathy/1008_right.jpeg'  # Replace with your test image path
try:
    predicted_label = predict_eye_disease(image_path)
    print(f"Predicted Eye Disease: {predicted_label}")
except Exception as e:
    print(f"Error: {e}")


"""
/Users/yogeshdevil/Desktop/eyescheck/ocular-disease/data/dataset/cataract/_13_3987009.jpg - diabetic_retinopathy
/Users/yogeshdevil/Desktop/eyescheck/2108_right.jpg - glaucoma
/Users/yogeshdevil/Desktop/eyescheck/2608_right.jpg - cataract

"""