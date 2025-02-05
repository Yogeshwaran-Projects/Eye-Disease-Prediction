from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Path to your training directory
train_dir = './data/dataset'  # Replace with the path to your training data

# Recreate the ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Use the same preprocessing as during training

# Generate the training data
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),  # Ensure this matches your model's input size
    batch_size=32,
    class_mode='categorical'
)

# Save class_indices
class_indices_path = './model/class_indices.json'
with open(class_indices_path, 'w') as f:
    json.dump(train_generator.class_indices, f)

print(f"class_indices saved to {class_indices_path}")
