import os
from dataset_loader import EyeDiseaseDataset
from utils import augment_data
from model import create_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

data_dir = './data/dataset'

# Load dataset
dataset = EyeDiseaseDataset(data_dir)
train_df, valid_df, test_df = dataset.split_()

# Augment data
train_gen, valid_gen, test_gen = augment_data(train_df, valid_df, test_df)

input_shape = (256, 256, 3)
num_classes = len(train_gen.class_indices)

# Check if model already exists and load it
model_path = './model/eye_disease_model.h5'
if os.path.exists(model_path):
    print("Loading saved model...")
    model = load_model(model_path)
else:
    print("Creating a new model...")
    model = create_model(input_shape, num_classes)

# Callbacks to save the best model and prevent overfitting
callbacks = [
    ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
]

# Train the model
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=15,
    callbacks=callbacks
)

# Save model after training if not already saved by ModelCheckpoint
if not os.path.exists(model_path):
    model.save(model_path)

# Optionally: Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
