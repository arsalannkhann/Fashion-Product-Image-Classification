import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os

IMG_SIZE = 128
MODEL_PATH = "app/multioutput_model.keras"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load label encoders
label_encoders = {}
for label in ["gender", "baseColour", "season", "masterCategory"]:
    with open(f"app/label_encoder_{label}.pkl", "rb") as f:
        label_encoders[label] = pickle.load(f)

def preprocess_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def predict(image_tensor):
    preds = model.predict(image_tensor)
    output = {
        "gender": label_encoders["gender"].inverse_transform([np.argmax(preds[0])])[0],
        "baseColour": label_encoders["baseColour"].inverse_transform([np.argmax(preds[1])])[0],
        "season": label_encoders["season"].inverse_transform([np.argmax(preds[2])])[0],
        "masterCategory": label_encoders["masterCategory"].inverse_transform([np.argmax(preds[3])])[0],
    }
    return output
