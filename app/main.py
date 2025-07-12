# app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os
import io

app = FastAPI()

# Allow all origins for now (you can restrict this later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = 128
MODEL_PATH = "app/multioutput_model.keras"

# Load model once at startup
model = tf.keras.models.load_model(MODEL_PATH)

# Load encoders
label_encoders = {}
for label in ["gender", "baseColour", "season", "masterCategory"]:
    with open(f"app/LabelEncoders/{label}.pkl", "rb") as f:
        label_encoders[label] = pickle.load(f)

def preprocess_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    contents = await file.read()
    try:
        image_tensor = preprocess_image(io.BytesIO(contents))
        preds = model.predict(image_tensor)

        result = {
            "gender": label_encoders["gender"].inverse_transform([np.argmax(preds[0])])[0],
            "baseColour": label_encoders["baseColour"].inverse_transform([np.argmax(preds[1])])[0],
            "season": label_encoders["season"].inverse_transform([np.argmax(preds[2])])[0],
            "masterCategory": label_encoders["masterCategory"].inverse_transform([np.argmax(preds[3])])[0],
        }

        return result
    except Exception as e:
        return {"error": str(e)}
