from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import pickle

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Preprocess
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Load model
    model = tf.keras.models.load_model("Model/best_model.h5")

    # Load label encoders
    with open("encoders/gender.pkl", "rb") as f:
        gender_enc = pickle.load(f)
    with open("encoders/color.pkl", "rb") as f:
        color_enc = pickle.load(f)
    with open("encoders/season.pkl", "rb") as f:
        season_enc = pickle.load(f)
    with open("encoders/product.pkl", "rb") as f:
        product_enc = pickle.load(f)

    # Predict
    preds = model.predict(image)
    gender = gender_enc.inverse_transform([np.argmax(preds[0])])[0]
    color = color_enc.inverse_transform([np.argmax(preds[1])])[0]
    season = season_enc.inverse_transform([np.argmax(preds[2])])[0]
    product = product_enc.inverse_transform([np.argmax(preds[3])])[0]

    return {
        "gender": gender,
        "baseColour": color,
        "season": season,
        "masterCategory": product
    }
