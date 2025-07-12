from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.model_utils import preprocess_image, predict

app = FastAPI(title="Fashion Product Classifier API")

# CORS if frontend hosted separately
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Fashion Multi-label Classifier API is up!"}

@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image_tensor = preprocess_image(img_bytes)
    results = predict(image_tensor)
    return results