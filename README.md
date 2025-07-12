Here’s your polished README.md—ready to paste into your GitHub repo. Scroll down further for instructions to generate a matching PDF version.

⸻


# 🧠 Fashion Product Image Classification

A multi-output deep learning application that classifies fashion product images by predicting:

- 👕 **Gender**  
- 🎨 **Base Color**  
- ❄️ **Season**  
- 🛍️ **Product Category**

This project uses **TensorFlow**, **FastAPI**, **Docker**, and **Streamlit** to serve an end-to-end deployable machine learning system.

---

## 📁 Dataset

- **Source**: [Kaggle – Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)  
- **Size**: ~24 GB  
- **Files used**: `styles.csv` for metadata + `images/` folder for visuals

---

## 🧪 Model Overview

- **Backbone**: `MobileNetV2` (pretrained on ImageNet)  
- **Architecture**: Shared CNN → 4 output heads (gender, baseColour, season, masterCategory)  
- **Training**:  
  - Loss: `sparse_categorical_crossentropy` per output  
  - Metrics: `accuracy` per output  
  - Callbacks: `EarlyStopping`, `ModelCheckpoint`  
- **Output formats**: Saved as `.h5` / SavedModel + converted to TFLite

---

## 🛠️ Code Structure

fashion-product-image-classification/
├── Model/                     # Trained model & label encoders
│   ├── best_model.h5
│   ├── gender_encoder.pkl
│   ├── color_encoder.pkl
│   ├── season_encoder.pkl
│   └── product_encoder.pkl
│
├── api/                       # Backend (FastAPI)
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── streamlit_app/             # Frontend
│   ├── streamlit_app.py
│   └── requirements.txt
│
├── environment.yml           # Conda environment setup
└── README.md                 # Documentation you’re reading

---

## 🚀 Quick Start

### ✅ Backend

##### Locally
```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload

Docker

cd api
docker build -t fashion-api .
docker run -p 8000:8000 fashion-api

API Endpoints:
	•	GET  / → Health check
	•	POST /predict → Accepts multipart/form-data (key = file, value = image)

⸻

✅ Frontend (Streamlit)

cd streamlit_app
pip install -r requirements.txt
streamlit run streamlit_app.py

Uploads an image and displays predictions using the backend API.

⸻

🔧 Conda Environment Setup

Use this to avoid wheel errors:

conda env create -f environment.yml
conda activate fashion-ml-api
cd api && pip install -r requirements.txt
cd ../streamlit_app && pip install -r requirements.txt


⸻

✅ How to Test
	1.	Postman / curl example:

curl -X POST https://<your-backend-url>/predict \
     -F "file=@/path/to/test.jpg"


	2.	Streamlit app:
	•	Upload image
	•	View raw JSON response and displayed predictions

⸻

🧩 Example Output

Sample	Gender	Color	Season	Category
	Women	Black	Winter	Apparel


⸻

🛠️ Next Steps
	•	Add multiple-product detection via bounding boxes
	•	Quantize model with TFLite for edge deployment
	•	Enhance UI: live camera, voice input, batch upload
	•	Publish to Hugging Face Spaces or deploy on Vercel

⸻

👤 Author

Arsalan Khan – GitHub
Based on dataset by Param Aggarwal (Kaggle)

⸻

📄 License: For research and education only. No commercial use without permission.

---
