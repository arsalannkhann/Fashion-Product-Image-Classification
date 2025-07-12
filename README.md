🧠 Fashion Product Image Classification
This project is a Streamlit-based web application that classifies fashion product images into categories using a deep learning model. The model is trained on the Fashion Product Images Dataset and predicts attributes such as Gender, Base Color, Season, and Product Category (e.g., Shirts, Jeans, Watches). The application provides an intuitive interface for users to upload images and view predictions, making it suitable for e-commerce and fashion industry applications.
The live demo is hosted at: Fashion Product Image Classification App.

📁 Dataset

Source: Kaggle – Fashion Product Images Dataset
Size: ~24 GB
Files:
styles.csv: Metadata with 44,424 rows, including columns: id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName.
images/: Folder containing ~44,424 JPG images, named by id (e.g., 15970.jpg).




🧪 Model Overview

Backbone: Pre-trained CNN (e.g., MobileNetV2, ResNet50, or VGG16, assumed based on typical Kaggle workflows; update with your specific model).
Architecture: Shared CNN backbone with multiple output heads for:
Gender: Men, Women, etc.
Base Color: Navy Blue, Black, etc.
Season: Fall, Summer, Winter, etc.
Product Category (articleType): Shirts, Jeans, Watches, etc.


Training:
Loss: sparse_categorical_crossentropy for each output head.
Metrics: Accuracy per output.
Callbacks: EarlyStopping, ModelCheckpoint (assumed for best practices).


Output Format: Model saved as .h5 or TensorFlow SavedModel; optionally converted to TFLite for lightweight deployment.
Preprocessing: Images resized to 224x224 pixels, normalized, and augmented (e.g., flips, rotations).


🛠️ Code Structure
Fashion-Product-Image-Classification/
├── Model/                          # Trained model and label encoders
│   ├── best_model.h5               # Trained model weights
│   ├── gender_encoder.pkl          # Label encoder for gender
│   ├── color_encoder.pkl           # Label encoder for baseColour
│   ├── season_encoder.pkl          # Label encoder for season
│   └── product_encoder.pkl         # Label encoder for articleType
├── streamlit_app/                  # Streamlit frontend
│   ├── streamlit_app.py            # Main Streamlit app
│   └── requirements.txt            # Streamlit dependencies
├── fashion_product_analysis.ipynb  # Jupyter notebook for data loading and EDA
├── requirements.txt                # General dependencies
├── data/                           # Dataset (downloaded via kagglehub)
│   ├── fashion-dataset/
│       ├── images/                 # Image files
│       ├── styles.csv              # Metadata
├── environment.yml                 # Conda environment setup
└── README.md                       # This documentation

Note: The repository may not include a FastAPI backend or Docker setup, as these are not indicated in the notebook or previous inputs. If used, update this section with api/ folder details.

🚀 Quick Start
✅ Prerequisites

Python 3.8+
GPU recommended for faster training (e.g., T4 GPU, as used in the notebook)
Conda (optional, for environment setup)

✅ Setup Conda Environment
To avoid dependency conflicts:
conda env create -f environment.yml
conda activate fashion-ml
pip install -r requirements.txt
cd streamlit_app && pip install -r requirements.txt

Key Dependencies (in requirements.txt):

tensorflow (or pytorch, if used)
streamlit
pandas
numpy
pillow
kagglehub

✅ Download Dataset
Download the dataset using kagglehub:
import kagglehub
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")

Or manually download from Kaggle and place in data/.
✅ Run the Streamlit App

Navigate to the Streamlit app folder:cd streamlit_app


Run the app:streamlit run streamlit_app.py


Open http://localhost:8501 in your browser.
Upload a fashion product image (JPG/PNG) to view predictions for Gender, Base Color, Season, and Product Category.

✅ Run the Notebook

Open fashion_product_analysis.ipynb in Jupyter Notebook or Google Colab.
Run cells to:
Download the dataset via kagglehub.
Load and explore styles.csv and images/.
Perform EDA (e.g., check dataset shape, verify image files).


Ensure dependencies (kagglehub, pandas) are installed.

✅ Backend (Optional)
If using a FastAPI backend (not confirmed in provided inputs):

Navigate to the API folder:cd api


Install dependencies:pip install -r requirements.txt


Run the FastAPI server:uvicorn main:app --reload


Use Docker (if applicable):docker build -t fashion-api .
docker run -p 8000:8000 fashion-api


API Endpoints:
GET /: Health check.
POST /predict: Accepts multipart/form-data (key=file, value=image).



Note: Confirm if FastAPI/Docker is part of your setup. If not, this section can be removed.

✅ How to Test

Streamlit App:

Access the app at http://localhost:8501 or the deployed URL: Fashion Product Image Classification App.
Upload an image and view predictions for Gender, Color, Season, and Category.


API Testing (if FastAPI is used):Use Postman or curl:
curl -X POST http://localhost:8000/predict \
     -F "file=@/path/to/test.jpg"

Expected JSON response:
{
  "gender": "Women",
  "baseColour": "Black",
  "season": "Winter",
  "masterCategory": "Apparel"
}




🧩 Example Output



Sample Image
Gender
Color
Season
Category



15970.jpg
Men
Navy Blue
Fall
Shirts


59263.jpg
Women
Silver
Winter
Watches



🛠️ Next Steps

Model Enhancements:
Add multi-product detection using bounding boxes (e.g., YOLO).
Quantize the model with TFLite for edge device deployment.


App Improvements:
Support live camera input or batch image uploads.
Add voice input for accessibility.


Deployment:
Publish to Hugging Face Spaces or Vercel for broader access.
Optimize API for faster inference (e.g., batch processing).




👤 Author

Arsalan Khan – GitHub
Dataset by Param Aggarwal (Kaggle)


📄 License
For research and educational purposes only. No commercial use without permission. See the LICENSE file for details.

