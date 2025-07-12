🧠 Fashion Product Image Classification
A multi-output deep learning system for classifying fashion product images, predicting:

👕 Gender (e.g., Men, Women)  
🎨 Base Color (e.g., Black, Blue)  
❄️ Season (e.g., Winter, Summer)  
🛍️ Product Category (e.g., Apparel, Accessories)

Built with TensorFlow, this project provides a robust pipeline for training, evaluation, and inference, with plans for FastAPI and Streamlit integration for deployment.

📁 Dataset

Source: Kaggle – Fashion Product Images Dataset  
Size: ~24 GB  
Files Used:  
styles.csv: Metadata for fashion products  
images/: Directory of product images




🧪 Model Overview

Backbone: ResNet50 (pretrained on ImageNet)  
Architecture: Shared CNN with four output heads for gender, baseColour, season, and masterCategory.  
Loss Function: Custom SparseCategoricalFocalLoss (gamma=1.5) to handle class imbalance.  
Metrics: Accuracy, Precision, Recall, F1-score per output head.  
Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.  
Exported Formats: .h5, .keras, and pickled LabelEncoder objects.


🛠️ Project Structure
Fashion-Product-Image-Classification/
├── model/                     # Trained models and label encoders
│   ├── multioutput_model.keras
│   ├── best_model.h5
│   ├── label_encoder_gender.pkl
│   ├── label_encoder_baseColour.pkl
│   ├── label_encoder_season.pkl
│   └── label_encoder_masterCategory.pkl
├── data/                      # Sample images for testing
│   └── test_samples/
├── training/                  # Training and preprocessing scripts
│   ├── fashion_product_analysis.py  # Main script
│   └── config.yaml            # Configuration (optional)
├── backend/                   # FastAPI backend (planned)
│   ├── app/
│   │   ├── main.py            # FastAPI routes
│   │   ├── predict.py         # Inference logic
│   │   └── utils.py           # Image preprocessing
│   ├── Dockerfile
│   ├── requirements.txt
│   └── test/
├── frontend/                  # Streamlit frontend (planned)
│   ├── streamlit_app.py
│   └── requirements.txt
├── environment.yml            # Conda environment configuration
├── README.md                  # This documentation
└── .gitignore                 # Git ignore file


🚀 Quick Start
📋 Prerequisites

Python 3.8+  
TensorFlow 2.10+  
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, Pillow, kagglehub, imblearn  
Optional: Docker, Conda (for environment management)

🔧 Setup

Clone the Repository  
git clone https://github.com/your-username/Fashion-Product-Image-Classification.git
cd Fashion-Product-Image-Classification


Set Up Conda Environment  
conda env create -f environment.yml
conda activate fashion-ml


Download DatasetEnsure you have a Kaggle account and API token set up (~/.kaggle/kaggle.json). Then run:
python training/fashion_product_analysis.py

This downloads the dataset via kagglehub to ~/.kaggle/ or a specified path.

Train the ModelRun the training script:
python training/fashion_product_analysis.py

The script handles data preprocessing, model training, and saves the model to model/multioutput_model.keras.



🚀 Inference
Local Inference
Run the demo inference section of the script to test on sample images:
python training/fashion_product_analysis.py

This loads the trained model and predicts on sample Amazon image URLs.
Planned API Deployment (FastAPI)

Navigate to the backend directory:cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload


Access the API at http://localhost:8000.

Planned UI Deployment (Streamlit)

Navigate to the frontend directory:cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py


Upload images via the Streamlit UI to view predictions.


🧪 Testing

API Testing (once backend is implemented):  
curl -X POST http://localhost:8000/predict -F "file=@data/test_samples/sample.jpg"


Local Testing: Use the demo inference section in fashion_product_analysis.py to predict on sample images.



📊 Sample Output



Gender
Color
Season
Category



Women
Black
Winter
Apparel



🛠️ Future Enhancements

Implement FastAPI backend for production-grade inference.  
Deploy Streamlit app for interactive UI with webcam support.  
Add object detection for multiple products in a single image.  
Optimize model with TFLite for edge deployment.  
Publish to Hugging Face Spaces or Vercel for public access.


❓ Troubleshooting

Dataset Download Issues: Ensure Kaggle API token is configured (~/.kaggle/kaggle.json).  
Missing Images: Verify the images/ folder exists and matches styles.csv IDs.  
Training Errors: Check for sufficient GPU memory or reduce batch size in fashion_product_analysis.py.  
Dependency Conflicts: Use the provided environment.yml to ensure consistent library versions.


👤 Author
Arsalan Khan 
📄 License
For research and educational purposes only.Commercial use requires explicit permission from the dataset owner and author.
