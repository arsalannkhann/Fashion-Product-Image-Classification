import streamlit as st
import requests
from PIL import Image
import io

API_URL = "https://your-fastapi-service.onrender.com/predict/"  # Replace with your Render FastAPI URL

st.set_page_config(page_title="Fashion Product Classifier", layout="centered")
st.title("🧥 Fashion Product Multi-Label Classifier")
st.write("Upload an image and get predictions for Gender, Color, Season, and Product Category.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files={"file": uploaded_file.getvalue()})

        if response.status_code == 200:
            prediction = response.json()
            st.success("Predictions:")
            st.markdown(f"""
            - **Gender**: {prediction['gender']}
            - **Color**: {prediction['baseColour']}
            - **Season**: {prediction['season']}
            - **Product Category**: {prediction['masterCategory']}
            """)
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")