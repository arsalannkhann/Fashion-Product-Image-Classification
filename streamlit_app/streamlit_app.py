import streamlit as st
import requests
from PIL import Image
import os
import io
import json

st.set_page_config(page_title="Fashion Image Classifier", layout="centered")

st.title("🧠 Fashion Product Image Classifier")
st.markdown("Upload a product image or choose from sample images to classify **gender**, **color**, **season**, and **category**.")

API_URL = "https://fashion-product-image-classification.onrender.com/predict"

# --- Sidebar with samples ---
st.sidebar.header("📂 Sample Images")
sample_dir = "streamlit_app/sample_images"
sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(("jpg", "png", "jpeg"))]
sample_choice = st.sidebar.selectbox("Or try a sample image:", ["None"] + sample_files)

# --- Image selection logic ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if sample_choice != "None":
    image_path = os.path.join(sample_dir, sample_choice)
    image = Image.open(image_path)
    st.image(image, caption="Sample Image", use_container_width=True)
    image_bytes = open(image_path, "rb").read()
elif uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    image_bytes = uploaded_file.read()
else:
    st.warning("Please upload an image or select a sample.")
    st.stop()

# --- Predict Button ---
if st.button("🔍 Predict"):
    with st.spinner("Analyzing image..."):
        try:
            files = {'file': ("image.jpg", image_bytes, "image/jpeg")}
            response = requests.post(API_URL, files=files)
            result = response.json()

            if response.status_code == 200 and all(k in result for k in ["gender", "baseColour", "season", "masterCategory"]):
                st.success("Predictions:")
                st.markdown(f"""
                - 🧍 **Gender**: {result['gender']}
                - 🎨 **Color**: {result['baseColour']}
                - ❄️ **Season**: {result['season']}
                - 🛍️ **Product Category**: {result['masterCategory']}
                """)

                with st.expander("🧪 Raw API Response"):
                    st.code(json.dumps(result, indent=2))

            else:
                st.error("❌ Incomplete prediction. Check API logs or model output.")
                st.json(result)

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
