import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image

# ===========================
#  STREAMLIT CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# ===========================
#  MODEL LOADING (CACHED)
# ===========================
@st.cache_resource
def load_trained_model():
    import os
    model_path = "mobilenetv2_best_tuned.keras"
    if not os.path.exists(model_path):
        st.sidebar.error("⚠️ Model file not found. Make sure it's uploaded to the same repo folder.")
        st.stop()
    model = load_model(model_path)
    st.sidebar.success("✅ Model loaded successfully")
    return model


# ===========================
#  IMAGE PREPROCESSING
# ===========================
def prepare_image(uploaded_image):
    """Resize, normalize and preprocess image for MobileNetV2."""
    image = Image.open(uploaded_image).convert("RGB")
    image = image.resize((32, 32))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# ===========================
#  UI DESIGN
# ===========================
st.title("🐱🐶 Cat vs Dog Classifier")
st.markdown("""
Upload an image below to classify whether it's a **Cat** or **Dog**.
The model was trained using **MobileNetV2 + Hyperparameter Tuning** on the **CIFAR-10 dataset**.
""")

uploaded_file = st.file_uploader(
    "📤 Upload your image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
   if st.button("➡️ Next (Classify)"):
    with st.spinner("Processing image... 🔍"):
        # Preprocess and predict
        image_array = prepare_image(uploaded_file)
        predictions = model.predict(image_array)
        class_idx = np.argmax(predictions)
        confidence = float(predictions[0][class_idx]) * 100

else:
    st.info("Please upload an image to start classification.")

# ===========================
#  SIDEBAR INFORMATION
# ===========================
st.sidebar.header("ℹ️ About this App")
st.sidebar.markdown("""
**Cat vs Dog Classifier** built with:
- 🧠 TensorFlow + MobileNetV2 (Transfer Learning)
- 🎨 Streamlit for interactive web UI
- 🐾 CIFAR-10 Dataset (Cat & Dog classes only)
""")

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻Developed by Group 1 ")
