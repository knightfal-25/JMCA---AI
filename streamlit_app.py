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
    try:
        model = load_model("mobilenetv2_best_tuned.keras")
        st.sidebar.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.sidebar.error("⚠️ Model not found. Please upload 'mobilenetv2_best_tuned.keras'.")
        st.stop()

model = load_trained_model()
class_names = ['Cat', 'Dog']

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
    st.info("Feature not implemented yet.")

        try:
            image_array = prepare_image(uploaded_file)
            st.write("✅ Image processed successfully!")
            st.write(f"Image shape: {image_array.shape}")

            predictions = model.predict(image_array)
            st.write("✅ Prediction executed successfully!")

            class_idx = np.argmax(predictions)
            confidence = float(predictions[0][class_idx]) * 100

            st.success("✅ Prediction Complete!")
            st.subheader(f"Result: **{class_names[class_idx]}**")
            st.progress(confidence / 100)
            st.caption(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")


else:
    st.info("Please upload an image to start classification.")

# ===========================
#  SIDEBAR INFORMATION
# ===========================
st.sidebar.header("ℹ️ About this App")
st.sidebar.markdown("""
**Cat vs Dog Classifier** built with:
- TensorFlow + MobileNetV2 (Transfer Learning)
- Streamlit for interactive web UI
- CIFAR-10 Dataset (Cat & Dog classes only)
""")

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Developed by Group 1")
