import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# ==============================
# STREAMLIT CONFIG
# ==============================
st.set_page_config(
    page_title="Image Classifier App",
    page_icon="📸",
    layout="centered"
)

# ==============================
# SESSION STATE
# ==============================
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "page" not in st.session_state:
    st.session_state.page = "upload"

# ==============================
# CUSTOM CSS (biar mirip website converter)
# ==============================
st.markdown("""
<style>
[data-testid="stFileUploader"] {
    margin-top: 50px;
    text-align: center;
}
button {
    font-size: 18px !important;
    padding: 0.75em 2em !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# PAGE 1: UPLOAD IMAGE
# ==============================
if st.session_state.page == "upload":
    st.title("📸 Image Classification App")
    st.subheader("Upload your image below")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Preview", use_container_width=True)
        st.session_state.uploaded_image = image

    if st.button("Next ➡️"):
        if st.session_state.uploaded_image is not None:
            st.session_state.page = "result"
            st.rerun()
        else:
            st.warning("Please upload an image first!")

# ==============================
# PAGE 2: RESULT PAGE
# ==============================
elif st.session_state.page == "result":
    st.title("🧠 Prediction Result")

    if st.session_state.uploaded_image is None:
        st.warning("No image uploaded. Please go back and upload an image.")
        if st.button("⬅️ Back"):
            st.session_state.page = "upload"
            st.rerun()
    else:
        image = st.session_state.uploaded_image
        st.image(image, caption="Your Uploaded Image", use_container_width=True)

        with st.spinner("Processing image..."):
            # --- LOAD MODEL ---
            model = tf.keras.models.load_model("mobilenetv2_best_tuned.keras")

            # --- PREPROCESS IMAGE ---
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # --- PREDICT ---
            preds = model.predict(img_array)
            class_idx = np.argmax(preds)
            confidence = np.max(preds) * 100

        st.success(f"✅ Predicted Class: {class_idx}")
        st.info(f"Confidence: {confidence:.2f}%")

        if st.button("⬅️ Back"):
            st.session_state.page = "upload"
            st.rerun()
