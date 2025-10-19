import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Judul halaman
st.title("Klasifikasi Gambar Kucing dan Anjing")
st.write("Aplikasi ini menggunakan model MobileNetV2 yang telah dioptimasi dengan hyperparameter tuning.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mobilenetv2_best_tuned.keras')
    return model

model = load_model()

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar kucing atau anjing", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocessing gambar
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized)
    img_preprocessed = preprocess_input(img_array)
    img_input = np.expand_dims(img_preprocessed, axis=0)

    # Prediksi
    prediction = model.predict(img_input)
    class_idx = np.argmax(prediction, axis=1)[0]
    labels = ['Cat', 'Dog']
    result = labels[class_idx]
    confidence = float(np.max(prediction)) * 100

    # Hasil prediksi
    st.markdown(f"### Prediksi Model: **{result}**")
    st.markdown(f"**Tingkat Kepercayaan:** {confidence:.2f}%")

st.write("---")
st.caption("Dibuat oleh Kelompok IT Works | Tugas AI-TK2")
