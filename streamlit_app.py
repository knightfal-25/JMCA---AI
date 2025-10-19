import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.title("Animal Classifier - Cat/Dog/Rabbit")
model = tf.keras.models.load_model("model/best_model.h5")

IMG_SIZE = (224,224)
class_names = ['cat','dog','rabbit']

uploaded_file = st.file_uploader("Upload image", type=['jpg','jpeg','png'])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    # preprocess
    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    top_idx = np.argmax(preds)
    st.write(f"Prediction: **{class_names[top_idx]}** ({preds[top_idx]*100:.2f}%)")
    st.bar_chart(preds)

