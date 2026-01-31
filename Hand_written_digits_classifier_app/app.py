import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "mnist_ann_model.keras")
    return tf.keras.models.load_model(model_path)

model = load_model()

st.title("🧠 Handwritten Digit Recognition (MNIST)")
st.write("Upload a digit image (0-9)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess
    image = image.resize((28, 28))
    image = np.array(image)
    image = 255 - image  # invert (MNIST-style)
    image = image / 255.0
    image = image.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    st.subheader(f"Predicted Digit: **{predicted_digit}**")

    st.bar_chart(prediction[0])