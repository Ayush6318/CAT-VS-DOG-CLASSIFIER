import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("cat_dog_model.keras")

st.title("🐶🐱 Cat vs Dog Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((150,150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img)

    # Result
    if pred[0][0] > 0.5:
        st.success("🐶 Dog")
    else:
        st.success("🐱 Cat")