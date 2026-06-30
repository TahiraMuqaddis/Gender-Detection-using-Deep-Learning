"""
Streamlit Live Test App — Gender Detection Model
--------------------------------------------------
Loads the gender.keras model trained in the notebook and lets you
upload an image to get a live Male/Female prediction.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from sklearn import preprocessing
import os

# ============================================================
# CONFIG — must match the training notebook exactly
# ============================================================
IMG_SIZE = 128
MODEL_PATH = "gender.keras"
CLASS_NAMES = {0: "Female", 1: "Male"}

st.set_page_config(
    page_title="Gender Detection",
    page_icon="🧑‍🤝‍🧑",
    layout="centered"
)

# ============================================================
# LOAD MODEL (cached so it only loads once per session)
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file '{MODEL_PATH}' not found! Please make sure it exists.")
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# ============================================================
# PREPROCESSING — replicates the training pipeline exactly
# ============================================================
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Preprocess image exactly like training:
    1. Convert to grayscale
    2. Resize to 128x128
    3. Flatten to (1, 16384)
    4. L2-normalize with sklearn.preprocessing.normalize
    """
    # Convert to grayscale
    img = np.array(pil_image.convert("L"))
    
    # Resize to training size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Flatten 2D image -> 1D vector, shape (1, 16384)
    img = img.reshape(1, IMG_SIZE * IMG_SIZE).astype("float64")
    
    # L2-normalize exactly like training
    img = preprocessing.normalize(img)
    
    return img

# ============================================================
# MAIN UI
# ============================================================

st.title("🧑‍🤝‍🧑 Gender Detection — Live Test")
st.write("Upload a face image and the model will predict **Male** or **Female**.")

# Load model
model = load_model()

if model is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read the image
    pil_image = Image.open(uploaded_file)
    
    # Display uploaded image and prediction side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(pil_image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("Predicting..."):
            # Preprocess and predict
            processed = preprocess_image(pil_image)
            prediction = model.predict(processed, verbose=0)[0]
            predicted_class = int(np.argmax(prediction))
            confidence = float(prediction[predicted_class]) * 100
        
        # Display results
        st.subheader("Prediction")
        
        # Color-coded result
        if predicted_class == 0:
            st.success(f"👩 **{CLASS_NAMES[predicted_class]}**")
        else:
            st.info(f"👨 **{CLASS_NAMES[predicted_class]}**")
        
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        
        # Show class probabilities
        st.write("---")
        st.write("**Class Probabilities**")
        
        col_f, col_m = st.columns(2)
        with col_f:
            st.write(f"👩 Female: {prediction[0] * 100:.2f}%")
            st.progress(float(prediction[0]))
        with col_m:
            st.write(f"👨 Male: {prediction[1] * 100:.2f}%")
            st.progress(float(prediction[1]))
else:
    st.info("👆 Upload an image to get started.")

# ============================================================
# FOOTER
# ============================================================

st.write("---")

with st.expander("ℹ️ About this app"):
    st.write("""
    ### Model Details
    - **Architecture**: Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.2) → Dense(2, softmax)
    - **Input**: Grayscale image, resized to 128×128, flattened, L2-normalized
    - **Dataset**: Labeled men/women face images
    
    ### How to Use
    1. Click 'Browse files' to upload an image
    2. The model will analyze the image
    3. View the prediction and confidence score
    
    ### Note
    - For best results, upload clear face images
    - The model was trained on grayscale images
    - Supports JPG, JPEG, and PNG formats
    """)

# Display app status
st.sidebar.title("📊 Status")
st.sidebar.success("✅ Model loaded successfully")
st.sidebar.write(f"📁 Model: {MODEL_PATH}")
st.sidebar.write(f"📐 Input size: {IMG_SIZE}×{IMG_SIZE}")
st.sidebar.write("🎯 Classes: Female (0), Male (1)")