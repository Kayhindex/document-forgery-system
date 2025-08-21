import streamlit as st
import numpy as np
import cv2
import pytesseract
import gdown
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import os

# -------------------------------
# Google Drive File IDs
# (replace with your own file IDs)
# -------------------------------
DOCUMENT_MODEL_ID = "1VTzUIddlLttbLAZhldzI8GMQsrvrhG7"
SCHOOL_MODEL_ID = "12UHOLzK2tb8BaMXoTVzV00kOpixjOz5C"

# -------------------------------
# Helper function: download from Drive
# -------------------------------
def download_from_drive(file_id, output):
    if not os.path.exists(output):  # Avoid re-downloading
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# -------------------------------
# Load Models (cached)
# -------------------------------
@st.cache_resource
def load_forgery_model():
    model_path = "document_forgery_model.h5"
    download_from_drive(DOCUMENT_MODEL_ID, model_path)
    model = load_model(model_path)
    return model

@st.cache_resource
def load_school_model():
    model_path = "school_forgery_model.h5"
    download_from_drive(SCHOOL_MODEL_ID, model_path)
    model2 = load_model(model_path)
    return model2

model = load_forgery_model()
model2 = load_school_model()

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img, (224, 224))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 224, 224, 3))
    return reshaped, img

# -------------------------------
# OCR Function
# -------------------------------
def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh)
    return text

# -------------------------------
# Mark Fake Documents
# -------------------------------
def mark_fake_document(image_path, is_fake):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    if is_fake:
        try:
            font = ImageFont.truetype("arial.ttf", size=40)
        except:
            font = ImageFont.load_default()
        draw.text((width // 4, height // 2), "FAKE DOCUMENT", fill="red", font=font)
        draw.line((0, height // 2, width, height // 2), fill="red", width=5)

    return img
