import streamlit as st
import numpy as np
import cv2
from paddleocr import PaddleOCR
import gdown
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import os

# -------------------------------
# Google Drive File IDs
# (replace with your own file IDs)
# -------------------------------
DOCUMENT_MODEL_ID = "1VTzUIddlLttbLAZhldzI8GMQsrvrhG7-"
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
# ------------------------------
# -------------------------------
@st.cache_resource
def load_paddleocr_reader():
    """
    Loads PaddleOCR reader instance (cached for performance).
    """
    # Use English only for speed, angle classification for rotated text
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    return ocr

# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess_for_ocr(image):
    """
    Preprocess image for better OCR accuracy and speed.
    """
    max_dim = 1024
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (binarization)
    _, thresh = cv2.threshold(
        gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh
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
