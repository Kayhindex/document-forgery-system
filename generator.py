import streamlit as st
import numpy as np
import cv2
import easyocr
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
# -------------------------------
@st.cache_resource
def load_easyocr_reader():
    """
    Loads EasyOCR models (downloaded from Google Drive if not present)
    and returns a cached EasyOCR reader instance.
    """
    # Create directories
    model_dir = "models/easyocr"
    os.makedirs(f"{model_dir}/detection", exist_ok=True)
    os.makedirs(f"{model_dir}/recognition", exist_ok=True)

    # Google Drive file IDs (make sure they are PUBLIC)
    DETECTION_MODEL_ID = "1BrPH3TOdDhUTUkOkYeGeWBBAEHMBWo6c"
    RECOGNITION_MODEL_ID = "14KoCV69J2V___XSug8KxMkBHRA10avSa"

    # Local paths
    detection_path = f"{model_dir}/detection/craft_mlt_25k.pth"
    recognition_path = f"{model_dir}/recognition/english_g2.pth"

    # Download detection model if not exists
    if not os.path.exists(detection_path):
        url = f"https://drive.google.com/uc?id={DETECTION_MODEL_ID}"
        gdown.download(url, detection_path, quiet=False, fuzzy=True)

    # Download recognition model if not exists
    if not os.path.exists(recognition_path):
        url = f"https://drive.google.com/uc?id={RECOGNITION_MODEL_ID}"
        gdown.download(url, recognition_path, quiet=False, fuzzy=True)

    # Initialize EasyOCR reader using local models
    return easyocr.Reader(
        ['en'],
        model_storage_directory=model_dir,
        user_network_directory=model_dir
    )

def extract_text(image):
    """
    Extracts text from an image using EasyOCR with preprocessing.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding for better OCR
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Load EasyOCR reader (cached)
    reader = load_easyocr_reader()

    # Perform OCR
    result = reader.readtext(thresh)

    # Extract text only
    text = " ".join([res[1] for res in result])
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
