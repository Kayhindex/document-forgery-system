import os
import cv2
import gdown
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model
import streamlit as st
from transformers import pipeline

# -------------------------------
# Google Drive File IDs
# -------------------------------
DOCUMENT_MODEL_ID = "1VTzUIddlLttbLAZhldzI8GMQsrvrhG7-"
SCHOOL_MODEL_ID = "12UHOLzK2tb8BaMXoTVzV00kOpixjOz5C"

# -------------------------------
# Helper: download from Drive
# -------------------------------
def download_from_drive(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# -------------------------------
# Load models (cached)
# -------------------------------
@st.cache_resource
def load_forgery_model():
    model_path = "document_forgery_model.h5"
    download_from_drive(DOCUMENT_MODEL_ID, model_path)
    return load_model(model_path)

@st.cache_resource
def load_school_model():
    model_path = "school_forgery_model.h5"
    download_from_drive(SCHOOL_MODEL_ID, model_path)
    return load_model(model_path)

model = load_forgery_model()
model2 = load_school_model()

# Hugging Face document classifier (lightweight text/image model)
@st.cache_resource
def load_document_classifier():
    return pipeline("image-classification", model="microsoft/dit-base-finetuned-documents")

doc_classifier = load_document_classifier()

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_image(image):
    """Prepare image for forgery models."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img, (224, 224))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 224, 224, 3))
    return reshaped, img

@st.cache_resource
def preprocess_for_ocr(image_path):
    """Preprocess image for OCR (resize + grayscale)."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (800, 800))
    return resized

# -------------------------------
# OCR + Text extraction
# -------------------------------
def extract_text(image_input):
    """Extract text using EasyOCR."""
    ocr = easyocr.Reader(['en'])
    result = ocr.readtext(image_input)
    text = " ".join([res[1] for res in result])
    return text.strip()

# -------------------------------
# Fake document marking
# -------------------------------
def mark_fake_document(image_path, is_fake):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    if is_fake:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=40)
        except:
            font = ImageFont.load_default()

        draw.text(
            (width // 4, height // 2),
            "FAKE DOCUMENT",
            fill="red",
            font=font
        )
        draw.line((0, height // 2, width, height // 2), fill="red", width=5)

    return img

# -------------------------------
# Document validation
# -------------------------------
def is_document(image):
    """
    Validate if uploaded image is a document.
    Uses contour geometry + OCR + Hugging Face classifier.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Contour check (documents are rectangles)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:  # rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2.0 and w > 200 and h > 200:
                break
    else:
        return False

    # Step 2: OCR density check
    ocr = easyocr.Reader(['en'])
    result = ocr.readtext(gray)
    if len(result) < 3:  # must have some text
        return False

    # Step 3: Hugging Face check (classifier confidence)
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    predictions = doc_classifier(pil_img)
    top_label = predictions[0]["label"].lower()
    confidence = predictions[0]["score"]

    if "document" in top_label and confidence > 0.7:
        return True

    return False
