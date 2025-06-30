import streamlit as st
import numpy as np
import cv2
import pytesseract
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

# Optional: Uncomment and modify if using Windows and Tesseract isn't in PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load model only once
@st.cache_resource
def load_forgery_model():
    model = load_model("document_forgery_model.keras")
    return model

model = load_forgery_model()

@st.cache_resource
def load_2nd_model():
    model2 = load_model("document1_forgery_model.keras")
    return model2

model2 = load_2nd_model()

# Preprocessing function for model input
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img, (224, 224))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 224, 224, 3))
    return reshaped, img

# OCR function
def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh)
    return text

def mark_fake_document(image_path, is_fake):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    if is_fake:
        try:
            font = ImageFont.truetype("arial.ttf", size=40)
        except:
            font = ImageFont.load_default()
        draw.text((width//4, height//2), "FAKE DOCUMENT", fill="red", font=font,)
        draw.line((0, height//2, width, height//2), fill="red", width=5)
    return img

