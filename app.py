import streamlit as st
import cv2
import numpy as np
import time
import os
import io
import base64
from PIL import Image
from datetime import datetime
from streamlit_option_menu import option_menu

# Import helpers
from generator import (
    preprocess_image,
    extract_text,
    model,
    model2,
    mark_fake_document,
    is_document
)

# ---------------- Streamlit Config ----------------
st.set_page_config(
    page_title="Document Forgery Detection",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Background / Styling ----------------
def set_background(image_path: str):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("image/new.jpg")

# ---------------- Navigation ----------------
selected = option_menu(
    menu_title=None,
    options=["Home", "Detect Forgery", "About"],
    icons=["house", "search", "info-circle"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"background-color": "#001F3F"},
        "icon": {"color": "#00CFFF", "font-size": "20px"},
        "nav-link": {
            "font-size": "18px",
            "color": "#66FCF1",
            "margin": "5px",
            "--hover-color": "#003B73",
        },
        "nav-link-selected": {
            "background-color": "#003B73",
            "color": "#00CFFF",
        },
    },
)

# ---------------- Header ----------------
def header():
    st.markdown(
        """
        <style>
        .custom-header {
            font-size: 40px;
            font-weight: bold;
            color: #1f4e79;
            text-align: center;
            padding: 10px;
            border-bottom: 3px solid #1f4e79;
            margin-bottom: 25px;
            background-color: rgba(255,255,255,0.85);
        }
        </style>
        <div class="custom-header">🛡️ Document Forgery Detection System</div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Home ----------------
if selected == "Home":
    header()
    st.markdown(
        """
        <div style='background-color: rgba(0, 31, 63, 0.9); border-radius: 20px; padding: 30px; 
        color: #E0F7FA; box-shadow: 0 0 15px rgba(0, 255, 255, 0.2); margin-top: 20px;'>
            <h2 style="color:#00CFFF; text-align:center;">👋 Welcome to DocumentGuard</h2>
            <p style="text-align:center;">AI-Powered System for Real-Time Document Forgery Detection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Detect Forgery ----------------
elif selected == "Detect Forgery":
    if "history" not in st.session_state:
        st.session_state.history = []
    if "show_history" not in st.session_state:
        st.session_state.show_history = False

    # Sidebar
    with st.sidebar:
        doc_type = st.selectbox("Select Document Type", options=["National ID", "School ID"])
        st.image("image/new2.jpg", use_container_width=True)
        st.divider()
        if st.button("📜 History"):
            st.session_state.show_history = not st.session_state.show_history
        if st.button("🧹 Clear"):
            st.session_state.history = []
            st.session_state.show_history = False
            st.rerun()

        if st.session_state.show_history and st.session_state.history:
            st.markdown("### 🕘 Last 3 Predictions")
            for record in reversed(st.session_state.history[-3:]):
                st.markdown(f"**{record['timestamp']}**")
                img_bytes = base64.b64decode(record["image"])
                st.image(Image.open(io.BytesIO(img_bytes)), caption=record["result"], use_container_width=True)
                st.markdown("---")

    # Upload/Camera input
    input_method = st.radio("Choose Input Method", ["Upload Image", "Use Camera"], horizontal=True)
    uploaded_file = st.file_uploader("Upload Document", type=["jpg", "jpeg", "png"]) if input_method == "Upload Image" else None
    camera_image = st.camera_input("Take a picture") if input_method == "Use Camera" else None
    image_input = uploaded_file or camera_image

    if image_input is not None and st.button("🔍 Check Document"):
        st.info("Validating document...")

        # Convert to CV2 image
        file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
        original_cv2 = cv2.imdecode(file_bytes, 1)

        # 🔒 Validation gate
        if not is_document(original_cv2):
            st.error("❌ This image does not look like a valid document. Please upload a valid ID or certificate.")
        else:
            st.info("Image processing in progress...")
            progress_bar = st.progress(0)
            for percent in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent + 1)

            # Preprocess for model
            gray_cv2 = cv2.cvtColor(original_cv2, cv2.COLOR_BGR2GRAY)
            gray_display = cv2.cvtColor(gray_cv2, cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(gray_display)
            img.save("output.jpg")

            input_img, _ = preprocess_image(img)

            # Select model
            prediction = model.predict(input_img)[0][0] if doc_type == "National ID" else model2.predict(input_img)[0][0]
            is_fake = prediction < 0.5
            label = "🔴 Fake" if is_fake else "🟢 Original"
            confidence = (1 - prediction) if is_fake else prediction

            # Results
            st.image(image_input, caption="📤 Uploaded Document", use_container_width=True)
            st.success(f"**Result:** {label}")
            st.markdown(f"**Confidence Score:** `{confidence:.2f}`")
            prediction_time = datetime.now().strftime("%Y-%m-%d / %H:%M:%S")

            # Marked output
            ocr_img = mark_fake_document("output.jpg", is_fake)
            st.image(ocr_img, caption="🧠 OCR & Forgery Marked Result", use_container_width=True)

            # OCR text extraction
            st.subheader("📄 Extracted Text")
            text = extract_text(original_cv2)
            st.text_area("Detected Text", text, height=200)
            st.download_button("⬇ Download Extracted Text", text, file_name="ocr_output.txt")

            # Save history
            img_buffer = io.BytesIO()
            ocr_img.save(img_buffer, format="JPEG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            st.session_state.history.append({"timestamp": prediction_time, "image": img_base64, "result": label})

            os.remove("output.jpg")

# ---------------- About ----------------
elif selected == "About":
    st.markdown(
        """
        <div style='background:#f7f9fa;padding:40px;border-radius:16px;'>
            <h2 style="text-align:center;color:#003366;">DocumentGuard</h2>
            <p style="text-align:center;">AI-Powered Document Validation Tool</p>
            <ul>
                <li>📷 Upload or capture document image</li>
                <li>🤖 CNN detects forgery or tampering</li>
                <li>🧠 OCR extracts text</li>
                <li>📊 Confidence scores displayed</li>
                <li>📂 View history of scans</li>
            </ul>
            <h3>Developer</h3>
            <p><b>Olarinde Olateju Rachael</b><br>Data Scientist | Python Programmer</p>
            <p>📍 Ogbomoso, Oyo State, Nigeria</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
