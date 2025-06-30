import streamlit as st
import cv2
from PIL import Image
from generator import preprocess_image, extract_text, model, mark_fake_document, model2
import time
import os
import base64
from streamlit_option_menu import option_menu
import numpy as np
from datetime import datetime
import io

# ---------------- Streamlit App ----------------
# Set up the page

st.set_page_config(
    page_title="Document Forgery Detection",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Responsive mobile-friendly CSS
st.markdown("""
<style>
.stApp {
    background-color: #010a14;
}
[data-testid="stSidebar"] {
    background-color: #0b1e2d;
}
[data-testid="stSidebar"] * {
    color: #00ccff;
}
@media only screen and (max-width: 768px) {
    .custom-header {
        font-size: 26px !important;
        padding: 6px !important;
        margin-bottom: 15px;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        white-space: normal !important;
        display: block !important;
    }
    .welcome-box, .card {
        padding: 15px !important;
        font-size: 16px !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        white-space: normal !important;
        display: block !important;
    }
    .stButton button, .stRadio > div {
        font-size: 15px !important;
    }
    .element-container {
        padding-left: 10px !important;
        padding-right: 10px !important;
    }
    .block-container {
        padding: 1rem !important;
    }
    .stImage img {
        width: 100% !important;
        height: auto !important;
    }
    .nav-link {
        font-size: 14px !important;
        padding: 5px 8px !important;
    }
    .stTextArea textarea {
        font-size: 14px !important;
    }
    .css-1d391kg {
        position: fixed !important;
        top: 0;
        left: 0;
        right: 0;
        z-index: 9999;
    }
    .block-container {
        padding-top: 100px !important;
    }
    .welcome-box h1, .welcome-box h2, .welcome-box p {
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        white-space: normal !important;
        display: block !important;
    }
}

@media only screen and (min-width: 769px) {
    .css-1d391kg {
        position: sticky !important;
        top: 0;
        z-index: 9999;
    }
    .block-container {
        padding-top: 60px !important;
    }
    .welcome-box h1, .welcome-box h2, .welcome-box p {
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        white-space: normal !important;
        display: block !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Navigation Menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Detect Forgery", "About"],
    icons=["house", "cloud-upload", "search", "info-circle"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"background-color": "#001F3F"},
        "icon": {"color": "#00CFFF", "font-size": "20px"},
        "nav-link": {
            "font-size": "18px",
            "color": "#66FCF1",
            "margin": "5px",
            "--hover-color": "#003B73"
        },
        "nav-link-selected": {
            "background-color": "#003B73",
            "color": "#00CFFF"
        },
    }
)


# Header
def header():
    st.markdown("""
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
    """, unsafe_allow_html=True)

# Background Image
def set_background(image_path):
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

set_background('image/new.jpg')

# Home Page
if selected == 'Home':
    header()

    st.markdown("""
        <style>
        .home-container {
            background-color: rgba(0, 31, 63, 0.9);
            border-radius: 20px;
            padding: 30px;
            color: #E0F7FA;
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.2);
            margin-top: 20px;
            font-family: 'Segoe UI', sans-serif;
            max-width: 100%;
            box-sizing: border-box;
        }

        .home-title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            color: #00CFFF;
            margin-bottom: 20px;
        }

        .home-subtitle {
            text-align: center;
            font-size: 1.2em;
            color: #B2EBF2;
            margin-bottom: 30px;
        }

        .feature-grid {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
        }

        .feature-card {
            flex: 1 1 200px;
            max-width: 250px;
            background-color: #012742;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s ease;
            box-sizing: border-box;
        }

        .feature-card:hover {
            transform: scale(1.05);
        }

        .feature-icon {
            font-size: 2em;
            color: #66FCF1;
            margin-bottom: 10px;
        }

        .feature-text {
            font-size: 1.05em;
            color: #ffffff;
        }

        /* 🔧 Responsive Fixes */
        @media (max-width: 768px) {
            .home-title {
                font-size: 1.8em;
            }

            .home-subtitle {
                font-size: 1em;
            }

            .feature-grid {
                flex-direction: column;
                align-items: center;
            }

            .feature-card {
                width: 90%;
                max-width: none;
            }
        }
        </style>

        <div class="home-container">
            <div class="home-title">👋 Welcome to DocumentGuard</div>
            <div class="home-subtitle">
                AI-Powered System for Real-Time Document Forgery Detection
            </div>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">📤</div>
                    <div class="feature-text"><b>Upload Documents</b><br>Supported formats like School ID & National ID</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <div class="feature-text"><b>Detect Forgery</b><br>Deep Learning model checks for tampering</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-text"><b>Fast Results</b><br>Instant verdict: Original ✅ or Fake ❌</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📝</div>
                    <div class="feature-text"><b>OCR Support</b><br>Extract text from documents using AI</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# Detect Forgery
elif selected == 'Detect Forgery':
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'show_history' not in st.session_state:
        st.session_state.show_history = False

    # Sidebar: Document type, branding image, and tools
    with st.sidebar:
        doc_type = st.selectbox("Select Document Type", options=['National ID', 'School ID'], key='doc_type')
        st.image('image/new2.jpg', use_container_width=True)
        st.divider()
        st.markdown("## 🛠️ Prediction Tools")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📜 History"):
                st.session_state.show_history = not st.session_state.show_history
        with col2:
            if st.button("🧹 Clear"):
                st.session_state.history = []
                st.session_state.show_history = False
                st.rerun()

        if st.session_state.show_history:
            st.markdown("---")
            st.markdown("### 🕘 Last 3 Predictions")
            if st.session_state.history:
                for record in reversed(st.session_state.history[-3:]):
                    st.markdown(f"**{record['timestamp']}**")
                    img_bytes = base64.b64decode(record["image"])
                    image = Image.open(io.BytesIO(img_bytes))
                    st.image(image, caption=record["result"], use_container_width=True)
                    st.markdown("---")
            else:
                st.info("No predictions yet.")

    # Style block
    st.markdown("""
        <style>
            .header-title {
                text-align: center;
                font-size: 2.7em;
                font-weight: 800;
                margin-top: 20px;
                color: #002b45;
                background-color: red;
            }
            # .upload-box {
            #     background-color: #f0f4f8;
            #     padding: 30px;
            #     border-radius: 14px;
            #     box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
            #     margin-top: 20px;
            }
            .stButton>button {
                background-color: #0066cc;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: bold;
                transition: background-color 0.3s;
            }
            .stButton>button:hover {
                background-color: #004d99;
    
            }
        </style>
        <div class='header-title'>📄 Upload or Capture Document</div>
    """, unsafe_allow_html=True)

    # Input method
    with st.container():
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        input_method = st.radio("Choose Input Method", ["Upload Image", "Use Camera"], horizontal=True)
        uploaded_file = st.file_uploader("Upload Document Image", type=["jpg", "jpeg", "png"]) if input_method == "Upload Image" else None
        camera_image = st.camera_input("Take a picture") if input_method == "Use Camera" else None
        st.markdown("</div>", unsafe_allow_html=True)

    image_input = uploaded_file or camera_image

    if image_input is not None and st.button('🔍 Check Document'):
        st.info('Image processing in progress...')
        progress_bar = st.progress(0)
        for percent in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent + 1)

        # Convert and process image
        file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
        original_cv2 = cv2.imdecode(file_bytes, 1)
        gray_cv2 = cv2.cvtColor(original_cv2, cv2.COLOR_BGR2GRAY)
        gray_display = cv2.cvtColor(gray_cv2, cv2.COLOR_GRAY2RGB)

        img = Image.fromarray(gray_display)
        img.save('output.jpg')

        input_img, original_cv2 = preprocess_image(img)

        # Use different model based on document type
        if doc_type == 'National ID':
            prediction = model.predict(input_img)[0][0]
        else:
            prediction = model2.predict(input_img)[0][0]  # Dummy confidence score for School ID
        is_fake = prediction < 0.5
        label = "🔴 Fake" if is_fake else "🟢 Original"
        confidence = (1 - prediction) if is_fake else prediction

        # Display results
        st.image(image_input, caption="📤 Uploaded Document", use_container_width=True)
        st.success(f"**Result:** {label}")
        st.markdown(f"**Confidence Score:** `{confidence:.2f}`")
        prediction_time = datetime.now().strftime("%Y-%m-%d / %H:%M:%S")

        ocr_img = mark_fake_document("output.jpg", is_fake)
        st.image(ocr_img, caption="🧠 OCR & Forgery Marked Result", use_container_width=True)

        st.subheader("⏰ Prediction Time")
        st.info(f"🕒 {prediction_time}")

        st.subheader("📄 Extracted Text (OCR)")
        text = extract_text(original_cv2)
        st.text_area("Detected Text", text, height=200)
        st.download_button("⬇ Download Extracted Text", text, file_name="ocr_output.txt")

        img_buffer = io.BytesIO()
        ocr_img.save(img_buffer, format='JPEG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        st.session_state.history.append({"timestamp": prediction_time, "image": img_base64, "result": label})

        os.remove("output.jpg")


elif selected == 'About':
    st.markdown("""
    <style>
        .about-container {
            background-color: #f7f9fa;
            padding: 40px;
            border-radius: 16px;
            font-family: 'Segoe UI', sans-serif;
            color: #333333;
            margin-top: 30px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }

        .about-header {
            text-align: center;
            font-size: 2.5em;
            font-weight: 700;
            color: #003366;
            margin-bottom: 10px;
        }

        .about-subtitle {
            text-align: center;
            font-size: 1.2em;
            color: #666;
            margin-bottom: 40px;
        }

        .section {
            margin-bottom: 40px;
        }

        .section h3 {
            color: #005580;
            margin-bottom: 15px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
        }

        .section p, .section ul {
            font-size: 1.05em;
            line-height: 1.6;
        }

        .section ul {
            padding-left: 20px;
        }

        .section ul li {
            margin-bottom: 10px;
        }

        .developer-card {
            background-color: #e6f2ff;
            padding: 20px;
            border-radius: 12px;
            margin-top: 30px;
        }

        .developer-card h4 {
            margin-bottom: 5px;
            color: #004d80;
        }

        .contact-info a {
            color: #0066cc;
            text-decoration: none;
        }

        .footer-note {
            text-align: center;
            font-size: 0.95em;
            color: #999999;
            margin-top: 30px;
        }

        @media (max-width: 768px) {
            .about-container {
                padding: 20px;
            }

            .about-header {
                font-size: 2em;
            }
        }
    </style>
    <div class="about-container">
        <div class="about-header">DocumentGuard: Forgery Detection System</div>
        <div class="about-subtitle">AI-Powered Document Validation Tool</div>
        <div class="section">
            <h3>Overview</h3>
            <p>DocumentGuard helps individuals and institutions verify the authenticity of identity documents like <strong>School IDs</strong> and <strong>National IDs</strong> using AI-based image processing and computer vision models.</p>
        </div>
        <div class="section">
            <h3>Core Features</h3>
            <ul>
                <li>📷 Upload or use camera to submit document image</li>
                <li>🤖 Detect forged or altered regions using CNN models</li>
                <li>🧠 Use OCR to extract embedded text</li>
                <li>📊 View prediction scores and confidence levels</li>
                <li>📂 Review previous scans in the history panel</li>
                <li>🖼️ Visual result overlays for transparency</li>
            </ul>
        </div>
        <div class="section">
            <h3>How It Works</h3>
            <ul>
                <li>Image is resized and preprocessed (grayscale, normalization)</li>
                <li>Convolutional Neural Network (CNN) checks for tampering or forgery</li>
                <li>Results are displayed with textual extraction and image annotation</li>
                <li>History is logged for traceability and easy reference</li>
            </ul>
        </div>
        <div class="section developer-card">
            <h3>Developer</h3>
            <h4>Olarinde Olateju Rachael</h4>
            <p>Data Scientist | Python Programmer | Streamlit Developer</p>
            <p>📍 Ogbomoso, Oyo State, Nigeria</p>
        </div>
        <div class="section contact-info">
            <h3>Contact</h3>
            <ul>
                <li>📧 Email: <a href="mailto:olarindeolatejur@gmail.com">olarindeolatejur@gmail.com</a></li>
                <li>💻 GitHub: <a href="https://github.com/Olateju" target="_blank">github.com/Olateju</a></li>
                <li>🔗 LinkedIn: <a href="https://linkedin.com/in/Olateju" target="_blank">linkedin.com/in/Olateju</a></li>
            </ul>
        </div>
        <div class="footer-note">
            © 2025 DocumentGuard | Built using Streamlit and Python
        </div>
    </div>
    """, unsafe_allow_html=True)




