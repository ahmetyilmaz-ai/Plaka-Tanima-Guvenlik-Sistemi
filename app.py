"""
Vehicle License Plate Recognition and Security System - Streamlit Web Application
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
import streamlit as st
import database as db
from main import plaka_oku_coklu_deneme, vlm_ile_arac_analizi

# Project root directory
PROJECT_ROOT = Path(__file__).parent
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# GPU support
USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'

# Streamlit page settings
st.set_page_config(
    page_title="Vehicle Plate Recognition System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load models once with caching"""
    from ultralytics import YOLO

    with st.spinner("Loading models..."):
        coco_model = YOLO('yolo11n.pt')
        # The other models (OCR, Florence-2) are loaded inside their respective functions
    return coco_model

def process_image_logic(image, models):
    """Processes the uploaded image to perform vehicle and plate analysis."""
    coco_model = models

    img_array = np.array(image.convert('RGB'))
    
    # Save image to a temporary file
    temp_image_path = PROJECT_ROOT / "temp_image.jpg"
    image.convert('RGB').save(temp_image_path)

    # Vehicle detection
    results = coco_model(str(temp_image_path), verbose=False)
    arac_bulundu = False
    arac_tipi = "UNKNOWN"
    arac_bbox = None

    COCO_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    VEHICLE_NAMES = {'car': 'CAR', 'motorcycle': 'MOTORCYCLE', 'bus': 'BUS', 'truck': 'TRUCK'}

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in COCO_CLASSES:
                arac_tipi = COCO_CLASSES[cls_id]
                arac_bbox = box.xyxy[0].tolist()
                arac_bulundu = True
                break
        if arac_bulundu:
            break

    if not arac_bulundu:
        os.remove(temp_image_path)
        return None, "Vehicle not detected."

    arac_tipi_en = VEHICLE_NAMES.get(arac_tipi, arac_tipi.upper())

    x1, y1, x2, y2 = map(int, arac_bbox)
    arac_img = img_array[y1:y2, x1:x2]

    # License plate detection
    h, w = arac_img.shape[:2]
    plaka_img = arac_img[int(h/2):h, :]

    plaka_text = plaka_oku_coklu_deneme(plaka_img)
    vlm_yorumu = vlm_ile_arac_analizi(str(temp_image_path), arac_tipi_en)
    
    os.remove(temp_image_path)

    # Authorization
    if arac_tipi_en != 'CAR':
        karar = "DENIED"
        sebep = "Vehicle type not authorized"
    elif plaka_text == "OKUNAMADI":
        karar = "DENIED"
        sebep = "Plate not readable"
    elif db.plaka_izinli_mi(plaka_text):
        karar = "ALLOWED"
        sebep = "Authorized plate"
    else:
        karar = "DENIED"
        sebep = "Unauthorized plate"
        
    db.log_kaydet(plaka_text, arac_tipi_en, vlm_yorumu, karar)

    return {
        'karar': karar,
        'sebep': sebep,
        'arac_tipi': arac_tipi_en,
        'plaka': plaka_text,
        'vlm_yorumu': vlm_yorumu,
    }, None

def main_ui():
    """Main Streamlit UI"""
    st.markdown('<h1 class="main-header">🚗 Vehicle Plate Recognition System</h1>', unsafe_allow_html=True)

    db.init_database()

    if 'models' not in st.session_state:
        st.session_state.models = load_models()
        st.success("Models loaded!")

    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("💻 CPU Mode (GPU not available)" if not USE_GPU else f"🎮 GPU Active: {torch.cuda.get_device_name(0)}")
        st.divider()

        st.subheader("📋 Allowed Plates")
        allowed_plates = db.get_allowed_plates()
        st.write(f"Total {len(allowed_plates)} allowed plates:")
        for plate in allowed_plates:
            st.write(f"• {plate}")

        st.divider()
        new_plate = st.text_input("Add new plate (e.g., 34ABC123)")
        if st.button("➕ Add"):
            if new_plate:
                if db.add_allowed_plate(new_plate.upper()):
                    st.success(f"{new_plate.upper()} added!")
                    st.rerun()
                else:
                    st.error("Failed to add plate.")
            else:
                st.warning("Please enter a plate.")

        st.divider()
        st.subheader("📜 Recent Logs")
        logs = db.get_recent_logs(5)
        for log in logs:
            plate, tip, vlm, durum, zaman = log
            color = "🟢" if durum == "ALLOWED" else "🔴"
            st.write(f"{color} {zaman.split()[1]} - {plate}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Upload Image")
        uploaded_file = st.file_uploader("Upload a vehicle image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("🔍 Analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    result, error = process_image_logic(image, st.session_state.models)

                if error:
                    st.error(error)
                elif result:
                    with col2:
                        st.subheader("📊 Analysis Result")
                        status_color = "success" if result['karar'] == "ALLOWED" else "danger"
                        st.markdown(f"""
                        <div class="result-box {status_color}">
                            <h2 style="margin: 0;">{result['karar']}</h2>
                            <p style="margin: 0.5rem 0 0 0;">{result['sebep']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown('<div class="info-box"><h4>📋 Details</h4></div>', unsafe_allow_html=True)
                        st.write(f"**Vehicle Type:** {result['arac_tipi']}")
                        st.write(f"**Plate:** {result['plaka']}")
                        st.write(f"**VLM Comment:** {result['vlm_yorumu']}")

if __name__ == "__main__":
    main_ui()
