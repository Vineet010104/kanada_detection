import streamlit as st
import cv2
import numpy as np
from utils import detect_kannada_text, draw_boxes

st.set_page_config(page_title="Kannada Detection", layout="centered")

st.title("📖 Kannada Text Detection (Final System)")
st.write("Detect Kannada text from multilingual images using OCR.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.subheader("📷 Original Image")
    st.image(image, channels="BGR")

    # 🔄 Detection
    results = detect_kannada_text(image)
    output = draw_boxes(image.copy(), results)

    st.subheader("🔍 Detected Kannada Text")
    st.image(output, channels="BGR")

    st.success(f"✅ Kannada Words Detected: {len(results)}")