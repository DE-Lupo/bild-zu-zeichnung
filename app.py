import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.title("Bild zu Zeichnung")

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

def make_sketch(image):
    img = np.array(image.convert("RGB"))

    max_width = 1000
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sanft glätten
    smooth = cv2.bilateralFilter(gray, 7, 50, 50)

    # Bleistift-Schattierung
    inverted = 255 - smooth
    blur = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(smooth, 255 - blur, scale=230)

    # Gamma/Kontrast: weniger weiß, mehr Details
    sketch = cv2.normalize(sketch, None, 25, 235, cv2.NORM_MINMAX)
    sketch = cv2.convertScaleAbs(sketch, alpha=1.25, beta=-25)

    return sketch

if uploaded_file:
    image = Image.open(uploaded_file)
    sketch = make_sketch(image)

    st.subheader("Original")
    st.image(image)

    st.subheader("Zeichnung")
    st.image(sketch, channels="GRAY")

    result_image = Image.fromarray(sketch)
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")

    st.download_button(
        "Zeichnung herunterladen",
        data=buffer.getvalue(),
        file_name="zeichnung.png",
        mime="image/png"
    )
