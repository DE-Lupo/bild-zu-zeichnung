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

    # Rauschen glätten, aber Gesicht erhalten
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Klassischer weicher Sketch
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (35, 35), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)

    # Nur leicht abdunkeln, nicht übertreiben
    sketch = cv2.convertScaleAbs(sketch, alpha=1.15, beta=-12)

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
