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

    # Resize
    max_width = 1000
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Kontrast verbessern (moderater!)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Rauschen reduzieren
    blur = cv2.medianBlur(gray, 5)

    # Kanten sauber extrahieren
    edges = cv2.Canny(blur, 60, 120)

    # Linien etwas dicker machen
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Hintergrund weiß machen
    sketch = 255 - edges

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
