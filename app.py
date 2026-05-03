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

    # 🔥 KONTRAST RETTEN (wichtigster Schritt!)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # zusätzlich Histogramm strecken
    gray = cv2.equalizeHist(gray)

    # Rauschen entfernen
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Kanten sauber extrahieren
    edges = cv2.Canny(blur, 70, 140)

    # Linien leicht verstärken
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Invertieren → weißer Hintergrund
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
