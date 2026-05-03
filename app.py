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
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Kontrast erhöhen
    gray = cv2.equalizeHist(gray)

    # Kanten erkennen (feiner als vorher)
    edges = cv2.Canny(gray, 50, 150)

    # Invertieren (damit weißer Hintergrund)
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
