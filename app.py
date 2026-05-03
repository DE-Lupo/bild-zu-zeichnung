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

    # 1. Weiche Schattierung
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    shading = cv2.divide(gray, 255 - blur, scale=256)

    # 2. Klare Kanten
    edges = cv2.Canny(gray, 40, 120)
    edges = 255 - edges  # invertieren (weiß = Hintergrund)

    # 3. Kombination
    sketch = cv2.multiply(shading, edges, scale=1/255)

    # 4. Kontrast gezielt hochziehen
    sketch = cv2.convertScaleAbs(sketch, alpha=1.5, beta=-20)

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
