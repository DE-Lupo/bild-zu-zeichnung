import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Bild zu Zeichnung")

st.title("Bild zu Zeichnung")

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

style = st.selectbox(
    "Zeichenstil wählen",
    [
        "Clean Lineart",
        "Normal Sketch",
        "Detail Sketch",
        "Soft Pencil"
    ]
)

# ---------- Hilfsfunktionen ----------

def resize_image(img, max_width=1000):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))
    return img

def clean_lineart(gray):
    smooth = cv2.bilateralFilter(gray, 15, 100, 100)
    edges = cv2.Canny(smooth, 100, 200)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return 255 - edges

def normal_sketch(gray):
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(smooth, 80, 160)

    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return 255 - edges

def detail_sketch(gray):
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blur, 60, 120)

    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return 255 - edges

def soft_pencil(gray):
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)

    inverted = 255 - smooth
    blurred = cv2.GaussianBlur(inverted, (35, 35), 0)

    sketch = cv2.divide(smooth, 255 - blurred, scale=256)
    sketch = cv2.convertScaleAbs(sketch, alpha=1.15, beta=-12)

    return sketch

def make_sketch(image, style):
    img = np.array(image.convert("RGB"))
    img = resize_image(img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if style == "Clean Lineart":
        return clean_lineart(gray)
    elif style == "Normal Sketch":
        return normal_sketch(gray)
    elif style == "Detail Sketch":
        return detail_sketch(gray)
    elif style == "Soft Pencil":
        return soft_pencil(gray)
    else:
        return normal_sketch(gray)

# ---------- Hauptlogik ----------

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        sketch = make_sketch(image, style)

        st.subheader("Original")
        st.image(image, use_column_width=True)

        st.subheader(f"Zeichnung: {style}")
        st.image(sketch, channels="GRAY", use_column_width=True)

        result_image = Image.fromarray(sketch)
        buffer = BytesIO()
        result_image.save(buffer, format="PNG")

        st.download_button(
            "Zeichnung herunterladen",
            data=buffer.getvalue(),
            file_name="zeichnung.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"Fehler: {e}")
