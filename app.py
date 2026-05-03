import streamlit as st
import replicate
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Bild zu Zeichnung", layout="centered")
st.title("Bild zu Zeichnung")

token = os.environ.get("REPLICATE_API_TOKEN")

mode = st.selectbox(
    "Modus wählen",
    [
        "Gratis: Foto → Zeichnung",
        "Premium: Foto → Bleistiftzeichnung",
        "KI: Text → Zeichnung"
    ]
)

def get_result_url(output):
    result = output[0] if isinstance(output, list) else output
    return result.url if hasattr(result, "url") else str(result)

def resize_image(img, max_width=1200):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))
    return img

def simple_lineart(image):
    img = np.array(image.convert("RGB"))
    img = resize_image(img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(smooth, 80, 160)

    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return 255 - edges

def premium_pencil(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # starke Glättung für schöne Flächen
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    # Pencil shading (Hauptteil!)
    inverted = 255 - gray
    blur = cv2.GaussianBlur(inverted, (25, 25), 0)
    pencil = cv2.divide(gray, 255 - blur, scale=256)

    pencil = np.clip(pencil, 0, 255).astype(np.uint8)

    # Linien (SEHR schwach!)
    edges = cv2.Canny(gray, 80, 150)
    edges = cv2.GaussianBlur(edges, (7, 7), 0)
    edges = np.clip(edges * 0.1, 0, 255).astype(np.uint8)

    # Kombination
    sketch = cv2.subtract(pencil, edges)

    # KONTRAST BOOST (entscheidend!)
    sketch = cv2.convertScaleAbs(sketch, alpha=1.4, beta=-25)

    return sketch

def image_download(sketch, filename):
    result_image = Image.fromarray(sketch)
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")

    st.download_button(
        "Zeichnung herunterladen",
        data=buffer.getvalue(),
        file_name=filename,
        mime="image/png"
    )

# ---------------- MODUS 1 ----------------

if mode == "Gratis: Foto → Zeichnung":
    uploaded_file = st.file_uploader("Foto hochladen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.subheader("Original")
        st.image(image, width="stretch")

        sketch = simple_lineart(image)

        st.subheader("Einfache Zeichnung")
        st.image(sketch, channels="GRAY", width="stretch")

        image_download(sketch, "einfache_zeichnung.png")

# ---------------- MODUS 2 ----------------

elif mode == "Premium: Foto → Bleistiftzeichnung":
    uploaded_file = st.file_uploader("Foto hochladen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.subheader("Original")
        st.image(image, width="stretch")

        sketch = premium_pencil(image)

        st.subheader("Bleistiftzeichnung")
        st.image(sketch, channels="GRAY", width="stretch")

        image_download(sketch, "bleistiftzeichnung.png")

# ---------------- MODUS 3 ----------------

elif mode == "KI: Text → Zeichnung":
    prompt = st.text_area(
        "Prompt",
        value="black and white pencil sketch portrait, detailed face, hand drawn graphite drawing, white paper background"
    )

    style = st.selectbox(
        "Stil",
        [
            "digital_illustration/hand_drawn",
            "digital_illustration/hand_drawn_outline",
            "realistic_image/b_and_w",
            "any"
        ],
        index=0
    )

    if st.button("KI-Zeichnung erstellen"):
        if not token:
            st.error("REPLICATE_API_TOKEN fehlt in Render.")
            st.stop()

        with st.spinner("Recraft erstellt die Zeichnung..."):
            output = replicate.run(
                "recraft-ai/recraft-v3",
                input={
                    "prompt": prompt,
                    "size": "1024x1024",
                    "style": style
                }
            )

        result_url = get_result_url(output)

        st.subheader("KI-Ergebnis")
        st.image(result_url, width="stretch")
