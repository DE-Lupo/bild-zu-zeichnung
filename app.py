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
        "KI: Text → Zeichnung",
        "KI: Foto → gleiche Person als Zeichnung"
    ]
)

def file_to_bytes(file):
    return BytesIO(file.getvalue())

def get_result_url(output):
    result = output[0] if isinstance(output, list) else output
    return result.url if hasattr(result, "url") else str(result)

def opencv_sketch(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(smooth, 80, 160)

    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return 255 - edges

# ---------------- MODUS 1 ----------------

if mode == "Gratis: Foto → Zeichnung":
    uploaded_file = st.file_uploader("Foto hochladen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.subheader("Original")
        st.image(image, width="stretch")

        sketch = opencv_sketch(image)

        st.subheader("Zeichnung")
        st.image(sketch, channels="GRAY", width="stretch")

        result_image = Image.fromarray(sketch)
        buffer = BytesIO()
        result_image.save(buffer, format="PNG")

        st.download_button(
            "Zeichnung herunterladen",
            data=buffer.getvalue(),
            file_name="zeichnung.png",
            mime="image/png"
        )

# ---------------- MODUS 2 ----------------

elif mode == "KI: Text → Zeichnung":
    prompt = st.text_area(
        "Prompt / Beschreibung",
        value="realistic black and white pencil sketch portrait, detailed face, clean white background, hand drawn style"
    )

    size = st.selectbox(
        "Bildgröße",
        ["1024x1024", "1024x1365", "1365x1024"],
        index=0
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
                    "size": size,
                    "style": style
                }
            )

        result_url = get_result_url(output)

        st.subheader("Ergebnis")
        st.image(result_url, width="stretch")

# ---------------- MODUS 3 ----------------

elif mode == "KI: Foto → gleiche Person als Zeichnung":
    uploaded_file = st.file_uploader("Foto hochladen", type=["jpg", "jpeg", "png"])

    prompt = st.text_area(
        "KI-Anweisung",
        value="turn this exact person into a detailed pencil sketch, preserve facial features, black and white drawing, realistic shading, clean background"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.subheader("Original")
        st.image(image, width="stretch")

        if st.button("👉 Zeichnung erstellen"):
            st.session_state.run_ai = True

        if st.button("Zurücksetzen"):
            st.session_state.run_ai = False

        if st.session_state.get("run_ai", False):
            if not token:
                st.error("REPLICATE_API_TOKEN fehlt in Render.")
                st.stop()

            with st.spinner("KI erstellt Zeichnung aus deinem Foto..."):
                output = replicate.run(
    "stability-ai/sdxl:2f779eb9b23b34fe171f8eaa021b8261566f0d2c10cd2674063e7dbcd351509e",
    input={
        "image": file_to_bytes(uploaded_file),
        "prompt": prompt,
        "prompt_strength": 0.65,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "num_outputs": 1
    }
)

            result_url = get_result_url(output)

            st.subheader("KI-Zeichnung")
            st.image(result_url, width="stretch")
    else:
        st.info("Bitte zuerst ein Foto hochladen.")
