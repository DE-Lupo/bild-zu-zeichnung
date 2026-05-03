import streamlit as st
import replicate
import os
from PIL import Image
from io import BytesIO

st.title("KI Bild zu Zeichnung")

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

prompt = st.text_input(
    "Zeichenstil (optional)",
    value="pencil sketch, detailed, black and white drawing"
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.subheader("Original")
    st.image(image, width="stretch")

    with st.spinner("KI erstellt Zeichnung..."):

        # Bild speichern temporär
        image.save("temp.png")

        output = replicate.run(
            "stability-ai/sdxl:latest",
            input={
                "image": open("temp.png", "rb"),
                "prompt": prompt,
                "strength": 0.7,
                "num_inference_steps": 30
            }
        )

        # Ergebnis laden
        result_url = output[0]

    st.subheader("KI Zeichnung")
    st.image(result_url, width="stretch")

    st.download_button(
        "Download",
        data=result_url,
        file_name="ki_zeichnung.png"
    )
