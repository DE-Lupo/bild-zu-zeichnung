import streamlit as st
import replicate
import os
from PIL import Image

st.title("KI Bild zu Zeichnung")

# Token prüfen
token = os.environ.get("REPLICATE_API_TOKEN")
if not token:
    st.error("REPLICATE_API_TOKEN fehlt!")
    st.stop()

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

prompt = st.text_input(
    "Zeichenstil",
    value="pencil sketch, black and white drawing"
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.subheader("Original")
    st.image(image, width="stretch")

    with st.spinner("KI erstellt Zeichnung..."):
        image.save("temp.png")

        output = replicate.run(
            "adirik/t2i-adapter-sdxl-lineart:6626dfffc713a43dcd1bc79e8e80337a1aaaae6eda0570407c8a00f46dd83c21",
            input={
                "image": open("temp.png", "rb"),
                "prompt": prompt,
                "strength": 0.7,
                "num_inference_steps": 30
            }
        )

        result_url = output[0]

    st.subheader("KI Zeichnung")
    st.image(result_url, width="stretch")
