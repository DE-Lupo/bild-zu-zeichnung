import streamlit as st
import replicate
import os

st.title("KI Zeichnung mit Recraft")

token = os.environ.get("REPLICATE_API_TOKEN")
if not token:
    st.error("REPLICATE_API_TOKEN fehlt in Render.")
    st.stop()

prompt = st.text_area(
    "Prompt / Bildbeschreibung",
    value="realistic black and white pencil sketch portrait, detailed face, clean white background, hand drawn style"
)

size = st.selectbox(
    "Bildgröße",
    ["1024x1024", "1024x1365", "1365x1024"],
    index=0
)

style = st.selectbox(
    "Stil",
    ["illustration", "any"],
    index=0
)

if st.button("Zeichnung erstellen"):
    with st.spinner("Recraft erstellt die Zeichnung..."):
        output = replicate.run(
            "recraft-ai/recraft-v3",
            input={
                "prompt": prompt,
                "size": size,
                "style": style
            }
        )

    result = output[0] if isinstance(output, list) else output

# FileOutput in URL/String umwandeln
result_url = str(result)

st.subheader("Ergebnis")
st.image(result_url, width="stretch")
