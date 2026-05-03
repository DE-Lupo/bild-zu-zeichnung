import streamlit as st
import replicate
import os

st.title("KI Zeichnung mit Recraft")

# 🔐 Token prüfen
token = os.environ.get("REPLICATE_API_TOKEN")
if not token:
    st.error("REPLICATE_API_TOKEN fehlt in Render.")
    st.stop()

# 📝 Prompt
prompt = st.text_area(
    "Prompt / Bildbeschreibung",
    value="realistic black and white pencil sketch portrait, detailed face, clean white background, hand drawn style"
)

# 📐 Größe
size = st.selectbox(
    "Bildgröße",
    ["1024x1024", "1024x1365", "1365x1024"],
    index=0
)

# 🎨 Stil (nur erlaubte Werte!)
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

# 🚀 Button
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

    # ✅ Ergebnis korrekt holen
    result = output[0] if isinstance(output, list) else output

    # ✅ URL sauber extrahieren
    result_url = result.url if hasattr(result, "url") else str(result)

    st.subheader("Ergebnis")
    st.image(result_url, width="stretch")
