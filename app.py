import streamlit as st
from PIL import Image
from transformers import pipeline

# ----------------------------
# 🌱 Seite Setup
# ----------------------------
st.set_page_config(page_title="Pflanzen KI", page_icon="🌿")
st.title("🌿 KI Pflanzen- & Bodenanalyse App")
st.write("Lade ein Bild einer Wildpflanze hoch und erhalte Bodeninfos + Empfehlungen.")

# ----------------------------
# 🧠 Modell (leicht & stabil)
# ----------------------------
@st.cache_resource
def load_model():
    # robustes allgemeines Vision Modell (funktioniert sicher)
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_model()

# ----------------------------
# 🌱 Deine Wissensbasis (LOGIK)
# ----------------------------
plant_to_soil = {
    "daisy": "nährstoffarm bis mittel",
    "dandelion": "nährstoffreich",
    "nettle": "stickstoffreich, feucht",
    "clover": "stickstoffarm"
}

soil_to_plants = {
    "nährstoffarm bis mittel": ["Lavendel", "Rosmarin"],
    "nährstoffreich": ["Tomate", "Zucchini"],
    "stickstoffreich, feucht": ["Kohl", "Gurke"],
    "stickstoffarm": ["Klee", "Erbsen"]
}

# ----------------------------
# 📷 Upload
# ----------------------------
uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Dein Bild", use_column_width=True)

    st.write("🔍 Analysiere Pflanze...")

    # ----------------------------
    # 🤖 KI Prediction (Top 3)
    # ----------------------------
    results = classifier(image)

    st.subheader("🌿 Erkannte Möglichkeiten:")

    top_plant = None

    for r in results[:3]:
        label = r["label"]
        score = round(r["score"] * 100, 2)

        st.write(f"👉 {label} ({score}%)")

        if top_plant is None:
            top_plant = label

    # ----------------------------
    # 🌱 Boden ableiten
    # ----------------------------
    st.subheader("🌱 Bodenanalyse:")

    soil = plant_to_soil.get(top_plant.lower(), "unbekannt / gemischt")

    st.success(f"Wahrscheinlicher Boden: {soil}")

    # ----------------------------
    # 🌿 Empfehlungen
    # ----------------------------
    st.subheader("🌿 Empfohlene Pflanzen:")

    recommendations = soil_to_plants.get(soil, [])

    if recommendations:
        for plant in recommendations:
            st.write(f"🌿 {plant}")
    else:
        st.write("Keine Daten vorhanden – erweitere deine Datenbank 🙂")

    # ----------------------------
    # 💡 Erklärung
    # ----------------------------
    st.subheader("💡 Erklärung")

    st.write(
        "Die erkannte Pflanze bestimmt typische Bodenbedingungen. "
        "Diese werden mit einer Datenbank abgeglichen, um passende Pflanzen vorzuschlagen."
    )
