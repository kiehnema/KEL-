import streamlit as st
from PIL import Image
from transformers import pipeline

# ----------------------------
# 🌿 App Setup
# ----------------------------
st.set_page_config(page_title="🌿 Pflanzen KI", page_icon="🌱")
st.title("🌿 KI Pflanzen- & Bodenanalyse")
st.write("Lade ein Bild einer Pflanze hoch und erhalte Boden + Empfehlungen.")

# ----------------------------
# 🤖 Modell laden (Pflanzen-spezialisiert)
# ----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="marwaALzaabi/plant-identification-vit"
    )

classifier = load_model()

# ----------------------------
# 🌱 Wissensdatenbank (deine Logik)
# ----------------------------
plant_to_soil = {
    "dandelion": "nährstoffreich",
    "nettle": "stickstoffreich, feucht",
    "clover": "stickstoffarm",
    "daisy": "nährstoffarm bis mittel",
    "rapeseed": "nährstoffreich"
}

soil_to_plants = {
    "nährstoffreich": ["Tomate", "Zucchini"],
    "stickstoffreich, feucht": ["Kohl", "Gurke"],
    "stickstoffarm": ["Erbsen", "Klee"],
    "nährstoffarm bis mittel": ["Lavendel", "Rosmarin"]
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
    # 🤖 KI Prediction
    # ----------------------------
    results = classifier(image)

    st.subheader("🌿 Erkannte Pflanzen (Top 3):")

    top_plant = None

    for r in results[:3]:
        label = r["label"].lower()
        score = round(r["score"] * 100, 2)

        st.write(f"👉 {label} ({score}%)")

        if top_plant is None:
            top_plant = label

    # ----------------------------
    # 🌱 Boden bestimmen
    # ----------------------------
    st.subheader("🌱 Bodenanalyse:")

    soil = plant_to_soil.get(top_plant, "unbekannt / gemischt")

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
        st.write("Keine Empfehlungen vorhanden – Datenbank erweitern!")

    # ----------------------------
    # 💡 Erklärung
    # ----------------------------
    st.subheader("💡 Erklärung")

    st.write(
        "Die KI erkennt die wahrscheinlichste Pflanze im Bild. "
        "Diese wird mit einer Boden-Datenbank verknüpft, "
        "um passende Gartenpflanzen vorzuschlagen."
    )
