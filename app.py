import streamlit as st
from PIL import Image
from transformers import pipeline

# ----------------------------
# 🌿 App Setup
# ----------------------------
st.set_page_config(page_title="🌿 Pflanzen KI", page_icon="🌱")
st.title("🌿 KI Pflanzen- & Bodenanalyse App")
st.write("Lade ein Bild einer Pflanze hoch.")

# ----------------------------
# 🤖 Modell laden (DEIN MODELL)
# ----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="maxefrost/plant_image_classifier_random"
    )

classifier = load_model()

# ----------------------------
# 🌱 Pflanzen → Boden Logik
# ----------------------------
plant_to_soil = {
    "nettle": "stickstoffreich, feucht",
    "dandelion": "nährstoffreich",
    "clover": "stickstoffarm",
    "daisy": "nährstoffarm bis mittel",
    "plant": "unbekannt / gemischt"
}

soil_to_plants = {
    "stickstoffreich, feucht": ["Kohl", "Gurke"],
    "nährstoffreich": ["Tomate", "Zucchini"],
    "stickstoffarm": ["Erbsen", "Lavendel"],
    "nährstoffarm bis mittel": ["Rosmarin", "Lavendel"]
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
    # 🌱 Bodenanalyse
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
        st.write("Keine Daten vorhanden – erweitere deine Datenbank 🙂")

    # ----------------------------
    # 💡 Erklärung
    # ----------------------------
    st.subheader("💡 Erklärung")
    st.write(
        "Die KI erkennt die wahrscheinlichste Pflanze. "
        "Diese wird mit einer Boden-Datenbank verknüpft, "
        "um passende Gartenpflanzen vorzuschlagen."
    )
