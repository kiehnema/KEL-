import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ----------------------------
# 🌿 Setup
# ----------------------------
st.set_page_config(page_title="🌿 Wildpflanzen KI", page_icon="🌱")
st.title("🌿 Wildpflanzen & Bodenanalyse (Offline KI)")

st.write("Lade ein Bild einer Pflanze hoch.")

# ----------------------------
# 🤖 Modell laden
# ----------------------------
@st.cache_resource
def load_model():
    model_name = "marwaALzaabi/plant-identification-vit"

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)

    return processor, model

processor, model = load_model()

# ----------------------------
# 🌱 Pflanzen → Boden
# ----------------------------
plant_to_soil = {
    "dandelion": "nährstoffreich",
    "nettle": "stickstoffreich, feucht",
    "clover": "stickstoffarm",
    "daisy": "nährstoffarm bis mittel",
    "plant": "unbekannt"
}

soil_to_plants = {
    "nährstoffreich": ["Tomate", "Zucchini"],
    "stickstoffreich, feucht": ["Kohl", "Gurke"],
    "stickstoffarm": ["Erbsen", "Lavendel"],
    "nährstoffarm bis mittel": ["Rosmarin", "Lavendel"]
}

# ----------------------------
# 📷 Upload
# ----------------------------
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    st.write("🔍 Analysiere Pflanze...")

    # ----------------------------
    # 🤖 Prediction
    # ----------------------------
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    topk = torch.topk(probs, 3)

    labels = [model.config.id2label[i.item()] for i in topk.indices[0]]
    scores = topk.values[0]

    st.subheader("🌿 Ergebnisse:")

    top_plant = labels[0].lower()

    for label, score in zip(labels, scores):
        st.write(f"👉 {label} ({round(score.item()*100,2)}%)")

    # ----------------------------
    # 🌱 Boden
    # ----------------------------
    soil = plant_to_soil.get(top_plant, "unbekannt / gemischt")

    st.subheader("🌱 Bodenanalyse")
    st.success(soil)

    # ----------------------------
    # 🌿 Empfehlungen
    # ----------------------------
    st.subheader("🌿 Empfehlungen")

    for p in soil_to_plants.get(soil, []):
        st.write("🌿", p)

    # ----------------------------
    # 💡 Erklärung
    # ----------------------------
    st.subheader("💡 Erklärung")
    st.write(
        "Das Modell erkennt die wahrscheinlichste Pflanze im Bild. "
        "Diese wird mit einer eigenen Boden-Datenbank verknüpft."
    )
