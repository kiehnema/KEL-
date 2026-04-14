import streamlit as st
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# ----------------------------
# 🌿 App Setup
# ----------------------------
st.set_page_config(page_title="🌿 Pflanzen KI", page_icon="🌱")
st.title("🌿 KI Pflanzen- & Bodenanalyse")

# ----------------------------
# 🤖 Modell
# ----------------------------
@st.cache_resource
def load_model():
    model_name = "maxefrost/plant_image_classifier_random"

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)

    return feature_extractor, model

feature_extractor, model = load_model()

# ----------------------------
# 🌱 Logik
# ----------------------------
plant_to_soil = {
    "nettle": "stickstoffreich, feucht",
    "dandelion": "nährstoffreich",
    "clover": "stickstoffarm",
    "daisy": "nährstoffarm bis mittel"
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
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    st.write("🔍 Analysiere...")

    # ----------------------------
    # 🤖 Prediction
    # ----------------------------
    inputs = feature_extractor(images=image, return_tensors="pt")

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

    st.success(f"Boden: {soil}")

    # ----------------------------
    # 🌿 Empfehlungen
    # ----------------------------
    st.subheader("🌿 Empfehlungen:")

    for p in soil_to_plants.get(soil, []):
        st.write("🌿", p)
