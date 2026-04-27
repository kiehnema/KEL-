import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# =============================
# SEITE
# =============================
st.set_page_config(page_title="🌿 Wildpflanzen KI", page_icon="🌱")
st.title("🌿 Wildpflanzen & Bodenanalyse (leichtes KI-Modell)")

st.write("Lade ein Bild einer Pflanze hoch.")

# =============================
# MODELL LADEN (Keras Seedlings Model)
# =============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)

    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    return model, class_names

model, class_names = load_model()

# =============================
# NORMALISIERUNG (für DB)
# =============================
def normalize(label):
    label = label.lower()

    if "dandelion" in label or "taraxacum" in label:
        return "loewenzahn"
    if "nettle" in label or "urtica" in label:
        return "brennnessel"
    if "clover" in label or "trifolium" in label:
        return "klee"
    if "daisy" in label:
        return "gaensebluemchen"
    if "plant" in label:
        return "unbekannt"

    return "unbekannt"

# =============================
# UPLOAD
# =============================
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    st.write("🔍 Analysiere Pflanze...")

    # =============================
    # PREPROCESSING (Teachable Machine Style)
    # =============================
    size = (224, 224)
    image = image.resize(size)

    image_array = np.asarray(image).astype(np.float32)

    # Normalisierung wie Teachable Machine
    normalized = (image_array / 127.5) - 1

    data = np.expand_dims(normalized, axis=0)

    # =============================
    # PREDICTION
    # =============================
    prediction = model.predict(data)

    index = np.argmax(prediction)
    confidence = float(prediction[0][index])

    raw_label = class_names[index]

    st.subheader("🌿 Ergebnisse:")

    st.success(f"{raw_label}")
    st.write(f"Confidence: {round(confidence * 100, 2)} %")

    # =============================
    # NORMALISIERUNG
    # =============================
    plant_key = normalize(raw_label)

    st.subheader("🌱 Erkannte Pflanzenklasse")
    st.info(plant_key)

    # =============================
    # EINFACHE BODENLOGIK (offline)
    # =============================
    soil_map = {
        "loewenzahn": "nährstoffreich",
        "brennnessel": "stickstoffreich",
        "klee": "stickstoffarm",
        "gaensebluemchen": "mittel",
    }

    soil = soil_map.get(plant_key, "unbekannt")

    st.subheader("🌱 Bodenanalyse")
    st.success(soil)

    # =============================
    # EMPFEHLUNGEN
    # =============================
    recommendations = {
        "nährstoffreich": ["Tomate", "Zucchini"],
        "stickstoffreich": ["Kohl", "Gurke"],
        "stickstoffarm": ["Lavendel", "Rosmarin"],
        "mittel": ["Salat", "Karotten"]
    }

    st.subheader("🌿 Empfehlungen")

    for plant in recommendations.get(soil, []):
        st.write("🌿", plant)

    if soil == "unbekannt":
        st.warning("Keine passenden Empfehlungen gefunden")
