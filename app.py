import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from supabase import create_client
import pandas as pd
import plotly.express as px

# =============================
# 🌿 UI STYLES
# =============================
st.markdown("""
<style>
.status-box {
    padding: 15px;
    border-radius: 12px;
    margin: 10px 0;
    font-size: 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.success {
    background-color: #e6f4ea;
    border-left: 6px solid #2e7d32;
    color: #1b5e20;
}
.warning {
    background-color: #fff8e1;
    border-left: 6px solid #f9a825;
    color: #e65100;
}
.error {
    background-color: #fdecea;
    border-left: 6px solid #c62828;
    color: #b71c1c;
}
.stApp {
    background-color: #f5f7fa;
}
.stButton>button {
    background-color: #4caf50 !important;
    color: white !important;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# SUPABASE
# =============================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================
# SEITE
# =============================
st.set_page_config(
    page_title="🌿 Wildpflanzen KI",
    page_icon="🌱",
    layout="wide"
)

st.title("🌿 Wildpflanzen & Bodenanalyse")

# =============================
# MODELL (NEU – OHNE LOKALE DATEIEN)
# =============================
@st.cache_resource
def load_model():
    model_name = "janjibDEV/vit-plantnet300k"

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)

    model.eval()  # wichtig für Inference
    return processor, model


with st.spinner("📥 Lade KI-Modell (nur beim ersten Start)..."):
    processor, model = load_model()

# =============================
# MAPPING
# =============================
def map_plant(label):
    label = label.lower()
    result = {
        "raw": label,
        "db_key": "unbekannt",
        "group": "unbekannt",
        "note": None,
        "scientific_name": None
    }

    plant_mapping = {
        "urtica": {"db_key": "brennnessel", "group": "Echte Brennnessel (Urtica)", "scientific_name": "Urtica dioica"},
        "lamium": {"db_key": "taubnessel", "group": "Taubnessel (Lamium)", "scientific_name": "Lamium album"},
        "taraxacum": {"db_key": "loewenzahn", "group": "Löwenzahn", "scientific_name": "Taraxacum officinale"},
        "trifolium": {"db_key": "klee", "group": "Klee", "scientific_name": "Trifolium pratense"},
    }

    for key, value in plant_mapping.items():
        if key in label:
            result.update(value)
            break

    return result

# =============================
# SUPABASE QUERY
# =============================
def get_plant_data(plant_key):
    try:
        res = supabase.table("plants").select("*").eq("plant_key", plant_key).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        st.error(f"DB Fehler: {e}")
        return None

# =============================
# UPLOAD
# =============================
uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Bild", use_column_width=True)

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        topk = torch.topk(probs, 5)

        labels = [model.config.id2label[i.item()] for i in topk.indices[0]]
        scores = topk.values[0]

        st.subheader("🔍 Ergebnisse")
        for l, s in zip(labels, scores):
            st.write(f"{l} → {round(s.item()*100,2)}%")

        # Mapping
        mapped = map_plant(labels[0])
        st.subheader("🌱 Pflanze")
        st.write(mapped)

        # DB
        if mapped["db_key"] != "unbekannt":
            data = get_plant_data(mapped["db_key"])
            if data:
                st.success("Daten gefunden")
                st.write(data)
            else:
                st.warning("Keine DB-Daten gefunden")

    except Exception as e:
        st.error(f"❌ Fehler: {e}")
