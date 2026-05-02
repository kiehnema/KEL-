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
st.markdown("""<style>
.stApp { background-color: #f8f9fa; color: #212529; }
h1 { color: #2e7d32; text-align: center; }
</style>""", unsafe_allow_html=True)

# =============================
# SUPABASE
# =============================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================
# PAGE
# =============================
st.set_page_config(page_title="🌿 Wildpflanzen KI", layout="wide")
st.title("🌿 Wildpflanzen & Bodenanalyse")

# =============================
# MODEL
# =============================
@st.cache_resource
def load_model():
    model_name = "janjibDEV/vit-plantnet300k"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model.eval()
    return processor, model

with st.spinner("📥 Lade Modell..."):
    processor, model = load_model()

# =============================
# SAFE LABEL HANDLING 🔥
# =============================
def get_label(model, idx):
    try:
        label = model.config.id2label.get(int(idx), str(idx))
    except:
        label = str(idx)
    return str(label)

# =============================
# MAPPING (FIXED)
# =============================
def map_plant(label):
    label = str(label).lower()  # 🔥 FIX

    result = {
        "raw": label,
        "db_key": "unbekannt",
        "group": "unbekannt",
        "scientific_name": None
    }

    mapping = {
        "urtica": ("brennnessel", "Urtica dioica"),
        "lamium": ("taubnessel", "Lamium album"),
        "taraxacum": ("loewenzahn", "Taraxacum officinale"),
        "trifolium": ("klee", "Trifolium pratense"),
    }

    for key, (db, sci) in mapping.items():
        if key in label:
            result["db_key"] = db
            result["group"] = key
            result["scientific_name"] = sci

    return result

# =============================
# DB
# =============================
def get_plant_data(key):
    try:
        res = supabase.table("plants").select("*").eq("plant_key", key).execute()
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
        st.image(image, use_column_width=True)

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        topk = torch.topk(probs, 5)

        indices = topk.indices[0]
        scores = topk.values[0]

        # 🔥 SAFE LABELS
        labels = [get_label(model, i.item()) for i in indices]

        # Anzeige
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
