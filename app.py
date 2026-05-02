import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from supabase import create_client
import pandas as pd
import plotly.express as px

# =============================
# 🌿 UI STYLES (LESBAR & ANSPRECHEND)
# =============================
st.markdown("""
<style>
/* --- Grundlegende Styles für bessere Lesbarkeit --- */
.stApp {
    background-color: #f8f9fa;  /* Hellgrauer Hintergrund */
    color: #212529;            /* Dunkler Text für bessere Lesbarkeit */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* --- Überschriften --- */
h1 {
    color: #2e7d32;           /* Dunkelgrün für Titel */
    text-align: center;
    margin-bottom: 20px;
}
h2, h3 {
    color: #1b5e20;           /* Etwas heller als h1 */
    margin-top: 20px;
}

/* --- Status-Boxen (für Erfolge, Warnungen, Fehler) --- */
.status-box {
    padding: 15px;
    border-radius: 12px;
    margin: 10px 0;
    font-size: 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    color: #212529;  /* Dunkler Text für Lesbarkeit */
}
.success {
    background-color: #e6f4ea;
    border-left: 6px solid #2e7d32;
}
.warning {
    background-color: #fff8e1;
    border-left: 6px solid #f9a825;
}
.error {
    background-color: #fdecea;
    border-left: 6px solid #c62828;
}

/* --- Buttons & File Uploader --- */
.stButton>button {
    background-color: #4caf50 !important;
    color: white !important;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
    margin: 10px 0;
}
.stButton>button:hover {
    background-color: #388e3c !important;
}
.stFileUploader>div>div>div>div {
    border: 2px dashed #4caf50;
    border-radius: 8px;
    padding: 20px;
    background-color: white;
    text-align: center;
}

/* --- Text & Tabellen --- */
.stTextInput>div>div>input, .stFileUploader>div>div>div>span {
    color: #212529 !important;
}
.stDataFrame {
    background-color: white;
    color: #212529;
    border-radius: 8px;
}

/* --- Fortschrittsbalken --- */
.stProgress>div>div>div>div {
    background-color: #4caf50;
}

/* --- Bilder --- */
.stImage {
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin: 10px 0;
}

/* --- Abstände & Layout --- */
.stContainer {
    padding: 10px;
}
.stMarkdown {
    color: #212529;
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

# Titel mit Beschreibungs-Text
st.title("🌿 Wildpflanzen & Bodenanalyse")
st.markdown("""
<div style="text-align: center; margin-bottom: 30px; color: #495057;">
    <p>Lade ein Bild einer Wildpflanze hoch, um sie zu identifizieren und passende Boden- und Pflanzempfehlungen zu erhalten.</p>
</div>
""", unsafe_allow_html=True)

# =============================
# MODELL (DIREKT VON HUGGING FACE)
# =============================
@st.cache_resource
def load_model():
    model_name = "janjibDEV/vit-plantnet300k"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()  # Wichtig für Inference
    return processor, model

# Lade das Modell mit Fortschrittsbalken
with st.spinner("📥 Lade KI-Modell (nur beim ersten Start)..."):
    processor, model = load_model()
st.success("✅ Modell erfolgreich geladen!")

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
        "calluna": {"db_key": "heidekraut", "group": "Heidekraut", "scientific_name": "Calluna vulgaris"},
        "thymus": {"db_key": "thymian", "group": "Thymian", "scientific_name": "Thymus vulgaris"},
        "matricaria": {"db_key": "kamille", "group": "Kamille", "scientific_name": "Matricaria chamomilla"},
        "chamomilla": {"db_key": "kamille", "group": "Kamille", "scientific_name": "Matricaria chamomilla"},
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
        st.error(f"❌ Datenbankfehler: {e}")
        return None

# =============================
# UPLOAD & ANALYSE
# =============================
st.markdown("---")  # Trennlinie für bessere Struktur
st.subheader("📷 Pflanze hochladen")

uploaded_file = st.file_uploader(
    "Wähle ein Bild einer Wildpflanze aus (JPG/PNG)",
    type=["jpg", "png", "jpeg"],
    help="Lade ein klares Foto mit gut sichtbarer Pflanze hoch."
)

if uploaded_file:
    try:
        # Bild anzeigen
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

        st.markdown("---")  # Trennlinie
        st.subheader("🔍 KI-Analyse läuft...")

        # Modellvorhersage
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        topk = torch.topk(probs, 5)

        labels = [model.config.id2label[i.item()] for i in topk.indices[0]]
        scores = topk.values[0]

        # Ergebnisse anzeigen
        st.subheader("🌿 Erkennungsergebnisse")
        results_df = pd.DataFrame({
            "Pflanze": labels,
            "Wahrscheinlichkeit": [f"{round(score.item()*100, 2)}%" for score in scores]
        })
        st.dataframe(results_df, hide_index=True, use_container_width=True)

        # Top-Erkennung hervorheben
        st.success(f"🎯 **Beste Übereinstimmung**: {labels[0]} ({round(scores[0].item()*100, 2)}%)")

        # Mapping
        mapped = map_plant(labels[0])
        st.markdown("---")
        st.subheader("🌱 Pflanzeninformationen")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Wissenschaftlicher Name:**")
            st.write(mapped.get("scientific_name", "Unbekannt"))
            st.markdown("**Gruppe:**")
            st.write(mapped["group"])
        with col2:
            if mapped["note"]:
                st.warning(mapped["note"])

        # Datenbankabfrage
        if mapped["db_key"] != "unbekannt":
            st.markdown("---")
            st.subheader("📚 Datenbankabgleich")
            data = get_plant_data(mapped["db_key"])
            if data:
                st.markdown(f"""
                <div class="status-box success">
                ✅ <b>Pflanze in Datenbank gefunden!</b><br>
                Boden: {data.get('soil', 'Unbekannt')}<br>
                Feuchtigkeit: {data.get('moisture', 'Unbekannt')}<br>
                Lichtbedarf: {data.get('sun', 'Unbekannt')}
                </div>
                """, unsafe_allow_html=True)

                if "recommendations" in data:
                    st.markdown("**🌿 Empfehlungen:**")
                    if isinstance(data["recommendations"], list):
                        for rec in data["recommendations"]:
                            st.write(f"- {rec}")
                    else:
                        st.write(data["recommendations"])
            else:
                st.markdown(f"""
                <div class="status-box warning">
                ⚠️ <b>Pflanze erkannt, aber keine Datenbank-Übereinstimmung.</b><br>
                Versuche eine andere Pflanze oder ergänze die Datenbank.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-box error">
            ❌ <b>Pflanze nicht erkannt.</b><br>
            Tipps:
            <ul>
                <li>Verwende ein klareres Bild mit besserer Beleuchtung.</li>
                <li>Fokussiere die Pflanze (Blätter + Blüten).</li>
                <li>Probiere eine andere Perspektive.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Ein Fehler ist aufgetreten: {e}")
