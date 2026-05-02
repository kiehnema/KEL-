import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from supabase import create_client
import pandas as pd
import plotly.express as px

# =============================
# 🌿 UI STYLES (ERWEITERT)
# =============================
st.markdown("""
<style>
/* --- Status-Boxen (behalten) --- */
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

/* --- Neue Styles --- */
.stApp {
    background-color: #f5f7fa;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stButton>button {
    background-color: #4caf50 !important;
    color: white !important;
    border-radius: 8px;
    border: none;
    padding: 8px 20px;
}
.stFileUploader>div>div>div>div {
    border: 2px dashed #4caf50;
    border-radius: 8px;
    padding: 20px;
    background-color: white;
}
h1 {
    color: #2e7d32;
    text-align: center;
}
h2, h3 {
    color: #1b5e20;
}
</style>
""", unsafe_allow_html=True)

# =============================
# SUPABASE & KONFIGURATION
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
st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <p>Lade ein Bild einer Wildpflanze hoch, um sie zu identifizieren und passende Boden- und Pflanzempfehlungen zu erhalten.</p>
</div>
""", unsafe_allow_html=True)

# =============================
# MODELL (ANGEPASST FÜR janjibDEV/vit-plantnet300k)
# =============================
@st.cache_resource
def load_model():
    try:
        # Lade Processor und Modell aus dem lokalen Ordner
        processor = AutoImageProcessor.from_pretrained("./model/processor")
        model = AutoModelForImageClassification.from_pretrained("./model/model")
        return processor, model
    except Exception as e:
        st.error(f"❌ Fehler beim Laden des Modells: {e}")
        st.error("Stelle sicher, dass der Ordner 'model/' mit den Dateien 'config.json', 'model.safetensors' und 'preprocessor_config.json' existiert.")
        raise

# Lade das Modell (wird nur einmal geladen dank @st.cache_resource)
processor, model = load_model()

# =============================
# 🌿 BOTANISCHES MAPPING (ERWEITERT)
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
        "lamium": {"db_key": "taubnessel", "group": "Taubnessel (Lamium)", "scientific_name": "Lamium album", "note": "⚠️ KEINE echte Brennnessel – nur ähnliche Blätter"},
        "taraxacum": {"db_key": "loewenzahn", "group": "Löwenzahn", "scientific_name": "Taraxacum officinale"},
        "trifolium": {"db_key": "klee", "group": "Klee", "scientific_name": "Trifolium pratense"},
        "calluna": {"db_key": "heidekraut", "group": "Heidekraut", "scientific_name": "Calluna vulgaris"},
        "thymus": {"db_key": "thymian", "group": "Thymian", "scientific_name": "Thymus vulgaris"},
        "matricaria": {"db_key": "kamille", "group": "Kamille", "scientific_name": "Matricaria chamomilla"},
        "chamomilla": {"db_key": "kamille", "group": "Kamille", "scientific_name": "Matricaria chamomilla"},
        "dryopteris": {"db_key": "farn", "group": "Farn", "scientific_name": "Dryopteris filix-mas"},
        "pteridium": {"db_key": "farn", "group": "Adlerfarn", "scientific_name": "Pteridium aquilinum"},
        "achillea": {"db_key": "schafgarbe", "group": "Schafgarbe", "scientific_name": "Achillea millefolium"},
        "caltha": {"db_key": "sumpfdotterblume", "group": "Sumpfdotterblume", "scientific_name": "Caltha palustris"},
        "carex": {"db_key": "seggen", "group": "Seggen", "scientific_name": "Carex spp."},
        "plantago": {"db_key": "wegeraich", "group": "Wegerich", "scientific_name": "Plantago major"},
        "rumex": {"db_key": "ampfer", "group": "Ampfer", "scientific_name": "Rumex acetosa"},
    }

    for key, value in plant_mapping.items():
        if key in label:
            result.update(value)
            break

    return result

# =============================
# SUPABASE (FEHLERBEHANDLUNG)
# =============================
def get_plant_data(plant_key):
    try:
        res = supabase.table("plants") \
            .select("*") \
            .eq("plant_key", plant_key) \
            .execute()
        return res.data[0] if res.data else None
    except Exception as e:
        st.error(f"❌ Fehler bei der Datenbankabfrage: {e}")
        return None

# Fallback-Daten für unbekannte Pflanzen
def get_fallback_data(plant_group):
    fallback_db = {
        "Echte Brennnessel (Urtica)": {
            "soil": "stickstoffreich, feucht",
            "moisture": "hoch",
            "sun": "Schatten bis Halbschatten",
            "recommendations": "Brennnesseljauche, Kompost, Spinat, Minze"
        },
        "Taubnessel (Lamium)": {
            "soil": "nährstoffreich, humusreich",
            "moisture": "mittel",
            "sun": "Schatten bis Halbschatten",
            "recommendations": "Stauden, Farne, Hostas"
        },
        "Löwenzahn": {
            "soil": "durchlässig, lehmig",
            "moisture": "mittel",
            "sun": "Sonne bis Halbschatten",
            "recommendations": "Kräuter wie Thymian, Salbei, Lavendel"
        },
        "Klee": {
            "soil": "sandig-lehmig, kalkhaltig",
            "moisture": "mittel",
            "sun": "Sonne",
            "recommendations": "Luzerne, Esparcette, Bohnen"
        },
        "Heidekraut": {
            "soil": "sauer, sandig",
            "moisture": "trocken",
            "sun": "Sonne",
            "recommendations": "Erika, Preiselbeeren, Moos"
        },
        "Farn": {
            "soil": "feucht, humusreich",
            "moisture": "hoch",
            "sun": "Schatten bis Halbschatten",
            "recommendations": "Farne, Moose, Schattenstauden"
        },
        "Schafgarbe": {
            "soil": "durchlässig, mager",
            "moisture": "trocken bis mittel",
            "sun": "Sonne",
            "recommendations": "Lavendel, Thymian, Fetthenne"
        },
    }
    return fallback_db.get(plant_group, None)

# =============================
# UPLOAD & ANALYSE
# =============================
uploaded_file = st.file_uploader(
    "📷 Bild hochladen (JPG/PNG)",
    type=["jpg", "png", "jpeg"],
    help="Lade ein klares Foto einer Wildpflanze hoch."
)

if uploaded_file:
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Bild laden und anzeigen
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True, caption="Hochgeladenes Bild")

        progress_bar.progress(20)
        status_text.text("🔍 Bild wird analysiert...")

        # KI PREDICTION
        inputs = processor(images=image, return_tensors="pt")
        progress_bar.progress(40)

        with torch.no_grad():
            outputs = model(**inputs)
        progress_bar.progress(60)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        topk = torch.topk(probs, 5)

        labels = [model.config.id2label[i.item()] for i in topk.indices[0]]
        scores = topk.values[0]

        raw_label = labels[0]
        confidence = float(scores[0])

        progress_bar.progress(80)
        status_text.text("🌿 Pflanze wird identifiziert...")

        # ERGEBNISSE ANZEIGEN
        st.subheader("🔍 KI-Erkennungsergebnisse")
        results_df = pd.DataFrame({
            "Pflanze": labels,
            "Wahrscheinlichkeit": [f"{round(score.item()*100, 2)}%" for score in scores]
        })
        st.dataframe(results_df, hide_index=True, use_container_width=True)

        st.success(f"🌿 **Top-Erkennung**: {raw_label} ({round(confidence*100, 2)}%)")

        # MAPPING & DATENBANK
        mapped = map_plant(raw_label)
        plant_key = mapped["db_key"]
        plant_data = get_plant_data(plant_key) if plant_key != "unbekannt" else None

        st.subheader("🌱 Pflanzen-Informationen")
        col1, col2 = st.columns(2)
        with col1:
            st.write("🔬 **Wissenschaftlicher Name**:", mapped.get("scientific_name", "Unbekannt"))
            st.write("🌿 **Gruppe**:", mapped["group"])
        with col2:
            if mapped["note"]:
                st.warning(mapped["note"])

        progress_bar.progress(90)

        # UI LOGIK
        if plant_key == "unbekannt":
            st.markdown(f"""
            <div class="status-box error">
            ❌ <b>Pflanze nicht erkannt</b><br><br>
            Die KI konnte keine passende Pflanze identifizieren.<br>
            💡 <b>Tipps:</b>
            <ul>
                <li>Versuche ein klareres Foto mit besserer Beleuchtung.</li>
                <li>Fokussiere die Pflanze (Blätter + Blüten).</li>
                <li>Probiere eine andere Perspektive.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        elif plant_data is None:
            fallback_data = get_fallback_data(mapped["group"])
            if fallback_data:
                st.markdown(f"""
                <div class="status-box warning">
                ⚠️ <b>Pflanze erkannt – keine exakte Datenbank-Übereinstimmung</b><br><br>
                🔬 Erkannt: <b>{mapped['group']}</b> ({mapped.get('scientific_name', '')})<br>
                📊 Es wird auf <b>Gruppendaten</b> zurückgegriffen.
                </div>
                """, unsafe_allow_html=True)

                st.subheader("🌱 Bodenanalyse (basierend auf Pflanzengruppe)")
                st.write("🌍 **Bodenart**:", fallback_data["soil"])
                st.write("💧 **Feuchtigkeit**:", fallback_data["moisture"])
                st.write("☀️ **Lichtbedarf**:", fallback_data["sun"])

                st.subheader("🌿 Empfehlungen")
                st.success(fallback_data["recommendations"])
            else:
                st.error("❌ Keine Fallback-Daten für diese Pflanzengruppe verfügbar.")

        else:
            st.markdown(f"""
            <div class="status-box success">
            ✅ <b>Pflanze erkannt & Datenbank-Übereinstimmung gefunden!</b><br><br>
            {mapped['group']} (<i>{mapped.get('scientific_name', '')}</i>)<br>
            Exakter Eintrag in der Datenbank vorhanden.
            </div>
            """, unsafe_allow_html=True)

            st.subheader("🌱 Bodenanalyse")
            soil_data = {
                "Eigenschaft": ["Nährstoffe", "Feuchtigkeit", "pH-Wert", "Durchlässigkeit"],
                "Wert": [
                    5 if "nährstoffreich" in plant_data.get("soil", "").lower() else 3,
                    5 if plant_data.get("moisture") == "hoch" else 3 if plant_data.get("moisture") == "mittel" else 1,
                    7,
                    4 if "durchlässig" in plant_data.get("soil", "").lower() else 2
                ]
            }
            df_soil = pd.DataFrame(soil_data)
            fig = px.radar(df_soil, x="Eigenschaft", y="Wert", range_y=[0, 5], title="Bodenprofil")
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🌍 Boden")
                st.write("**Art**:", plant_data.get("soil", "Unbekannt"))
                st.write("**Feuchtigkeit**:", plant_data.get("moisture", "Unbekannt"))
                st.write("**Licht**:", plant_data.get("sun", "Unbekannt"))
            with col2:
                st.markdown("### 🌿 Empfehlungen")
                recommendations = plant_data.get("recommendations", "Keine Empfehlungen")
                if isinstance(recommendations, list):
                    for rec in recommendations:
                        st.write(f"- {rec}")
                else:
                    st.write(recommendations)

        progress_bar.progress(100)
        status_text.text("✅ Analyse abgeschlossen!")

    except Exception as e:
        st.error(f"❌ Ein Fehler ist aufgetreten: {e}")
        progress_bar.empty()
