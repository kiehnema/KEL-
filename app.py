import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from supabase import create_client

# =============================
# 🌿 UI STYLES
# =============================
st.markdown("""
<style>
.status-box {
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
    font-size: 16px;
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
st.set_page_config(page_title="🌿 Wildpflanzen KI", page_icon="🌱")
st.title("🌿 Wildpflanzen & Bodenanalyse (AI + DB)")
st.write("Lade ein Bild einer Pflanze hoch.")

# =============================
# MODELL LADEN
# =============================
@st.cache_resource
def load_model():
    model_name = "marwaALzaabi/plant-identification-vit"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

processor, model = load_model()

# =============================
# 🌿 NEUES 3-LEVEL PLANT MAPPING
# =============================
def map_plant(label):

    label = label.lower()

    result = {
        "raw": label,
        "db_key": "unbekannt",
        "group": None,
        "note": None
    }

    # 🌿 Brennnessel / Taubnessel Unterscheidung
    if "urtica" in label:
        result["db_key"] = "brennnessel"
        result["group"] = "Echte Brennnessel (Urtica)"
        result["note"] = "Echte Brennnessel"

    elif "lamium" in label:
        result["db_key"] = "brennnessel"
        result["group"] = "Taubnessel (Lamium)"
        result["note"] = "⚠️ KEINE echte Brennnessel – nur ähnliche Pflanze"

    # 🌿 Löwenzahn
    elif "taraxacum" in label:
        result["db_key"] = "loewenzahn"
        result["group"] = "Löwenzahn"

    # 🌿 Klee
    elif "trifolium" in label:
        result["db_key"] = "klee"
        result["group"] = "Klee"

    # 🌿 Heidekraut
    elif "calluna" in label:
        result["db_key"] = "heidekraut"
        result["group"] = "Heidekraut"

    # 🌿 Thymian
    elif "thymus" in label:
        result["db_key"] = "thymian"
        result["group"] = "Thymian"

    # 🌿 Kamille
    elif "matricaria" in label or "chamomilla" in label:
        result["db_key"] = "kamille"
        result["group"] = "Kamille"

    # 🌿 Farn
    elif "dryopteris" in label or "pteridium" in label:
        result["db_key"] = "farn"
        result["group"] = "Farn"

    # 🌿 Schafgarbe
    elif "achillea" in label:
        result["db_key"] = "schafgarbe"
        result["group"] = "Schafgarbe"

    # 🌿 Sumpfdotterblume
    elif "caltha" in label:
        result["db_key"] = "sumpfdotterblume"
        result["group"] = "Sumpfdotterblume"

    # 🌿 Seggen
    elif "carex" in label:
        result["db_key"] = "seggen"
        result["group"] = "Seggen"

    else:
        result["db_key"] = "unbekannt"

    return result

# =============================
# SUPABASE
# =============================
def get_plant_data(plant_key):
    res = supabase.table("plants") \
        .select("*") \
        .eq("plant_key", plant_key) \
        .execute()

    if res.data:
        return res.data[0]
    return None

# =============================
# UPLOAD
# =============================
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    st.write("🔍 Analysiere Pflanze...")

    # =============================
    # PREDICTION
    # =============================
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    topk = torch.topk(probs, 3)

    labels = [model.config.id2label[i.item()] for i in topk.indices[0]]
    scores = topk.values[0]

    raw_label = labels[0]
    confidence = float(scores[0])

    st.subheader("🌿 Ergebnisse:")

    for label, score in zip(labels, scores):
        st.write(f"👉 {label} ({round(score.item()*100,2)}%)")

    st.success(f"🌿 Top-Erkennung: {raw_label} ({round(confidence*100,2)}%)")

    # =============================
    # 🌿 MAPPING (NEU)
    # =============================
    mapped = map_plant(raw_label)
    plant_key = mapped["db_key"]

    st.subheader("🌱 Pflanzen-Analyse")

    st.write("🔬 Exakte Art:", mapped["raw"])
    st.write("🌿 Gruppe:", mapped["group"] or "unbekannt")

    if mapped["note"]:
        st.warning(mapped["note"])

    st.info(f"DB-Key: {plant_key}")

    # =============================
    # SUPABASE
    # =============================
    plant_data = None

    if plant_key != "unbekannt":
        plant_data = get_plant_data(plant_key)

    # =============================
    # UI LOGIK
    # =============================

    if plant_key == "unbekannt":

        st.markdown(f"""
        <div class="status-box warning">
        ⚠️ <b>Unsichere Erkennung</b><br><br>
        Das Modell konnte die Pflanze nicht eindeutig zuordnen.
        </div>
        """, unsafe_allow_html=True)

    elif plant_data is None:

        st.markdown(f"""
        <div class="status-box error">
        🌿 <b>Pflanze erkannt, aber keine Daten gefunden</b><br><br>
        Erkannt: <b>{plant_key}</b><br>
        Gruppe: <b>{mapped['group']}</b><br>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="status-box success">
        🌿 <b>Pflanze erfolgreich erkannt</b><br><br>
        <b>{mapped['group']}</b><br>
        Datenbankeintrag gefunden.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🌱 Bodenanalyse (Supabase)")
        st.write("Boden:", plant_data.get("soil"))
        st.write("Feuchtigkeit:", plant_data.get("moisture"))
        st.write("Sonne:", plant_data.get("sun"))

        st.subheader("🌿 Empfehlungen")
        st.success(plant_data.get("recommendations"))
