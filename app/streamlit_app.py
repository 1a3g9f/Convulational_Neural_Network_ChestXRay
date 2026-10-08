"""Run:  streamlit run app/streamlit_app.py
Env vars: XRAY_MODEL (checkpoint name in models/, default efficientnet_b0)"""
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import streamlit as st
from PIL import Image

from src.config import DEFAULT_MODEL
from src.gradcam import GradCAM, overlay
from src.guard import InputGuard
from src.infer import load_model, load_threshold
from src.preprocess import prepare, to_model_input

MAX_MB = 10
MODEL_NAME = os.environ.get("XRAY_MODEL", DEFAULT_MODEL)

st.set_page_config(page_title="Pneumonia X-ray screening", layout="wide")
st.warning("**Not a diagnostic device.** For educational use only. Always consult a qualified clinician. "
           "The model was trained on paediatric (age 1-5) chest X-rays; adult X-rays or images from other sources can give unreliable results.")
st.title("Paediatric chest X-ray pneumonia screening")


@st.cache_resource
def load():
    model = load_model(MODEL_NAME)
    return model, load_threshold(MODEL_NAME), GradCAM(model), InputGuard()


try:
    model, thr, cam_fn, guard = load()
except Exception as e:
    st.error(f"Could not load model '{MODEL_NAME}'. Train and evaluate it first (see README). Details: {e}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

up = st.file_uploader("Upload a chest X-ray (JPG/PNG/BMP/TIFF/WEBP)",
                      type=["jpg", "jpeg", "jfif", "png", "bmp", "tif", "tiff", "webp"])
if up is not None:
    if up.size > MAX_MB * 1024 * 1024:
        st.error(f"File is larger than {MAX_MB} MB.")
    else:
        try:
            img = Image.open(up); img.load()
        except Exception:
            st.error("Could not read this file as an image."); img = None
        if img is not None:
            res = guard.check(img)
            if not res.ok:
                st.error(f"Image rejected: {res.reason} Please upload a frontal chest X-ray.")
            else:
                arr = prepare(img)
                cam, p = cam_fn(to_model_input(arr[None])); prob = float(p[0])
                label = "PNEUMONIA suspected" if prob >= thr else "No pneumonia detected"
                (st.error if prob >= thr else st.success)(f"**{label}** - probability {prob:.1%} (decision threshold {thr:.3f})")
                if res.warning:
                    st.warning(res.warning)
                c1, c2 = st.columns(2)
                c1.image(Image.fromarray(arr).resize((448, 448)), caption="Original (contrast-normalised, borders cropped)", use_container_width=True)
                c2.image(overlay(arr, cam[0]).resize((448, 448)), caption="Grad-CAM heatmap", use_container_width=True)
                thumb = Image.fromarray(arr); thumb.thumbnail((96, 96))
                st.session_state.history.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "name": up.name,
                                                    "label": label, "prob": prob, "thumb": thumb})

st.divider()
h1, h2 = st.columns([5, 1])
h1.subheader("Session history")
if h2.button("Clear history"):
    st.session_state.history = []; st.rerun()
for item in st.session_state.history:
    a, b = st.columns([1, 6])
    a.image(item["thumb"])
    b.write(f"**{item['label']}** - {item['prob']:.1%} - {item['name']} - {item['time']}")
st.caption("Uploaded images are processed in memory and are not stored beyond this session.")
