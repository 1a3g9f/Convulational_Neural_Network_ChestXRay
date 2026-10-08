"""Flask front end for the pneumonia X-ray screening project (TensorFlow).

Run from the project root:
    python app/flask_app.py
Then open http://127.0.0.1:5000

Env vars:
    XRAY_MODEL  checkpoint name in models/ (default: efficientnet_b0)
    PORT        port to listen on (default: 5000)
"""
import base64
import io
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.exceptions import HTTPException

from src.config import DEFAULT_MODEL
from src.gradcam import GradCAM, overlay
from src.guard import InputGuard
from src.infer import load_model, load_threshold
from src.preprocess import prepare, to_model_input

MAX_MB = 10
MODEL_NAME = os.environ.get("XRAY_MODEL", DEFAULT_MODEL)
Image.MAX_IMAGE_PIXELS = 200_000_000     # large scans are fine; a decompression bomb still raises

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def load():
    model = load_model(MODEL_NAME)
    return model, load_threshold(MODEL_NAME), GradCAM(model), InputGuard()


try:
    model, thr, cam_fn, guard = load()
except Exception as e:  # model not trained / evaluated yet
    sys.exit(f"Could not load model '{MODEL_NAME}'. Train and evaluate it first (see README).\nDetails: {e}")

infer_lock = threading.Lock()     # one inference at a time (keeps memory use predictable)


def data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def fail(message: str, status: int):
    return jsonify(ok=False, error=message), status


@app.get("/")
def index():
    return render_template("index.html", model_name=MODEL_NAME, threshold=round(thr, 4), max_mb=MAX_MB)


@app.post("/predict")
def predict():
    f = request.files.get("file")
    if f is None or not f.filename:
        return fail("No file was received. Choose a chest X-ray image.", 400)

    name = Path(f.filename).name
    if name.startswith("._"):
        return fail("This is a hidden macOS metadata file (its name starts with '._'), not an X-ray. "
                    "Upload the real image with the same name without the '._' prefix.", 400)
    if name.lower().endswith((".dcm", ".dicom")):
        return fail("DICOM files are not supported. Export the image as PNG or JPG first.", 400)

    # No file-extension whitelist: whatever Pillow can decode is accepted (.jfif, .jpe, .gif, no extension, ...).
    try:
        img = Image.open(f.stream)
        img.load()
    except Exception as e:
        return fail(f"Could not read this file as an image ({type(e).__name__}). Use a JPG, PNG, BMP, TIFF or WEBP file.", 400)

    res = guard.check(img)
    if not res.ok:
        return fail(f"Image rejected: {res.reason} Upload a frontal chest X-ray.", 422)

    arr = prepare(img)                                    # same preprocessing as training
    with infer_lock:
        cam, p = cam_fn(to_model_input(arr[None]))
    prob = float(p[0])
    positive = prob >= thr

    shown = Image.fromarray(arr)
    thumb = shown.copy()
    thumb.thumbnail((96, 96))
    return jsonify(
        ok=True,
        name=name,
        positive=positive,
        label="Pneumonia suspected" if positive else "No pneumonia detected",
        probability=prob,
        threshold=thr,
        warning=res.warning,
        original=data_url(shown.resize((448, 448))),
        heatmap=data_url(overlay(arr, cam[0]).resize((448, 448))),
        thumb=data_url(thumb),
    )


@app.errorhandler(Exception)
def unhandled(e):
    if isinstance(e, HTTPException):
        if e.code == 413:
            return fail(f"File is larger than {MAX_MB} MB.", 413)
        return fail(e.description or e.name, e.code or 500)
    app.logger.exception("Unhandled error while analysing an image")
    return fail(f"Server error while analysing this image: {type(e).__name__}: {e}", 500)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
