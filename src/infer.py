"""Loading saved models and running inference in memory-friendly chunks."""
import json

import keras
import numpy as np

from .config import MODELS_DIR
from .preprocess import to_model_input


def load_model(name: str) -> keras.Model:
    return keras.saving.load_model(MODELS_DIR / f"{name}.keras", compile=False)


def load_meta(name: str) -> dict:
    return json.load(open(MODELS_DIR / f"{name}_meta.json"))


def load_threshold(name: str) -> float:
    return json.load(open(MODELS_DIR / f"{name}_threshold.json"))["threshold"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict_prob(model, x_u8: np.ndarray, chunk: int = 128) -> np.ndarray:
    """x_u8: (N,224,224) uint8 from preprocess.prepare -> probabilities of pneumonia (N,)."""
    out = []
    for i in range(0, len(x_u8), chunk):
        out.append(model.predict(to_model_input(x_u8[i:i + chunk]), batch_size=64, verbose=0).reshape(-1))
    return sigmoid(np.concatenate(out))
