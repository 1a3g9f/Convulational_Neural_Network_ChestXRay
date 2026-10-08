"""Grad-CAM for the Keras models (tf.GradientTape on the backbone's last feature map) + overlay helpers."""
import keras
import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image

from .config import IMG_SIZE


class GradCAM:
    def __init__(self, model):
        layers = [l for l in model.layers if not isinstance(l, keras.layers.InputLayer)]
        i = [l.name for l in layers].index("backbone")
        self.pre, self.backbone, self.post = layers[:i], layers[i], layers[i + 1:]

    def __call__(self, x):
        """x: float32 (B,224,224,3) in 0..255. Returns cams (B,H,W) in [0,1] and probabilities (B,)."""
        h = tf.convert_to_tensor(x, tf.float32)
        for l in self.pre:
            h = l(h)
        with tf.GradientTape() as tape:
            feats = self.backbone(h, training=False)
            tape.watch(feats)
            z = feats
            for l in self.post:
                z = l(z)
            logit = z[:, 0]
        grads = tape.gradient(logit, feats)
        w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(w * feats, axis=-1, keepdims=True))
        cam = tf.image.resize(cam, (IMG_SIZE, IMG_SIZE), method="bilinear")[..., 0]
        mn = tf.reduce_min(cam, axis=(1, 2), keepdims=True)
        mx = tf.reduce_max(cam, axis=(1, 2), keepdims=True)
        cam = (cam - mn) / (mx - mn + 1e-8)
        return cam.numpy(), tf.sigmoid(logit).numpy()


def overlay(gray_u8: np.ndarray, cam: np.ndarray, alpha=0.45) -> Image.Image:
    base = Image.fromarray(gray_u8).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    heat = Image.fromarray((matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8))
    return Image.blend(base, heat.resize(base.size), alpha)


def border_fraction(cam: np.ndarray, band: float = 0.10) -> float:
    """Share of heatmap energy in the outer band of the image (border/collimation shortcuts)."""
    h, w = cam.shape
    b = int(band * min(h, w))
    mask = np.ones_like(cam, bool); mask[b:h - b, b:w - b] = False
    return float(cam[mask].sum() / (cam.sum() + 1e-8))


def corner_fraction(cam: np.ndarray, size: float = 0.25) -> float:
    """Share of energy in the four image corners (where L/R markers and text often sit)."""
    h, w = cam.shape
    ch, cw = int(size * h), int(size * w)
    m = np.zeros_like(cam, bool)
    m[:ch, :cw] = m[:ch, -cw:] = m[-ch:, :cw] = m[-ch:, -cw:] = True
    return float(cam[m].sum() / (cam.sum() + 1e-8))


def central_fraction(cam: np.ndarray) -> float:
    """Crude lung proxy: energy inside the central region (10-90% width, 15-85% height)."""
    h, w = cam.shape
    m = np.zeros_like(cam, bool); m[int(.15 * h):int(.85 * h), int(.10 * w):int(.90 * w)] = True
    return float(cam[m].sum() / (cam.sum() + 1e-8))
