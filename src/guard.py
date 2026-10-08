"""Input guard. Hard-rejects only images that are clearly NOT a chest X-ray; borderline images get a
WARNING but are still analysed (the old guard rejected ~1% of genuine X-rays by design, plus any
tinted, wide-aspect or 16-bit image).
  Layer 1 (heuristics): undecodable/tiny, absurd aspect ratio, blank, real colour photos (tinted greyscale is fine).
  Layer 2 (embedding distance to training X-rays, MobileNetV2 ImageNet features):
        similarity < hard  -> reject      hard <= similarity < soft -> accept with a warning
  python -m src.guard fit [--neg_dir folder_of_non_xray_images]   # --neg_dir calibrates the hard limit
  python -m src.guard eval --neg_dir folder_of_non_xray_images
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
from PIL import Image

from .config import IMG_SIZE, MODELS_DIR, NO_PRETRAINED
from .data import load_images, load_splits
from .preprocess import prepare

GUARD_PATH = MODELS_DIR / "guard.npz"


@dataclass
class GuardResult:
    ok: bool
    reason: str = ""
    warning: str = ""


def _embedder():
    return keras.applications.MobileNetV2(include_top=False, pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                          weights=None if NO_PRETRAINED else "imagenet")


def embed(emb, x_u8):
    x = np.repeat(np.asarray(x_u8, np.float32)[..., None], 3, axis=-1) / 127.5 - 1.0
    v = emb.predict(x, batch_size=64, verbose=0)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


class InputGuard:
    def __init__(self, min_side=100, aspect=(0.5, 2.0)):
        self.min_side, self.aspect = min_side, aspect
        self.emb = self.centroid = self.soft = self.hard = None
        if GUARD_PATH.exists():
            try:
                d = np.load(GUARD_PATH)
                self.centroid, self.soft, self.hard = d["centroid"], float(d["soft"]), float(d["hard"])
                self.emb = _embedder()
            except Exception as e:
                print(f"[guard] embedding layer disabled ({e}); using heuristics only")
                self.emb = None

    @staticmethod
    def _looks_like_colour_photo(img):
        if img.mode not in ("RGB", "RGBA", "P", "CMYK"):
            return False
        small = img.convert("RGB"); small.thumbnail((128, 128))
        hsv = np.asarray(small.convert("HSV"), np.float32) / 255.0
        mask = hsv[..., 1] > 0.2
        if mask.mean() < 0.25:                      # mostly grey -> not a colour photo
            return False
        r = abs(np.mean(np.exp(2j * np.pi * hsv[..., 0][mask])))   # 1.0 = a single tint, 0 = many hues
        return r < 0.85                             # many hues -> a real colour photo; one tint -> tinted X-ray

    def check(self, img: Image.Image) -> GuardResult:
        w, h = img.size
        if min(w, h) < self.min_side:
            return GuardResult(False, f"Image is too small ({w}x{h}px); at least {self.min_side}px on each side is needed.")
        if not (self.aspect[0] <= w / h <= self.aspect[1]):
            return GuardResult(False, "Unusual aspect ratio for a chest X-ray (is it a screenshot or a strip?).")
        if self._looks_like_colour_photo(img):
            return GuardResult(False, "This looks like a colour photo, not an X-ray.")
        arr = prepare(img)
        if arr.std() < 8:
            return GuardResult(False, "Image is almost blank.")
        if self.emb is not None:
            sim = float(embed(self.emb, arr[None])[0] @ self.centroid)
            if sim < self.hard:
                return GuardResult(False, "This doesn't look like a chest X-ray.")
            if sim < self.soft:
                return GuardResult(True, warning="This image looks different from the training X-rays "
                                                 "(different scanner, body part or age group). Treat the result as unreliable.")
        return GuardResult(True)


def fit(n=800, neg_dir=None):
    df = load_splits(); tr = df[df.split == "train"]
    tr, x = load_images(tr.sample(n=min(n, len(tr)), random_state=0))
    emb = _embedder()
    V = embed(emb, x)
    centroid = V.mean(0); centroid /= np.linalg.norm(centroid)
    sims = V @ centroid
    soft = float(np.percentile(sims, 1.0))
    hard = float(soft - 3 * sims.std())
    if neg_dir:
        negs = [p for p in Path(neg_dir).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        ns = np.array([embed(emb, prepare(Image.open(p))[None])[0] @ centroid for p in negs])
        if len(ns):
            hard = float(min(max(ns.max() + 0.02, hard), soft - 0.02))
            print(f"calibrated on {len(ns)} non-X-rays (max similarity {ns.max():.3f})")
    np.savez(GUARD_PATH, centroid=centroid, soft=soft, hard=hard)
    print(f"saved guard | warn below {soft:.3f} | reject below {hard:.3f}")


def evaluate(neg_dir):
    g = InputGuard()
    df = load_splits(); pos = df[df.split == "test"].sample(n=min(200, (df.split == "test").sum()), random_state=0)
    res = [g.check(Image.open(p)) for p in pos["path"]]
    msg = f"X-rays accepted: {np.mean([r.ok for r in res]):.1%} (with warning: {np.mean([bool(r.warning) for r in res]):.1%})"
    if neg_dir:
        negs = [p for p in Path(neg_dir).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if negs:
            msg += f" | non-X-rays rejected: {np.mean([not g.check(Image.open(p)).ok for p in negs]):.1%} (n={len(negs)})"
    print(msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["fit", "eval"]); ap.add_argument("--neg_dir")
    a = ap.parse_args()
    fit(neg_dir=a.neg_dir) if a.cmd == "fit" else evaluate(a.neg_dir)
