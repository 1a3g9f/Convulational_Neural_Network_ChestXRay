"""Plot feature maps from an early, middle and late layer of the backbone for one Normal and one Pneumonia image.
  python -m src.featuremaps --model efficientnet_b0                # trained weights if models/<name>.keras exists
  python -m src.featuremaps --model efficientnet_b0 --random       # untrained, to compare
"""
import argparse

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import FIG_DIR, MODEL_NAMES, MODELS_DIR
from .data import load_splits
from .infer import load_model
from .models import build_model
from .preprocess import prepare, to_model_input


def pick_layers(backbone):
    """early = last layer at 112x112, middle = last layer at 28x28, late = backbone output."""
    last = {}
    for l in backbone.layers:
        try:
            shp = l.output.shape
        except Exception:
            continue
        if len(shp) == 4:
            last[int(shp[1])] = l
    return [last[112], last[28], backbone.layers[-1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0", choices=MODEL_NAMES)
    ap.add_argument("--random", action="store_true", help="use untrained weights")
    ap.add_argument("--channels", type=int, default=16)
    a = ap.parse_args()
    model = build_model(a.model, pretrained=False) if (a.random or not (MODELS_DIR / f"{a.model}.keras").exists()) else load_model(a.model)
    trained = not (a.random or not (MODELS_DIR / f"{a.model}.keras").exists())
    bb = model.get_layer("backbone")
    layers = pick_layers(bb)
    fm = keras.Model(bb.input, [l.output for l in layers])
    pre = [l for l in model.layers if l.name.startswith("pre_")]

    df = load_splits(); te = df[df.split == "test"]
    samples = [te[te.label == 0].iloc[0], te[te.label == 1].iloc[0]]
    for s, tag in zip(samples, ["normal", "pneumonia"]):
        with Image.open(s["path"]) as im:
            arr = prepare(im)
        h = to_model_input(arr[None])
        for l in pre:
            h = l(h)
        acts = fm(h, training=False)
        fig, axes = plt.subplots(3, a.channels + 1, figsize=(2 * (a.channels + 1) * 0.6, 3 * 1.3))
        for r, (act, l) in enumerate(zip(acts, layers)):
            act = np.asarray(act)[0]
            axes[r, 0].imshow(arr, cmap="gray"); axes[r, 0].set_ylabel(["early", "middle", "late"][r] + f"\n{l.name[:14]}", fontsize=7)
            for c in range(a.channels):
                axes[r, c + 1].imshow(act[..., c], cmap="viridis")
            for ax in axes[r]:
                ax.set_xticks([]); ax.set_yticks([])
        plt.suptitle(f"{a.model} feature maps - {tag}" + (" (trained)" if trained else " (random init)"))
        plt.tight_layout(); out = FIG_DIR / f"featuremaps_{a.model}_{tag}{'' if trained else '_random'}.png"
        plt.savefig(out, dpi=130); plt.close(); print("saved", out)


if __name__ == "__main__":
    main()
