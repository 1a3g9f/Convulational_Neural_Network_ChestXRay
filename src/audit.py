"""Grad-CAM audit: sample TP/TN/FP/FN, save overlays, quantify border/corner attention,
and run an occlusion test (mask borders/corners -> does the prediction change?).
  python -m src.audit --model efficientnet_b0 --n 20
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .config import FIG_DIR, MODELS_DIR, PREDS_DIR, RESULTS_DIR
from .gradcam import GradCAM, border_fraction, central_fraction, corner_fraction, overlay
from .infer import load_model, sigmoid
from .preprocess import prepare, to_model_input


def occlude(a, kind):
    """a: uint8 (H,W) prepared image. Fill with the image mean."""
    y = a.copy(); fill = int(a.mean()); h, w = a.shape
    if kind == "border":
        b = int(0.10 * h); y[:b] = fill; y[-b:] = fill; y[:, :b] = fill; y[:, -b:] = fill
    elif kind == "corners":
        c = int(0.25 * h); y[:c, :c] = fill; y[:c, -c:] = fill; y[-c:, :c] = fill; y[-c:, -c:] = fill
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20, help="images per group (TP/TN/FP/FN)")
    a = ap.parse_args()
    model = load_model(a.model); cam_fn = GradCAM(model)
    thr = json.load(open(MODELS_DIR / f"{a.model}_threshold.json"))["threshold"]
    preds = pd.read_csv(PREDS_DIR / f"{a.model}_test.csv")
    preds["pred"] = (preds.prob >= thr).astype(int)

    def prob(arr):
        return float(sigmoid(model(to_model_input(arr[None]), training=False).numpy().ravel()[0]))

    rows, rng = [], np.random.default_rng(0)
    for g, (yt, yp) in {"TP": (1, 1), "TN": (0, 0), "FP": (0, 1), "FN": (1, 0)}.items():
        sub = preds[(preds.label == yt) & (preds.pred == yp)]
        sub = sub.iloc[rng.permutation(len(sub))[: a.n]]
        fig, axes = plt.subplots(4, 5, figsize=(15, 12)); axes = axes.ravel()
        for k, (_, r) in enumerate(sub.iterrows()):
            with Image.open(r["path"]) as im:
                arr = prepare(im)
            cam, p = cam_fn(to_model_input(arr[None])); cam = cam[0]
            pb, pc = prob(occlude(arr, "border")), prob(occlude(arr, "corners"))
            rows.append({"group": g, "path": r["path"], "prob": float(p[0]),
                         "border_frac": border_fraction(cam), "corner_frac": corner_fraction(cam),
                         "central_frac": central_fraction(cam),
                         "prob_border_masked": pb, "prob_corners_masked": pc,
                         "flip_border": int((pb >= thr) != (p[0] >= thr)),
                         "flip_corners": int((pc >= thr) != (p[0] >= thr))})
            if k < len(axes):
                axes[k].imshow(overlay(arr, cam)); axes[k].axis("off"); axes[k].set_title(f"{g} p={p[0]:.2f}", fontsize=9)
        for ax in axes[len(sub):]:
            ax.axis("off")
        plt.tight_layout(); plt.savefig(FIG_DIR / f"gradcam_{a.model}_{g}.png", dpi=110); plt.close()

    df = pd.DataFrame(rows); df.to_csv(RESULTS_DIR / f"{a.model}_audit.csv", index=False)
    summary = df.groupby("group")[["border_frac", "corner_frac", "central_frac", "flip_border", "flip_corners"]].mean()
    summary.to_csv(RESULTS_DIR / f"{a.model}_audit_summary.csv")
    print(summary.round(3))
    print("\nA high border/corner share or many prediction flips after masking suggests a shortcut. "
          "Inspect the saved overlays and document what you see (markers, borders, tubes, etc.).")


if __name__ == "__main__":
    main()
