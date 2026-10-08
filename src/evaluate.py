"""Pick the decision threshold on VAL only, then evaluate ONCE on TEST.
  python -m src.evaluate --model efficientnet_b0                       # balanced threshold (Youden's J), default
  python -m src.evaluate --model efficientnet_b0 --criterion sens --target_sens 0.95   # sensitivity-first
Why Youden by default: the old setup forced 96% sensitivity, which on a shifted val set pushed the
threshold down to ~0.13, so many normal X-rays scored above it. Youden's J maximises
sensitivity + specificity - 1, giving a threshold that does not favour either error.
Both operating points are reported so you can discuss the trade-off.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from .config import FIG_DIR, MODELS_DIR, PREDS_DIR, RESULTS_DIR
from .data import load_images, load_splits
from .infer import load_meta, load_model, predict_prob


def choose_threshold(y, p, criterion="youden", target_sens=0.95):
    fpr, tpr, thr = roc_curve(y, p)
    if criterion == "sens":                       # highest specificity subject to sensitivity >= target
        ok = np.where(tpr >= target_sens)[0]
        i = ok[np.argmin(fpr[ok])]
    else:                                         # Youden's J
        i = int(np.argmax(tpr - fpr))
    return float(min(thr[i], 1.0))


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(thr), "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "precision": tp / max(tp + fp, 1), "roc_auc": float(roc_auc_score(y, p)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def bootstrap_ci(y, p, thr, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = {"sensitivity": [], "specificity": [], "roc_auc": []}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        m = metrics(y[idx], p[idx], thr)
        for k in out:
            out[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def plots(name, yv, pv, yt, pt, m):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for y, p, lab in [(yv, pv, "val"), (yt, pt, "test")]:
        f, t, _ = roc_curve(y, p)
        ax[0].plot(f, t, label=f"{lab} AUC={roc_auc_score(y, p):.3f}")
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_xlabel("1 - specificity"); ax[0].set_ylabel("sensitivity")
    ax[0].legend(); ax[0].set_title(f"ROC - {name}")
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax[1].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[1].text(j, i, v, ha="center", va="center", fontsize=14)
    ax[1].set_xticks([0, 1], ["Normal", "Pneumonia"]); ax[1].set_yticks([0, 1], ["Normal", "Pneumonia"])
    ax[1].set_xlabel("predicted"); ax[1].set_ylabel("true"); ax[1].set_title(f"Test confusion (thr={m['threshold']:.3f})")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"eval_{name}.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint name in models/, e.g. efficientnet_b0")
    ap.add_argument("--criterion", choices=["youden", "sens"], default="youden")
    ap.add_argument("--target_sens", type=float, default=0.95)
    a = ap.parse_args()

    meta = load_meta(a.model)
    model = load_model(a.model)
    df = load_splits(meta["split_mode"])
    va, xv = load_images(df[df.split == "val"])
    te, xt = load_images(df[df.split == "test"])
    yv, yt = va.label.values, te.label.values
    pv, pt = predict_prob(model, xv), predict_prob(model, xt)       # test is touched once, thresholds come from val

    thr_main = choose_threshold(yv, pv, a.criterion, a.target_sens)
    alt = {"youden": choose_threshold(yv, pv, "youden"), f"sens{int(a.target_sens * 100)}": choose_threshold(yv, pv, "sens", a.target_sens)}
    pd.DataFrame({"path": te["path"].values, "label": yt, "prob": pt}).to_csv(PREDS_DIR / f"{a.model}_test.csv", index=False)

    m = metrics(yt, pt, thr_main)
    m["ci95"] = bootstrap_ci(yt, pt, thr_main)
    m["val"] = metrics(yv, pv, thr_main)
    m["criterion"] = a.criterion
    m["operating_points_on_test"] = {k: metrics(yt, pt, t) for k, t in alt.items()}
    m["split_mode"] = meta["split_mode"]
    json.dump(m, open(RESULTS_DIR / f"{a.model}_test.json", "w"), indent=2)
    json.dump({"threshold": thr_main, "arch": meta["arch"], "criterion": a.criterion},
              open(MODELS_DIR / f"{a.model}_threshold.json", "w"))
    plots(a.model, yv, pv, yt, pt, m)

    print(f"split mode: {meta['split_mode']} | test n={len(te)} | threshold ({a.criterion}, chosen on val) = {thr_main:.4f}")
    for k, v in {"CHOSEN": m, **m["operating_points_on_test"]}.items():
        print(f"  {k:8s} thr={v['threshold']:.3f}  sens={v['sensitivity']:.3f}  spec={v['specificity']:.3f}  "
              f"prec={v['precision']:.3f}  AUC={v['roc_auc']:.3f}  (TN {v['tn']} FP {v['fp']} FN {v['fn']} TP {v['tp']})")
    print("95% CI:", {k: [round(x, 3) for x in v] for k, v in m["ci95"].items()})
    ok = m["sensitivity"] >= 0.95 and m["specificity"] >= 0.80
    print("SUCCESS CRITERIA MET" if ok else "success criteria NOT met - report honestly and analyse why")


if __name__ == "__main__":
    main()
