"""Train a model with TensorFlow/Keras. Examples:
  python -m src.train --model efficientnet_b0                 # recommended default
  python -m src.train --model custom_cnn --epochs 30
  python -m src.train --model efficientnet_b0 --no_class_weights --tag noweights   # ablation
  python -m src.train --model mobilenet_v2 --limit 300 --epochs 2                  # quick dry run
Phase 1: backbone frozen, head learns (--freeze_epochs). Phase 2: backbone fine-tuned at a lower LR
(BatchNorm layers stay frozen). The checkpoint with the best validation AUC is kept.
"""
import argparse
import json
import math
import time

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import MODEL_NAMES, MODELS_DIR, PREDS_DIR, RESULTS_DIR, SEED, SPLIT_MODE
from .data import class_weights, load_images, load_splits, make_dataset
from .infer import load_model, predict_prob
from .models import build_model, set_backbone_trainable


class BestByAUC(keras.callbacks.Callback):
    """Computes exact validation AUC each epoch, saves the best model, early-stops on no improvement."""

    def __init__(self, path, x_val, y_val, patience):
        super().__init__()
        self.path, self.x_val, self.y_val, self.patience = path, x_val, y_val, patience
        self.best, self.bad, self.log = -1.0, 0, []

    def on_epoch_end(self, epoch, logs=None):
        auc = float(roc_auc_score(self.y_val, predict_prob(self.model, self.x_val)))
        self.log.append({"epoch": epoch + 1, "train_loss": float(logs.get("loss", np.nan)), "val_auc": auc})
        print(f"   epoch {epoch + 1}: train loss {logs.get('loss', float('nan')):.4f} | val AUC {auc:.4f}")
        if auc > self.best:
            self.best, self.bad = auc, 0
            self.model.save(self.path)
        else:
            self.bad += 1
            if self.bad >= self.patience:
                print("   early stopping")
                self.model.stop_training = True


def compile_(model, lr, total_steps):
    sched = keras.optimizers.schedules.CosineDecay(lr, max(int(total_steps), 1), alpha=0.05)
    model.compile(optimizer=keras.optimizers.AdamW(sched, weight_decay=1e-4),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--freeze_epochs", type=int, default=3, help="epochs with the pretrained backbone frozen")
    ap.add_argument("--lr", type=float, default=1e-3, help="LR for the head phase (and for custom_cnn)")
    ap.add_argument("--ft_lr", type=float, default=1e-4, help="LR for the fine-tuning phase")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--split_mode", default=SPLIT_MODE, choices=["pooled", "official"])
    ap.add_argument("--limit", type=int, default=0, help="use only N training images (dry run)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    df = load_splits(a.split_mode)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    if a.limit:
        tr = tr.sample(n=min(a.limit, len(tr)), random_state=SEED)
    tr, x_tr = load_images(tr)
    va, x_va = load_images(va)
    print(f"split={a.split_mode} | train {len(tr)} ({int(tr.label.sum())} pneumonia) | val {len(va)} ({int(va.label.sum())} pneumonia)")

    cw = [1.0, 1.0] if a.no_class_weights else class_weights(tr)
    print("class weights [normal, pneumonia]:", [round(c, 3) for c in cw], "(disabled)" if a.no_class_weights else "")
    train_ds = make_dataset(x_tr, tr.label.values, train=True, batch_size=a.batch_size, cw=cw)

    scratch = a.model == "custom_cnn"
    model = build_model(a.model, pretrained=not scratch)
    name = a.model + (f"_{a.tag}" if a.tag else "")
    cb = BestByAUC(MODELS_DIR / f"{name}.keras", x_va, va.label.values, a.patience)
    steps = math.ceil(len(tr) / a.batch_size)
    t0 = time.time()

    freeze = 0 if scratch else min(a.freeze_epochs, a.epochs)
    if freeze:
        print(f"phase 1: backbone frozen for {freeze} epoch(s)")
        set_backbone_trainable(model, False)
        compile_(model, a.lr, freeze * steps)
        model.fit(train_ds, epochs=freeze, callbacks=[cb], verbose=0)
    if freeze < a.epochs and not (cb.bad >= a.patience):
        print(f"phase 2: {'training from scratch' if scratch else 'fine-tuning'} for epochs {freeze + 1}-{a.epochs}")
        set_backbone_trainable(model, True, keep_bn_frozen=not scratch)
        compile_(model, a.lr if scratch else a.ft_lr, (a.epochs - freeze) * steps)
        cb.bad = 0
        model.fit(train_ds, initial_epoch=freeze, epochs=a.epochs, callbacks=[cb], verbose=0)

    pd.DataFrame(cb.log).to_csv(RESULTS_DIR / f"{name}_trainlog.csv", index=False)
    best = load_model(name)
    pd.DataFrame({"path": va["path"].values, "label": va.label.values,
                  "prob": predict_prob(best, x_va)}).to_csv(PREDS_DIR / f"{name}_val.csv", index=False)
    json.dump({"arch": a.model, "split_mode": a.split_mode, "class_weights": cw, "best_val_auc": cb.best},
              open(MODELS_DIR / f"{name}_meta.json", "w"), indent=2)
    print(f"best val AUC {cb.best:.4f} -> models/{name}.keras  ({(time.time() - t0) / 60:.1f} min)")
    print(f"next: python -m src.evaluate --model {name}")


if __name__ == "__main__":
    main()
