"""Dataset scan, patient-level splits, preprocessed-image cache and tf.data pipelines."""
import re
from concurrent.futures import ThreadPoolExecutor

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from .config import CACHE_PATH, CLASS_NAMES, DATA_ROOT, IMG_SIZE, SEED, SPLIT_MODE, split_csv
from .preprocess import prepare

EXTS = {".jpeg", ".jpg", ".png"}


def patient_id(fname: str) -> str:
    """person12_bacteria_45.jpeg -> person12 ; NORMAL2-IM-0373-0001.jpeg / IM-0115-0001-0002.jpeg -> IM-0373 / IM-0115"""
    stem = fname.rsplit(".", 1)[0]
    m = re.match(r"person(\d+)", stem, re.I)
    if m:
        return f"person{m.group(1)}"
    m = re.search(r"IM-(\d+)", stem)
    if m:
        return f"IM-{m.group(1)}"
    return stem


def scan(folder_name: str) -> pd.DataFrame:
    rows = []
    for label, cls in enumerate(CLASS_NAMES):
        for p in sorted((DATA_ROOT / folder_name / cls).glob("*")):
            if p.name.startswith("._") or p.suffix.lower() not in EXTS:   # skip macOS '._' junk files
                continue
            rows.append({"rel": p.relative_to(DATA_ROOT).as_posix(), "label": label,
                         "patient": patient_id(p.name), "orig_folder": folder_name})
    df = pd.DataFrame(rows, columns=["rel", "label", "patient", "orig_folder"])
    df["label"] = df["label"].astype(int)
    return df


def _fold(df: pd.DataFrame, n_splits: int) -> np.ndarray:
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    _, idx = next(sgkf.split(df, df["label"], groups=df["patient"]))
    return idx


def build_splits(mode: str = SPLIT_MODE) -> pd.DataFrame:
    """mode='pooled'  : merge train+val+test, then a patient-grouped, class-stratified 70/15/15 split.
                        Train, val and test now come from the SAME distribution, so the threshold chosen
                        on val actually transfers to test (this removes the Kaggle train->test shift).
       mode='official': Kaggle train+val -> train/val (80/20, patient-grouped); official test untouched."""
    parts = {f: scan(f) for f in ("train", "val", "test")}
    for f, p in parts.items():
        print(f"found {len(p):5d} images in {DATA_ROOT / f}")
    if sum(len(p) for p in parts.values()) == 0:
        raise FileNotFoundError(f"No images found under {DATA_ROOT}. See data/README.md.")
    parts = {f: p for f, p in parts.items() if len(p)}      # an empty/missing folder must not break the split
    if mode == "official" and not {"train", "test"} <= set(parts):
        raise FileNotFoundError("--split_mode official needs both train and test folders; use the default pooled mode.")
    if mode == "official":
        tv = pd.concat([parts["train"], parts.get("val", parts["train"].iloc[0:0])], ignore_index=True)
        tv["split"] = "train"
        tv.loc[_fold(tv, 5), "split"] = "val"
        te = parts["test"].copy()
        te["split"] = "test"
        df = pd.concat([tv, te], ignore_index=True)
    elif mode == "pooled":
        df = pd.concat(list(parts.values()), ignore_index=True)
        df["label"] = df["label"].astype(int)
        df["split"] = "train"
        df.loc[_fold(df, 7), "split"] = "test"                       # ~14%
        rest = df[df["split"] != "test"].reset_index()               # keeps original index in column 'index'
        df.loc[rest.loc[_fold(rest, 6), "index"].values, "split"] = "val"   # ~14%
    else:
        raise ValueError(mode)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = set(df[df.split == a].patient) & set(df[df.split == b].patient)
        assert not shared, f"patient leakage between {a} and {b}: {sorted(shared)[:5]}"
    df.to_csv(split_csv(mode), index=False)
    return df


def load_splits(mode: str = SPLIT_MODE) -> pd.DataFrame:
    f = split_csv(mode)
    df = pd.read_csv(f) if f.exists() else build_splits(mode)
    df["path"] = [str(DATA_ROOT / r) for r in df["rel"]]       # relative in the CSV -> works if the folder moves
    return df


def _prep_file(rel: str):
    try:
        with Image.open(DATA_ROOT / rel) as im:
            im.load()
            return prepare(im)
    except Exception as e:
        print(f"WARNING: skipping unreadable image {rel}: {e}")
        return None


def load_images(df: pd.DataFrame, workers: int = 8):
    """Returns (df_ok, x_uint8[N,224,224]). Preprocessed images are cached in one .npz file."""
    cache = {}
    if CACHE_PATH.exists():
        z = np.load(CACHE_PATH, allow_pickle=False)
        cache = dict(zip(z["rel"].tolist(), z["x"]))
    missing = [r for r in df["rel"] if r not in cache]
    if missing:
        with ThreadPoolExecutor(workers) as ex:
            for r, a in zip(missing, tqdm(ex.map(_prep_file, missing), total=len(missing), desc="preprocessing")):
                if a is not None:
                    cache[r] = a
        np.savez(CACHE_PATH, rel=np.array(list(cache)), x=np.stack(list(cache.values())))
    ok = df[df["rel"].isin(cache)].reset_index(drop=True)
    return ok, np.stack([cache[r] for r in ok["rel"]])


def class_weights(df_train: pd.DataFrame) -> list:
    counts = df_train["label"].value_counts().sort_index().values.astype(float)
    return (counts.sum() / (2 * counts)).tolist()          # n_total / (2 * n_class)


# ---------------------------------------------------------------- tf.data
_GEOM = keras.Sequential([
    keras.layers.RandomRotation(10 / 360, fill_mode="constant"),
    keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant"),
    keras.layers.RandomZoom((-0.10, 0.10), fill_mode="constant"),
])   # no horizontal flip: it would mirror the anatomy and the L/R marker


def _cutout(x):
    """Small random grey square (p=0.3): stops the net leaning on any single corner/marker."""
    def apply(x):
        size = tf.cast(tf.random.uniform([], 0.08, 0.20) * IMG_SIZE, tf.int32)
        cy = tf.random.uniform([], 0, IMG_SIZE, tf.int32)
        cx = tf.random.uniform([], 0, IMG_SIZE, tf.int32)
        yy, xx = tf.range(IMG_SIZE)[:, None], tf.range(IMG_SIZE)[None, :]
        m = (tf.abs(yy - cy) < size // 2) & (tf.abs(xx - cx) < size // 2)
        return tf.where(m[..., None], tf.reduce_mean(x), x)
    return tf.cond(tf.random.uniform([]) < 0.3, lambda: apply(x), lambda: x)


def _augment(img_u8, y, w):
    x = tf.cast(img_u8, tf.float32)[..., None]                     # (H,W,1)
    x = _GEOM(x[None], training=True)[0]
    x = tf.image.random_contrast(x, 0.8, 1.2)
    x = x + tf.random.uniform([], -20.0, 20.0)
    x = _cutout(tf.clip_by_value(x, 0.0, 255.0))
    return tf.repeat(x, 3, axis=-1), y, w


def _plain(img_u8, y):
    return tf.repeat(tf.cast(img_u8, tf.float32)[..., None], 3, axis=-1), y


def make_dataset(x_u8, labels, train=False, batch_size=32, cw=None):
    y = np.asarray(labels, dtype="float32")[:, None]
    if train:
        w = np.asarray(cw, dtype="float32")[np.asarray(labels).astype(int)]
        ds = tf.data.Dataset.from_tensor_slices((x_u8, y, w)).shuffle(len(y), seed=SEED)
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((x_u8, y)).map(_plain, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_mode", default=SPLIT_MODE, choices=["pooled", "official"])
    a = ap.parse_args()
    df = build_splits(a.split_mode)
    print(df.groupby(["split", "label"]).size().unstack())
    print("patient leakage between splits: none (asserted)")
    print("saved", split_csv(a.split_mode))
