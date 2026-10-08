import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's INFO/WARNING spam

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("XRAY_DATA_ROOT", ROOT / "data" / "chest_xray"))
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
PREDS_DIR = REPORTS_DIR / "preds"

SEED = 42
IMG_SIZE = 224
BORDER_CROP = 0.04          # fraction cropped from every edge (removes L/R markers, text, film borders)
SPLIT_MODE = os.environ.get("XRAY_SPLIT", "pooled")   # "pooled" or "official" (see README)
NO_PRETRAINED = os.environ.get("XRAY_NO_PRETRAINED") == "1"   # offline / smoke-test switch

MODEL_NAMES = ["custom_cnn", "mobilenet_v2", "efficientnet_b0", "resnet50v2", "densenet121"]
DEFAULT_MODEL = "efficientnet_b0"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # label 0 / 1 (1 = positive = pneumonia)

# Changing IMG_SIZE / BORDER_CROP / the preprocessing automatically creates a new cache file.
CACHE_PATH = ROOT / "data" / f"cache_{IMG_SIZE}_c{int(BORDER_CROP * 100)}_v1.npz"


def split_csv(mode: str) -> Path:
    return ROOT / "data" / f"splits_{mode}.csv"


for _d in (MODELS_DIR, FIG_DIR, RESULTS_DIR, PREDS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
