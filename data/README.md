# Data

1. Download "Chest X-Ray Images (Pneumonia)" from Kaggle (free login needed):
   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Unzip so the layout is:

```
data/chest_xray/
    train/NORMAL, train/PNEUMONIA
    val/NORMAL,   val/PNEUMONIA
    test/NORMAL,  test/PNEUMONIA
```
(If the zip has a nested `chest_xray/chest_xray` folder, use the inner one.)

Or point to another location with the environment variable `XRAY_DATA_ROOT`.

Then run `python -m src.data` to build `data/splits.csv`.
