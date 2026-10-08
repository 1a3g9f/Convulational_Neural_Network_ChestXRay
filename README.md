# Pneumonia screening from paediatric chest X-rays (TensorFlow + Grad-CAM)

Not a diagnostic device. Educational project. Trained on paediatric (age 1-5) chest X-rays only.

## Setup (Windows / VS Code terminal)
```bash
py -3.12 -m venv .venv          # Python 3.10-3.12
.venv\Scripts\activate
pip install -r requirements.txt
```
Download the dataset (see `data/README.md`), then run everything **from the project root**.
TensorFlow on native Windows uses the CPU, so prefer `efficientnet_b0` or `mobilenet_v2` (fast).

## Run order
```bash
# 1. Patient-level split (pooled 70/15/15 by default; prints counts, asserts no patient leakage)
python -m src.data

# 2. Train (backbone frozen for 3 epochs, then fine-tuned; best val-AUC checkpoint is kept)
python -m src.train --model efficientnet_b0
python -m src.train --model mobilenet_v2
python -m src.train --model custom_cnn --epochs 30           # from scratch, for comparison
python -m src.train --model efficientnet_b0 --no_class_weights --tag noweights   # imbalance ablation
#    quick dry run first:  python -m src.train --model mobilenet_v2 --limit 300 --epochs 2

# 3. Threshold chosen on VAL (Youden's J) -> single evaluation on TEST
python -m src.evaluate --model efficientnet_b0
#    sensitivity-first alternative:  --criterion sens --target_sens 0.95

# 4. Feature maps, Grad-CAM audit
python -m src.featuremaps --model efficientnet_b0
python -m src.featuremaps --model efficientnet_b0 --random
python -m src.audit --model efficientnet_b0 --n 20

# 5. Input guard (optionally calibrate with a folder of non-X-ray images)
python -m src.guard fit
python -m src.guard eval --neg_dir path\to\non_xray_images

# 6. Web apps
python app/flask_app.py                     # http://127.0.0.1:5000
streamlit run app/streamlit_app.py
#    other checkpoint:  set XRAY_MODEL=mobilenet_v2   (PowerShell: $env:XRAY_MODEL="mobilenet_v2")
```

## What changed from the PyTorch version, and why
| Problem | Cause | Fix |
|---|---|---|
| Normal X-rays flagged as pneumonia with high probability | Threshold forced to 96% sensitivity on a val set that looked nothing like test (it came out at 0.13), on top of class weights; plus brightness/border shortcuts | Default threshold = Youden's J on val; both operating points are reported. Per-image contrast stretch + 4% border crop at train and inference; random cut-out and brightness/contrast augmentation; BatchNorm kept frozen while fine-tuning |
| Train/val/test mismatch | Kaggle's official test folder is distributed differently from train (known issue), and its val folder has only 16 images | `--split_mode pooled` (default): all folders pooled, patient-grouped stratified 70/15/15. `--split_mode official` keeps the old behaviour. Report both if you present this |
| Some images would not upload | Guard rejected ~1% of genuine X-rays by design, any image with aspect ratio outside 0.6-1.7, any "coloured" (tinted) X-ray, and the app only accepted a fixed list of extensions; 16-bit and RGBA files were mis-converted | One robust loader (16-bit, RGBA, palette, EXIF); no extension whitelist; tinted greyscale accepted; guard hard-rejects only clear non-X-rays and otherwise shows a warning; errors always come back as readable messages |
| PyTorch | - | TensorFlow/Keras 3 throughout. ResNet-18 is not in `keras.applications`, so the ResNet option is `resnet50v2` (ViT dropped) |

## Layout
- `src/` data, preprocess, models, train, evaluate, infer, gradcam, audit, featuremaps, guard
- `app/flask_app.py` + `app/templates/index.html`, `app/streamlit_app.py`
- `reports/` figures, results (JSON/CSV), predictions, `report_template.md`
- `models/` `*.keras` checkpoints, `*_meta.json`, `*_threshold.json`, `guard.npz`

## Limitations to state in your presentation
- Paediatric, single-source dataset. Adult X-rays and images from other scanners are out of distribution; the model can output confident but meaningless probabilities for them.
- A pooled split gives optimistic numbers compared with truly external data. Use `--split_mode official` to show the distribution-shift gap honestly.
- Probabilities are not calibrated. Never tune on the test set.
- If Grad-CAM lights up borders, corners, text markers or the abdomen instead of the lungs, treat the prediction as a shortcut, not as evidence.
