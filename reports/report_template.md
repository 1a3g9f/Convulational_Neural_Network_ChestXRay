# Pneumonia screening from chest X-rays - Report

## 1. Problem and dataset
Dataset counts per split, patient-level stratified split method, class imbalance.

## 2. Feature map analysis
Figures: `figures/featuremaps_*`. Describe early (edges, rib/lung outlines), middle (textures, vessels, hazy regions), late (abstract, class-relevant regions). Compare trained vs random init.

## 3. Models and training setup
Custom CNN, MobileNetV2, EfficientNet-B0, ResNet50V2, DenseNet121 (TensorFlow/Keras). Augmentation, optimiser, LR schedule, epochs, freeze/unfreeze.

## 4. Class imbalance handling and ablation
Class weights used; with vs without weights table (sensitivity, specificity).

## 5. Results
Table for all models on TEST at the chosen threshold: sensitivity, specificity, precision, ROC-AUC (95% bootstrap CI). ROC curves and confusion matrices (`figures/eval_*.png`).

## 6. Threshold justification
Chosen on the validation set (Youden's J, or sensitivity target) and frozen before test. State the split mode (pooled vs official).

## 7. Grad-CAM audit
Overlays for TP/TN/FP/FN (`figures/gradcam_*`), border/corner/central attention table (`results/*_audit_summary.csv`), occlusion test results. **Shortcuts found:** ... **Mitigation and before/after metrics:** ...

## 8. Input guard evaluation
Acceptance of X-rays, rejection of non-X-rays (photos, screenshots, colour images).

## 9. Limitations
Paediatric-only, single source, dataset shift between train and test, no clinical validation.

## 10. Ethics
Not a diagnostic device; privacy; risk of false negatives.
