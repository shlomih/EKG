"""
validate_hybrid.py
==================
Compare CNN-only vs hybrid_classifier (voltage gating + confidence thresholds)
on PTB-XL fold 10 (test set, 2,158 records).

Reports per-class precision/recall/F1 for both approaches,
highlighting delta for HYP.

Usage:
    python validate_hybrid.py
"""

import sys
import os
import numpy as np
import wfdb
from pathlib import Path

from cnn_classifier import load_cnn_classifier, predict_cnn, load_dataset, LABEL_TO_IDX, SUPERCLASS_LABELS
from hybrid_classifier import hybrid_predict
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

print("\n" + "="*70)
print("  HYBRID CLASSIFIER VALIDATION  (CNN vs CNN+Voltage+Threshold)")
print("="*70)

# Load models
print("\nLoading CNN model...")
cnn_model = load_cnn_classifier()
if cnn_model is None:
    print("ERROR: CNN model not found at models/ecg_cnn.pt")
    sys.exit(1)
print("  OK")

# Load PTB-XL test set paths
print("Loading PTB-XL test fold (fold 10)...")
paths, labels, folds = load_dataset()
paths  = np.array(paths)
labels = np.array(labels)
folds  = np.array(folds)

test_mask  = (folds == 10)
test_paths = paths[test_mask]
test_labels = labels[test_mask]
print(f"  {len(test_paths)} records")

# ---------------------------------------------------------------
# Phase 1: CNN-only baseline
# ---------------------------------------------------------------
print("\n" + "-"*70)
print("  PHASE 1: CNN baseline (no voltage gating)")
print("-"*70)

cnn_preds = []
for i, rec_path in enumerate(test_paths):
    if (i + 1) % 300 == 0 or (i + 1) == len(test_paths):
        print(f"  {i+1}/{len(test_paths)}", end="\r")
    try:
        record = wfdb.rdrecord(rec_path)
        result = predict_cnn(cnn_model, record.p_signal, fs=record.fs)
        cnn_preds.append(LABEL_TO_IDX[result["prediction"]])
    except Exception:
        cnn_preds.append(0)

cnn_preds = np.array(cnn_preds)
prec_cnn, rec_cnn, f1_cnn, _ = precision_recall_fscore_support(
    test_labels, cnn_preds, average=None, labels=list(range(len(SUPERCLASS_LABELS)))
)

print("\n\nCNN-only metrics:")
print(f"  {'Class':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}")
print("  " + "-"*30)
for i, lbl in enumerate(SUPERCLASS_LABELS):
    print(f"  {lbl:>6s}  {prec_cnn[i]:.3f}   {rec_cnn[i]:.3f}   {f1_cnn[i]:.3f}")

# ---------------------------------------------------------------
# Phase 2: Hybrid (voltage + thresholds)
# ---------------------------------------------------------------
print("\n" + "-"*70)
print("  PHASE 2: Hybrid (voltage gating + confidence thresholds)")
print("-"*70)

hybrid_preds = []
n_adjusted = 0

for i, rec_path in enumerate(test_paths):
    if (i + 1) % 300 == 0 or (i + 1) == len(test_paths):
        print(f"  {i+1}/{len(test_paths)}", end="\r")
    try:
        record = wfdb.rdrecord(rec_path)
        result = hybrid_predict(
            cnn_model,
            record.p_signal,
            fs=record.fs,
            lead_names=record.sig_name,
        )
        hybrid_preds.append(LABEL_TO_IDX[result["prediction"]])
        if result["adjustment_applied"]:
            n_adjusted += 1
    except Exception:
        hybrid_preds.append(0)

hybrid_preds = np.array(hybrid_preds)
prec_h, rec_h, f1_h, _ = precision_recall_fscore_support(
    test_labels, hybrid_preds, average=None, labels=list(range(len(SUPERCLASS_LABELS)))
)

print(f"\n\n  Adjustments applied: {n_adjusted}/{len(test_paths)} records")

print("\nHybrid metrics (delta vs CNN baseline):")
print(f"  {'Class':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'dPrec':>7s}  {'dRec':>7s}  {'dF1':>6s}")
print("  " + "-"*55)
for i, lbl in enumerate(SUPERCLASS_LABELS):
    dp = prec_h[i] - prec_cnn[i]
    dr = rec_h[i] - rec_cnn[i]
    df = f1_h[i] - f1_cnn[i]
    flag = " [+]" if df > 0.01 else (" [-]" if df < -0.02 else "")
    print(f"  {lbl:>6s}  {prec_h[i]:.3f}   {rec_h[i]:.3f}   {f1_h[i]:.3f}  "
          f"{dp:+.3f}   {dr:+.3f}   {df:+.3f}{flag}")

# ---------------------------------------------------------------
# HYP Focus
# ---------------------------------------------------------------
hyp_idx = LABEL_TO_IDX["HYP"]

print("\n" + "="*70)
print("  HYP CLASS DETAIL")
print("="*70)
print(f"\n  Baseline:  Prec={prec_cnn[hyp_idx]:.1%}  Rec={rec_cnn[hyp_idx]:.1%}  F1={f1_cnn[hyp_idx]:.3f}")
print(f"  Hybrid:    Prec={prec_h[hyp_idx]:.1%}  Rec={rec_h[hyp_idx]:.1%}  F1={f1_h[hyp_idx]:.3f}")
print(f"  dPrec: {prec_h[hyp_idx] - prec_cnn[hyp_idx]:+.1%}   "
      f"dRec: {rec_h[hyp_idx] - rec_cnn[hyp_idx]:+.1%}   "
      f"dF1: {f1_h[hyp_idx] - f1_cnn[hyp_idx]:+.3f}")

# Confusion matrix for HYP
cm = confusion_matrix(test_labels, hybrid_preds, labels=list(range(len(SUPERCLASS_LABELS))))
print("\nHybrid Confusion Matrix:")
print("          " + "  ".join(f"{l:>5s}" for l in SUPERCLASS_LABELS))
for i, lbl in enumerate(SUPERCLASS_LABELS):
    print(f"  {lbl:>6s}: {cm[i]}")

print("\n" + "="*70)
print("  VALIDATION COMPLETE")
print("="*70)
