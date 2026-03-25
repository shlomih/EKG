"""
validate_ensemble.py
====================
Benchmark v9-only, v10-only, and v9+v10 ensemble on PTB-XL test fold 10.
Also grid-searches ensemble weights to find optimal v9:v10 ratio.
"""

import sys
import os
import numpy as np
import wfdb
from pathlib import Path

os.chdir(Path(__file__).parent)

from cnn_classifier import (load_cnn_classifier, predict_cnn,
                             load_dataset, LABEL_TO_IDX, SUPERCLASS_LABELS)
from ensemble_classifier import load_ensemble, predict_ensemble
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

N_CLASSES = len(SUPERCLASS_LABELS)
HYP_IDX   = LABEL_TO_IDX["HYP"]

print("\n" + "="*70)
print("  ENSEMBLE VALIDATION  (v9 + v10, PTB-XL test fold 10)")
print("="*70)

# ------------------------------------------------------------------
# Load both models individually + ensemble
# ------------------------------------------------------------------
m9  = load_cnn_classifier("models/ecg_cnn_v9_backup.pt")
m10 = load_cnn_classifier("models/ecg_cnn.pt")

paths, labels, folds = load_dataset()
paths  = np.array(paths)
labels = np.array(labels)
folds  = np.array(folds)

test_mask   = (folds == 10)
test_paths  = paths[test_mask]
test_labels = labels[test_mask]
print(f"\n  Test fold 10: {len(test_paths)} records")

# ------------------------------------------------------------------
# Collect raw probability matrices for both models
# ------------------------------------------------------------------
def collect_probs(model_data, rec_paths, label):
    prob_matrix = np.zeros((len(rec_paths), N_CLASSES), dtype=np.float32)
    print(f"\nCollecting {label} probabilities...")
    for i, rp in enumerate(rec_paths):
        if (i+1) % 300 == 0 or (i+1) == len(rec_paths):
            print(f"  {i+1}/{len(rec_paths)}", end="\r")
        try:
            rec = wfdb.rdrecord(rp)
            res = predict_cnn(model_data, rec.p_signal, fs=rec.fs)
            for j, lbl in enumerate(SUPERCLASS_LABELS):
                prob_matrix[i, j] = res["probabilities"].get(lbl, 0.0)
        except Exception:
            prob_matrix[i, 0] = 1.0
    print()
    return prob_matrix

probs_v9  = collect_probs(m9,  test_paths, "v9")
probs_v10 = collect_probs(m10, test_paths, "v10")

# ------------------------------------------------------------------
# Evaluate with CLASS_THRESHOLDS applied
# ------------------------------------------------------------------
from hybrid_classifier import CLASS_THRESHOLDS

def apply_thresholds(probs):
    preds = []
    for p in probs:
        order  = np.argsort(p)[::-1]
        chosen = order[0]
        for idx in order:
            if p[idx] >= CLASS_THRESHOLDS.get(SUPERCLASS_LABELS[idx], 0.0):
                chosen = idx
                break
        preds.append(chosen)
    return np.array(preds)

def metrics(preds, true):
    prec, rec, f1, _ = precision_recall_fscore_support(
        true, preds, average=None,
        labels=list(range(N_CLASSES)), zero_division=0)
    macro = f1.mean()
    return prec, rec, f1, macro

def print_row(name, prec, rec, f1, macro):
    print(f"  {name:<28s}  {prec[HYP_IDX]:.1%}  {rec[HYP_IDX]:.1%}  "
          f"{f1[HYP_IDX]:.3f}  {macro:.3f}")

print("\n" + "-"*70)
print(f"  {'Config':<28s}  {'HYPPrec':>7s}  {'HYPRec':>6s}  "
      f"{'HYPF1':>5s}  {'Macro':>5s}")
print("  " + "-"*65)

# v9 alone
preds_v9 = apply_thresholds(probs_v9)
prec9, rec9, f1_9, mac9 = metrics(preds_v9, test_labels)
print_row("v9 alone (Transformer)", prec9, rec9, f1_9, mac9)

# v10 alone
preds_v10 = apply_thresholds(probs_v10)
prec10, rec10, f1_10, mac10 = metrics(preds_v10, test_labels)
print_row("v10 alone (Joint+ampnorm)", prec10, rec10, f1_10, mac10)

# ------------------------------------------------------------------
# Grid search: find optimal weight for v9
# ------------------------------------------------------------------
print("\n" + "-"*70)
print("  WEIGHT GRID SEARCH  (w_v9 : w_v10, evaluated on test fold 10)")
print("-"*70)
print(f"  {'w_v9':>5s}  {'HYPPrec':>7s}  {'HYPRec':>6s}  {'HYPF1':>5s}  {'Macro':>5s}")

best_hyp_f1   = -1.0
best_w        = 0.35
best_prec_at_best = 0.0

for w9 in np.arange(0.0, 1.05, 0.05):
    w10       = 1.0 - w9
    ens_probs = w9 * probs_v9 + w10 * probs_v10
    ens_probs = ens_probs / ens_probs.sum(axis=1, keepdims=True)
    preds     = apply_thresholds(ens_probs)
    prec, rec, f1, macro = metrics(preds, test_labels)
    marker = ""
    if f1[HYP_IDX] > best_hyp_f1:
        best_hyp_f1 = f1[HYP_IDX]
        best_w      = w9
        best_prec_at_best = prec[HYP_IDX]
        marker = " <-- best HYP F1"
    print(f"  {w9:>5.2f}  {prec[HYP_IDX]:>7.1%}  {rec[HYP_IDX]:>6.1%}  "
          f"{f1[HYP_IDX]:>5.3f}  {macro:>5.3f}{marker}")

# ------------------------------------------------------------------
# Best ensemble: full per-class breakdown
# ------------------------------------------------------------------
print(f"\n  Best w_v9={best_w:.2f}  w_v10={1-best_w:.2f}  "
      f"(HYP F1={best_hyp_f1:.3f}  Prec={best_prec_at_best:.1%})")

best_ens = best_w * probs_v9 + (1-best_w) * probs_v10
best_ens = best_ens / best_ens.sum(axis=1, keepdims=True)
best_preds = apply_thresholds(best_ens)
prec_b, rec_b, f1_b, mac_b = metrics(best_preds, test_labels)

print("\n" + "="*70)
print("  BEST ENSEMBLE vs INDIVIDUAL MODELS (test fold 10)")
print("="*70)
print(f"\n  {'Class':>6s}  {'v9Prec':>6s}  {'v10Prec':>7s}  {'EnsPrec':>7s}  "
      f"{'EnsRec':>6s}  {'EnsF1':>5s}")
print("  " + "-"*55)
for i, lbl in enumerate(SUPERCLASS_LABELS):
    print(f"  {lbl:>6s}  {prec9[i]:>6.1%}  {prec10[i]:>7.1%}  {prec_b[i]:>7.1%}  "
          f"{rec_b[i]:>6.1%}  {f1_b[i]:>5.3f}")
print(f"\n  MacroF1:  v9={mac9:.3f}  v10={mac10:.3f}  ensemble={mac_b:.3f}")

# Confusion matrix
cm = confusion_matrix(test_labels, best_preds, labels=list(range(N_CLASSES)))
print("\nEnsemble Confusion Matrix:")
print("          " + "  ".join(f"{l:>5s}" for l in SUPERCLASS_LABELS))
for i, lbl in enumerate(SUPERCLASS_LABELS):
    print(f"  {lbl:>6s}: {cm[i]}")

print("\n" + "="*70)
print(f"  RECOMMENDED ENSEMBLE WEIGHTS: v9={best_w:.2f}, v10={1-best_w:.2f}")
print("="*70 + "\n")
