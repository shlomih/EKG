"""
optimize_thresholds.py
======================
Grid-search per-class confidence thresholds on PTB-XL validation fold 9,
then evaluate on test fold 10.

Goal: find thresholds that maximize HYP F1 (and show precision/recall curve).
"""

import sys
import os
import numpy as np
import wfdb
from pathlib import Path

os.chdir(Path(__file__).parent)

from cnn_classifier import load_cnn_classifier, predict_cnn, load_dataset, LABEL_TO_IDX, SUPERCLASS_LABELS
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

HYP_IDX = LABEL_TO_IDX["HYP"]
N_CLASSES = len(SUPERCLASS_LABELS)

print("\n" + "="*70)
print("  THRESHOLD OPTIMIZATION  (val fold 9 -> evaluate on test fold 10)")
print("="*70)

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
print("\nLoading CNN model...")
model = load_cnn_classifier()
if model is None:
    print("ERROR: no model found at models/ecg_cnn.pt")
    sys.exit(1)
print("  OK")

# ------------------------------------------------------------------
# Load dataset paths + folds
# ------------------------------------------------------------------
paths, labels, folds = load_dataset()
paths  = np.array(paths)
labels = np.array(labels)
folds  = np.array(folds)

val_mask  = (folds == 9)
test_mask = (folds == 10)
val_paths, val_labels   = paths[val_mask],  labels[val_mask]
test_paths, test_labels = paths[test_mask], labels[test_mask]
print(f"\n  Val fold 9 : {len(val_paths)} records")
print(f"  Test fold 10: {len(test_paths)} records")

# ------------------------------------------------------------------
# Collect raw probabilities on val + test
# ------------------------------------------------------------------
def collect_probs(rec_paths, split_name):
    prob_matrix = np.zeros((len(rec_paths), N_CLASSES), dtype=np.float32)
    for i, rp in enumerate(rec_paths):
        if (i + 1) % 300 == 0 or (i + 1) == len(rec_paths):
            print(f"  {i+1}/{len(rec_paths)}", end="\r")
        try:
            rec = wfdb.rdrecord(rp)
            res = predict_cnn(model, rec.p_signal, fs=rec.fs)
            for j, lbl in enumerate(SUPERCLASS_LABELS):
                prob_matrix[i, j] = res["probabilities"].get(lbl, 0.0)
        except Exception:
            prob_matrix[i, 0] = 1.0  # fallback to NORM
    print()
    return prob_matrix

print("\nCollecting val probabilities...")
val_probs  = collect_probs(val_paths, "val")
print("Collecting test probabilities...")
test_probs = collect_probs(test_paths, "test")


# ------------------------------------------------------------------
# Baseline: argmax (no threshold)
# ------------------------------------------------------------------
def apply_thresholds(probs, thresholds):
    """
    For each sample: start with argmax class.
    If prob[argmax] < threshold[argmax], fall back to next best class
    that meets its threshold. If none qualify, use argmax anyway.
    """
    preds = []
    for prob in probs:
        order = np.argsort(prob)[::-1]
        chosen = order[0]  # default argmax
        for idx in order:
            if prob[idx] >= thresholds[idx]:
                chosen = idx
                break
        preds.append(chosen)
    return np.array(preds)

baseline_thresholds = np.zeros(N_CLASSES)  # 0 = always use argmax
base_val_preds  = apply_thresholds(val_probs,  baseline_thresholds)
base_test_preds = apply_thresholds(test_probs, baseline_thresholds)

prec_base, rec_base, f1_base, _ = precision_recall_fscore_support(
    test_labels, base_test_preds, average=None, labels=list(range(N_CLASSES))
)

print("\nBaseline (argmax, no threshold) on test fold 10:")
print(f"  {'Class':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}")
print("  " + "-"*30)
for i, lbl in enumerate(SUPERCLASS_LABELS):
    print(f"  {lbl:>6s}  {prec_base[i]:.3f}   {rec_base[i]:.3f}   {f1_base[i]:.3f}")


# ------------------------------------------------------------------
# Grid search: HYP threshold (most impactful)
# ------------------------------------------------------------------
print("\n" + "-"*70)
print("  GRID SEARCH: HYP threshold (all other classes at 0)")
print("-"*70)
print(f"  {'tau_HYP':>8s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'n_HYP_pred':>10s}")

hyp_tau_range = np.arange(0.25, 0.75, 0.025)
best_hyp_f1   = -1.0
best_hyp_tau  = 0.45
best_hyp_row  = None

for tau in hyp_tau_range:
    thresholds = np.zeros(N_CLASSES)
    thresholds[HYP_IDX] = tau
    val_preds = apply_thresholds(val_probs, thresholds)
    prec, rec, f1, support = precision_recall_fscore_support(
        val_labels, val_preds, average=None,
        labels=list(range(N_CLASSES)), zero_division=0
    )
    n_pred = int((val_preds == HYP_IDX).sum())
    marker = ""
    if f1[HYP_IDX] > best_hyp_f1:
        best_hyp_f1  = f1[HYP_IDX]
        best_hyp_tau = tau
        best_hyp_row = (prec[HYP_IDX], rec[HYP_IDX], f1[HYP_IDX], n_pred)
        marker = " <-- best F1"
    print(f"  {tau:>8.3f}  {prec[HYP_IDX]:.3f}   {rec[HYP_IDX]:.3f}   "
          f"{f1[HYP_IDX]:.3f}  {n_pred:>10d}{marker}")

print(f"\n  Best HYP tau on val: {best_hyp_tau:.3f}  (F1={best_hyp_f1:.3f})")


# ------------------------------------------------------------------
# Also find precision-optimized threshold (prec >= 0.50, best recall)
# ------------------------------------------------------------------
print("\n" + "-"*70)
print("  HYP PRECISION-RECALL CURVE (val fold 9)")
print("-"*70)
print(f"  {'tau_HYP':>8s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}")

best_prec50_rec = -1.0
best_prec50_tau = None
best_prec60_rec = -1.0
best_prec60_tau = None

for tau in np.arange(0.20, 0.80, 0.01):
    thresholds = np.zeros(N_CLASSES)
    thresholds[HYP_IDX] = tau
    val_preds = apply_thresholds(val_probs, thresholds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        val_labels, val_preds, average=None,
        labels=list(range(N_CLASSES)), zero_division=0
    )
    p, r = prec[HYP_IDX], rec[HYP_IDX]
    if p >= 0.50 and r > best_prec50_rec:
        best_prec50_rec = r
        best_prec50_tau = tau
    if p >= 0.60 and r > best_prec60_rec:
        best_prec60_rec = r
        best_prec60_tau = tau

    # print every 5th point
    if round(tau * 100) % 5 == 0:
        print(f"  {tau:>8.2f}  {p:.3f}   {r:.3f}   {f1[HYP_IDX]:.3f}")

print(f"\n  Best tau for Prec>=50%: {best_prec50_tau}  (Rec={best_prec50_rec:.3f})")
print(f"  Best tau for Prec>=60%: {best_prec60_tau}  (Rec={best_prec60_rec:.3f})")


# ------------------------------------------------------------------
# Joint optimization: all 5 classes simultaneously
# ------------------------------------------------------------------
print("\n" + "-"*70)
print("  JOINT THRESHOLD OPTIMIZATION (optimize macro F1 on val)")
print("-"*70)

# Coarse grid: each class threshold in {0, 0.25, 0.35, 0.45, 0.55}
from itertools import product as iproduct

class_grids = [
    [0.0, 0.25, 0.35],          # NORM
    [0.0, 0.20, 0.30],          # MI
    [0.0, 0.20, 0.30],          # STTC
    [0.35, 0.45, 0.55, 0.65],   # HYP  (finer)
    [0.0, 0.20, 0.30],          # CD
]

best_macro_f1 = -1.0
best_joint_thresholds = np.zeros(N_CLASSES)

for combo in iproduct(*class_grids):
    thresholds = np.array(combo, dtype=np.float32)
    val_preds = apply_thresholds(val_probs, thresholds)
    _, _, f1, _ = precision_recall_fscore_support(
        val_labels, val_preds, average=None,
        labels=list(range(N_CLASSES)), zero_division=0
    )
    macro = f1.mean()
    # Bonus weight for HYP F1 (we care more about it)
    weighted = 0.6 * macro + 0.4 * f1[HYP_IDX]
    if weighted > best_macro_f1:
        best_macro_f1 = weighted
        best_joint_thresholds = thresholds

print(f"\n  Best joint thresholds (val):")
for i, lbl in enumerate(SUPERCLASS_LABELS):
    print(f"    {lbl}: {best_joint_thresholds[i]:.3f}")


# ------------------------------------------------------------------
# Evaluate candidate threshold sets on TEST fold 10
# ------------------------------------------------------------------
print("\n" + "="*70)
print("  FINAL EVALUATION ON TEST FOLD 10")
print("="*70)

candidates = {
    "Baseline (argmax)":           np.zeros(N_CLASSES),
    "Current (hybrid_classifier)": np.array([0.30, 0.25, 0.25, 0.45, 0.25]),
    f"HYP-only tau={best_hyp_tau:.2f}": np.array([0.0, 0.0, 0.0, best_hyp_tau, 0.0]),
    f"Prec50 tau={best_prec50_tau}":     np.array([0.0, 0.0, 0.0, best_prec50_tau or 0.45, 0.0]),
    f"Prec60 tau={best_prec60_tau}":     np.array([0.0, 0.0, 0.0, best_prec60_tau or 0.55, 0.0]),
    "Joint optimized":             best_joint_thresholds,
}

print(f"\n  {'Config':<30s}  {'Prec_HYP':>8s}  {'Rec_HYP':>8s}  {'F1_HYP':>7s}  {'MacroF1':>8s}")
print("  " + "-"*75)

for name, thresholds in candidates.items():
    test_preds = apply_thresholds(test_probs, thresholds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average=None,
        labels=list(range(N_CLASSES)), zero_division=0
    )
    macro = f1.mean()
    print(f"  {name:<30s}  {prec[HYP_IDX]:>8.1%}  {rec[HYP_IDX]:>8.1%}  "
          f"{f1[HYP_IDX]:>7.3f}  {macro:>8.3f}")

# Detailed breakdown for joint-optimized
print("\nJoint-optimized: per-class breakdown on test fold 10:")
test_preds = apply_thresholds(test_probs, best_joint_thresholds)
prec_j, rec_j, f1_j, _ = precision_recall_fscore_support(
    test_labels, test_preds, average=None, labels=list(range(N_CLASSES))
)
prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(
    test_labels, base_test_preds, average=None, labels=list(range(N_CLASSES))
)
print(f"  {'Class':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'dPrec':>7s}  {'dRec':>7s}  {'dF1':>6s}")
print("  " + "-"*55)
for i, lbl in enumerate(SUPERCLASS_LABELS):
    dp = prec_j[i] - prec_b[i]
    dr = rec_j[i]  - rec_b[i]
    df = f1_j[i]   - f1_b[i]
    print(f"  {lbl:>6s}  {prec_j[i]:.3f}   {rec_j[i]:.3f}   {f1_j[i]:.3f}  "
          f"{dp:+.3f}   {dr:+.3f}   {df:+.3f}")

print("\n" + "="*70)
print("  RECOMMENDED THRESHOLDS TO UPDATE IN hybrid_classifier.py:")
print("="*70)
print(f"\n  CLASS_THRESHOLDS = {{")
for i, lbl in enumerate(SUPERCLASS_LABELS):
    t = best_joint_thresholds[i]
    print(f'      "{lbl}": {t:.2f},')
print("  }")
print()
