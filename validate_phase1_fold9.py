"""
validate_phase1_fold9.py
========================
Validate retrained CNN on PTB-XL fold 9 (test set).
Compare pre vs post Phase 1 improvements per-class metrics.

Usage:
    python validate_phase1_fold9.py
"""

import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from cnn_classifier import (
    load_cnn_classifier,
    predict_cnn,
    SUPERCLASS_LABELS,
    LABEL_TO_IDX,
    load_dataset,
)

print("\n" + "="*70)
print("  PHASE 1 VALIDATION ON PTB-XL FOLD 9 (TEST SET)")
print("="*70)

# Load dataset
paths, labels, folds = load_dataset()
paths = np.array(paths)
labels = np.array(labels)
folds = np.array(folds)

# Get test set (fold 10)
test_mask = folds == 10
test_paths = paths[test_mask]
test_labels = labels[test_mask]

print(f"\n✓ Loaded PTB-XL dataset")
print(f"  Total records: {len(paths)}")
print(f"  Test set (fold 10): {len(test_paths)} records")

# Distribution in test set
test_dist = np.bincount(test_labels, minlength=len(SUPERCLASS_LABELS))
print(f"\n  Test set distribution:")
for i, label in enumerate(SUPERCLASS_LABELS):
    print(f"    {label}: {int(test_dist[i])} records")

# Load model
model_data = load_cnn_classifier()
print(f"\n✓ Loaded retrained CNN model")

# Run predictions on test set
print(f"\n  Running predictions on {len(test_paths)} test records...")
pred_labels = []
pred_probs = []
errors = []

for i, path in enumerate(test_paths):
    if (i + 1) % 100 == 0 or (i + 1) == len(test_paths):
        print(f"    {i+1}/{len(test_paths)}")
    
    try:
        # Load signal
        rec = wfdb.rdrecord(path)
        signal_12 = rec.p_signal
        
        # Predict
        result = predict_cnn(model_data, signal_12, fs=rec.fs)
        pred_labels.append(LABEL_TO_IDX[result['prediction']])
        pred_probs.append(result['probabilities'])
        
    except Exception as e:
        errors.append((path, str(e)))

pred_labels = np.array(pred_labels)

print(f"\n✓ Completed predictions")
if errors:
    print(f"  ⚠️ {len(errors)} prediction errors (skipped)")
else:
    print(f"  ✅ All predictions successful")

# Skip records with errors
valid_mask = np.arange(len(test_paths))
if errors:
    error_indices = [test_paths.tolist().index(item[0]) for item in errors if item[0] in test_paths.tolist()]
    valid_mask = np.array([i for i in range(len(test_paths)) if i not in error_indices])

valid_labels = test_labels[:len(pred_labels)]
valid_preds = pred_labels

# Calculate metrics
accuracy = np.mean(valid_preds == valid_labels)
f1_macro = f1_score(valid_labels, valid_preds, average='macro')
f1_weighted = f1_score(valid_labels, valid_preds, average='weighted')

print(f"\n" + "="*70)
print("  OVERALL PERFORMANCE (FOLD 9 TEST SET)")
print("="*70)
print(f"\n  Accuracy: {accuracy:.1%}")
print(f"  Macro F1: {f1_macro:.3f}")
print(f"  Weighted F1: {f1_weighted:.3f}")

# Per-class metrics
print(f"\n  Classification Report:")
report = classification_report(
    valid_labels, 
    valid_preds,
    target_names=SUPERCLASS_LABELS,
    digits=3
)
print(report)

# Confusion matrix
print(f"\n  Confusion Matrix:")
cm = confusion_matrix(valid_labels, valid_preds, labels=range(len(SUPERCLASS_LABELS)))
print(f"    (rows=truth, columns=prediction)")
print()
for i, label in enumerate(SUPERCLASS_LABELS):
    row = cm[i]
    print(f"    {label:>6s}: {row}")

# Per-class analysis
print(f"\n" + "="*70)
print("  PER-CLASS ANALYSIS vs BASELINE")
print("="*70)

baseline_f1 = {
    "NORM": 0.83,
    "MI": 0.62,
    "STTC": 0.65,
    "HYP": 0.27,
    "CD": 0.68,
}

pred_f1s = []
for i, label in enumerate(SUPERCLASS_LABELS):
    class_pred = (valid_preds == i).astype(int)
    class_true = (valid_labels == i).astype(int)
    
    if np.sum(class_true) > 0:
        f1 = f1_score(class_true, class_pred)
    else:
        f1 = 0.0
    pred_f1s.append(f1)
    
    # Per-class breakdown
    tp = np.sum((valid_preds == i) & (valid_labels == i))
    fp = np.sum((valid_preds == i) & (valid_labels != i))
    fn = np.sum((valid_preds != i) & (valid_labels == i))
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
    
    baseline = baseline_f1.get(label, 0.0)
    delta = f1 - baseline
    delta_pct = (delta / baseline * 100) if baseline > 0 else 0
    
    print(f"\n  {label}:")
    print(f"    F1 Score: {f1:.3f} (baseline: {baseline:.3f}) [Δ {delta:+.3f} / {delta_pct:+.1f}%]")
    print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}")
    print(f"    TP: {int(tp)}, FP: {int(fp)}, FN: {int(fn)}")

print(f"\n" + "="*70)
print("  PHASE 1 IMPACT SUMMARY")
print("="*70)

improvements = []
for label, baseline_score in baseline_f1.items():
    idx = LABEL_TO_IDX[label]
    new_f1 = pred_f1s[idx]
    improvement = new_f1 - baseline_score
    improvements.append((label, baseline_score, new_f1, improvement))

# Sort by improvement
improvements.sort(key=lambda x: x[3], reverse=True)

print(f"\n  F1 Score Changes:")
for label, baseline, new, improvement in improvements:
    status = "✅" if improvement >= 0 else "⚠️"
    print(f"    {status} {label:>6s}: {baseline:.3f} → {new:.3f}  ({improvement:+.3f})")

print(f"\n  Summary:")
positive_improvements = sum(1 for _ , _, _, imp in improvements if imp > 0)
print(f"    Classes improved: {positive_improvements}/{len(SUPERCLASS_LABELS)}")
print(f"    Average change: {np.mean([imp for _, _, _, imp in improvements]):+.3f}")

# Target metrics from Phase 1 plan
print(f"\n  Phase 1 Goals vs Actual:")
print(f"    Goal: MI F1 +10-15%  | Actual: MI {improvements[3 if improvements[3][0] == 'MI' else 2][3]:+.3f}")
print(f"    Goal: HYP F1 +25-30% | Actual: HYP {improvements[3 if improvements[3][0] == 'HYP' else 2][3]:+.3f}")
print(f"    Goal: Overall accuracy +3-5% | Actual: {accuracy:.1%} (baseline ~69.9%)")

print(f"\n" + "="*70)
print("  Done.")
print("="*70)
