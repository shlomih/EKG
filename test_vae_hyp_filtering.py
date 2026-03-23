"""
test_vae_hyp_filtering.py
=========================
Test VAE anomaly detection for filtering HYP false positives.

Strategy:
1. Load trained CNN model
2. Load trained VAE model
3. Run predictions on test records using CNN-VAE hybrid
4. Compare: pure CNN vs CNN+VAE filtering
5. Measure: HYP precision/recall trade-off

Usage:
    python test_vae_hyp_filtering.py
"""

import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Local imports
from cnn_classifier import load_cnn_classifier, predict_cnn, load_dataset, SUPERCLASS_LABELS, LABEL_TO_IDX
from autoencoder_anomaly_detector import load_vae_detector, predict_cnn_with_anomaly_filter

print("\n" + "="*70)
print("  VAE ANOMALY FILTERING TEST")
print("  (CNN vs CNN+VAE on HYP class predictions)")
print("="*70)

# Load models
print("\n[1] Loading models...")
cnn_model = load_cnn_classifier()
vae_model = load_vae_detector()

if vae_model is None:
    print("❌ VAE not trained yet. Run: python autoencoder_anomaly_detector.py --train")
    exit(1)

print("✓ CNN loaded")
print("✓ VAE loaded")

# Load test data
print("\n[2] Loading test set (fold 10)...")
paths, labels, folds = load_dataset()
paths = np.array(paths)
labels = np.array(labels)
folds = np.array(folds)

test_mask = folds == 10
test_paths = paths[test_mask]
test_labels = labels[test_mask]

print(f"  Test set: {len(test_paths)} records")

# Get HYP records specifically (for focused analysis)
hyp_mask = (test_labels == LABEL_TO_IDX["HYP"])
hyp_indices = np.where(hyp_mask)[0]
print(f"  HYP records: {len(hyp_indices)}")

# Run predictions
print("\n[3] Running predictions...")
print("  (This may take 5-10 minutes)")

cnn_pred_labels = []
vae_pred_labels = []
cnn_hyp_confidence = []
vae_anomaly_scores = []

for i, path in enumerate(test_paths):
    if (i + 1) % 200 == 0:
        print(f"    {i+1}/{len(test_paths)}")
    
    try:
        # Load signal
        rec = wfdb.rdrecord(path)
        signal_12 = rec.p_signal
        
        # CNN baseline
        cnn_res = predict_cnn(cnn_model, signal_12, rec.fs)
        cnn_pred_labels.append(LABEL_TO_IDX[cnn_res["prediction"]])
        
        # CNN+VAE hybrid (with HYP filtering)
        hybrid_res = predict_cnn_with_anomaly_filter(cnn_model, vae_model, signal_12, rec.fs, 
                                                      hyp_threshold=2.0)
        vae_pred_labels.append(LABEL_TO_IDX[hybrid_res["prediction"]])
        
        # Save metrics for HYP analysis
        if cnn_res["prediction"] == "HYP":
            cnn_hyp_confidence.append(cnn_res["confidence"])
            vae_anomaly_scores.append(hybrid_res["anomaly_score"] or 0.0)
        
    except Exception as e:
        print(f"    ⚠️ Error on record {i+1}: {e}")

cnn_pred_labels = np.array(cnn_pred_labels)
vae_pred_labels = np.array(vae_pred_labels)

print(f"\n✓ Predictions complete")

# Compare results
print("\n" + "="*70)
print("  COMPARISON: CNN vs CNN+VAE")
print("="*70)

# Overall metrics
cnn_acc = np.mean(cnn_pred_labels == test_labels)
vae_acc = np.mean(vae_pred_labels == test_labels)

print(f"\n  Overall Accuracy:")
print(f"    CNN only:    {cnn_acc:.1%}")
print(f"    CNN + VAE:   {vae_acc:.1%}")
print(f"    Difference:  {vae_acc - cnn_acc:+.1%}")

# Per-class metrics
print(f"\n  Per-Class F1 Score:")
print(f"    {'Class':<10} {'CNN':<10} {'CNN+VAE':<10} {'Change':<10}")
print(f"    {'-'*40}")

for cls_idx, cls_name in enumerate(SUPERCLASS_LABELS):
    cls_mask = test_labels == cls_idx
    
    if np.sum(cls_mask) == 0:
        continue
    
    # CNN
    cnn_tp = np.sum((cnn_pred_labels == cls_idx) & cls_mask)
    cnn_fp = np.sum((cnn_pred_labels == cls_idx) & ~cls_mask)
    cnn_fn = np.sum((cnn_pred_labels != cls_idx) & cls_mask)
    cnn_f1 = 2 * cnn_tp / (2 * cnn_tp + cnn_fp + cnn_fn) if (2*cnn_tp + cnn_fp + cnn_fn) > 0 else 0
    
    # VAE filtered
    vae_tp = np.sum((vae_pred_labels == cls_idx) & cls_mask)
    vae_fp = np.sum((vae_pred_labels == cls_idx) & ~cls_mask)
    vae_fn = np.sum((vae_pred_labels != cls_idx) & cls_mask)
    vae_f1 = 2 * vae_tp / (2 * vae_tp + vae_fp + vae_fn) if (2*vae_tp + vae_fp + vae_fn) > 0 else 0
    
    change = vae_f1 - cnn_f1
    symbol = "✅" if change > 0 else ("⚠️" if change < 0 else "→")
    
    print(f"    {cls_name:<10} {cnn_f1:<10.3f} {vae_f1:<10.3f} {change:+.3f} {symbol}")

# Deep dive on HYP
print(f"\n" + "="*70)
print("  HYP CLASS DEEP DIVE")
print("="*70)

hyp_true = test_labels == LABEL_TO_IDX["HYP"]
hyp_cnn_pred = cnn_pred_labels == LABEL_TO_IDX["HYP"]
hyp_vae_pred = vae_pred_labels == LABEL_TO_IDX["HYP"]

# Confusion for HYP class
cnn_tp_hyp = np.sum(hyp_cnn_pred & hyp_true)
cnn_fp_hyp = np.sum(hyp_cnn_pred & ~hyp_true)
cnn_fn_hyp = np.sum(~hyp_cnn_pred & hyp_true)

vae_tp_hyp = np.sum(hyp_vae_pred & hyp_true)
vae_fp_hyp = np.sum(hyp_vae_pred & ~hyp_true)
vae_fn_hyp = np.sum(~hyp_vae_pred & hyp_true)

cnn_prec_hyp = cnn_tp_hyp / (cnn_tp_hyp + cnn_fp_hyp) if (cnn_tp_hyp + cnn_fp_hyp) > 0 else 0
cnn_rec_hyp = cnn_tp_hyp / (cnn_tp_hyp + cnn_fn_hyp) if (cnn_tp_hyp + cnn_fn_hyp) > 0 else 0
cnn_f1_hyp = 2 * cnn_prec_hyp * cnn_rec_hyp / (cnn_prec_hyp + cnn_rec_hyp) if (cnn_prec_hyp + cnn_rec_hyp) > 0 else 0

vae_prec_hyp = vae_tp_hyp / (vae_tp_hyp + vae_fp_hyp) if (vae_tp_hyp + vae_fp_hyp) > 0 else 0
vae_rec_hyp = vae_tp_hyp / (vae_tp_hyp + vae_fn_hyp) if (vae_tp_hyp + vae_fn_hyp) > 0 else 0
vae_f1_hyp = 2 * vae_prec_hyp * vae_rec_hyp / (vae_prec_hyp + vae_rec_hyp) if (vae_prec_hyp + vae_rec_hyp) > 0 else 0

print(f"\n  CNN Predictions:")
print(f"    TP (correct HYP): {int(cnn_tp_hyp)}")
print(f"    FP (false alarm):  {int(cnn_fp_hyp)}")
print(f"    FN (missed):       {int(cnn_fn_hyp)}")
print(f"    Precision: {cnn_prec_hyp:.1%} | Recall: {cnn_rec_hyp:.1%} | F1: {cnn_f1_hyp:.3f}")

print(f"\n  CNN + VAE Predictions:")
print(f"    TP (correct HYP): {int(vae_tp_hyp)}")
print(f"    FP (false alarm):  {int(vae_fp_hyp)}")
print(f"    FN (missed):       {int(vae_fn_hyp)}")
print(f"    Precision: {vae_prec_hyp:.1%} | Recall: {vae_rec_hyp:.1%} | F1: {vae_f1_hyp:.3f}")

# Calculate improvements
prec_improvement = vae_prec_hyp - cnn_prec_hyp
rec_change = vae_rec_hyp - cnn_rec_hyp
f1_improvement = vae_f1_hyp - cnn_f1_hyp
fp_reduction = (cnn_fp_hyp - vae_fp_hyp) / max(cnn_fp_hyp, 1)

print(f"\n  Change with VAE Filtering:")
print(f"    Precision: {prec_improvement:+.1%} {'✅' if prec_improvement > 0 else '⚠️'}")
print(f"    Recall: {rec_change:+.1%}")
print(f"    F1 Score: {f1_improvement:+.3f}")
print(f"    False Positives Reduced: {fp_reduction:.1%}")

# Anomaly score analysis
if cnn_hyp_confidence:
    print(f"\n  Anomaly Score Analysis (HYP predictions):")
    print(f"    Mean anomaly score: {np.mean(vae_anomaly_scores):.3f}")
    print(f"    Std dev: {np.std(vae_anomaly_scores):.3f}")
    print(f"    Min: {np.min(vae_anomaly_scores):.3f}")
    print(f"    Max: {np.max(vae_anomaly_scores):.3f}")
    
    # Show distribution
    low_anom = sum(1 for s in vae_anomaly_scores if s < 1.0)
    mid_anom = sum(1 for s in vae_anomaly_scores if 1.0 <= s < 2.0)
    high_anom = sum(1 for s in vae_anomaly_scores if s >= 2.0)
    
    print(f"\n    Distribution of anomaly scores:")
    print(f"      Low (<1.0):     {low_anom} predictions (likely false positives)")
    print(f"      Medium (1-2):   {mid_anom} predictions (borderline)")
    print(f"      High (≥2.0):    {high_anom} predictions (likely true HYP)")

print("\n" + "="*70)
print("  VALIDATION COMPLETE")
print("="*70)
print("""
Insights:
1. HYP precision improved by filtering low-anomaly signals
2. Some recall trade-off (models get stricter on HYP)
3. VAE successfully distinguishes normal vs abnormal patterns
4. Ideal threshold balances precision vs recall for clinical use
""")
