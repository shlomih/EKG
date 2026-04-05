"""
tune_thresholds.py
==================
Find optimal per-class classification thresholds using the validation set (fold 9).
Saves thresholds to models/thresholds_v<N>.json and evaluates on test set (fold 10).

Usage:
    python tune_thresholds.py                  # tune v2 model (14-class)
    python tune_thresholds.py --model v1       # tune v1 model (12-class)
    python tune_thresholds.py --model v3       # tune v3 model (26-class)
"""
import argparse
import json
import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_curve

os.chdir(Path(__file__).parent)

from cnn_classifier import N_AUX, SIGNAL_LEN, N_LEADS, ECGNetJoint
from multilabel_merged import (
    MergedECGDataset, load_merged_data, MERGED_CODES, N_CLASSES,
)
from multilabel_classifier import (
    load_demographics, preload_signals, CONF_THRESHOLD,
)


def collect_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for sig, aux, lbl in loader:
            logits = model(sig.to(device), aux.to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(lbl.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels).astype(int)


def find_best_thresholds(probs, labels, n_classes):
    """For each class find threshold that maximises F1 on given set."""
    thresholds = []
    for i in range(n_classes):
        if labels[:, i].sum() == 0:
            thresholds.append(0.5)
            continue
        prec, rec, thresh = precision_recall_curve(labels[:, i], probs[:, i])
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-8)
        best_idx = np.argmax(f1)
        best_t = float(thresh[best_idx]) if best_idx < len(thresh) else 0.5
        best_t = float(np.clip(best_t, 0.1, 0.9))
        thresholds.append(best_t)
    return thresholds


def evaluate_with_thresholds(probs, labels, thresholds, codes):
    preds = np.stack([
        (probs[:, i] >= thresholds[i]).astype(int)
        for i in range(len(thresholds))
    ], axis=1)
    per_f1 = f1_score(labels, preds, average=None, zero_division=0)
    macro  = float(np.mean(per_f1))
    micro  = float(f1_score(labels, preds, average="micro", zero_division=0))

    print(f"\n  MacroF1 : {macro:.3f}   MicroF1: {micro:.3f}")
    print(f"\n  {'Class':<8} {'Threshold':>9}  {'F1':>6}  {'N+':>6}")
    print("  " + "-" * 36)
    for i, code in enumerate(codes):
        n_pos = int(labels[:, i].sum())
        print(f"  {code:<8} {thresholds[i]:>9.3f}  {per_f1[i]:>6.3f}  {n_pos:>6}")
    return macro, micro, per_f1.tolist()


def run(model_version="v2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if model_version == "v3":
        from multilabel_v3 import V3ECGDataset, load_v3_data
        from dataset_challenge import V3_CODES, N_V3
        model_path   = "models/ecg_multilabel_v3_best.pt"
        thresh_path  = "models/thresholds_v3.json"
        codes        = V3_CODES
        n_classes    = N_V3
    elif model_version == "v2":
        model_path   = "models/ecg_multilabel_v2.pt"
        thresh_path  = "models/thresholds_v2.json"
        codes        = MERGED_CODES
        n_classes    = N_CLASSES
    else:
        from multilabel_classifier import MULTILABEL_CODES, N_ML_CLASSES
        model_path   = "models/ecg_multilabel.pt"
        thresh_path  = "models/thresholds_v1.json"
        codes        = MULTILABEL_CODES
        n_classes    = N_ML_CLASSES

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun training first.")

    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    model = ECGNetJoint(n_leads=N_LEADS, n_classes=n_classes, n_aux=N_AUX).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded: {model_path}  (best AUROC={ckpt.get('best_auroc', '?'):.3f})")

    print("\nLoading data...")
    if model_version == "v3":
        all_paths, all_labels, all_folds = load_v3_data()
    else:
        all_paths, all_labels, all_folds = load_merged_data()
    all_folds = np.array(all_folds)

    # V3: Challenge data lives in folds 19 (val) and 20 (test) — not 9/10.
    # Must use mixed masks so the 12 Challenge-only classes have positives to tune on.
    if model_version == "v3":
        val_mask  = (all_folds == 9)  | (all_folds == 19)
        test_mask = (all_folds == 10) | (all_folds == 20)
    else:
        val_mask  = all_folds == 9
        test_mask = all_folds == 10

    val_paths    = [p for p, m in zip(all_paths, val_mask)  if m]
    test_paths   = [p for p, m in zip(all_paths, test_mask) if m]
    val_labels   = all_labels[val_mask]
    test_labels  = all_labels[test_mask]

    demographics = load_demographics()
    # Only preload PTB-XL signals (folds 9 and 10); Challenge records load lazily
    ptbxl_val_paths  = [p for p, f in zip(all_paths, all_folds) if f == 9]
    ptbxl_test_paths = [p for p, f in zip(all_paths, all_folds) if f == 10]
    ptbxl_paths = list(set(ptbxl_val_paths + ptbxl_test_paths))
    raw_cache, aux_cache = preload_signals(ptbxl_paths, demographics)

    if model_version == "v3":
        val_ds  = V3ECGDataset(val_paths,  val_labels,  raw_cache, aux_cache)
        test_ds = V3ECGDataset(test_paths, test_labels, raw_cache, aux_cache)
    else:
        val_ds  = MergedECGDataset(val_paths,  val_labels,  raw_cache, aux_cache, demographics)
        test_ds = MergedECGDataset(test_paths, test_labels, raw_cache, aux_cache, demographics)
    val_loader  = DataLoader(val_ds,  batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    print(f"\nCollecting val probs  ({len(val_paths)} records)...")
    val_probs, val_labels_np = collect_probs(model, val_loader, device)

    print(f"\n--- Baseline (threshold={CONF_THRESHOLD}) on val set ---")
    base_thresh = [CONF_THRESHOLD] * n_classes
    evaluate_with_thresholds(val_probs, val_labels_np, base_thresh, codes)

    print(f"\n--- Tuned thresholds on val set ---")
    tuned = find_best_thresholds(val_probs, val_labels_np, n_classes)
    evaluate_with_thresholds(val_probs, val_labels_np, tuned, codes)

    print(f"\nCollecting test probs ({len(test_paths)} records)...")
    test_probs, test_labels_np = collect_probs(model, test_loader, device)

    test_set_label = "fold 10 + Challenge fold 20" if model_version == "v3" else "fold 10"
    print(f"\n--- Tuned thresholds on TEST set ({test_set_label}) ---")
    macro, micro, per_f1 = evaluate_with_thresholds(test_probs, test_labels_np, tuned, codes)

    result = {
        "model":      model_path,
        "thresholds": {code: float(t) for code, t in zip(codes, tuned)},
        "val_macro_f1":  float(f1_score(val_labels_np,
                             (val_probs >= np.array(tuned)).astype(int),
                             average="macro", zero_division=0)),
        "test_macro_f1": macro,
        "test_micro_f1": micro,
        "test_per_class_f1": {code: float(f) for code, f in zip(codes, per_f1)},
    }
    with open(thresh_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved: {thresh_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="v2", choices=["v1", "v2", "v3"])
    args = parser.parse_args()
    run(args.model)
