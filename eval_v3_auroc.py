"""
eval_v3_auroc.py
================
Per-class AUROC + F1 report for the V3 26-class model.
Evaluates on both PTB-XL test (fold 10) and Challenge test (fold 20) combined,
and separately, to distinguish generalization vs. calibration issues.

Usage:
    python eval_v3_auroc.py
    python eval_v3_auroc.py --model models/ecg_multilabel_v3_best.pt  # Colab checkpoint

Output: console table + saves eval_v3_auroc_results.json
"""

import argparse
import json
import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

os.chdir(Path(__file__).parent)

from cnn_classifier import N_AUX, SIGNAL_LEN, N_LEADS, ECGNetJoint
from multilabel_v3 import (
    V3ECGDataset, load_v3_data, V3_CODES, N_CLASSES,
    V3_URGENCY,
)
from multilabel_classifier import load_demographics, preload_signals
from dataset_chapman import MERGED_CODES
from dataset_code15 import build_code15_demo_cache, CODE15_INDEX, CODE15_PATH_PREFIX


def collect_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for sig, aux, lbl in loader:
            logits = model(sig.to(device), aux.to(device))
            all_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            all_labels.append(lbl.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels).astype(int)


def print_table(title, probs, labels, tuned_thresholds, codes, new_codes):
    preds = np.stack([
        (probs[:, i] >= tuned_thresholds.get(code, 0.5)).astype(int)
        for i, code in enumerate(codes)
    ], axis=1)

    per_f1 = f1_score(labels, preds, average=None, zero_division=0)
    macro_f1 = float(np.mean(per_f1))
    micro_f1 = float(f1_score(labels, preds, average="micro", zero_division=0))

    valid = np.where(labels.sum(axis=0) > 0)[0]
    per_auroc = np.full(len(codes), float("nan"))
    if len(valid) > 0:
        per_auroc[valid] = roc_auc_score(
            labels[:, valid], probs[:, valid], average=None
        )
    macro_auroc = float(np.nanmean(per_auroc))

    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"  Records: {len(labels)}   MacroAUROC: {macro_auroc:.4f}   MacroF1: {macro_f1:.3f}   MicroF1: {micro_f1:.3f}")
    print(f"{'=' * 72}")
    print(f"  {'Class':<8} {'Tag':<5} {'Urg':>3}  {'AUROC':>6}  {'F1':>6}  {'Thresh':>7}  {'N+':>6}  {'Diagnosis'}")
    print(f"  {'-' * 68}")

    rows = []
    for i, code in enumerate(codes):
        tag = "[NEW]" if code in new_codes else "     "
        urg = V3_URGENCY.get(code, 0)
        n_pos = int(labels[:, i].sum())
        auroc_str = f"{per_auroc[i]:.4f}" if not np.isnan(per_auroc[i]) else "   n/a"
        thresh = tuned_thresholds.get(code, 0.5)
        rows.append((code, tag, urg, per_auroc[i], per_f1[i], thresh, n_pos))
        print(f"  {code:<8} {tag} {urg:>3}  {auroc_str}  {per_f1[i]:>6.3f}  {thresh:>7.3f}  {n_pos:>6}")

    # Flag weak classes (AUROC < 0.85 or F1 < 0.40 with enough positives)
    print(f"\n  ⚠️  Attention flags (AUROC < 0.85 or F1 < 0.40 with N+ ≥ 10):")
    flagged = False
    for code, tag, urg, auroc, f1_val, thresh, n_pos in rows:
        if n_pos >= 10:
            if (not np.isnan(auroc) and auroc < 0.85) or f1_val < 0.40:
                issue = []
                if not np.isnan(auroc) and auroc < 0.85:
                    issue.append(f"AUROC={auroc:.3f}")
                if f1_val < 0.40:
                    issue.append(f"F1={f1_val:.3f}")
                print(f"    {code:<8}: {', '.join(issue)}  (n={n_pos})")
                flagged = True
    if not flagged:
        print("    None — all classes with sufficient data are performing well.")

    return {
        "macro_auroc": macro_auroc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class": {
            code: {
                "auroc": float(per_auroc[i]) if not np.isnan(per_auroc[i]) else None,
                "f1": float(per_f1[i]),
                "threshold": tuned_thresholds.get(code, 0.5),
                "n_pos": int(labels[:, i].sum()),
                "is_new": code in new_codes,
            }
            for i, code in enumerate(codes)
        },
    }


def run(model_path: str = "models/ecg_multilabel_v3.pt",
        thresholds_path: str = "models/thresholds_v3.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    model = ECGNetJoint(n_leads=N_LEADS, n_classes=N_CLASSES, n_aux=N_AUX).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded: {model_path}  (best_auroc={ckpt.get('best_auroc', '?'):.4f})")

    with open(thresholds_path) as f:
        tuned_thresholds = json.load(f)["thresholds"]
    print(f"Thresholds: {thresholds_path}")

    print("\nLoading data (this takes ~2 min on CPU)...")
    all_paths, all_labels, all_folds = load_v3_data()
    all_folds = np.array(all_folds)

    ptbxl_test_mask = all_folds == 10
    chal_test_mask  = all_folds == 20
    c15_test_mask   = all_folds == 30
    combined_mask   = ptbxl_test_mask | chal_test_mask

    demographics = load_demographics()
    ptbxl_test_paths = [p for p, m in zip(all_paths, ptbxl_test_mask) if m]
    raw_cache, aux_cache = preload_signals(ptbxl_test_paths, demographics)

    # CODE-15% demographics cache (age + sex)
    c15_demo_cache = build_code15_demo_cache(CODE15_INDEX) if CODE15_INDEX.exists() else {}

    def make_loader(mask):
        paths  = [p for p, m in zip(all_paths, mask) if m]
        labels = all_labels[mask]
        ds = V3ECGDataset(paths, labels, raw_cache, aux_cache)
        ds.demo_cache = c15_demo_cache   # needed for CODE-15% HDF5 signal loading
        return DataLoader(ds, batch_size=128, shuffle=False, num_workers=0), labels

    new_codes = set(V3_CODES) - set(MERGED_CODES)

    results = {}

    print("\nRunning inference on PTB-XL test (fold 10)...")
    loader, labels = make_loader(ptbxl_test_mask)
    probs, lbl_np = collect_probs(model, loader, device)
    results["ptbxl_test"] = print_table(
        "PTB-XL Test (fold 10) — original 14 classes", probs, lbl_np, tuned_thresholds, V3_CODES, new_codes
    )

    print("\nRunning inference on Challenge test (fold 20)...")
    loader, labels = make_loader(chal_test_mask)
    probs, lbl_np = collect_probs(model, loader, device)
    results["challenge_test"] = print_table(
        "Challenge Test (fold 20) — all 26 classes", probs, lbl_np, tuned_thresholds, V3_CODES, new_codes
    )

    print("\nRunning inference on Combined test (fold 10 + fold 20)...")
    loader, labels = make_loader(combined_mask)
    probs, lbl_np = collect_probs(model, loader, device)
    results["combined_test"] = print_table(
        "Combined Test (fold 10 + 20) — full 26-class evaluation", probs, lbl_np, tuned_thresholds, V3_CODES, new_codes
    )

    if c15_test_mask.sum() > 0:
        print("\nRunning inference on CODE-15% test (fold 30)...")
        loader, labels = make_loader(c15_test_mask)
        probs, lbl_np = collect_probs(model, loader, device)
        results["code15_test"] = print_table(
            "CODE-15% Test (fold 30) — AF/1dAVb/RBBB/LBBB/SB/ST + NORM", probs, lbl_np,
            tuned_thresholds, V3_CODES, new_codes
        )

    out_path = "eval_v3_auroc_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ecg_multilabel_v3_best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--thresholds", default="models/thresholds_v3.json",
                        help="Path to thresholds JSON")
    args = parser.parse_args()
    run(args.model, args.thresholds)
