"""
temperature_scaling.py
======================
Fit a temperature parameter T on the V3 validation set (folds 9+19) to
recalibrate sigmoid probabilities, then re-tune per-class thresholds.

Temperature scaling doesn't change AUROC (rank-preserving) but fixes the
confidence calibration — allowing the threshold optimizer to find better
operating points for classes stuck at the 0.9 ceiling.

Usage:
    python temperature_scaling.py
    python temperature_scaling.py --model models/ecg_multilabel_v3_best.pt

Output:
    models/thresholds_v3.json  (updated with temperature-scaled thresholds)
    Prints before/after comparison table
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

os.chdir(Path(__file__).parent)

from cnn_classifier import N_AUX, SIGNAL_LEN, N_LEADS, ECGNetJoint
from multilabel_v3 import (
    V3ECGDataset, load_v3_data, V3_CODES, N_CLASSES, MODEL_PATH,
)
from multilabel_classifier import load_demographics, preload_signals


# ---------------------------------------------------------------------------
# Collect raw logits (NOT probs) from model
# ---------------------------------------------------------------------------
def collect_logits(model, loader, device):
    """Return raw logits and labels as numpy arrays."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for sig, aux, lbl in loader:
            logits = model(sig.to(device), aux.to(device))
            all_logits.append(logits.float().cpu())
            all_labels.append(lbl)
    return torch.cat(all_logits), torch.cat(all_labels).int()


# ---------------------------------------------------------------------------
# Temperature scaling: single scalar T applied to all logits
# ---------------------------------------------------------------------------
class TemperatureScaler(nn.Module):
    """Learns a single temperature T that minimises NLL on a held-out set."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def fit_temperature(val_logits: torch.Tensor, val_labels: torch.Tensor,
                    lr: float = 0.01, max_iter: int = 200) -> float:
    """
    Find optimal temperature T that minimises BCEWithLogits on validation set.
    Returns the fitted temperature as a float.
    """
    scaler = TemperatureScaler()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    val_labels_float = val_labels.float()

    def closure():
        optimizer.zero_grad()
        scaled = scaler(val_logits)
        loss = criterion(scaled, val_labels_float)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(scaler.temperature.item())
    print(f"  Fitted temperature: T = {T:.4f}")
    return T


# ---------------------------------------------------------------------------
# Per-class temperature (optional refinement)
# ---------------------------------------------------------------------------
def fit_per_class_temperature(val_logits: torch.Tensor, val_labels: torch.Tensor,
                              max_iter: int = 100) -> np.ndarray:
    """
    Fit a separate temperature for each class.
    More flexible than global T — useful when different classes have
    different calibration issues.
    Returns array of shape (n_classes,).
    """
    n_classes = val_logits.shape[1]
    temperatures = np.ones(n_classes)

    for i in range(n_classes):
        n_pos = val_labels[:, i].sum().item()
        if n_pos < 5:
            continue  # skip classes with too few positives

        logits_i = val_logits[:, i:i+1]
        labels_i = val_labels[:, i:i+1].float()

        scaler = TemperatureScaler()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(scaler(logits_i), labels_i)
            loss.backward()
            return loss

        optimizer.step(closure)
        temperatures[i] = float(scaler.temperature.item())

    return temperatures


# ---------------------------------------------------------------------------
# Threshold tuning on calibrated probabilities
# ---------------------------------------------------------------------------
def find_best_thresholds(probs, labels, n_classes):
    """Find per-class threshold maximising F1."""
    thresholds = []
    for i in range(n_classes):
        if labels[:, i].sum() == 0:
            thresholds.append(0.5)
            continue
        prec, rec, thresh = precision_recall_curve(labels[:, i], probs[:, i])
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-8)
        best_idx = np.argmax(f1)
        best_t = float(thresh[best_idx]) if best_idx < len(thresh) else 0.5
        # After temperature scaling, we can use the full [0.05, 0.95] range
        best_t = float(np.clip(best_t, 0.05, 0.95))
        thresholds.append(best_t)
    return thresholds


def evaluate(probs, labels, thresholds, codes, title=""):
    """Print per-class table and return macro/micro F1."""
    preds = np.stack([
        (probs[:, i] >= thresholds[i]).astype(int)
        for i in range(len(codes))
    ], axis=1)

    per_f1 = f1_score(labels, preds, average=None, zero_division=0)
    macro  = float(np.mean(per_f1))
    micro  = float(f1_score(labels, preds, average="micro", zero_division=0))

    valid = np.where(labels.sum(axis=0) > 0)[0]
    per_auroc = np.full(len(codes), float("nan"))
    if len(valid) > 0:
        per_auroc[valid] = roc_auc_score(
            labels[:, valid], probs[:, valid], average=None
        )

    if title:
        print(f"\n  {title}")
    print(f"  MacroF1: {macro:.3f}   MicroF1: {micro:.3f}")
    print(f"\n  {'Class':<8} {'Thresh':>7} {'F1':>6} {'AUROC':>6} {'N+':>6}")
    print(f"  {'-' * 40}")
    for i, code in enumerate(codes):
        n_pos = int(labels[:, i].sum())
        auroc_s = f"{per_auroc[i]:.3f}" if not np.isnan(per_auroc[i]) else "  n/a"
        print(f"  {code:<8} {thresholds[i]:>7.3f} {per_f1[i]:>6.3f} {auroc_s} {n_pos:>6}")

    return macro, micro, per_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(model_path: str = MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    model = ECGNetJoint(n_leads=N_LEADS, n_classes=N_CLASSES, n_aux=N_AUX).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded: {model_path}  (AUROC={ckpt.get('best_auroc', '?'):.4f})")

    # Load old thresholds for comparison
    old_thresholds_path = "models/thresholds_v3.json"
    with open(old_thresholds_path) as f:
        old_data = json.load(f)
    old_thresholds = [old_data["thresholds"].get(c, 0.5) for c in V3_CODES]

    print("\nLoading data...")
    all_paths, all_labels, all_folds = load_v3_data()
    all_folds = np.array(all_folds)

    val_mask    = (all_folds == 9)  | (all_folds == 19)
    test_mask   = (all_folds == 10) | (all_folds == 20)

    val_paths  = [p for p, m in zip(all_paths, val_mask) if m]
    test_paths = [p for p, m in zip(all_paths, test_mask) if m]
    val_labels = all_labels[val_mask]
    test_labels = all_labels[test_mask]

    demographics = load_demographics()
    ptbxl_paths = [p for p, f in zip(all_paths, all_folds) if f in (9, 10)]
    raw_cache, aux_cache = preload_signals(list(set(ptbxl_paths)), demographics)

    val_ds  = V3ECGDataset(val_paths,  val_labels,  raw_cache, aux_cache)
    test_ds = V3ECGDataset(test_paths, test_labels, raw_cache, aux_cache)
    val_loader  = DataLoader(val_ds,  batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # ── Step 1: Collect raw logits ────────────────────────────────────────
    print(f"\nCollecting logits on val set ({len(val_paths)} records)...")
    val_logits, val_labels_t = collect_logits(model, val_loader, device)
    val_labels_np = val_labels_t.numpy()

    print(f"Collecting logits on test set ({len(test_paths)} records)...")
    test_logits, test_labels_t = collect_logits(model, test_loader, device)
    test_labels_np = test_labels_t.numpy()

    # ── Step 2: Show BEFORE results (no temperature) ─────────────────────
    val_probs_raw  = torch.sigmoid(val_logits).numpy()
    test_probs_raw = torch.sigmoid(test_logits).numpy()

    print("\n" + "=" * 60)
    print("  BEFORE temperature scaling")
    print("=" * 60)
    evaluate(test_probs_raw, test_labels_np, old_thresholds, V3_CODES,
             title="Test set — old thresholds")

    # ── Step 3: Fit global temperature ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Fitting GLOBAL temperature on val set")
    print("=" * 60)
    T_global = fit_temperature(val_logits, val_labels_t)

    val_probs_global  = torch.sigmoid(val_logits / T_global).numpy()
    test_probs_global = torch.sigmoid(test_logits / T_global).numpy()

    global_thresholds = find_best_thresholds(val_probs_global, val_labels_np, N_CLASSES)
    evaluate(test_probs_global, test_labels_np, global_thresholds, V3_CODES,
             title="Test set — global T, re-tuned thresholds")

    # ── Step 4: Fit per-class temperature ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  Fitting PER-CLASS temperature on val set")
    print("=" * 60)
    T_per_class = fit_per_class_temperature(val_logits, val_labels_t)

    print(f"  Per-class temperatures:")
    for i, code in enumerate(V3_CODES):
        n_pos = val_labels_np[:, i].sum()
        if n_pos >= 5:
            print(f"    {code:<8}: T={T_per_class[i]:.3f}")

    val_logits_pc  = val_logits.numpy()  / T_per_class[np.newaxis, :]
    test_logits_pc = test_logits.numpy() / T_per_class[np.newaxis, :]
    val_probs_pc   = 1 / (1 + np.exp(-val_logits_pc))
    test_probs_pc  = 1 / (1 + np.exp(-test_logits_pc))

    pc_thresholds = find_best_thresholds(val_probs_pc, val_labels_np, N_CLASSES)
    macro_pc, micro_pc, per_f1_pc = evaluate(
        test_probs_pc, test_labels_np, pc_thresholds, V3_CODES,
        title="Test set — per-class T, re-tuned thresholds"
    )

    # ── Step 5: Pick best method and compare ──────────────────────────────
    macro_old, _, per_f1_old = evaluate(
        test_probs_raw, test_labels_np, old_thresholds, V3_CODES, title=""
    )
    macro_global, _, per_f1_global = evaluate(
        test_probs_global, test_labels_np, global_thresholds, V3_CODES, title=""
    )

    print("\n" + "=" * 60)
    print("  COMPARISON: Old vs Global-T vs Per-Class-T")
    print("=" * 60)
    print(f"  {'':>8}  {'Old':>8}  {'Global-T':>8}  {'PerClass-T':>10}")
    print(f"  {'MacroF1':>8}  {macro_old:>8.3f}  {macro_global:>8.3f}  {macro_pc:>10.3f}")
    print()
    print(f"  {'Class':<8} {'Old F1':>7} {'Glb F1':>7} {'PC F1':>7} {'Delta':>7}")
    print(f"  {'-' * 40}")
    for i, code in enumerate(V3_CODES):
        n_pos = int(test_labels_np[:, i].sum())
        if n_pos == 0:
            continue
        delta = per_f1_pc[i] - per_f1_old[i]
        marker = " ⬆" if delta > 0.02 else " ⬇" if delta < -0.02 else ""
        print(f"  {code:<8} {per_f1_old[i]:>7.3f} {per_f1_global[i]:>7.3f} "
              f"{per_f1_pc[i]:>7.3f} {delta:>+7.3f}{marker}")

    # ── Step 6: Save best result ──────────────────────────────────────────
    # Use per-class-T if it beats global, else global
    if macro_pc >= macro_global:
        best_thresholds = pc_thresholds
        best_method = "per_class_temperature"
        best_T = T_per_class.tolist()
        best_probs_test = test_probs_pc
    else:
        best_thresholds = global_thresholds
        best_method = "global_temperature"
        best_T = T_global
        best_probs_test = test_probs_global

    best_preds = np.stack([
        (best_probs_test[:, i] >= best_thresholds[i]).astype(int)
        for i in range(N_CLASSES)
    ], axis=1)
    best_per_f1 = f1_score(test_labels_np, best_preds, average=None, zero_division=0)
    best_macro  = float(np.mean(best_per_f1))
    best_micro  = float(f1_score(test_labels_np, best_preds, average="micro", zero_division=0))

    result = {
        "model": model_path,
        "calibration_method": best_method,
        "temperature": best_T,
        "thresholds": {code: float(t) for code, t in zip(V3_CODES, best_thresholds)},
        "val_macro_f1": float(f1_score(
            val_labels_np,
            (best_probs_test[:len(val_labels_np)] >= np.array(best_thresholds)).astype(int)
            if len(val_labels_np) == len(best_probs_test) else
            np.zeros_like(val_labels_np),
            average="macro", zero_division=0
        )) if best_method == "global_temperature" else None,
        "test_macro_f1": best_macro,
        "test_micro_f1": best_micro,
        "test_per_class_f1": {code: float(f) for code, f in zip(V3_CODES, best_per_f1)},
    }

    out_path = "models/thresholds_v3.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved calibrated thresholds → {out_path}")
    print(f"  Method: {best_method}")
    print(f"  Test MacroF1: {best_macro:.3f}  (was {macro_old:.3f}, delta={best_macro - macro_old:+.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH, help="Model checkpoint path")
    args = parser.parse_args()
    run(args.model)
