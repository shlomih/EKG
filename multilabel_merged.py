"""
multilabel_merged.py
====================
Train 14-class multi-label ECG classifier on merged PTB-XL + Chapman-Shaoxing.

New vs existing 12-class model:
  Adds AFIB  (48 -> 9,840 samples after merge)
  Adds STACH (  4 -> 7,255 samples after merge)

Label set (14 classes, MERGED_CODES from dataset_chapman.py):
  NORM, AFIB, PVC, LVH, IMI, ASMI, CLBBB, CRBBB, LAFB, 1AVB, ISC_, NDT, IRBBB, STACH

Split strategy:
  Test  : PTB-XL fold 10 only  (same as baseline -- allows direct comparison)
  Val   : PTB-XL fold 9 only
  Train : PTB-XL folds 1-8  +  ALL Chapman records

Usage:
    python multilabel_merged.py           # train + eval
    python multilabel_merged.py --eval    # eval saved model
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support

warnings.filterwarnings("ignore")
os.chdir(Path(__file__).parent)

from cnn_classifier import (
    N_AUX, SIGNAL_LEN, N_LEADS,
    ECGNetJoint,
    _load_raw_signal, extract_voltage_features, augment_signal,
)
from torch.utils.data import Dataset
from multilabel_classifier import (
    MULTILABEL_CODES,        # 12-class PTB-XL label set
    load_multilabel_dataset,
    preload_signals,
    load_demographics,
    compute_pos_weights,
    CONF_THRESHOLD,
)
from dataset_chapman import MERGED_CODES, N_MERGED, load_chapman_multilabel

# -- Constants -----------------------------------------------------------------
MODEL_PATH = "models/ecg_multilabel_v2.pt"
N_CLASSES  = N_MERGED   # 14

# Mapping from 12-class PTB-XL index -> 14-class MERGED index
PTBXL_TO_MERGED = np.array([
    MERGED_CODES.index(c) for c in MULTILABEL_CODES
], dtype=int)   # shape (12,)


# -- Evaluate (14-class aware) -------------------------------------------------
def evaluate(model, loader, device, criterion=None):
    """Like multilabel_classifier.evaluate but uses N_CLASSES (14) not N_ML_CLASSES (12)."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for sig, aux, lbl in loader:
            sig, aux, lbl = sig.to(device), aux.to(device), lbl.to(device)
            logits = model(sig, aux)
            if criterion is not None:
                total_loss += criterion(logits, lbl).item()
                n_batches  += 1
            all_logits.append(logits.cpu().numpy())
            all_labels.append(lbl.cpu().numpy())

    logits_np = np.vstack(all_logits)      # (N, 14)
    labels_np = np.vstack(all_labels)      # (N, 14)
    probs_np  = 1 / (1 + np.exp(-logits_np))
    preds_np  = (probs_np >= CONF_THRESHOLD).astype(int)

    aurocs = []
    for i in range(N_CLASSES):
        if labels_np[:, i].sum() > 0 and labels_np[:, i].sum() < len(labels_np):
            aurocs.append(roc_auc_score(labels_np[:, i], probs_np[:, i]))
        else:
            aurocs.append(float("nan"))

    _, _, f1s, _ = precision_recall_fscore_support(
        labels_np, preds_np, average=None, labels=list(range(N_CLASSES)), zero_division=0
    )

    macro_auroc = float(np.nanmean(aurocs))
    macro_f1    = float(np.nanmean(f1s))
    avg_loss    = total_loss / max(n_batches, 1)

    return {
        "loss": avg_loss,
        "macro_auroc": macro_auroc,
        "macro_f1": macro_f1,
        "per_class_auroc": aurocs,
        "per_class_f1": f1s.tolist(),
    }


# -- Dataset with lazy fallback ------------------------------------------------
class MergedECGDataset(Dataset):
    """Like MultiLabelECGDataset but loads from disk if not in cache (lazy fallback)."""

    def __init__(self, paths, label_matrix, raw_cache, aux_cache, demographics, augment=False):
        self.paths        = paths
        self.label_matrix = label_matrix
        self.raw_cache    = raw_cache      # may not contain all paths (Chapman lazy-loaded)
        self.aux_cache    = aux_cache
        self.demographics = demographics
        self.augment      = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if path in self.raw_cache:
            sig = self.raw_cache[path].copy()
            aux = self.aux_cache[path]
        else:
            # Lazy load from disk (Chapman records not preloaded)
            sig = _load_raw_signal(path)
            aux = extract_voltage_features(sig)
        if self.augment:
            sig = augment_signal(sig)
        sig_norm = (sig / 5.0).astype(np.float32)
        return (
            torch.from_numpy(sig_norm),
            torch.from_numpy(aux),
            torch.from_numpy(self.label_matrix[idx]),
        )


# -- Data loading --------------------------------------------------------------
def load_merged_data():
    """
    Returns paths, label_matrix (N, 14), fold_array (N,).
    Fold meanings:
      1-8  : PTB-XL train folds
      9    : PTB-XL val fold
      10   : PTB-XL test fold  (never used for training)
      0    : Chapman records  (treated as training-only, excluded from test)
    """
    # -- PTB-XL ----------------------------------------------------------------
    ptb_paths, ptb_labels12, ptb_folds = load_multilabel_dataset()
    ptb_folds = np.array(ptb_folds)

    # Expand 12-class -> 14-class
    N_ptb = len(ptb_paths)
    ptb_labels14 = np.zeros((N_ptb, N_CLASSES), dtype=np.float32)
    ptb_labels14[:, PTBXL_TO_MERGED] = ptb_labels12

    # -- Chapman ---------------------------------------------------------------
    chap_paths, chap_labels14 = load_chapman_multilabel()
    N_chap = len(chap_paths)
    chap_folds = np.zeros(N_chap, dtype=int)   # fold 0 = train-only

    # -- Merge -----------------------------------------------------------------
    all_paths  = ptb_paths + chap_paths
    all_labels = np.concatenate([ptb_labels14, chap_labels14], axis=0)
    all_folds  = np.concatenate([ptb_folds, chap_folds], axis=0)

    print(f"  Merged: {N_ptb} PTB-XL + {N_chap} Chapman = {len(all_paths)} total")
    print(f"  Per-class positives:")
    for i, code in enumerate(MERGED_CODES):
        n = int(all_labels[:, i].sum())
        print(f"    {code:<6}: {n:>6}", end="")
        if (i + 1) % 4 == 0:
            print()
    print()
    return all_paths, all_labels, all_folds


# -- Training ------------------------------------------------------------------
def train(batch_size: int = 64, n_epochs: int = 60, patience: int = 12):
    print("\n" + "=" * 60)
    print("  Merged Multi-Label ECG Classifier  (14 conditions)")
    print("  PTB-XL + Chapman-Shaoxing")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    all_paths, all_labels, all_folds = load_merged_data()

    # Splits: train = folds 1-8 + Chapman (0), val = fold 9, test = fold 10
    train_mask = (all_folds <= 8)   # includes fold 0 (Chapman)
    val_mask   = (all_folds == 9)
    test_mask  = (all_folds == 10)

    train_paths  = [p for p, m in zip(all_paths, train_mask) if m]
    val_paths    = [p for p, m in zip(all_paths, val_mask)   if m]
    test_paths   = [p for p, m in zip(all_paths, test_mask)  if m]
    train_labels = all_labels[train_mask]
    val_labels   = all_labels[val_mask]
    test_labels  = all_labels[test_mask]

    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)} (PTB-XL fold 10)")

    # Only preload PTB-XL signals (~4.4 GB); Chapman loads lazily from SSD
    print("  Pre-loading PTB-XL signals into RAM (~4.4 GB)...")
    demographics = load_demographics()
    ptbxl_paths  = [p for p in set(train_paths + val_paths + test_paths)
                    if "ptbxl" in p]
    raw_cache, aux_cache = preload_signals(ptbxl_paths, demographics)
    print(f"  Cached {len(raw_cache)} PTB-XL signals. Chapman records load lazily from SSD.")

    train_ds = MergedECGDataset(train_paths, train_labels, raw_cache, aux_cache, demographics, augment=True)
    val_ds   = MergedECGDataset(val_paths,   val_labels,   raw_cache, aux_cache, demographics, augment=False)
    test_ds  = MergedECGDataset(test_paths,  test_labels,  raw_cache, aux_cache, demographics, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    model      = ECGNetJoint(n_leads=N_LEADS, n_classes=N_CLASSES, n_aux=N_AUX).to(device)
    n_params   = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params/1e6:.1f}M  ({N_CLASSES} output classes)")

    pos_weight = compute_pos_weights(train_labels).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs, pct_start=0.1,
    )

    best_auroc = 0.0
    best_state = None
    no_improve = 0

    print(f"\n  {'Ep':>3}  {'Loss':>7}  {'ValAUROC':>8}  {'ValF1':>7}")
    print("  " + "-" * 35)

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for sig, aux, lbl in train_loader:
            sig, aux, lbl = sig.to(device), aux.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(sig, aux), lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(sig)

        m = evaluate(model, val_loader, device, criterion)
        improved = m["macro_auroc"] > best_auroc
        if improved:
            best_auroc = m["macro_auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = " <*>"
            # Save best model locally on every improvement
            torch.save({"model_state": best_state, "best_auroc": best_auroc,
                        "label_codes": MERGED_CODES, "n_classes": N_CLASSES}, MODEL_PATH)
            # Also save to Drive if running on Colab
            drive_ckpt = "/content/drive/MyDrive/EKG_models/ecg_multilabel_v2_best.pt"
            if os.path.exists("/content/drive/MyDrive/EKG_models"):
                torch.save({"model_state": best_state, "best_auroc": best_auroc,
                            "label_codes": MERGED_CODES, "n_classes": N_CLASSES}, drive_ckpt)
                print(f"  Checkpoint saved to Drive (AUROC={best_auroc:.3f})", flush=True)
        else:
            no_improve += 1
            marker = ""

        elapsed = time.time() - t0
        print(f"  {epoch:>3}  {m['loss']:>7.4f}  {m['macro_auroc']:>8.3f}  {m['macro_f1']:>7.3f}  [{elapsed:.0f}s]{marker}",
              flush=True)

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # Final test evaluation
    if best_state:
        model.load_state_dict(best_state)
    torch.save({
        "model_state": best_state or model.state_dict(),
        "best_auroc":  best_auroc,
        "label_codes": MERGED_CODES,
        "n_classes":   N_CLASSES,
    }, MODEL_PATH)
    print(f"\n  Saved: {MODEL_PATH}")

    tm = evaluate(model, test_loader, device)
    _print_merged_results(tm, model, test_loader, device)


def _print_merged_results(m, model, loader, device):
    """Print per-class results, highlight AFIB and STACH (new classes)."""
    # Collect all probs + labels
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for sig, aux, lbl in loader:
            logits = model(sig.to(device), aux.to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(lbl.numpy())
    probs  = np.concatenate(all_probs,  axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(int)
    preds  = (probs >= 0.5).astype(int)

    macro_f1    = f1_score(labels, preds, average="macro",   zero_division=0)
    micro_f1    = f1_score(labels, preds, average="micro",   zero_division=0)
    # Only compute AUROC for classes that have at least one positive in test set
    valid = np.where(labels.sum(axis=0) > 0)[0]
    per_auroc = np.full(len(MERGED_CODES), float("nan"))
    if len(valid) > 0:
        per_auroc[valid] = roc_auc_score(labels[:, valid], probs[:, valid], average=None)
    macro_auroc = np.nanmean(per_auroc)

    per_f1 = f1_score(labels, preds, average=None, zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"  Merged Model - Test Results (PTB-XL fold 10)")
    print(f"{'=' * 60}")
    print(f"  MacroF1   : {macro_f1:.3f}   (12-class CNN baseline: 0.699)")
    print(f"  MicroF1   : {micro_f1:.3f}")
    print(f"  MacroAUROC: {macro_auroc:.3f}  (12-class CNN baseline: 0.972)")
    print(f"\n  Per-class ([NEW] = added from Chapman):")
    for i, code in enumerate(MERGED_CODES):
        tag   = " [NEW]" if code in ("AFIB", "STACH") else "      "
        n_pos = int(labels[:, i].sum())
        auroc_str = f"{per_auroc[i]:.3f}" if not np.isnan(per_auroc[i]) else "  n/a"
        print(f"    {code:>6}{tag}: F1={per_f1[i]:.3f}  AUROC={auroc_str}  (n={n_pos})")


# -- Eval only -----------------------------------------------------------------
def eval_saved():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model  = ECGNetJoint(n_leads=N_LEADS, n_classes=N_CLASSES, n_aux=N_AUX).to(device)
    model.load_state_dict(ckpt["model_state"])

    all_paths, all_labels, all_folds = load_merged_data()
    all_folds   = np.array(all_folds)
    test_mask   = all_folds == 10
    test_paths  = [p for p, m in zip(all_paths, test_mask) if m]
    test_labels = all_labels[test_mask]

    demographics = load_demographics()
    raw_cache, aux_cache = preload_signals(test_paths, demographics)
    test_ds     = MergedECGDataset(test_paths, test_labels, raw_cache, aux_cache, demographics)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    _print_merged_results(None, model, test_loader, device)


# -- Main ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluate saved model only")
    args = parser.parse_args()

    if args.eval:
        eval_saved()
    else:
        train()
