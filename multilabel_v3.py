"""
multilabel_v3.py
================
Train 26-class multi-label ECG classifier.
Builds on the 14-class v2 model by adding 12 new conditions from PhysioNet 2021 Challenge.

Label set (26 classes, V3_CODES from dataset_challenge.py):
  From PTB-XL+Chapman (14): NORM, AFIB, PVC, LVH, IMI, ASMI, CLBBB, CRBBB, LAFB,
                             1AVB, ISC_, NDT, IRBBB, STACH
  New from Challenge  (12): PAC, Brady, SVT, LQTP, TAb, LAD, RAD, NSIVC,
                             AFL, STc, STD, LAE

Data sources:
  PTB-XL     : 18,524 records  (folds 1-10, test=fold10, val=fold9)
  Chapman    : 42,390 records  (train only, fold 0)
  Challenge  : 50,842 records  (Georgia + CPSC + Ningbo, train only, fold 0)
  Total      : ~111,756 records

Usage:
    python multilabel_v3.py               # train from v2 checkpoint
    python multilabel_v3.py --scratch     # train from random init
    python multilabel_v3.py --eval        # eval saved v3 model
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support

warnings.filterwarnings("ignore")
os.chdir(Path(__file__).parent)

from cnn_classifier import (
    N_AUX, SIGNAL_LEN, N_LEADS,
    ECGNetJoint,
    _load_raw_signal, extract_voltage_features, augment_signal,
)
from multilabel_classifier import (
    MULTILABEL_CODES,
    load_multilabel_dataset,
    preload_signals,
    load_demographics,
    compute_pos_weights,
    CONF_THRESHOLD,
)
from dataset_chapman import MERGED_CODES, load_chapman_multilabel
from dataset_challenge import V3_CODES, N_V3, load_challenge_multilabel

# -- Constants -----------------------------------------------------------------
MODEL_PATH    = "models/ecg_multilabel_v3.pt"
V2_MODEL_PATH = "models/ecg_multilabel_v2.pt"
N_CLASSES     = N_V3   # 26

# Mapping: 12-class PTB-XL index -> 26-class V3 index
PTBXL_TO_V3 = np.array([
    V3_CODES.index(c) for c in MULTILABEL_CODES
], dtype=int)

# Mapping: 14-class Chapman/merged index -> 26-class V3 index
MERGED_TO_V3 = np.array([
    V3_CODES.index(c) for c in MERGED_CODES
], dtype=int)


# -- Evaluate ------------------------------------------------------------------
def evaluate(model, loader, device, criterion=None):
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

    logits_np = np.vstack(all_logits)
    labels_np = np.vstack(all_labels)
    probs_np  = 1 / (1 + np.exp(-logits_np))
    preds_np  = (probs_np >= CONF_THRESHOLD).astype(int)

    aurocs = []
    for i in range(N_CLASSES):
        if 0 < labels_np[:, i].sum() < len(labels_np):
            aurocs.append(roc_auc_score(labels_np[:, i], probs_np[:, i]))
        else:
            aurocs.append(float("nan"))

    _, _, f1s, _ = precision_recall_fscore_support(
        labels_np, preds_np, average=None,
        labels=list(range(N_CLASSES)), zero_division=0
    )

    return {
        "loss":            total_loss / max(n_batches, 1),
        "macro_auroc":     float(np.nanmean(aurocs)),
        "macro_f1":        float(np.nanmean(f1s)),
        "per_class_auroc": aurocs,
        "per_class_f1":    f1s.tolist(),
    }


# -- Dataset -------------------------------------------------------------------
class V3ECGDataset(Dataset):
    def __init__(self, paths, label_matrix, raw_cache, aux_cache, augment=False):
        self.paths        = paths
        self.label_matrix = label_matrix
        self.raw_cache    = raw_cache
        self.aux_cache    = aux_cache
        self.augment      = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if path in self.raw_cache:
            sig = self.raw_cache[path].copy()
            aux = self.aux_cache[path]
        else:
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
def load_v3_data():
    """
    Returns paths, label_matrix (N, 26), fold_array (N,).
    Fold: 1-8=PTB-XL train, 9=val, 10=test, 0=Chapman+Challenge (train-only)
    """
    # PTB-XL (12 classes -> 26 classes)
    ptb_paths, ptb_labels12, ptb_folds = load_multilabel_dataset()
    ptb_folds = np.array(ptb_folds)
    N_ptb = len(ptb_paths)
    ptb_labels26 = np.zeros((N_ptb, N_CLASSES), dtype=np.float32)
    ptb_labels26[:, PTBXL_TO_V3] = ptb_labels12

    # Chapman (14 classes -> 26 classes)
    chap_paths, chap_labels14 = load_chapman_multilabel()
    N_chap = len(chap_paths)
    chap_labels26 = np.zeros((N_chap, N_CLASSES), dtype=np.float32)
    chap_labels26[:, MERGED_TO_V3] = chap_labels14
    chap_folds = np.zeros(N_chap, dtype=int)

    # Challenge (already 26-class aligned)
    # Reserve 10% as test (fold 20), 5% as val (fold 19), rest as train (fold 0)
    print("Loading Challenge datasets...")
    chal_paths, chal_labels26 = load_challenge_multilabel(codes=V3_CODES)
    N_chal = len(chal_paths)
    rng = np.random.default_rng(42)
    perm = rng.permutation(N_chal)
    n_test = int(N_chal * 0.10)
    n_val  = int(N_chal * 0.05)
    chal_folds = np.zeros(N_chal, dtype=int)
    chal_folds[perm[:n_test]]           = 20   # challenge test
    chal_folds[perm[n_test:n_test+n_val]] = 19  # challenge val

    # Merge all
    all_paths  = ptb_paths + chap_paths + chal_paths
    all_labels = np.concatenate([ptb_labels26, chap_labels26, chal_labels26], axis=0)
    all_folds  = np.concatenate([ptb_folds, chap_folds, chal_folds], axis=0)

    n_chal_train = int((chal_folds == 0).sum())
    n_chal_val   = int((chal_folds == 19).sum())
    n_chal_test  = int((chal_folds == 20).sum())
    print(f"  V3 dataset: {N_ptb} PTB-XL + {N_chap} Chapman + {N_chal} Challenge"
          f" = {len(all_paths)} total")
    print(f"  Challenge split: {n_chal_train} train / {n_chal_val} val / {n_chal_test} test")
    print(f"  Per-class positives:")
    for i, code in enumerate(V3_CODES):
        n = int(all_labels[:, i].sum())
        print(f"    {code:<8}: {n:>7}", end="")
        if (i + 1) % 4 == 0:
            print()
    print()
    return all_paths, all_labels, all_folds


# -- Training ------------------------------------------------------------------
def train(batch_size=64, n_epochs=60, patience=12, from_scratch=False):
    print("\n" + "=" * 60)
    print("  V3 Multi-Label ECG Classifier  (26 conditions)")
    print("  PTB-XL + Chapman + PhysioNet 2021 Challenge")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    all_paths, all_labels, all_folds = load_v3_data()

    train_mask  = (all_folds <= 8) | (all_folds == 0)
    val_mask    = all_folds == 9
    test_mask   = all_folds == 10
    ctest_mask  = all_folds == 20

    train_paths   = [p for p, m in zip(all_paths, train_mask) if m]
    val_paths     = [p for p, m in zip(all_paths, val_mask)   if m]
    test_paths    = [p for p, m in zip(all_paths, test_mask)  if m]
    ctest_paths   = [p for p, m in zip(all_paths, ctest_mask) if m]
    train_labels  = all_labels[train_mask]
    val_labels    = all_labels[val_mask]
    test_labels   = all_labels[test_mask]
    ctest_labels  = all_labels[ctest_mask]

    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}"
          f" | Challenge-test: {len(ctest_paths)}")

    # Preload PTB-XL only; Chapman + Challenge load lazily
    print("  Pre-loading PTB-XL signals into RAM...")
    demographics = load_demographics()
    ptbxl_paths  = [p for p in set(train_paths + val_paths + test_paths)
                    if "ptbxl" in p.lower()]
    raw_cache, aux_cache = preload_signals(ptbxl_paths, demographics)
    print(f"  Cached {len(raw_cache)} PTB-XL signals.")

    train_ds  = V3ECGDataset(train_paths,  train_labels,  raw_cache, aux_cache, augment=True)
    val_ds    = V3ECGDataset(val_paths,    val_labels,    raw_cache, aux_cache, augment=False)
    test_ds   = V3ECGDataset(test_paths,   test_labels,   raw_cache, aux_cache, augment=False)
    ctest_ds  = V3ECGDataset(ctest_paths,  ctest_labels,  raw_cache, aux_cache, augment=False)

    train_loader  = DataLoader(train_ds,  batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader    = DataLoader(val_ds,    batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader   = DataLoader(test_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    ctest_loader  = DataLoader(ctest_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    model = ECGNetJoint(n_leads=N_LEADS, n_classes=N_CLASSES, n_aux=N_AUX).to(device)

    # Transfer weights from v2 model (14-class -> 26-class)
    if not from_scratch and os.path.exists(V2_MODEL_PATH):
        v2_ckpt = torch.load(V2_MODEL_PATH, map_location=device, weights_only=False)
        v2_state = v2_ckpt["model_state"]
        v3_state = model.state_dict()
        transferred = 0
        for k, v in v2_state.items():
            if k in v3_state and v3_state[k].shape == v.shape:
                v3_state[k] = v
                transferred += 1
        model.load_state_dict(v3_state)
        print(f"  Transferred {transferred}/{len(v3_state)} layers from v2 model "
              f"(AUROC={v2_ckpt.get('best_auroc', 0):.3f})")
    else:
        print("  Training from random initialization")

    n_params = sum(p.numel() for p in model.parameters())
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
            # Save locally on every improvement
            torch.save({"model_state": best_state, "best_auroc": best_auroc,
                        "label_codes": V3_CODES, "n_classes": N_CLASSES}, MODEL_PATH)
            # Save to Drive if on Colab
            drive_ckpt = "/content/drive/MyDrive/EKG_models/ecg_multilabel_v3_best.pt"
            if os.path.exists("/content/drive/MyDrive/EKG_models"):
                torch.save({"model_state": best_state, "best_auroc": best_auroc,
                            "label_codes": V3_CODES, "n_classes": N_CLASSES}, drive_ckpt)
                print(f"  Checkpoint saved to Drive (AUROC={best_auroc:.3f})", flush=True)
        else:
            no_improve += 1
            marker = ""

        elapsed = time.time() - t0
        print(f"  {epoch:>3}  {m['loss']:>7.4f}  {m['macro_auroc']:>8.3f}"
              f"  {m['macro_f1']:>7.3f}  [{elapsed:.0f}s]{marker}", flush=True)

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # Final test evaluation
    if best_state:
        model.load_state_dict(best_state)
    torch.save({"model_state": best_state or model.state_dict(),
                "best_auroc": best_auroc, "label_codes": V3_CODES,
                "n_classes": N_CLASSES}, MODEL_PATH)
    print(f"\n  Saved: {MODEL_PATH}")
    _print_results(model, test_loader,  device, title="PTB-XL fold 10 (14 original classes)")
    _print_results(model, ctest_loader, device, title="Challenge test set (all 26 classes)")


def _print_results(model, loader, device, title="Test Results"):
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

    macro_f1    = f1_score(labels, preds, average="macro",  zero_division=0)
    micro_f1    = f1_score(labels, preds, average="micro",  zero_division=0)
    valid       = np.where(labels.sum(axis=0) > 0)[0]
    per_auroc   = np.full(N_CLASSES, float("nan"))
    if len(valid) > 0:
        per_auroc[valid] = roc_auc_score(
            labels[:, valid], probs[:, valid], average=None
        )
    macro_auroc = np.nanmean(per_auroc)
    per_f1      = f1_score(labels, preds, average=None, zero_division=0)

    new_codes = set(V3_CODES) - set(MERGED_CODES)

    print(f"\n{'=' * 60}")
    print(f"  V3 Model - {title}")
    print(f"{'=' * 60}")
    print(f"  MacroF1   : {macro_f1:.3f}")
    print(f"  MicroF1   : {micro_f1:.3f}")
    print(f"  MacroAUROC: {macro_auroc:.3f}")
    print(f"\n  Per-class ([NEW] = added in v3):")
    for i, code in enumerate(V3_CODES):
        tag   = " [NEW]" if code in new_codes else "      "
        n_pos = int(labels[:, i].sum())
        auroc_str = f"{per_auroc[i]:.3f}" if not np.isnan(per_auroc[i]) else "  n/a"
        print(f"    {code:>6}{tag}: F1={per_f1[i]:.3f}  AUROC={auroc_str}  (n={n_pos})")


def eval_saved():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model  = ECGNetJoint(n_leads=N_LEADS, n_classes=N_CLASSES, n_aux=N_AUX).to(device)
    model.load_state_dict(ckpt["model_state"])

    all_paths, all_labels, all_folds = load_v3_data()
    all_folds = np.array(all_folds)

    test_mask   = all_folds == 10
    ctest_mask  = all_folds == 20
    test_paths   = [p for p, m in zip(all_paths, test_mask)  if m]
    ctest_paths  = [p for p, m in zip(all_paths, ctest_mask) if m]
    test_labels  = all_labels[test_mask]
    ctest_labels = all_labels[ctest_mask]

    demographics = load_demographics()
    ptbxl_paths  = [p for p in test_paths if "ptbxl" in p.lower()]
    raw_cache, aux_cache = preload_signals(ptbxl_paths, demographics)

    test_ds    = V3ECGDataset(test_paths,  test_labels,  raw_cache, aux_cache)
    ctest_ds   = V3ECGDataset(ctest_paths, ctest_labels, raw_cache, aux_cache)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=0)
    ctest_loader = DataLoader(ctest_ds, batch_size=128, shuffle=False, num_workers=0)

    _print_results(model, test_loader,  device, title="PTB-XL fold 10 (14 original classes)")
    _print_results(model, ctest_loader, device, title="Challenge test set (all 26 classes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",    action="store_true", help="Evaluate saved model only")
    parser.add_argument("--scratch", action="store_true", help="Train from random init (no v2 transfer)")
    args = parser.parse_args()

    if args.eval:
        eval_saved()
    else:
        train(from_scratch=args.scratch)
