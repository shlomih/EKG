"""
multilabel_v3.py
================
Train 26-class multi-label ECG classifier.
Builds on the 14-class v2 model by adding 12 new conditions from PhysioNet 2021 Challenge.

Supports: GPU (CUDA), TPU (torch_xla), and CPU — auto-detected at runtime.

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
    python multilabel_v3.py               # train from v2 checkpoint (GPU/TPU auto)
    python multilabel_v3.py --scratch     # train from random init
    python multilabel_v3.py --eval        # eval saved v3 model
    python multilabel_v3.py --batch_size 256  # override batch size (default: 256 TPU, 64 GPU)
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

# -- TPU / XLA support (optional) ---------------------------------------------
_HAS_XLA = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
    _HAS_XLA = True
except ImportError:
    pass


def _get_device():
    """Auto-detect best available device: TPU > GPU > CPU."""
    if _HAS_XLA:
        dev = torch_xla.device()
        print(f"  Device: TPU ({dev})")
        return dev, "tpu"
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  Device: GPU ({torch.cuda.get_device_name(0)})")
        return dev, "gpu"
    dev = torch.device("cpu")
    print("  Device: CPU (training will be slow)")
    return dev, "cpu"

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
MODEL_PATH      = "models/ecg_multilabel_v3.pt"
MODEL_BEST_PATH = "models/ecg_multilabel_v3_best.pt"
V2_MODEL_PATH   = "models/ecg_multilabel_v2.pt"
N_CLASSES       = N_V3   # 26

# AFIB class weight multiplier — boosts loss penalty for missed AFIB detections.
# AFIB has 11,590 positives but low AUROC (0.875) due to .mat bug during prior training.
# Multiplying pos_weight pushes the model to prioritise recall on AFIB.
AFIB_WEIGHT_BOOST = 4.0   # increase to 5-6 if AFIB recall remains poor after retrain

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
    mdtype = next(model.parameters()).dtype  # bfloat16 on TPU, float32 otherwise
    all_logits, all_labels = [], []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for sig, aux, lbl in loader:
            sig = sig.to(device=device, dtype=mdtype)
            aux = aux.to(device=device, dtype=mdtype)
            lbl = lbl.to(device=device, dtype=mdtype)
            logits = model(sig, aux)
            if criterion is not None:
                total_loss += criterion(logits, lbl).item()
                n_batches  += 1
            all_logits.append(logits.float().detach().cpu().numpy())
            all_labels.append(lbl.float().detach().cpu().numpy())

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
def train(batch_size=None, n_epochs=60, patience=12, from_scratch=False):
    print("\n" + "=" * 60)
    print("  V3 Multi-Label ECG Classifier  (26 conditions)")
    print("  PTB-XL + Chapman + PhysioNet 2021 Challenge")
    print("=" * 60)

    device, dev_type = _get_device()
    is_tpu = (dev_type == "tpu")

    # Default batch size: 256 for TPU (single v6e core), 64 for GPU/CPU
    if batch_size is None:
        batch_size = 256 if is_tpu else 64
    print(f"  Batch size: {batch_size}")

    # Scale learning rate with batch size (linear scaling rule, base=64)
    base_lr = 3e-4
    lr = base_lr * (batch_size / 64)
    print(f"  Learning rate: {lr:.1e} (scaled from {base_lr:.1e})")

    all_paths, all_labels, all_folds = load_v3_data()

    train_mask  = (all_folds <= 8) | (all_folds == 0)
    # Mixed val: PTB-XL fold 9 + Challenge fold 19
    # This ensures early stopping sees all 26 classes (new classes only exist in Challenge)
    val_mask    = (all_folds == 9) | (all_folds == 19)
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

    # DataLoader config — TPU needs drop_last=True to avoid XLA recompilation
    # on a smaller final batch, and num_workers>0 for CPU-side prefetch
    dl_workers = 2 if is_tpu else 0
    train_loader  = DataLoader(train_ds,  batch_size=batch_size, shuffle=True,
                               num_workers=dl_workers, drop_last=is_tpu)
    val_loader    = DataLoader(val_ds,    batch_size=batch_size, shuffle=False,
                               num_workers=dl_workers, drop_last=False)
    test_loader   = DataLoader(test_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=dl_workers, drop_last=False)
    ctest_loader  = DataLoader(ctest_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=dl_workers, drop_last=False)

    # Wrap loaders with MpDeviceLoader for async CPU→TPU data transfer
    if is_tpu:
        train_loader = MpDeviceLoader(train_loader, device)
        val_loader   = MpDeviceLoader(val_loader, device)
        test_loader  = MpDeviceLoader(test_loader, device)
        ctest_loader = MpDeviceLoader(ctest_loader, device)

    model = ECGNetJoint(n_leads=N_LEADS, n_classes=N_CLASSES, n_aux=N_AUX).to(device)

    # Use bfloat16 on TPU for native hardware acceleration
    if is_tpu:
        model = model.to(torch.bfloat16)
        print("  Using bfloat16 precision (TPU native)")

    # Weight initialisation priority:
    #   1. V3 best checkpoint (fine-tune from best known weights)
    #   2. V2 checkpoint      (transfer learning, 14→26 class)
    #   3. Random init        (--scratch flag)
    if not from_scratch and os.path.exists(MODEL_BEST_PATH):
        v3_ckpt = torch.load(MODEL_BEST_PATH, map_location="cpu", weights_only=False)
        state = {k: v.to(model.state_dict()[k].dtype)
                 for k, v in v3_ckpt["model_state"].items()
                 if k in model.state_dict()}
        model.load_state_dict(state, strict=True)
        prev_auroc = v3_ckpt.get("best_auroc", 0)
        print(f"  Loaded V3 best checkpoint (AUROC={prev_auroc:.4f}) — fine-tuning")
        # Lower LR for fine-tuning from a strong V3 checkpoint
        base_lr = 1e-4
        lr = base_lr * (batch_size / 64)
        print(f"  Fine-tune LR: {lr:.1e} (reduced from default for V3 continuation)")
    elif not from_scratch and os.path.exists(V2_MODEL_PATH):
        v2_ckpt = torch.load(V2_MODEL_PATH, map_location="cpu", weights_only=False)
        v2_state = v2_ckpt["model_state"]
        v3_state = model.state_dict()
        transferred = 0
        for k, v in v2_state.items():
            if k in v3_state and v3_state[k].shape == v.shape:
                v3_state[k] = v.to(v3_state[k].dtype)
                transferred += 1
        model.load_state_dict(v3_state)
        print(f"  Transferred {transferred}/{len(v3_state)} layers from v2 model "
              f"(AUROC={v2_ckpt.get('best_auroc', 0):.3f})")
    else:
        print("  Training from random initialization")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params/1e6:.1f}M  ({N_CLASSES} output classes)")

    pos_weight = compute_pos_weights(train_labels).to(device)

    # Boost AFIB weight — AUROC=0.875 is the weakest class, limited by .mat bug
    # during prior training runs. 4x multiplier pushes recall without destabilising
    # other classes. Adjust AFIB_WEIGHT_BOOST constant if needed.
    afib_idx = V3_CODES.index("AFIB")
    pos_weight[afib_idx] *= AFIB_WEIGHT_BOOST
    print(f"  AFIB pos_weight boosted {AFIB_WEIGHT_BOOST}x "
          f"(idx={afib_idx}, new weight={pos_weight[afib_idx].item():.2f})")

    if is_tpu:
        pos_weight = pos_weight.to(torch.bfloat16)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs, pct_start=0.05,   # shorter warmup for fine-tuning
    )

    best_auroc = 0.0
    best_state = None
    no_improve = 0

    if is_tpu:
        print("\n  [NOTE] First epoch is slow due to XLA graph compilation — this is normal!")

    print(f"\n  {'Ep':>3}  {'Loss':>7}  {'ValAUROC':>8}  {'ValF1':>7}")
    print("  " + "-" * 35)

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_samples  = 0
        for sig, aux, lbl in train_loader:
            if not is_tpu:
                sig, aux, lbl = sig.to(device), aux.to(device), lbl.to(device)
            if is_tpu:
                sig = sig.to(torch.bfloat16)
                aux = aux.to(torch.bfloat16)
                lbl = lbl.to(torch.bfloat16)
            optimizer.zero_grad()
            loss = criterion(model(sig, aux), lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if is_tpu:
                xm.optimizer_step(optimizer)
                xm.mark_step()
            else:
                optimizer.step()
            scheduler.step()
            total_loss += loss.item() * sig.size(0)
            n_samples  += sig.size(0)

        m = evaluate(model, val_loader, device, criterion)
        improved = m["macro_auroc"] > best_auroc
        if improved:
            best_auroc = m["macro_auroc"]
            best_state = {k: v.cpu().clone().float() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = " <*>"
            # Save locally on every improvement
            _save_ckpt = {"model_state": best_state, "best_auroc": best_auroc,
                          "label_codes": V3_CODES, "n_classes": N_CLASSES}
            if is_tpu:
                xm.save(_save_ckpt, MODEL_PATH)
            else:
                torch.save(_save_ckpt, MODEL_PATH)
            # Save to Drive if on Colab
            drive_ckpt = "/content/drive/MyDrive/EKG/models/ecg_multilabel_v3_best.pt"
            if os.path.exists("/content/drive/MyDrive/EKG/models"):
                if is_tpu:
                    xm.save(_save_ckpt, drive_ckpt)
                else:
                    torch.save(_save_ckpt, drive_ckpt)
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
        model = model.to(device)
    _final_ckpt = {"model_state": best_state or {k: v.cpu().clone().float()
                   for k, v in model.state_dict().items()},
                   "best_auroc": best_auroc, "label_codes": V3_CODES,
                   "n_classes": N_CLASSES}
    if is_tpu:
        xm.save(_final_ckpt, MODEL_PATH)
    else:
        torch.save(_final_ckpt, MODEL_PATH)
    print(f"\n  Saved: {MODEL_PATH}")
    _print_results(model, test_loader,  device, title="PTB-XL fold 10 (14 original classes)")
    _print_results(model, ctest_loader, device, title="Challenge test set (all 26 classes)")


def _print_results(model, loader, device, title="Test Results"):
    model.eval()
    mdtype = next(model.parameters()).dtype  # bfloat16 on TPU, float32 otherwise
    all_probs, all_labels = [], []
    with torch.no_grad():
        for sig, aux, lbl in loader:
            sig = sig.to(device=device, dtype=mdtype)
            aux = aux.to(device=device, dtype=mdtype)
            logits = model(sig, aux)
            all_probs.append(torch.sigmoid(logits.float()).detach().cpu().numpy())
            all_labels.append(lbl.detach().cpu().numpy())
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
    device, dev_type = _get_device()
    ckpt   = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
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


# ---------------------------------------------------------------------------
# Inference API  (used by app.py)
# ---------------------------------------------------------------------------
import json as _json
import scipy.signal as _scipy_signal

# Extend condition metadata with all 26 V3 classes
V3_CONDITION_DESCRIPTIONS = {
    "NORM":  "Normal ECG",
    "AFIB":  "Atrial Fibrillation",
    "PVC":   "Premature Ventricular Contraction",
    "LVH":   "Left Ventricular Hypertrophy",
    "IMI":   "Inferior Myocardial Infarction",
    "ASMI":  "Anteroseptal Myocardial Infarction",
    "CLBBB": "Complete Left Bundle Branch Block",
    "CRBBB": "Complete Right Bundle Branch Block",
    "LAFB":  "Left Anterior Fascicular Block",
    "1AVB":  "First-Degree AV Block",
    "ISC_":  "Non-Specific Ischemic ST Changes",
    "NDT":   "Non-Diagnostic T Abnormalities",
    "IRBBB": "Incomplete Right Bundle Branch Block",
    "STACH": "Sinus Tachycardia",
    "PAC":   "Premature Atrial Contraction",
    "Brady": "Bradycardia",
    "SVT":   "Supraventricular Tachycardia",
    "LQTP":  "Prolonged QT Interval",
    "TAb":   "T-Wave Abnormality",
    "LAD":   "Left Axis Deviation",
    "RAD":   "Right Axis Deviation",
    "NSIVC": "Non-Specific Intraventricular Conduction Delay",
    "AFL":   "Atrial Flutter",
    "STc":   "ST-T Change",
    "STD":   "ST Depression",
    "LAE":   "Left Atrial Enlargement",
}

V3_URGENCY = {
    # Urgency 3 — critical / immediate action
    "AFIB": 3, "AFL": 3, "IMI": 3, "ASMI": 3, "CLBBB": 3,
    # Urgency 2 — significant / prompt review
    "LVH": 2, "PVC": 2, "CRBBB": 2, "ISC_": 2, "SVT": 2,
    "LQTP": 2, "STD": 2, "STc": 2,
    # Urgency 1 — mild / monitor
    "LAFB": 1, "1AVB": 1, "NDT": 1, "IRBBB": 1, "STACH": 1,
    "PAC": 1, "Brady": 1, "TAb": 1, "LAD": 1, "RAD": 1,
    "NSIVC": 1, "LAE": 1,
    # Urgency 0 — normal
    "NORM": 0,
}

V3_CLINICAL_GUIDANCE = {
    "NORM":  {"action": "No acute findings. Routine follow-up as indicated.", "note": ""},
    "AFIB":  {"action": "Assess stroke risk (CHA2DS2-VASc). Consider anticoagulation. Rate/rhythm control.",
              "note": "Irregular rhythm — no distinct P waves."},
    "AFL":   {"action": "Rate control or rhythm control. Assess for anticoagulation (similar risk to AFIB).",
              "note": "Atrial flutter — typically 2:1 or 3:1 block with sawtooth flutter waves."},
    "PVC":   {"action": "Assess frequency and symptoms. If >10% burden or symptomatic, refer for Holter + echo.",
              "note": "Isolated PVCs are common and often benign."},
    "LVH":   {"action": "Evaluate for hypertension or hypertrophic cardiomyopathy. Echo recommended.",
              "note": "Voltage criteria met — Cornell/Sokolow-Lyon thresholds exceeded."},
    "IMI":   {"action": "If acute: activate cath lab. Check reciprocal changes in I/aVL. Evaluate RV involvement.",
              "note": "Inferior territory (RCA or LCx). ST elevation in II, III, aVF."},
    "ASMI":  {"action": "If acute: activate cath lab. Anteroseptal STEMI protocol.",
              "note": "Anterior territory (LAD). ST elevation in V1–V4."},
    "CLBBB": {"action": "New LBBB with chest pain: treat as STEMI equivalent — activate cath lab.",
              "note": "Complete LBBB — Sgarbossa criteria if ischaemia suspected."},
    "CRBBB": {"action": "Isolated CRBBB often benign. New CRBBB with symptoms — assess for PE or acute MI.",
              "note": "Complete RBBB — RSR' in V1, wide S in I/V6."},
    "LAFB":  {"action": "Usually benign in isolation. Monitor for progression to bifascicular block.",
              "note": "Left anterior fascicular block — LAD with small q in I/aVL."},
    "1AVB":  {"action": "Usually benign. Review medications (beta-blockers, digoxin). Annual follow-up.",
              "note": "PR interval > 200ms. No treatment usually required."},
    "ISC_":  {"action": "Compare with prior ECG. If new or symptomatic, evaluate for ACS.",
              "note": "Non-specific ischaemic ST changes — may indicate demand ischaemia."},
    "NDT":   {"action": "Non-diagnostic. Correlate with clinical history and symptoms.",
              "note": "Non-diagnostic T-wave abnormalities — many potential causes."},
    "IRBBB": {"action": "Usually benign. No immediate action required.",
              "note": "Incomplete RBBB — RSR' pattern in V1, QRS < 120ms."},
    "STACH": {"action": "Identify and treat underlying cause (pain, fever, hypovolaemia, anaemia).",
              "note": "Sinus tachycardia — rate > 100bpm with normal P waves."},
    "PAC":   {"action": "Usually benign. If frequent or symptomatic, evaluate for structural heart disease.",
              "note": "Premature atrial contraction — early narrow beat with abnormal P wave."},
    "Brady": {"action": "If symptomatic (syncope, hypotension), assess for pacemaker indication. Review medications.",
              "note": "Bradycardia — HR < 60bpm."},
    "SVT":   {"action": "Vagal manoeuvres or adenosine for acute termination. Refer for EP study if recurrent.",
              "note": "Supraventricular tachycardia — narrow complex tachycardia."},
    "LQTP":  {"action": "Review QT-prolonging medications. Electrolyte correction. Consider cardiology referral.",
              "note": "Prolonged QTc — risk of Torsades de Pointes. QTc ≥ 500ms is high risk."},
    "TAb":   {"action": "Correlate with symptoms. Compare with prior ECG. Evaluate electrolytes.",
              "note": "T-wave abnormality — inversion or flattening."},
    "LAD":   {"action": "Usually incidental. Rule out LAFB, inferior MI, or ventricular hypertrophy.",
              "note": "Left axis deviation — QRS axis between −30° and −90°."},
    "RAD":   {"action": "Evaluate for RVH, RBBB, lateral MI, or PE if new.",
              "note": "Right axis deviation — QRS axis > +90°."},
    "NSIVC": {"action": "Monitor. Evaluate for structural heart disease if new or symptomatic.",
              "note": "Non-specific intraventricular conduction delay — QRS 110–119ms."},
    "STc":   {"action": "Compare with prior ECG. If new or symptomatic, evaluate for ischaemia or pericarditis.",
              "note": "ST-T change — non-specific."},
    "STD":   {"action": "If new or ≥1mm in multiple leads with symptoms, evaluate urgently for ACS.",
              "note": "ST depression — may indicate subendocardial ischaemia or reciprocal change."},
    "LAE":   {"action": "Evaluate for mitral valve disease, hypertension, or LV dysfunction. Echo recommended.",
              "note": "Left atrial enlargement — broad/notched P wave in II, or biphasic in V1."},
}

_V3_THRESHOLDS_PATH = "models/thresholds_v3.json"
_v3_calibration_cache: dict | None = None


def _load_v3_calibration() -> dict:
    """Load thresholds + optional temperature from JSON, cached after first read."""
    global _v3_calibration_cache
    if _v3_calibration_cache is None:
        with open(_V3_THRESHOLDS_PATH) as f:
            data = _json.load(f)
        _v3_calibration_cache = {
            "thresholds": data["thresholds"],
            "temperature": data.get("temperature", None),
            "method": data.get("calibration_method", None),
        }
    return _v3_calibration_cache


def load_v3_cnn(model_path: str = MODEL_PATH):
    """Load V3 26-class model for inference. Returns model in eval mode."""
    ckpt  = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ECGNetJoint(n_leads=N_LEADS, n_classes=N_CLASSES, n_aux=N_AUX)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_v3(model, signal_12: np.ndarray, fs: int = 500,
               sex: str = "M", age: float = 50.0) -> dict:
    """
    Run V3 26-class inference on a single ECG.

    Args:
        model:      loaded ECGNetJoint (from load_v3_cnn)
        signal_12:  (12, N) numpy array in mV
        fs:         sampling frequency in Hz
        sex:        "M" or "F" (affects voltage feature extraction)
        age:        patient age in years

    Returns dict compatible with predict_multilabel output:
        primary, description, confidence, conditions, scores, per_class
    """
    if signal_12 is None or signal_12.ndim != 2 or signal_12.shape[0] != 12:
        raise ValueError(f"signal_12 must be (12, N), got {getattr(signal_12, 'shape', type(signal_12))}")

    sig = signal_12.copy()
    if sig.shape[1] != SIGNAL_LEN:
        sig = _scipy_signal.resample(sig, SIGNAL_LEN, axis=1)

    sig_norm = (sig / 5.0).astype(np.float32)
    aux      = extract_voltage_features(sig, sex=sex, age=age)

    with torch.no_grad():
        logits = model(
            torch.from_numpy(sig_norm).unsqueeze(0),
            torch.from_numpy(aux).unsqueeze(0),
        )
        logits_np = logits.squeeze(0).numpy()  # (26,)

    # Apply temperature scaling if calibration was fitted
    cal = _load_v3_calibration()
    thresholds = cal["thresholds"]
    T = cal["temperature"]
    if T is not None:
        if isinstance(T, list):
            # Per-class temperature
            logits_np = logits_np / np.array(T, dtype=np.float32)
        else:
            # Global temperature
            logits_np = logits_np / float(T)
    probs = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid
    conditions = [
        code for i, code in enumerate(V3_CODES)
        if probs[i] >= thresholds.get(code, 0.5)
    ]
    conditions.sort(key=lambda c: (-V3_URGENCY.get(c, 0), -float(probs[V3_CODES.index(c)])))

    scores  = {code: float(probs[i]) for i, code in enumerate(V3_CODES)}
    primary = conditions[0] if conditions else V3_CODES[int(np.argmax(probs))]
    primary_idx = V3_CODES.index(primary)

    return {
        "primary":     primary,
        "description": V3_CONDITION_DESCRIPTIONS.get(primary, primary),
        "confidence":  float(probs[primary_idx]),
        "conditions":  conditions,
        "scores":      scores,
        "per_class": {
            code: {
                "prob":        float(probs[i]),
                "detected":    bool(probs[i] >= thresholds.get(code, 0.5)),
                "description": V3_CONDITION_DESCRIPTIONS.get(code, code),
                "urgency":     V3_URGENCY.get(code, 0),
                "action":      V3_CLINICAL_GUIDANCE.get(code, {}).get("action", ""),
                "note":        V3_CLINICAL_GUIDANCE.get(code, {}).get("note", ""),
            }
            for i, code in enumerate(V3_CODES)
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",    action="store_true", help="Evaluate saved model only")
    parser.add_argument("--scratch", action="store_true", help="Train from random init (no v2 transfer)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (default: 256 TPU, 64 GPU)")
    args = parser.parse_args()

    if args.eval:
        eval_saved()
    else:
        train(batch_size=args.batch_size, from_scratch=args.scratch)
