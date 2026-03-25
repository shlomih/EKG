"""
multilabel_classifier.py
========================
Multi-label ECG classifier: expands from 5 PTB-XL superclasses to 12 specific
clinically actionable conditions using exact SCP codes.

12 Target Labels (all ≥ 500 samples in PTB-XL, confidence ≥ 50%):
  NORM   - Normal ECG                         (9,438)
  AFIB   - Atrial fibrillation                (1,514) ← NEW
  LVH    - Left ventricular hypertrophy       (1,751)
  IMI    - Inferior myocardial infarction     (1,714)
  ASMI   - Anteroseptal MI                    (2,007)
  CLBBB  - Complete LBBB                      (  536)
  CRBBB  - Complete RBBB                      (  540)
  LAFB   - Left anterior fascicular block     (1,622)
  1AVB   - First-degree AV block              (  790)
  ISC_   - Non-specific ischemic ST changes   (1,260)
  NDT    - Non-diagnostic T abnormalities     (1,824)
  STACH  - Sinus tachycardia                  (  826) ← NEW

Architecture:
  Reuses ECGNetJoint backbone from cnn_classifier.py (same CNN, same aux features)
  Output head: Linear(288→12) with sigmoid — BCEWithLogitsLoss
  Training uses per-class positive-frequency weighting (handles class imbalance)

Usage:
    python multilabel_classifier.py              # train
    python multilabel_classifier.py --eval       # evaluate saved model

    # From app.py:
    from multilabel_classifier import load_multilabel_cnn, predict_multilabel
    model = load_multilabel_cnn()
    result = predict_multilabel(model, signal_12, fs=500)
"""

import ast
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

os.chdir(Path(__file__).parent)

from cnn_classifier import (
    N_AUX, SIGNAL_LEN, N_LEADS,
    ECGNetJoint,
    _load_raw_signal, extract_voltage_features, augment_signal,
)

# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------

MULTILABEL_CODES = [
    "NORM",   # 0
    "AFIB",   # 1  ← rhythm code
    "LVH",    # 2
    "IMI",    # 3
    "ASMI",   # 4
    "CLBBB",  # 5
    "CRBBB",  # 6
    "LAFB",   # 7
    "1AVB",   # 8
    "ISC_",   # 9
    "NDT",    # 10
    "STACH",  # 11 ← rhythm code
]

N_ML_CLASSES  = len(MULTILABEL_CODES)
CODE_TO_IDX   = {c: i for i, c in enumerate(MULTILABEL_CODES)}

CONDITION_DESCRIPTIONS = {
    "NORM":  "Normal ECG",
    "AFIB":  "Atrial Fibrillation",
    "LVH":   "Left Ventricular Hypertrophy",
    "IMI":   "Inferior Myocardial Infarction",
    "ASMI":  "Anteroseptal Myocardial Infarction",
    "CLBBB": "Complete Left Bundle Branch Block",
    "CRBBB": "Complete Right Bundle Branch Block",
    "LAFB":  "Left Anterior Fascicular Block",
    "1AVB":  "First-Degree AV Block",
    "ISC_":  "Non-Specific Ischemic ST Changes",
    "NDT":   "Non-Diagnostic T Abnormalities",
    "STACH": "Sinus Tachycardia",
}

# Clinical urgency tier (used for sorting output)
URGENCY = {
    "AFIB": 3, "IMI": 3, "ASMI": 3, "CLBBB": 3,
    "LVH": 2, "CRBBB": 2, "ISC_": 2, "STACH": 2,
    "LAFB": 1, "1AVB": 1, "NDT": 1,
    "NORM": 0,
}

MODEL_PATH = "models/ecg_multilabel.pt"
CONF_THRESHOLD = 0.40   # sigmoid threshold for positive prediction

# ---------------------------------------------------------------------------
# Label extraction from SCP codes
# ---------------------------------------------------------------------------

def extract_multilabel_vector(scp_codes_dict: dict, conf_threshold: float = 50.0) -> np.ndarray:
    """Convert SCP code dict {code: likelihood} → 12-hot float32 vector."""
    vec = np.zeros(N_ML_CLASSES, dtype=np.float32)
    for code, likelihood in scp_codes_dict.items():
        if likelihood >= conf_threshold and code in CODE_TO_IDX:
            vec[CODE_TO_IDX[code]] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_multilabel_dataset(base_path: str = "ekg_datasets/ptbxl"):
    """
    Load PTB-XL with 12-class multi-hot labels.
    Returns: paths (list), label_matrix (np.ndarray N×12), folds (list[int])
    Only keeps records that have at least one of the 12 target codes.
    """
    base = Path(base_path)
    meta = pd.read_csv(base / "ptbxl_database.csv", index_col="ecg_id")
    meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)

    print(f"  Indexing {len(meta)} records for multi-label...")

    paths, label_rows, folds = [], [], []
    skipped_no_label, skipped_no_file = 0, 0

    for ecg_id, row in meta.iterrows():
        vec = extract_multilabel_vector(row["scp_codes"])
        if vec.sum() == 0:
            skipped_no_label += 1
            continue

        rec_path = str(base / row["filename_hr"])
        if not os.path.exists(rec_path + ".dat"):
            skipped_no_file += 1
            continue

        paths.append(rec_path)
        label_rows.append(vec)
        folds.append(int(row["strat_fold"]))

    label_matrix = np.stack(label_rows, axis=0)   # (N, 12)
    print(f"  Kept {len(paths)} records  (skipped: {skipped_no_label} no-label, {skipped_no_file} no-file)")
    print(f"  Per-class positives:", end="")
    for i, code in enumerate(MULTILABEL_CODES):
        print(f"  {code}={int(label_matrix[:, i].sum())}", end="")
    print()
    return paths, label_matrix, folds


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class MultiLabelECGDataset(Dataset):
    """Returns (signal_12_5000, aux_14, label_vec_12)."""

    def __init__(self, paths, label_matrix, raw_cache, aux_cache, augment=False):
        self.paths        = paths
        self.label_matrix = label_matrix   # (N, 12) float32
        self.raw_cache    = raw_cache
        self.aux_cache    = aux_cache
        self.augment      = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        sig  = self.raw_cache[path].copy()    # (12, 5000) raw mV
        if self.augment:
            sig = augment_signal(sig)
        sig_norm = (sig / 5.0).astype(np.float32)   # same GLOBAL_NORM_SCALE as cnn_classifier
        aux  = self.aux_cache[path]
        lbl  = self.label_matrix[idx]
        return (
            torch.from_numpy(sig_norm),
            torch.from_numpy(aux),
            torch.from_numpy(lbl),
        )


# ---------------------------------------------------------------------------
# Signal pre-loading (same pattern as ecgfm_finetune.py)
# ---------------------------------------------------------------------------

def preload_signals(paths, demographics):
    """Load all raw signals and aux features into RAM dicts."""
    raw_cache, aux_cache = {}, {}
    t0 = time.time()
    for i, path in enumerate(paths):
        if path in raw_cache:
            continue
        sig = _load_raw_signal(path)
        if sig is None:
            continue
        raw_cache[path] = sig
        sex_raw, age_raw = demographics.get(path, (0, 50))
        aux_cache[path]  = extract_voltage_features(sig, sex=("F" if sex_raw else "M"), age=age_raw)
        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{len(paths)}  ({(i+1)/(time.time()-t0):.0f} rec/s)", flush=True)
    return raw_cache, aux_cache


def load_demographics(base_path="ekg_datasets/ptbxl"):
    """Returns dict: path → (sex_raw, age) — same as cnn_classifier logic."""
    base = Path(base_path)
    meta = pd.read_csv(base / "ptbxl_database.csv", index_col="ecg_id")
    demo = {}
    for ecg_id, row in meta.iterrows():
        path = str(base / row["filename_hr"])
        sex  = 0 if pd.isna(row.get("sex", 0)) else int(row.get("sex", 0))
        age  = 50 if pd.isna(row.get("age", 50)) else float(row.get("age", 50))
        demo[path] = (sex, age)
    return demo


# ---------------------------------------------------------------------------
# Model: reuse ECGNetJoint with n_classes=12
# ---------------------------------------------------------------------------

def build_model() -> ECGNetJoint:
    return ECGNetJoint(n_leads=N_LEADS, n_classes=N_ML_CLASSES, n_aux=N_AUX)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_pos_weights(label_matrix: np.ndarray) -> torch.Tensor:
    """Per-class BCEWithLogitsLoss pos_weight = neg_count / pos_count."""
    n = label_matrix.shape[0]
    pos = label_matrix.sum(axis=0).clip(min=1)
    neg = n - pos
    return torch.from_numpy((neg / pos).astype(np.float32))


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

    logits_np = np.vstack(all_logits)      # (N, 12)
    labels_np = np.vstack(all_labels)      # (N, 12)
    probs_np  = 1 / (1 + np.exp(-logits_np))   # sigmoid
    preds_np  = (probs_np >= CONF_THRESHOLD).astype(int)

    # Per-class AUROC (skip if only one class present)
    aurocs = []
    for i in range(N_ML_CLASSES):
        if labels_np[:, i].sum() > 0 and labels_np[:, i].sum() < len(labels_np):
            aurocs.append(roc_auc_score(labels_np[:, i], probs_np[:, i]))
        else:
            aurocs.append(float("nan"))

    # Per-class F1
    _, _, f1s, _ = precision_recall_fscore_support(
        labels_np, preds_np, average=None, labels=list(range(N_ML_CLASSES)), zero_division=0
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


def print_results(m, label="Val"):
    print(f"\n  {label}: MacroAUROC={m['macro_auroc']:.3f}  MacroF1={m['macro_f1']:.3f}")
    print(f"  {'Code':<8} {'AUROC':>6}  {'F1':>6}")
    print("  " + "-" * 24)
    for i, code in enumerate(MULTILABEL_CODES):
        auroc = m["per_class_auroc"][i]
        f1    = m["per_class_f1"][i]
        auroc_s = f"{auroc:.3f}" if not np.isnan(auroc) else "  —  "
        print(f"  {code:<8} {auroc_s:>6}  {f1:>6.3f}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(batch_size: int = 64, n_epochs: int = 50, patience: int = 12):
    print("\n" + "=" * 60)
    print("  Multi-Label ECG Classifier  (12 conditions)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    paths, label_matrix, folds = load_multilabel_dataset()
    folds_arr = np.array(folds)

    train_mask = folds_arr <= 8
    val_mask   = folds_arr == 9
    test_mask  = folds_arr == 10

    train_paths  = [p for p, m in zip(paths, train_mask) if m]
    val_paths    = [p for p, m in zip(paths, val_mask)   if m]
    test_paths   = [p for p, m in zip(paths, test_mask)  if m]
    train_labels = label_matrix[train_mask]
    val_labels   = label_matrix[val_mask]
    test_labels  = label_matrix[test_mask]

    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    print("  Pre-loading signals...")
    demographics = load_demographics()
    all_unique = list(set(train_paths + val_paths + test_paths))
    raw_cache, aux_cache = preload_signals(all_unique, demographics)

    train_ds = MultiLabelECGDataset(train_paths, train_labels, raw_cache, aux_cache, augment=True)
    val_ds   = MultiLabelECGDataset(val_paths,   val_labels,   raw_cache, aux_cache, augment=False)
    test_ds  = MultiLabelECGDataset(test_paths,  test_labels,  raw_cache, aux_cache, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params/1e6:.1f}M")

    pos_weight = compute_pos_weights(train_labels).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs, pct_start=0.1,
    )

    best_auroc  = 0.0
    best_state  = None
    no_improve  = 0

    hdr = f"  {'Ep':>3}  {'Loss':>7}  {'ValAUROC':>8}  {'ValF1':>7}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        for sig, aux, lbl in train_loader:
            sig, aux, lbl = sig.to(device), aux.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits = model(sig, aux)
            loss   = criterion(logits, lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        m = evaluate(model, val_loader, device, criterion)
        improved = m["macro_auroc"] > best_auroc
        marker = ""
        if improved:
            best_auroc = m["macro_auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = " <*>"
        else:
            no_improve += 1

        elapsed = time.time() - t0
        print(f"  {epoch:>3}  {m['loss']:>7.4f}  {m['macro_auroc']:>8.3f}  {m['macro_f1']:>7.3f}  [{elapsed:.0f}s]{marker}", flush=True)

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # Test evaluation with best checkpoint
    if best_state:
        model.load_state_dict(best_state)
    tm = evaluate(model, test_loader, device)
    print_results(tm, label="Test")

    torch.save({
        "model_state": best_state or model.state_dict(),
        "best_auroc": best_auroc,
        "test_metrics": tm,
        "codes": MULTILABEL_CODES,
        "threshold": CONF_THRESHOLD,
    }, MODEL_PATH)
    print(f"\n  Saved -> {MODEL_PATH}")
    return model


# ---------------------------------------------------------------------------
# Inference API (compatible with app.py)
# ---------------------------------------------------------------------------

def load_multilabel_cnn(model_path: str = MODEL_PATH):
    ckpt  = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_multilabel(model, signal_12: np.ndarray, fs: int = 500,
                       sex: str = "M", age: float = 50.0,
                       threshold: float = CONF_THRESHOLD) -> dict:
    """
    Args:
        signal_12: (12, N) raw mV signal
        fs: sampling frequency
    Returns dict with keys: primary, conditions, scores, confidence, per_class
    """
    from cnn_classifier import _load_raw_signal  # already imported above; just clarity

    # Resample to 5000 if needed
    sig = signal_12.copy()
    if sig.shape[1] != SIGNAL_LEN:
        import scipy.signal
        sig = scipy.signal.resample(sig, SIGNAL_LEN, axis=1)

    sig_norm = (sig / 5.0).astype(np.float32)
    aux      = extract_voltage_features(sig, sex=sex, age=age)

    sig_t  = torch.from_numpy(sig_norm).unsqueeze(0)   # (1, 12, 5000)
    aux_t  = torch.from_numpy(aux).unsqueeze(0)         # (1, 14)

    with torch.no_grad():
        logits = model(sig_t, aux_t)
        probs  = torch.sigmoid(logits).squeeze(0).numpy()   # (12,)

    scores      = {code: float(probs[i]) for i, code in enumerate(MULTILABEL_CODES)}
    conditions  = [code for i, code in enumerate(MULTILABEL_CODES) if probs[i] >= threshold]
    # Sort detected conditions by urgency then confidence
    conditions.sort(key=lambda c: (-URGENCY.get(c, 0), -scores[c]))

    primary     = conditions[0] if conditions else MULTILABEL_CODES[int(np.argmax(probs))]
    confidence  = float(probs[MULTILABEL_CODES.index(primary)])

    return {
        "primary":     primary,
        "description": CONDITION_DESCRIPTIONS.get(primary, primary),
        "confidence":  confidence,
        "conditions":  conditions,
        "scores":      scores,
        "per_class":   {code: {
                            "prob": float(probs[i]),
                            "detected": bool(probs[i] >= threshold),
                            "description": CONDITION_DESCRIPTIONS[code],
                            "urgency": URGENCY.get(code, 0),
                        }
                        for i, code in enumerate(MULTILABEL_CODES)},
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="Evaluate saved model on test fold")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs",     type=int, default=50)
    args = ap.parse_args()

    if args.eval:
        print("Loading saved model for evaluation...")
        model  = load_multilabel_cnn()
        paths, label_matrix, folds = load_multilabel_dataset()
        folds_arr  = np.array(folds)
        test_paths = [p for p, m in zip(paths, folds_arr == 10) if m]
        test_labels = label_matrix[folds_arr == 10]
        demo   = load_demographics()
        rc, ac = preload_signals(test_paths, demo)
        test_ds = MultiLabelECGDataset(test_paths, test_labels, rc, ac, augment=False)
        loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
        tm      = evaluate(model, loader, torch.device("cpu"))
        print_results(tm, label="Test")
    else:
        train(batch_size=args.batch_size, n_epochs=args.epochs)
