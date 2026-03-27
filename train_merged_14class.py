"""
train_merged_14class.py
=======================
Merged 14-class multi-label ECG classifier: PTB-XL + Chapman-Shaoxing.

Why 14 classes vs the 12-class PTB-XL model:
  AFIB  (atrial fibrillation)  : only 48 confirmed in PTB-XL → useless
                                  Chapman adds 3,889 confirmed cases
  STACH (sinus tachycardia)    : only 4 in PTB-XL
                                  Chapman adds 2,760 cases

14 labels (MERGED_CODES from dataset_chapman.py):
  0  NORM   Normal ECG
  1  AFIB   Atrial fibrillation         ← KEY new label from Chapman
  2  PVC    Premature ventricular complex
  3  LVH    Left ventricular hypertrophy
  4  IMI    Inferior MI
  5  ASMI   Anteroseptal MI
  6  CLBBB  Complete LBBB
  7  CRBBB  Complete RBBB
  8  LAFB   Left anterior fascicular block
  9  1AVB   First-degree AV block
  10 ISC_   Non-specific ischemic ST changes
  11 NDT    Non-diagnostic T abnormalities
  12 IRBBB  Incomplete RBBB
  13 STACH  Sinus tachycardia            ← KEY new label from Chapman

Training splits:
  PTB-XL  : strat_fold 1–8 = train, 9 = val, 10 = test  (21,837 records)
  Chapman : random 80 / 10 / 10 split                    (up to 45,152 records)

Output: models/ecg_multilabel_14class.pt

Usage:
    python train_merged_14class.py               # full training
    python train_merged_14class.py --eval        # eval saved model on test split
    python train_merged_14class.py --ptbxl_only  # train without Chapman (debug)
"""

import argparse
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
from dataset_chapman import (
    MERGED_CODES, MERGED_CODE_TO_IDX, N_MERGED,
    CHAPMAN_BASE, CHAPMAN_INDEX,
    load_chapman_multilabel, load_chapman_signal, parse_snomed_codes,
    snomed_to_multilabel, build_chapman_index,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH    = "models/ecg_multilabel_14class.pt"
CONF_THRESHOLD = 0.40

PTBXL_BASE   = "ekg_datasets/ptbxl"
PTBXL_META   = "ptbxl_database.csv"
PTBXL_SCP    = "scp_statements.csv"

# PTB-XL SCP code names that map directly to our merged label set
# (PTB-XL uses its own diagnostic abbreviations, same as MERGED_CODES)
PTBXL_SCP_TO_MERGED = {code: code for code in MERGED_CODES}
# PTB-XL has AFIB and STACH as SCP codes — they map directly
# Aliases for any PTB-XL codes that use different abbreviations:
PTBXL_SCP_TO_MERGED.update({
    "AF":   "AFIB",   # some PTB-XL records use "AF" not "AFIB"
    "STach": "STACH", # PTB-XL capitalizes differently in some versions
})


# ---------------------------------------------------------------------------
# PTB-XL: 14-class label extraction
# ---------------------------------------------------------------------------

def extract_ptbxl_merged_vector(scp_codes_dict: dict,
                                conf_threshold: float = 50.0) -> np.ndarray:
    """
    Convert PTB-XL scp_codes dict {code: likelihood} → 14-hot float32 vector.
    Codes that exceed conf_threshold and map to MERGED_CODES are set to 1.
    """
    vec = np.zeros(N_MERGED, dtype=np.float32)
    for code, likelihood in scp_codes_dict.items():
        if likelihood < conf_threshold:
            continue
        merged = PTBXL_SCP_TO_MERGED.get(code)
        if merged and merged in MERGED_CODE_TO_IDX:
            vec[MERGED_CODE_TO_IDX[merged]] = 1.0
    return vec


def load_ptbxl_merged(base_path: str = PTBXL_BASE):
    """
    Load PTB-XL with 14-class multi-hot labels.
    Returns: paths, label_matrix (N×14 float32), folds (list[int])
    """
    base = Path(base_path)
    meta = pd.read_csv(base / PTBXL_META, index_col="ecg_id")
    meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)

    print(f"  Indexing PTB-XL ({len(meta)} records) for 14-class labels...")
    paths, label_rows, folds, demos = [], [], [], {}

    skipped_no_label = skipped_no_file = 0
    for ecg_id, row in meta.iterrows():
        vec = extract_ptbxl_merged_vector(row["scp_codes"])
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

        sex = 0 if pd.isna(row.get("sex", 0)) else int(row.get("sex", 0))
        age = 50.0 if pd.isna(row.get("age", 50)) else float(row.get("age", 50))
        demos[rec_path] = (sex, age)

    label_matrix = np.stack(label_rows, axis=0)
    print(f"  PTB-XL: {len(paths)} records  "
          f"(skipped {skipped_no_label} no-label, {skipped_no_file} no-file)")
    _print_label_counts("PTB-XL", label_matrix)
    return paths, label_matrix, folds, demos


# ---------------------------------------------------------------------------
# Chapman: load with demographic info
# ---------------------------------------------------------------------------

def load_chapman_demographics(base_path: str = CHAPMAN_BASE) -> dict:
    """
    Parse age and sex from Chapman .hea comment lines.
    Returns dict: rec_path → (sex_int, age_float)
    Sex: 0=Male/unknown, 1=Female
    """
    demo = {}
    for hea in sorted(Path(base_path).rglob("*.hea")):
        rec = str(hea.with_suffix(""))
        age, sex = 50.0, 0
        try:
            with open(str(hea), encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("#Age:"):
                        try:
                            age = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
                    elif line.startswith("#Sex:"):
                        s = line.split(":", 1)[1].strip().upper()
                        sex = 1 if s in ("F", "FEMALE") else 0
        except Exception:
            pass
        demo[rec] = (sex, age)
    return demo


def load_chapman_merged(index_path: str = CHAPMAN_INDEX,
                        base_path: str = CHAPMAN_BASE):
    """
    Load Chapman-Shaoxing for the merged model.
    Returns: paths, label_matrix (N×14), demos dict
    """
    if not Path(index_path).exists():
        print(f"  Chapman index not found — building from {base_path}...")
        build_chapman_index(base_path, index_path)

    paths, label_matrix = load_chapman_multilabel(index_path)
    print(f"  Loading Chapman demographics ({base_path})...")
    demos = load_chapman_demographics(base_path)
    return paths, label_matrix, demos


# ---------------------------------------------------------------------------
# Unified dataset
# ---------------------------------------------------------------------------

def _load_signal_unified(path: str):
    """
    Load raw (12, 5000) float32 signal from either PTB-XL (.dat) or Chapman (.mat).
    Returns None on failure.
    """
    # Chapman records have .mat alongside the .hea
    if Path(path + ".mat").exists():
        return load_chapman_signal(path)
    return _load_raw_signal(path)


def preload_signals_merged(paths, demos):
    """
    Pre-load all signals and aux features into RAM.
    Returns: raw_cache {path: (12,5000)}, aux_cache {path: (14,)}
    """
    raw_cache, aux_cache = {}, {}
    t0 = time.time()
    missing = 0
    for i, path in enumerate(paths):
        if path in raw_cache:
            continue
        sig = _load_signal_unified(path)
        if sig is None:
            missing += 1
            continue
        raw_cache[path] = sig
        sex_raw, age_raw = demos.get(path, (0, 50.0))
        aux_cache[path] = extract_voltage_features(
            sig, sex=("F" if sex_raw else "M"), age=float(age_raw)
        )
        if (i + 1) % 3000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(paths)}  "
                  f"({(i+1)/elapsed:.0f} rec/s)  missing={missing}", flush=True)

    print(f"  Loaded {len(raw_cache)} / {len(paths)} signals  "
          f"({missing} unreadable)")
    return raw_cache, aux_cache


class MergedECGDataset(Dataset):
    """Unified dataset for PTB-XL + Chapman. Returns (sig, aux, label_14)."""

    def __init__(self, paths, label_matrix, raw_cache, aux_cache, augment=False):
        # Filter out paths that failed to load
        valid = [i for i, p in enumerate(paths) if p in raw_cache]
        self.paths        = [paths[i] for i in valid]
        self.label_matrix = label_matrix[valid]
        self.raw_cache    = raw_cache
        self.aux_cache    = aux_cache
        self.augment      = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        sig  = self.raw_cache[path].copy()       # (12, 5000)
        if self.augment:
            sig = augment_signal(sig)
        sig_norm = (sig / 5.0).astype(np.float32)
        aux  = self.aux_cache[path]
        lbl  = self.label_matrix[idx]
        return (
            torch.from_numpy(sig_norm),
            torch.from_numpy(aux),
            torch.from_numpy(lbl),
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_pos_weights(label_matrix: np.ndarray) -> torch.Tensor:
    n   = label_matrix.shape[0]
    pos = label_matrix.sum(axis=0).clip(min=1)
    neg = n - pos
    return torch.from_numpy((neg / pos).astype(np.float32))


def evaluate_merged(model, loader, device, criterion=None):
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n_batches = 0.0, 0

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
    for i in range(N_MERGED):
        n_pos = labels_np[:, i].sum()
        if 0 < n_pos < len(labels_np):
            aurocs.append(roc_auc_score(labels_np[:, i], probs_np[:, i]))
        else:
            aurocs.append(float("nan"))

    _, _, f1s, _ = precision_recall_fscore_support(
        labels_np, preds_np, average=None,
        labels=list(range(N_MERGED)), zero_division=0
    )

    return {
        "loss":           total_loss / max(n_batches, 1),
        "macro_auroc":    float(np.nanmean(aurocs)),
        "macro_f1":       float(np.nanmean(f1s)),
        "per_class_auroc": aurocs,
        "per_class_f1":   f1s.tolist(),
    }


def print_results_merged(m, label="Test"):
    print(f"\n  {label}: MacroAUROC={m['macro_auroc']:.3f}  MacroF1={m['macro_f1']:.3f}")
    print(f"  {'Code':<8} {'AUROC':>6}  {'F1':>6}")
    print("  " + "-" * 24)
    for i, code in enumerate(MERGED_CODES):
        auroc = m["per_class_auroc"][i]
        f1    = m["per_class_f1"][i]
        auroc_s = f"{auroc:.3f}" if not np.isnan(auroc) else "  —  "
        print(f"  {code:<8} {auroc_s:>6}  {f1:>6.3f}")


def _print_label_counts(name, label_matrix):
    counts = label_matrix.sum(axis=0).astype(int)
    parts  = [f"{c}={counts[i]}" for i, c in enumerate(MERGED_CODES)]
    print(f"  {name} label counts: {' '.join(parts)}")


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(batch_size: int = 64, n_epochs: int = 50, patience: int = 12,
          use_chapman: bool = True):

    print("\n" + "=" * 65)
    print("  Merged 14-Class ECG Classifier  (PTB-XL + Chapman-Shaoxing)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── Load PTB-XL ─────────────────────────────────────────────────────────
    ptb_paths, ptb_labels, ptb_folds, ptb_demos = load_ptbxl_merged()
    folds_arr = np.array(ptb_folds)

    ptb_train_mask = folds_arr <= 8
    ptb_val_mask   = folds_arr == 9
    ptb_test_mask  = folds_arr == 10

    # ── Load Chapman ─────────────────────────────────────────────────────────
    chap_train_paths = chap_val_paths = chap_test_paths = []
    chap_train_labels = chap_val_labels = chap_test_labels = np.zeros((0, N_MERGED), dtype=np.float32)
    chap_demos = {}

    if use_chapman and Path(CHAPMAN_BASE).exists():
        chap_paths, chap_labels, chap_demos = load_chapman_merged()
        n_chap = len(chap_paths)
        rng = np.random.default_rng(42)
        idx = rng.permutation(n_chap)
        n_val  = int(n_chap * 0.10)
        n_test = int(n_chap * 0.10)
        test_idx  = idx[:n_test]
        val_idx   = idx[n_test : n_test + n_val]
        train_idx = idx[n_test + n_val:]

        chap_train_paths  = [chap_paths[i] for i in train_idx]
        chap_val_paths    = [chap_paths[i] for i in val_idx]
        chap_test_paths   = [chap_paths[i] for i in test_idx]
        chap_train_labels = chap_labels[train_idx]
        chap_val_labels   = chap_labels[val_idx]
        chap_test_labels  = chap_labels[test_idx]
        print(f"  Chapman split: {len(chap_train_paths)} train / "
              f"{len(chap_val_paths)} val / {len(chap_test_paths)} test")
    elif use_chapman:
        print(f"  WARNING: Chapman dataset not found at {CHAPMAN_BASE}")
        print(f"  Run: python dataset_chapman.py --download")
        print(f"  Proceeding with PTB-XL only (AFIB/STACH will have few positives)")

    # ── Merge splits ─────────────────────────────────────────────────────────
    train_paths  = [p for p, m in zip(ptb_paths, ptb_train_mask) if m] + chap_train_paths
    val_paths    = [p for p, m in zip(ptb_paths, ptb_val_mask)   if m] + chap_val_paths
    test_paths   = [p for p, m in zip(ptb_paths, ptb_test_mask)  if m] + chap_test_paths
    train_labels = np.vstack([ptb_labels[ptb_train_mask], chap_train_labels]) \
                   if len(chap_train_labels) else ptb_labels[ptb_train_mask]
    val_labels   = np.vstack([ptb_labels[ptb_val_mask],   chap_val_labels]) \
                   if len(chap_val_labels)   else ptb_labels[ptb_val_mask]
    test_labels  = np.vstack([ptb_labels[ptb_test_mask],  chap_test_labels]) \
                   if len(chap_test_labels)  else ptb_labels[ptb_test_mask]

    all_demos = {**ptb_demos, **chap_demos}

    print(f"  Total: {len(train_paths)} train | "
          f"{len(val_paths)} val | {len(test_paths)} test")
    _print_label_counts("Train combined", train_labels)

    # ── Pre-load signals ─────────────────────────────────────────────────────
    print("  Pre-loading signals into RAM (train+val — test deferred)...")
    all_preload = list(set(train_paths + val_paths))
    raw_cache, aux_cache = preload_signals_merged(all_preload, all_demos)

    train_ds = MergedECGDataset(train_paths, train_labels, raw_cache, aux_cache, augment=True)
    val_ds   = MergedECGDataset(val_paths,   val_labels,   raw_cache, aux_cache, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────────
    model = ECGNetJoint(n_leads=N_LEADS, n_classes=N_MERGED, n_aux=N_AUX).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: ECGNetJoint  ({n_params/1e6:.1f}M params, {N_MERGED} outputs)")

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

    hdr = f"  {'Ep':>3}  {'Loss':>7}  {'ValAUROC':>8}  {'ValF1':>7}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        for sig, aux, lbl in train_loader:
            sig, aux, lbl = sig.to(device), aux.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(sig, aux), lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        m = evaluate_merged(model, val_loader, device, criterion)
        improved = m["macro_auroc"] > best_auroc
        if improved:
            best_auroc = m["macro_auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = " <*>"
        else:
            no_improve += 1
            marker = ""

        elapsed = time.time() - t0
        print(f"  {epoch:>3}  {m['loss']:>7.4f}  "
              f"{m['macro_auroc']:>8.3f}  {m['macro_f1']:>7.3f}  "
              f"[{elapsed:.0f}s]{marker}", flush=True)

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # ── Test evaluation ───────────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)

    print("\n  Loading test signals...")
    test_raw, test_aux = preload_signals_merged(list(set(test_paths)), all_demos)
    test_ds     = MergedECGDataset(test_paths, test_labels, test_raw, test_aux, augment=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    tm = evaluate_merged(model, test_loader, device)
    print_results_merged(tm, label="Test")

    Path("models").mkdir(exist_ok=True)
    torch.save({
        "model_state":  best_state or model.state_dict(),
        "best_auroc":   best_auroc,
        "test_metrics": tm,
        "codes":        MERGED_CODES,
        "threshold":    CONF_THRESHOLD,
        "n_classes":    N_MERGED,
    }, MODEL_PATH)
    print(f"\n  Saved → {MODEL_PATH}")
    return model


# ---------------------------------------------------------------------------
# Inference API  (drop-in replacement for predict_multilabel from multilabel_classifier.py)
# ---------------------------------------------------------------------------

from multilabel_classifier import CONDITION_DESCRIPTIONS, URGENCY, CLINICAL_GUIDANCE


def load_merged_14class(model_path: str = MODEL_PATH):
    """Load the 14-class merged model. Returns (model, codes, threshold)."""
    ckpt  = torch.load(model_path, map_location="cpu", weights_only=False)
    codes = ckpt.get("codes", MERGED_CODES)
    n_cls = ckpt.get("n_classes", len(codes))
    model = ECGNetJoint(n_leads=N_LEADS, n_classes=n_cls, n_aux=N_AUX)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    threshold = ckpt.get("threshold", CONF_THRESHOLD)
    return model, codes, threshold


def predict_merged(model, signal_12: np.ndarray, codes=None,
                   threshold: float = CONF_THRESHOLD,
                   fs: int = 500, sex: str = "M", age: float = 50.0) -> dict:
    """
    Run inference with the 14-class merged model.
    Same output format as predict_multilabel() in multilabel_classifier.py.

    Args:
        model     : loaded ECGNetJoint (14-class)
        signal_12 : (12, N) raw mV numpy array
        codes     : list of label codes (default: MERGED_CODES)
        threshold : sigmoid threshold for positive detection
    """
    if codes is None:
        codes = MERGED_CODES

    if signal_12 is None or not isinstance(signal_12, np.ndarray) or signal_12.ndim != 2:
        raise ValueError(
            f"signal_12 must be a 2D numpy array (12, N), got {type(signal_12)}"
        )

    sig = signal_12.copy()
    if sig.shape[1] != SIGNAL_LEN:
        import scipy.signal
        sig = scipy.signal.resample(sig, SIGNAL_LEN, axis=1)

    sig_norm = (sig / 5.0).astype(np.float32)
    aux      = extract_voltage_features(sig, sex=sex, age=age)

    sig_t = torch.from_numpy(sig_norm).unsqueeze(0)
    aux_t = torch.from_numpy(aux).unsqueeze(0)

    with torch.no_grad():
        probs = torch.sigmoid(model(sig_t, aux_t)).squeeze(0).numpy()

    scores     = {c: float(probs[i]) for i, c in enumerate(codes)}
    conditions = [c for i, c in enumerate(codes) if probs[i] >= threshold]
    conditions.sort(key=lambda c: (-URGENCY.get(c, 0), -scores[c]))

    primary     = conditions[0] if conditions else codes[int(np.argmax(probs))]
    primary_idx = {c: i for i, c in enumerate(codes)}.get(primary, int(np.argmax(probs)))
    confidence  = float(probs[primary_idx])

    return {
        "primary":     primary,
        "description": CONDITION_DESCRIPTIONS.get(primary, primary),
        "confidence":  confidence,
        "conditions":  conditions,
        "scores":      scores,
        "per_class": {
            c: {
                "prob":        float(probs[i]),
                "detected":    bool(probs[i] >= threshold),
                "description": CONDITION_DESCRIPTIONS.get(c, c),
                "urgency":     URGENCY.get(c, 0),
                "action":      CLINICAL_GUIDANCE.get(c, {}).get("action", ""),
                "note":        CLINICAL_GUIDANCE.get(c, {}).get("note", ""),
            }
            for i, c in enumerate(codes)
        },
    }


# ---------------------------------------------------------------------------
# Eval-only mode
# ---------------------------------------------------------------------------

def eval_saved(model_path: str = MODEL_PATH):
    model, codes, threshold = load_merged_14class(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ptb_paths, ptb_labels, ptb_folds, ptb_demos = load_ptbxl_merged()
    folds_arr  = np.array(ptb_folds)
    test_paths = [p for p, m in zip(ptb_paths, folds_arr == 10) if m]
    test_labels = ptb_labels[folds_arr == 10]

    if Path(CHAPMAN_INDEX).exists():
        chap_paths, chap_labels, chap_demos = load_chapman_merged()
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(chap_paths))
        n_test = int(len(chap_paths) * 0.10)
        test_idx = idx[:n_test]
        test_paths  += [chap_paths[i] for i in test_idx]
        test_labels  = np.vstack([test_labels, chap_labels[test_idx]])
        all_demos    = {**ptb_demos, **chap_demos}
    else:
        all_demos = ptb_demos

    print("  Loading test signals...")
    raw, aux = preload_signals_merged(list(set(test_paths)), all_demos)
    ds = MergedECGDataset(test_paths, test_labels, raw, aux, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    m = evaluate_merged(model, loader, device)
    print_results_merged(m, label="Test (saved model)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train 14-class merged ECG model")
    ap.add_argument("--eval",        action="store_true",
                    help="Evaluate saved model on test split")
    ap.add_argument("--ptbxl_only",  action="store_true",
                    help="Train without Chapman-Shaoxing (debug mode)")
    ap.add_argument("--batch_size",  type=int, default=64)
    ap.add_argument("--epochs",      type=int, default=50)
    ap.add_argument("--patience",    type=int, default=12)
    ap.add_argument("--model_path",  default=MODEL_PATH)
    args = ap.parse_args()

    MODEL_PATH = args.model_path

    if args.eval:
        eval_saved(MODEL_PATH)
    else:
        train(
            batch_size  = args.batch_size,
            n_epochs    = args.epochs,
            patience    = args.patience,
            use_chapman = not args.ptbxl_only,
        )
