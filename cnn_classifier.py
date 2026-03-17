"""
cnn_classifier.py
=================
1D CNN classifier for 12-lead ECG superclass prediction.
Operates on raw signal (5000 x 12) -- no hand-crafted features needed.

Superclasses: NORM, MI, STTC, HYP, CD

v2 improvements:
  - Data augmentation (noise, scaling, time shift, lead dropout)
  - Oversampling of minority classes (HYP, STTC)
  - Squeeze-and-Excitation attention
  - 50 epochs with OneCycleLR
  - Label smoothing

Usage:
    # Train
    python cnn_classifier.py

    # From app.py:
    from cnn_classifier import load_cnn_classifier, predict_cnn
    model = load_cnn_classifier()
    result = predict_cnn(model, signal_12, fs=500)
"""

import ast
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------

SUPERCLASS_LABELS = ["NORM", "MI", "STTC", "HYP", "CD"]
SUPERCLASS_DESCRIPTIONS = {
    "NORM": "Normal ECG",
    "MI":   "Myocardial Infarction",
    "STTC": "ST/T Change",
    "HYP":  "Hypertrophy",
    "CD":   "Conduction Disturbance",
}
LABEL_TO_IDX = {label: i for i, label in enumerate(SUPERCLASS_LABELS)}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}

MODEL_PATH = "models/ecg_cnn.pt"
SIGNAL_LEN = 5000  # 10s at 500Hz
N_LEADS = 12
N_CLASSES = 5


# -------------------------------------------------------------
# Model Architecture
# -------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    """Channel attention: learns which channels (features) matter most."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)  # (B, C, 1)
        return x * w


class ECGResBlock(nn.Module):
    """Residual block for 1D convolution with optional SE attention."""
    def __init__(self, channels, kernel_size=7, dropout=0.1, use_se=False):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.se = SqueezeExcitation(channels) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return self.relu(x + out)


class ECGNet(nn.Module):
    """
    1D ResNet-style CNN with SE attention for 12-lead ECG classification.
    Input: (batch, 12, 5000)
    Output: (batch, 5) logits
    """
    def __init__(self, n_leads=12, n_classes=5):
        super().__init__()

        # Stem: 12 leads -> 64 channels
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),  # 5000 -> 1250
        )

        # Block 1: 64 channels
        self.layer1 = nn.Sequential(
            ECGResBlock(64, kernel_size=7, dropout=0.1, use_se=True),
            ECGResBlock(64, kernel_size=7, dropout=0.1),
            nn.MaxPool1d(4),  # 1250 -> 312
        )

        # Block 2: 128 channels
        self.expand2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            ECGResBlock(128, kernel_size=7, dropout=0.2, use_se=True),
            ECGResBlock(128, kernel_size=5, dropout=0.2),
            nn.MaxPool1d(4),  # 312 -> 78
        )

        # Block 3: 256 channels
        self.expand3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            ECGResBlock(256, kernel_size=5, dropout=0.3, use_se=True),
            ECGResBlock(256, kernel_size=3, dropout=0.3),
            nn.AdaptiveAvgPool1d(1),  # -> 1
        )

        # Classifier head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.expand2(x)
        x = self.layer2(x)
        x = self.expand3(x)
        x = self.layer3(x)
        return self.head(x)


# -------------------------------------------------------------
# Data Augmentation
# -------------------------------------------------------------

def augment_signal(sig):
    """
    Apply random augmentations to a (12, 5000) signal.
    Each augmentation is applied independently with some probability.
    """
    # Gaussian noise
    if np.random.random() < 0.5:
        noise_level = np.random.uniform(0.01, 0.15)
        sig = sig + np.random.randn(*sig.shape).astype(np.float32) * noise_level

    # Amplitude scaling per-lead
    if np.random.random() < 0.5:
        scale = np.random.uniform(0.8, 1.2, size=(sig.shape[0], 1)).astype(np.float32)
        sig = sig * scale

    # Time shift (circular)
    if np.random.random() < 0.3:
        shift = np.random.randint(-250, 250)
        sig = np.roll(sig, shift, axis=1)

    # Lead dropout (zero 1-2 random leads)
    if np.random.random() < 0.2:
        n_drop = np.random.randint(1, 3)
        drop_leads = np.random.choice(sig.shape[0], n_drop, replace=False)
        sig[drop_leads] = 0.0

    # Baseline wander (low-frequency sinusoidal noise)
    if np.random.random() < 0.3:
        t = np.linspace(0, 1, sig.shape[1], dtype=np.float32)
        freq = np.random.uniform(0.1, 0.5)
        amplitude = np.random.uniform(0.05, 0.2)
        wander = amplitude * np.sin(2 * np.pi * freq * t)
        sig = sig + wander[np.newaxis, :]

    return sig


# -------------------------------------------------------------
# Dataset
# -------------------------------------------------------------

class ECGDataset(Dataset):
    """Lazy-loading dataset -- reads WFDB records from disk per item.
    Handles variable-length signals by padding/truncating to SIGNAL_LEN.
    """
    def __init__(self, record_paths, labels, augment=False):
        self.record_paths = record_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.record_paths)

    def __getitem__(self, idx):
        try:
            rec = wfdb.rdrecord(self.record_paths[idx])
            sig = rec.p_signal  # (N, channels)
        except Exception:
            sig = None

        if sig is None:
            sig = np.zeros((SIGNAL_LEN, N_LEADS), dtype=np.float32)
        else:
            # Handle non-12-lead signals
            n_ch = sig.shape[1]
            if n_ch < N_LEADS:
                sig = np.hstack([sig, np.zeros((sig.shape[0], N_LEADS - n_ch))])
            elif n_ch > N_LEADS:
                sig = sig[:, :N_LEADS]

            # Resample if needed (detect by checking record fs)
            if hasattr(rec, 'fs') and rec.fs != 500 and rec.fs > 0:
                from scipy.signal import resample
                target_n = int(sig.shape[0] * 500 / rec.fs)
                sig = resample(sig, target_n, axis=0)

            # Pad or truncate to SIGNAL_LEN
            if sig.shape[0] < SIGNAL_LEN:
                pad = np.zeros((SIGNAL_LEN - sig.shape[0], N_LEADS))
                sig = np.vstack([sig, pad])
            else:
                sig = sig[:SIGNAL_LEN]

        sig = sig.T.astype(np.float32)  # (12, 5000)

        # Normalize per-lead
        mean = sig.mean(axis=1, keepdims=True)
        std = sig.std(axis=1, keepdims=True) + 1e-8
        sig = (sig - mean) / std

        # Apply augmentation (training only)
        if self.augment:
            sig = augment_signal(sig)

        label = self.labels[idx]
        return torch.from_numpy(sig), torch.tensor(label, dtype=torch.long)


# -------------------------------------------------------------
# Data loading (reuses PTB-XL superclass mapping)
# -------------------------------------------------------------

def build_scp_to_superclass(scp_path):
    df = pd.read_csv(scp_path, index_col=0)
    diag = df[df["diagnostic"] == 1.0]
    return diag["diagnostic_class"].to_dict()


def get_primary_superclass(scp_codes_str, scp_map):
    try:
        codes = ast.literal_eval(scp_codes_str)
    except Exception:
        return None
    best_class = None
    best_score = -1
    for code, likelihood in codes.items():
        superclass = scp_map.get(code)
        if superclass and likelihood > best_score:
            best_score = likelihood
            best_class = superclass
    return best_class


def load_dataset(base_path="ekg_datasets/ptbxl"):
    """Index all records and return paths + labels (no signal loading)."""
    base = Path(base_path)
    meta = pd.read_csv(base / "ptbxl_database.csv", index_col="ecg_id")
    scp_map = build_scp_to_superclass(str(base / "scp_statements.csv"))

    meta["superclass"] = meta["scp_codes"].apply(
        lambda x: get_primary_superclass(x, scp_map)
    )
    meta = meta.dropna(subset=["superclass"])
    meta = meta[meta["superclass"].isin(SUPERCLASS_LABELS)]

    paths, labels, folds = [], [], []

    print(f"  Indexing {len(meta)} records...")
    for ecg_id, row in meta.iterrows():
        rec_path = str(base / row["filename_hr"])
        if os.path.exists(rec_path + ".dat"):
            paths.append(rec_path)
            labels.append(LABEL_TO_IDX[row["superclass"]])
            folds.append(int(row["strat_fold"]))

    print(f"  Found {len(paths)} valid records on disk")
    return paths, labels, folds


def load_unified_dataset():
    """
    Load the unified multi-dataset index (if available).
    Falls back to PTB-XL only if unified index doesn't exist.

    Returns: paths, labels, folds, dataset_names
    PTB-XL records keep their original folds (1-10).
    External records get fold=0 (used for training only, never test).
    """
    index_path = Path("ekg_datasets/unified_index.csv")

    if not index_path.exists():
        print("  No unified index found. Building it now...")
        try:
            from dataset_pipeline import build_unified_index
            build_unified_index()
        except Exception as e:
            print(f"  Could not build unified index: {e}")
            print("  Falling back to PTB-XL only")
            p, l, f = load_dataset()
            return p, l, f, ["PTB-XL"] * len(p)

    if not index_path.exists():
        print("  Falling back to PTB-XL only")
        p, l, f = load_dataset()
        return p, l, f, ["PTB-XL"] * len(p)

    df = pd.read_csv(index_path)
    print(f"  Unified index: {len(df)} total records")

    paths, labels, folds, datasets = [], [], [], []
    skipped = 0

    for _, row in df.iterrows():
        rec_path = row["path"]
        superclass = row["superclass"]

        if superclass not in LABEL_TO_IDX:
            skipped += 1
            continue

        # Verify file exists (check .dat or .mat)
        if not (os.path.exists(rec_path + ".dat") or os.path.exists(rec_path + ".mat")):
            skipped += 1
            continue

        paths.append(rec_path)
        labels.append(LABEL_TO_IDX[superclass])
        datasets.append(row["dataset"])

        # PTB-XL has stratified folds; external datasets get fold=0
        if "strat_fold" in row and pd.notna(row.get("strat_fold")):
            folds.append(int(row["strat_fold"]))
        else:
            folds.append(0)  # External = training only

    print(f"  Loaded {len(paths)} valid records ({skipped} skipped)")
    for ds_name, count in pd.Series(datasets).value_counts().items():
        print(f"    {ds_name}: {count}")

    return paths, labels, folds, datasets


# -------------------------------------------------------------
# Training
# -------------------------------------------------------------

def train(use_multi=False):
    version = "v3 (multi-dataset)" if use_multi else "v2 (PTB-XL)"
    print("\n" + "=" * 60)
    print(f"  ECG 1D CNN {version} -- Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    if use_multi:
        paths, labels, folds, datasets = load_unified_dataset()
    else:
        paths, labels, folds = load_dataset()
        datasets = ["PTB-XL"] * len(paths)

    if not paths:
        print("  No records found.")
        return

    paths = np.array(paths)
    labels = np.array(labels)
    folds = np.array(folds)

    # Split: folds 1-8 + external (0) = train, fold 9 = val, fold 10 = test
    # External datasets (fold=0) go to training only
    # Val and test are always PTB-XL to keep evaluation consistent
    train_mask = (folds <= 8)  # includes fold 0 (external)
    val_mask = folds == 9
    test_mask = folds == 10

    # Partial oversampling: bring minority classes up to median count (not max)
    # This avoids the HYP false-positive explosion from full oversampling
    train_paths = paths[train_mask].tolist()
    train_labels = labels[train_mask].tolist()

    class_counts = np.bincount(train_labels, minlength=N_CLASSES)
    target_count = int(np.median(class_counts))  # ~2656 (STTC/CD level)

    oversampled_paths = list(train_paths)
    oversampled_labels = list(train_labels)

    for cls_idx in range(N_CLASSES):
        cls_count = int(class_counts[cls_idx])
        if cls_count < target_count:
            cls_indices = [i for i, l in enumerate(train_labels) if l == cls_idx]
            n_extra = target_count - cls_count
            extra_indices = np.random.choice(cls_indices, n_extra, replace=True)
            for ei in extra_indices:
                oversampled_paths.append(train_paths[ei])
                oversampled_labels.append(train_labels[ei])

    print(f"\n  Original training: {len(train_paths)} records")
    print(f"  After oversampling: {len(oversampled_paths)} records")
    os_counts = np.bincount(oversampled_labels, minlength=N_CLASSES)
    for i, label in enumerate(SUPERCLASS_LABELS):
        print(f"    {label}: {int(class_counts[i])} -> {int(os_counts[i])}")

    train_ds = ECGDataset(oversampled_paths, oversampled_labels, augment=True)
    val_ds = ECGDataset(paths[val_mask].tolist(), labels[val_mask].tolist(), augment=False)
    test_ds = ECGDataset(paths[test_mask].tolist(), labels[test_mask].tolist(), augment=False)

    print(f"  Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = ECGNet(n_leads=N_LEADS, n_classes=N_CLASSES).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {param_count:,}")

    # Class weights: inverse-frequency on the oversampled distribution
    os_counts_f = os_counts.astype(np.float32)
    class_weights = 1.0 / (os_counts_f + 1e-6)
    class_weights = class_weights / class_weights.sum() * N_CLASSES
    class_weights = torch.from_numpy(class_weights).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    n_epochs = 50
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
        pct_start=0.1,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # Training loop
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    patience = 10
    no_improve = 0

    print(f"\n  Training for up to {n_epochs} epochs (patience={patience})...\n")
    print(f"  {'Epoch':>5} {'Loss':>8} {'TrainAcc':>9} {'ValAcc':>7} {'ValF1':>7} {'LR':>10}")
    print(f"  {'-'*5} {'-'*8} {'-'*9} {'-'*7} {'-'*7} {'-'*10}")

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * batch_x.size(0)
            train_correct += (logits.argmax(1) == batch_y).sum().item()
            train_total += batch_x.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_preds_all = []
        val_labels_all = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                preds = logits.argmax(1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)
                val_preds_all.extend(preds.cpu().numpy())
                val_labels_all.extend(batch_y.cpu().numpy())

        val_acc = val_correct / val_total

        # Compute macro F1 for val
        from sklearn.metrics import f1_score as compute_f1
        val_f1 = compute_f1(val_labels_all, val_preds_all, average="macro", zero_division=0)

        lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improve = 0
            marker = " *"
            torch.save({
                "model_state_dict": model.state_dict(),
                "superclass_labels": SUPERCLASS_LABELS,
                "superclass_descriptions": SUPERCLASS_DESCRIPTIONS,
                "val_accuracy": best_val_acc,
                "val_f1_macro": best_val_f1,
                "epoch": best_epoch,
                "n_params": param_count,
            }, MODEL_PATH)
        else:
            no_improve += 1

        print(f"  {epoch+1:>5} {train_loss:>8.4f} {train_acc:>9.1%} {val_acc:>7.1%} {val_f1:>7.3f} {lr:>10.6f}{marker}")

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    print(f"\n  Best val F1: {best_val_f1:.3f} | Acc: {best_val_acc:.1%} (epoch {best_epoch})")

    # Test evaluation
    checkpoint = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = np.mean(all_preds == all_labels)

    print(f"  Test accuracy: {test_acc:.1%}")
    print(f"\n  Classification report:")

    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(
        all_labels, all_preds,
        target_names=SUPERCLASS_LABELS,
        zero_division=0,
    ))
    print("  Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print(f"\n  Model saved to {MODEL_PATH}")
    print(f"  Done.\n")


# -------------------------------------------------------------
# Inference (used by app.py)
# -------------------------------------------------------------

def load_cnn_classifier(path=MODEL_PATH):
    """Load the trained CNN model."""
    if not os.path.exists(path):
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    model = ECGNet(n_leads=N_LEADS, n_classes=N_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return {
        "model": model,
        "device": device,
        "superclass_labels": checkpoint["superclass_labels"],
        "superclass_descriptions": checkpoint["superclass_descriptions"],
        "val_accuracy": checkpoint["val_accuracy"],
    }


def predict_cnn(model_data, signal_12, fs=500):
    """
    Predict ECG superclass from a 12-lead signal.

    Args:
        model_data: dict from load_cnn_classifier()
        signal_12: (N, 12) numpy array
        fs: sampling rate

    Returns:
        dict: prediction, description, confidence, probabilities
    """
    model = model_data["model"]
    device = model_data["device"]

    # Pad or truncate to SIGNAL_LEN
    sig = signal_12.copy()
    if len(sig) < SIGNAL_LEN:
        pad = np.zeros((SIGNAL_LEN - len(sig), sig.shape[1]))
        sig = np.vstack([sig, pad])
    else:
        sig = sig[:SIGNAL_LEN]

    # Normalize per-lead
    sig = sig.T.astype(np.float32)  # (12, 5000)
    mean = sig.mean(axis=1, keepdims=True)
    std = sig.std(axis=1, keepdims=True) + 1e-8
    sig = (sig - mean) / std

    x = torch.from_numpy(sig).unsqueeze(0).to(device)  # (1, 12, 5000)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = SUPERCLASS_LABELS[pred_idx]

    prob_dict = {SUPERCLASS_LABELS[i]: round(float(p), 3) for i, p in enumerate(probs)}

    return {
        "prediction": pred_label,
        "description": SUPERCLASS_DESCRIPTIONS.get(pred_label, pred_label),
        "confidence": float(probs[pred_idx]),
        "probabilities": prob_dict,
    }


if __name__ == "__main__":
    import sys
    use_multi = "--multi" in sys.argv
    if use_multi:
        print("  Multi-dataset mode: training on all available datasets")
    train(use_multi=use_multi)
