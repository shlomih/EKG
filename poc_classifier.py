"""
poc_classifier.py
=================
ECG superclass classifier for the EKG Intelligence Platform POC.

Groups PTB-XL diagnoses into 5 superclasses:
    NORM — Normal ECG
    MI   — Myocardial Infarction
    STTC — ST/T Changes
    HYP  — Hypertrophy
    CD   — Conduction Disturbance

Uses feature extraction + Random Forest (suitable for small datasets).
Trained on PTB-XL records available locally.

Usage:
    # Train and save model
    python poc_classifier.py

    # From app.py:
    from poc_classifier import load_classifier, predict
    clf = load_classifier()
    result = predict(clf, signal_12lead, fs=500)
"""

import ast
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy import stats as sp_stats
from scipy.signal import welch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Superclass mapping
# ─────────────────────────────────────────────────────────────

SUPERCLASS_LABELS = ["NORM", "MI", "STTC", "HYP", "CD"]
SUPERCLASS_DESCRIPTIONS = {
    "NORM": "Normal ECG",
    "MI":   "Myocardial Infarction",
    "STTC": "ST/T Change",
    "HYP":  "Hypertrophy",
    "CD":   "Conduction Disturbance",
}

MODEL_PATH = "models/ecg_classifier.pkl"


def build_scp_to_superclass(scp_path):
    """Build a mapping from SCP code to superclass using scp_statements.csv."""
    df = pd.read_csv(scp_path, index_col=0)
    diag = df[df["diagnostic"] == 1.0]
    return diag["diagnostic_class"].to_dict()


def get_primary_superclass(scp_codes_str, scp_map):
    """
    Given the scp_codes column (string repr of a dict),
    return the superclass with highest likelihood.
    """
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


# ─────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────

def extract_features(signal_12, fs=500):
    """
    Extract a feature vector from a 12-lead ECG signal.

    signal_12: (N, 12) numpy array
    fs: sampling rate

    Returns: 1D numpy array of features
    """
    features = []
    n_samples = signal_12.shape[0]

    for lead_idx in range(signal_12.shape[1]):
        sig = signal_12[:, lead_idx]
        sig = np.nan_to_num(sig)

        # Statistical features
        features.append(np.mean(sig))
        features.append(np.std(sig))
        features.append(sp_stats.skew(sig))
        features.append(sp_stats.kurtosis(sig))
        features.append(np.max(sig) - np.min(sig))  # peak-to-peak

        # Percentiles
        features.append(np.percentile(sig, 5))
        features.append(np.percentile(sig, 25))
        features.append(np.percentile(sig, 75))
        features.append(np.percentile(sig, 95))

        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(sig - np.mean(sig))) != 0)
        features.append(zero_crossings / n_samples)

        # RMS
        features.append(np.sqrt(np.mean(sig ** 2)))

        # Power spectral density features
        if len(sig) > 256:
            freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))
            # Total power in bands
            low_mask = (freqs >= 0.5) & (freqs < 5)
            mid_mask = (freqs >= 5) & (freqs < 15)
            high_mask = (freqs >= 15) & (freqs < 40)

            total_power = np.sum(psd) + 1e-10
            features.append(np.sum(psd[low_mask]) / total_power)
            features.append(np.sum(psd[mid_mask]) / total_power)
            features.append(np.sum(psd[high_mask]) / total_power)
            features.append(freqs[np.argmax(psd)])  # dominant frequency
        else:
            features.extend([0, 0, 0, 0])

    return np.array(features, dtype=np.float64)


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_dataset(base_path="ekg_datasets/ptbxl"):
    """
    Load PTB-XL records and their superclass labels.

    Returns: X (list of (N,12) arrays), y (list of superclass strings),
             folds (list of ints), meta (DataFrame)
    """
    base = Path(base_path)
    meta = pd.read_csv(base / "ptbxl_database.csv", index_col="ecg_id")
    scp_map = build_scp_to_superclass(str(base / "scp_statements.csv"))

    # Map each record to its superclass
    meta["superclass"] = meta["scp_codes"].apply(
        lambda x: get_primary_superclass(x, scp_map)
    )

    # Drop records with no valid superclass
    meta = meta.dropna(subset=["superclass"])
    meta = meta[meta["superclass"].isin(SUPERCLASS_LABELS)]

    X = []
    y = []
    folds = []
    valid_ids = []

    rows = list(meta.iterrows())
    print(f"  Found {len(rows)} labeled records on disk — loading signals...")
    t0 = time.time()

    for ecg_id, row in tqdm(rows, unit="rec", ncols=72):
        rec_path = str(base / row["filename_hr"])
        if not os.path.exists(rec_path + ".dat"):
            continue

        try:
            rec = wfdb.rdrecord(rec_path)
            sig = rec.p_signal
            if sig is None or sig.shape[1] < 12:
                continue
            X.append(sig)
            y.append(row["superclass"])
            folds.append(int(row["strat_fold"]))
            valid_ids.append(ecg_id)
        except Exception:
            continue

    elapsed = time.time() - t0
    print(f"  Loaded {len(X)} records in {elapsed:.1f}s")
    return X, y, folds, valid_ids


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train():
    """Train the classifier and save it."""
    print("\n" + "=" * 60)
    print("  ECG Superclass Classifier — Training")
    print("=" * 60)

    X_raw, y, folds, _ = load_dataset()

    if not X_raw:
        print("  No records found. Run download_ekg_datasets.py first.")
        return

    # Extract features
    print(f"  Extracting features from {len(X_raw)} records...")
    t1 = time.time()
    X = np.array([
        extract_features(sig)
        for sig in tqdm(X_raw, unit="rec", ncols=72)
    ])
    y = np.array(y)
    folds = np.array(folds)
    print(f"  Feature extraction done in {time.time() - t1:.1f}s  |  shape: {X.shape}")

    # Replace NaN/inf in features
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Split: folds 1-8 train, 9 val, 10 test
    train_mask = folds <= 8
    val_mask = folds == 9
    test_mask = folds == 10

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\n  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Feature dimensions: {X_train.shape[1]}")

    # Class distribution
    for split_name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(split_y, return_counts=True)
        dist = ", ".join(f"{u}: {c}" for u, c in zip(unique, counts))
        print(f"  {split_name} classes: {dist}")

    # Train Gradient Boosting (handles imbalance better than RF)
    print(f"\n  Training Gradient Boosting on {len(X_train)} records...")
    t2 = time.time()
    le = LabelEncoder()
    le.fit(SUPERCLASS_LABELS)

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=3,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    print(f"  Training done in {time.time() - t2:.1f}s")

    # Validation results
    val_pred = clf.predict(X_val)
    val_acc = np.mean(val_pred == y_val)
    print(f"\n  Validation accuracy: {val_acc:.1%}")
    print("\n  Validation report:")
    print(classification_report(y_val, val_pred, zero_division=0))

    # Test results
    if len(X_test) > 0:
        test_pred = clf.predict(X_test)
        test_acc = np.mean(test_pred == y_test)
        print(f"  Test accuracy: {test_acc:.1%}")
        print("\n  Test report:")
        print(classification_report(y_test, test_pred, zero_division=0))
        print("  Confusion matrix:")
        print(confusion_matrix(y_test, test_pred, labels=clf.classes_))

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model_data = {
        "classifier": clf,
        "feature_names": None,
        "superclass_labels": SUPERCLASS_LABELS,
        "superclass_descriptions": SUPERCLASS_DESCRIPTIONS,
        "val_accuracy": float(val_acc),
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\n  Model saved to {MODEL_PATH}")
    print("  Done.\n")

    return model_data


# ─────────────────────────────────────────────────────────────
# Inference (used by app.py)
# ─────────────────────────────────────────────────────────────

def load_classifier(path=MODEL_PATH):
    """Load the trained classifier from disk."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(model_data, signal_12, fs=500):
    """
    Predict the ECG superclass from a 12-lead signal.

    Args:
        model_data: dict from load_classifier()
        signal_12: (N, 12) numpy array
        fs: sampling rate

    Returns:
        dict with keys: prediction, confidence, probabilities, descriptions
    """
    clf = model_data["classifier"]

    features = extract_features(signal_12, fs)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = features.reshape(1, -1)

    prediction = clf.predict(features)[0]
    proba = clf.predict_proba(features)[0]
    classes = clf.classes_

    prob_dict = {cls: round(float(p), 3) for cls, p in zip(classes, proba)}
    confidence = float(max(proba))

    return {
        "prediction": prediction,
        "description": SUPERCLASS_DESCRIPTIONS.get(prediction, prediction),
        "confidence": confidence,
        "probabilities": prob_dict,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
