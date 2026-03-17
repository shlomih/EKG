"""
validate_classifier.py
======================
Batch validation of the ECG classifier across the full PTB-XL test fold.
Generates accuracy metrics, confusion matrix, and per-class reports.

Supports both sklearn (poc_classifier) and CNN (cnn_classifier) models.

Usage:
    python validate_classifier.py              # validates whichever model is available
    python validate_classifier.py --model cnn  # force CNN
    python validate_classifier.py --model sklearn  # force sklearn
"""

import argparse
import ast
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wfdb
from tqdm import tqdm

SUPERCLASS_LABELS = ["NORM", "MI", "STTC", "HYP", "CD"]


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


def load_test_data(base_path="ekg_datasets/ptbxl", fold=10):
    """Load test fold records and labels."""
    base = Path(base_path)
    meta = pd.read_csv(base / "ptbxl_database.csv", index_col="ecg_id")
    scp_map = build_scp_to_superclass(str(base / "scp_statements.csv"))

    meta["superclass"] = meta["scp_codes"].apply(
        lambda x: get_primary_superclass(x, scp_map)
    )
    meta = meta.dropna(subset=["superclass"])
    meta = meta[meta["superclass"].isin(SUPERCLASS_LABELS)]
    meta = meta[meta["strat_fold"] == fold]

    records = []
    for ecg_id, row in meta.iterrows():
        rec_path = str(base / row["filename_hr"])
        if os.path.exists(rec_path + ".dat"):
            records.append({
                "ecg_id": ecg_id,
                "path": rec_path,
                "true_label": row["superclass"],
            })

    print(f"  Test fold {fold}: {len(records)} records")
    return records


def validate_sklearn(records):
    """Run validation with the sklearn model."""
    from poc_classifier import load_classifier, predict, extract_features
    model = load_classifier()
    if model is None:
        print("  sklearn model not found at models/ecg_classifier.pkl")
        return None

    predictions = []
    true_labels = []

    for rec_info in tqdm(records, desc="  Classifying (sklearn)", ncols=72):
        try:
            rec = wfdb.rdrecord(rec_info["path"])
            sig = rec.p_signal
            if sig is None or sig.shape[1] < 12:
                continue
            result = predict(model, sig, rec.fs)
            predictions.append(result["prediction"])
            true_labels.append(rec_info["true_label"])
        except Exception:
            continue

    return true_labels, predictions


def validate_cnn(records):
    """Run validation with the CNN model."""
    from cnn_classifier import load_cnn_classifier, predict_cnn
    model = load_cnn_classifier()
    if model is None:
        print("  CNN model not found at models/ecg_cnn.pt")
        return None

    predictions = []
    true_labels = []

    for rec_info in tqdm(records, desc="  Classifying (CNN)", ncols=72):
        try:
            rec = wfdb.rdrecord(rec_info["path"])
            sig = rec.p_signal
            if sig is None or sig.shape[1] < 12:
                continue
            result = predict_cnn(model, sig, rec.fs)
            predictions.append(result["prediction"])
            true_labels.append(rec_info["true_label"])
        except Exception:
            continue

    return true_labels, predictions


def generate_report(true_labels, predictions, model_name, out_dir="validation_results"):
    """Generate classification report, confusion matrix, and save to disk."""
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
    )

    os.makedirs(out_dir, exist_ok=True)

    y_true = np.array(true_labels)
    y_pred = np.array(predictions)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n  {'=' * 50}")
    print(f"  {model_name} — Validation Results")
    print(f"  {'=' * 50}")
    print(f"  Accuracy:          {acc:.1%}")
    print(f"  F1 (macro):        {f1_macro:.3f}")
    print(f"  F1 (weighted):     {f1_weighted:.3f}")
    print(f"  Total predictions: {len(y_true)}")

    report = classification_report(
        y_true, y_pred,
        labels=SUPERCLASS_LABELS,
        target_names=SUPERCLASS_LABELS,
        zero_division=0,
    )
    print(f"\n{report}")

    # Save text report
    with open(os.path.join(out_dir, f"{model_name}_report.txt"), "w") as f:
        f.write(f"{model_name} Validation Report\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Accuracy:      {acc:.4f}\n")
        f.write(f"F1 (macro):    {f1_macro:.4f}\n")
        f.write(f"F1 (weighted): {f1_weighted:.4f}\n")
        f.write(f"N:             {len(y_true)}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=SUPERCLASS_LABELS)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SUPERCLASS_LABELS, yticklabels=SUPERCLASS_LABELS,
                ax=axes[0])
    axes[0].set_title(f"{model_name} — Counts")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=SUPERCLASS_LABELS, yticklabels=SUPERCLASS_LABELS,
                ax=axes[1])
    axes[1].set_title(f"{model_name} — Recall (normalized)")
    axes[1].set_ylabel("True")
    axes[1].set_xlabel("Predicted")

    fig.suptitle(f"{model_name} — Accuracy: {acc:.1%} | F1: {f1_macro:.3f}", fontsize=13)
    fig.tight_layout()

    img_path = os.path.join(out_dir, f"{model_name}_confusion_matrix.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved -> {img_path}")

    # Save per-class CSV
    report_dict = classification_report(
        y_true, y_pred,
        labels=SUPERCLASS_LABELS,
        target_names=SUPERCLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    df_report = pd.DataFrame(report_dict).T
    csv_path = os.path.join(out_dir, f"{model_name}_per_class.csv")
    df_report.to_csv(csv_path)
    print(f"  Per-class metrics -> {csv_path}")

    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def main():
    parser = argparse.ArgumentParser(description="Validate ECG classifier")
    parser.add_argument("--model", choices=["sklearn", "cnn", "both"], default="both")
    parser.add_argument("--fold", type=int, default=10, help="Test fold (default: 10)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ECG Classifier — Batch Validation")
    print("=" * 60)

    records = load_test_data(fold=args.fold)
    if not records:
        print("  No test records found.")
        return

    results = {}

    if args.model in ("sklearn", "both"):
        print("\n  Running sklearn (GradientBoosting) validation...")
        t0 = time.time()
        out = validate_sklearn(records)
        if out:
            true_labels, preds = out
            metrics = generate_report(true_labels, preds, "GradientBoosting")
            metrics["time_s"] = time.time() - t0
            results["sklearn"] = metrics

    if args.model in ("cnn", "both"):
        print("\n  Running CNN validation...")
        t0 = time.time()
        out = validate_cnn(records)
        if out:
            true_labels, preds = out
            metrics = generate_report(true_labels, preds, "CNN")
            metrics["time_s"] = time.time() - t0
            results["cnn"] = metrics

    # Compare if both ran
    if len(results) == 2:
        print(f"\n  {'=' * 50}")
        print(f"  Model Comparison")
        print(f"  {'=' * 50}")
        print(f"  {'':>20} {'GradientBoosting':>18} {'CNN':>10}")
        print(f"  {'Accuracy':>20} {results['sklearn']['accuracy']:>17.1%} {results['cnn']['accuracy']:>9.1%}")
        print(f"  {'F1 (macro)':>20} {results['sklearn']['f1_macro']:>17.3f} {results['cnn']['f1_macro']:>9.3f}")
        print(f"  {'Time':>20} {results['sklearn']['time_s']:>16.1f}s {results['cnn']['time_s']:>8.1f}s")

    print(f"\n  Results saved to validation_results/")
    print()


if __name__ == "__main__":
    main()
