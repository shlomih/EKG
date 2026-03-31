"""
dataset_challenge.py
====================
Load PhysioNet 2021 Challenge ECG datasets (Georgia, CPSC-2018, CPSC-Extra, Ningbo).
Maps SNOMED-CT diagnosis codes to human-readable condition names.

Conditions added beyond the 14-class merged model:
  PAC   - Premature atrial contraction
  LBBB  - Left bundle branch block (complete, alias for CLBBB)
  RBBB  - Right bundle branch block (complete, alias for CRBBB)
  Brady - Bradycardia (sinus bradycardia)
  STACH - Sinus tachycardia (already in merged model)
  SVT   - Supraventricular tachycardia
  TAb   - T-wave abnormality / inversion
  LQTP  - Prolonged QT interval
  RAD   - Right axis deviation
  LAD   - Left axis deviation
  PR    - Prolonged PR (alias for 1AVB)
  NSIVC - Non-specific intraventricular conduction delay

Usage:
    from dataset_challenge import load_challenge_multilabel, CHALLENGE_CODES
    paths, labels = load_challenge_multilabel()
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# SNOMED-CT code -> short label mapping
# Source: PhysioNet/CinC Challenge 2020/2021 scored conditions
# ---------------------------------------------------------------------------
SNOMED_TO_LABEL = {
    # Already in merged model (14-class) -- kept for overlap detection
    426783006: "NORM",    # Normal sinus rhythm
    164889003: "AFIB",    # Atrial fibrillation
    427084000: "STACH",   # Sinus tachycardia
    164873001: "LVH",     # Left ventricular hypertrophy
    270492004: "CLBBB",   # Left bundle branch block (complete)
    164909002: "CLBBB",   # Left bundle branch block (alternate code)
    713427006: "CRBBB",   # Right bundle branch block (complete)
    59118001:  "CRBBB",   # Right bundle branch block (alternate code)
    713426002: "IRBBB",   # Incomplete right bundle branch block
    164947007: "1AVB",    # First-degree AV block / prolonged PR
    427172004: "PVC",     # Premature ventricular contraction
    17338001:  "PVC",     # Premature ventricular contraction (alternate)
    # New conditions
    284470004: "PAC",     # Premature atrial contraction
    63593006:  "PAC",     # Premature atrial contraction (alternate)
    426627000: "Brady",   # Bradycardia
    426177001: "Brady",   # Sinus bradycardia
    63386007:  "SVT",     # Supraventricular tachycardia
    54329005:  "SVT",     # SVT (alternate)
    111975006: "LQTP",    # Prolonged QT interval
    164934002: "TAb",     # T-wave abnormality (includes inversion)
    59931005:  "TAb",     # T-wave inversion
    164930006: "TAb",     # T-wave change
    39732003:  "LAD",     # Left axis deviation
    47665007:  "RAD",     # Right axis deviation
    698252002: "NSIVC",   # Non-specific intraventricular conduction delay
    445118002: "LAFB",    # Left anterior fascicular block (alternate)
    251146004: "LAFB",    # Left posterior fascicular block
    # Additional high-value conditions found in data
    164890007: "AFL",     # Atrial flutter (7,824 samples)
    55827005:  "LVH",     # Left ventricular hypertrophy (alternate code)
    428750005: "STc",     # ST-T change (3,159 samples)
    429622005: "STD",     # ST depression (2,095 samples)
    67741000119109: "LAE", # Left atrial enlargement (871 samples)
    164884008: "PVC",     # Ventricular ectopic beat (alternate for PVC)
}

# ---------------------------------------------------------------------------
# Full label set for the 20-class v3 model
# Starts with all 14 merged classes, adds new ones
# ---------------------------------------------------------------------------
from dataset_chapman import MERGED_CODES  # 14-class base

NEW_CONDITIONS = ["PAC", "Brady", "SVT", "LQTP", "TAb", "LAD", "RAD", "NSIVC", "AFL", "STc", "STD", "LAE"]
V3_CODES = MERGED_CODES + [c for c in NEW_CONDITIONS if c not in MERGED_CODES]
N_V3 = len(V3_CODES)  # 22 classes

CHALLENGE_DATASETS = ["georgia", "cpsc_2018", "cpsc_2018_extra", "ningbo"]
CHALLENGE_DIR = Path("ekg_datasets/challenge2021")


def _parse_hea(hea_path: Path):
    """Parse a .hea file and return list of SNOMED codes."""
    codes = []
    try:
        with open(hea_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# Dx:"):
                    parts = line.replace("# Dx:", "").strip().split(",")
                    for p in parts:
                        p = p.strip()
                        if p.isdigit():
                            codes.append(int(p))
    except Exception:
        pass
    return codes


def load_challenge_multilabel(
    challenge_dir: str = None,
    codes: list = None,
) -> tuple:
    """
    Load all challenge ECG records and return (paths, label_matrix).

    Returns:
        paths        : list of str, absolute paths to .mat files
        label_matrix : np.ndarray shape (N, len(codes)), float32
    """
    if challenge_dir is None:
        challenge_dir = CHALLENGE_DIR
    else:
        challenge_dir = Path(challenge_dir)

    if codes is None:
        codes = V3_CODES

    code_to_idx = {c: i for i, c in enumerate(codes)}
    n_classes = len(codes)

    paths = []
    labels = []
    skipped_no_label = 0
    skipped_no_file  = 0

    for ds in CHALLENGE_DATASETS:
        ds_dir = challenge_dir / ds
        if not ds_dir.exists():
            print(f"  [challenge] {ds}: not found, skipping")
            continue

        mat_files = sorted(ds_dir.rglob("*.mat"))
        ds_paths, ds_labels, ds_skip_lbl, ds_skip_file = [], [], 0, 0

        for mat_path in mat_files:
            hea_path = mat_path.with_suffix(".hea")
            if not hea_path.exists():
                ds_skip_file += 1
                continue

            snomed_codes = _parse_hea(hea_path)
            row = np.zeros(n_classes, dtype=np.float32)
            matched = False
            for sc in snomed_codes:
                label = SNOMED_TO_LABEL.get(sc)
                if label and label in code_to_idx:
                    row[code_to_idx[label]] = 1.0
                    matched = True

            if not matched:
                ds_skip_lbl += 1
                continue

            ds_paths.append(str(mat_path))
            ds_labels.append(row)

        n_kept = len(ds_paths)
        skipped_no_label += ds_skip_lbl
        skipped_no_file  += ds_skip_file
        paths.extend(ds_paths)
        labels.extend(ds_labels)
        print(f"  [challenge] {ds:<16}: {n_kept:>6} kept  "
              f"(skipped: {ds_skip_lbl} no-label, {ds_skip_file} no-hea)")

    label_matrix = np.stack(labels, axis=0) if labels else np.zeros((0, n_classes), dtype=np.float32)
    print(f"  Challenge total: {len(paths)} records  "
          f"(skipped: {skipped_no_label} no-label, {skipped_no_file} no-hea)")
    return paths, label_matrix


def print_challenge_stats(challenge_dir: str = None, codes: list = None):
    """Print per-class sample counts for challenge datasets."""
    paths, labels = load_challenge_multilabel(challenge_dir, codes)
    if codes is None:
        codes = V3_CODES
    print(f"\n  Per-class positives in challenge data ({len(paths)} records):")
    for i, code in enumerate(codes):
        n = int(labels[:, i].sum())
        if n > 0:
            print(f"    {code:<8}: {n:>6}")


if __name__ == "__main__":
    import sys
    os.chdir(Path(__file__).parent)
    print(f"V3 label set ({N_V3} classes): {V3_CODES}")
    print()
    print_challenge_stats()
