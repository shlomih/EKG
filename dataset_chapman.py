"""
dataset_chapman.py
==================
Download and integrate the Chapman-Shaoxing 12-lead ECG dataset into the
multi-label training pipeline. Adds AFIB (3,889 confirmed cases) and other
arrhythmia conditions missing from PTB-XL.

Dataset facts:
  - 45,152 records, 500 Hz, 10s (5000 samples), 12 leads
  - WFDB format: .hea header + .mat signal file
  - Labels: SNOMED-CT codes in header #Dx field
  - PhysioNet: physionet.org/content/ecg-arrhythmia/1.0.0/

Key conditions well-represented (vs PTB-XL gaps):
  AFIB  : 3,889  ← primary reason to use this dataset
  GSVT  :   869  supraventricular tachycardia
  SB    : 3,889  sinus bradycardia
  ST    : 2,760  sinus tachycardia

SNOMED-CT → our label mapping:
  164889003 → AFIB   Atrial fibrillation
  426783006 → NORM   Sinus rhythm (normal)
  426177001 → NORM   Sinus bradycardia (normal rhythm, just slow)
  427084000 → STACH  Sinus tachycardia  [new label added here]
  713422000 → CRBBB  Complete right bundle branch block
  164909002 → CLBBB  Complete left bundle branch block
  164884008 → PVC    Premature ventricular complex
  59118001  → CRBBB  Right bundle branch block (generic)
  164909002 → CLBBB  Left bundle branch block (generic)

Usage:
    python dataset_chapman.py --download    # download dataset (~20 GB)
    python dataset_chapman.py --index       # build index CSV (no download needed if already present)
    python dataset_chapman.py --stats       # print label distribution

    # From multilabel_classifier.py:
    from dataset_chapman import load_chapman_multilabel
    paths_c, labels_c = load_chapman_multilabel()   # returns (paths, N×13 matrix)
"""

import ast
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

CHAPMAN_BASE   = "ekg_datasets/chapman"
CHAPMAN_INDEX  = "ekg_datasets/chapman_index.csv"
PHYSIONET_DB   = "ecg-arrhythmia/1.0.0"

# SNOMED-CT code → our multi-label code
# Only map codes that appear in our label set (or AFIB which we add)
SNOMED_TO_LABEL = {
    "164889003": "AFIB",   # Atrial fibrillation
    "164890007": "AFIB",   # Atrial flutter (treat as AFIB-class for now)
    "426783006": "NORM",   # Sinus rhythm
    "426177001": "NORM",   # Sinus bradycardia (normal rhythm)
    "713422000": "CRBBB",  # Complete right bundle branch block
    "59118001":  "CRBBB",  # Right bundle branch block
    "164909002": "CLBBB",  # Complete left bundle branch block
    "164884008": "PVC",    # Premature ventricular complex
    "164917005": "1AVB",   # First degree atrioventricular block
    "164934002": "LAFB",   # Left anterior fascicular block
    "445118002": "IRBBB",  # Incomplete right bundle branch block
    "251146004": "IRBBB",  # Incomplete right bundle branch block (alt code)
    "55930002":  "ISC_",   # ST depression (ischemic)
    "164931005": "LVH",    # Left ventricular hypertrophy
    "67751000119106": "LVH",  # LVH (alt code)
    "427084000": "STACH",  # Sinus tachycardia  ← new label (rich in Chapman)
    "164934002": "LAFB",   # Left anterior fascicular block
}

# Our 13-label set for the merged model (adds AFIB + STACH to the 12-class PTB-XL set)
MERGED_CODES = [
    "NORM",   # 0
    "AFIB",   # 1  ← from Chapman-Shaoxing
    "PVC",    # 2
    "LVH",    # 3
    "IMI",    # 4
    "ASMI",   # 5
    "CLBBB",  # 6
    "CRBBB",  # 7
    "LAFB",   # 8
    "1AVB",   # 9
    "ISC_",   # 10
    "NDT",    # 11
    "IRBBB",  # 12
    "STACH",  # 13  ← from Chapman-Shaoxing (only 4 in PTB-XL, 2760 here)
]
MERGED_CODE_TO_IDX = {c: i for i, c in enumerate(MERGED_CODES)}
N_MERGED = len(MERGED_CODES)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_chapman(base_path: str = CHAPMAN_BASE):
    """Download Chapman-Shaoxing from PhysioNet (~20 GB). Takes ~30-60 min."""
    import wfdb
    Path(base_path).mkdir(parents=True, exist_ok=True)
    print(f"Downloading Chapman-Shaoxing to {base_path} ...")
    print("This takes 30–60 minutes depending on connection speed.")
    wfdb.dl_database(PHYSIONET_DB, base_path)
    print("Download complete.")


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_snomed_codes(hea_path: str) -> list:
    """Read SNOMED-CT codes from a .hea header file's #Dx field."""
    codes = []
    try:
        with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("#Dx:"):
                    raw = line.split(":", 1)[1].strip()
                    codes = [c.strip() for c in raw.split(",")]
                    break
    except Exception:
        pass
    return codes


def snomed_to_multilabel(snomed_codes: list) -> np.ndarray:
    """Convert list of SNOMED codes → 14-hot vector (MERGED_CODES)."""
    vec = np.zeros(N_MERGED, dtype=np.float32)
    for code in snomed_codes:
        label = SNOMED_TO_LABEL.get(code)
        if label and label in MERGED_CODE_TO_IDX:
            vec[MERGED_CODE_TO_IDX[label]] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_chapman_index(base_path: str = CHAPMAN_BASE,
                        output_path: str = CHAPMAN_INDEX) -> pd.DataFrame:
    """
    Walk Chapman-Shaoxing directory, parse all .hea files, build an index CSV.
    Columns: path, snomed_codes, <one column per MERGED_CODE>
    """
    base = Path(base_path)
    rows = []
    hea_files = sorted(base.rglob("*.hea"))
    print(f"Found {len(hea_files)} .hea files in {base_path}")

    for i, hea in enumerate(hea_files):
        rec_path = str(hea.with_suffix(""))  # path without extension
        mat_path = hea.with_suffix(".mat")
        if not mat_path.exists():
            continue

        codes = parse_snomed_codes(str(hea))
        vec   = snomed_to_multilabel(codes)

        if vec.sum() == 0:
            continue  # skip records with no mappable labels

        row = {"path": rec_path, "snomed_codes": ",".join(codes)}
        for j, label in enumerate(MERGED_CODES):
            row[label] = int(vec[j])
        rows.append(row)

        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(hea_files)} indexed...")

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nIndexed {len(df)} records → {output_path}")
    _print_label_stats(df)
    return df


def _print_label_stats(df: pd.DataFrame):
    print("\nLabel distribution:")
    for code in MERGED_CODES:
        if code in df.columns:
            n = int(df[code].sum())
            print(f"  {code:<8} {n:>6}")


# ---------------------------------------------------------------------------
# Dataset loading (for use in multilabel_classifier.py)
# ---------------------------------------------------------------------------

def load_chapman_multilabel(index_path: str = CHAPMAN_INDEX):
    """
    Load Chapman-Shaoxing index and return paths + label matrix aligned to MERGED_CODES.

    Returns:
        paths       : list of record paths (without extension)
        label_matrix: np.ndarray (N, 14) float32 multi-hot
    """
    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"Chapman index not found: {index_path}\n"
            f"Run: python dataset_chapman.py --index"
        )

    df = pd.read_csv(index_path)
    paths = df["path"].tolist()
    label_matrix = df[MERGED_CODES].values.astype(np.float32)
    print(f"Chapman-Shaoxing: {len(paths)} records loaded")
    return paths, label_matrix


# ---------------------------------------------------------------------------
# Signal loading (Chapman uses .mat, not .dat)
# ---------------------------------------------------------------------------

def load_chapman_signal(rec_path: str) -> np.ndarray | None:
    """
    Load a Chapman-Shaoxing signal from .mat file.
    Returns (12, 5000) float32 in mV, or None on failure.
    """
    try:
        import wfdb
        record = wfdb.rdrecord(rec_path)
        sig = record.p_signal.T.astype(np.float32)   # (12, N)
        if sig.shape[1] != 5000:
            import scipy.signal
            sig = scipy.signal.resample(sig, 5000, axis=1)
        return sig
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stats / CLI
# ---------------------------------------------------------------------------

def print_stats(index_path: str = CHAPMAN_INDEX):
    df = pd.read_csv(index_path)
    print(f"Total records: {len(df)}")
    _print_label_stats(df)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="Download dataset from PhysioNet")
    ap.add_argument("--index",    action="store_true", help="Build index from downloaded files")
    ap.add_argument("--stats",    action="store_true", help="Print label distribution")
    ap.add_argument("--base",     default=CHAPMAN_BASE, help="Local dataset directory")
    args = ap.parse_args()

    if args.download:
        download_chapman(args.base)
    if args.index:
        build_chapman_index(args.base)
    if args.stats:
        print_stats()
    if not any([args.download, args.index, args.stats]):
        ap.print_help()
