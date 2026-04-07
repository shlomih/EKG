"""
dataset_code15.py
=================
Download and integrate the CODE-15% ECG dataset into the multilabel V3 pipeline.

Dataset: "CODE-15%: a large scale annotated dataset of 12-lead ECGs"
Source:  https://zenodo.org/record/4916206  (open-access, no credentials needed)
Paper:   Ribeiro et al., Nature Communications 2020

Files on Zenodo:
  exams.csv              -- metadata + binary labels (345,779 rows)
  exams_part0.zip        -- HDF5 file with first batch of ECGs
  exams_part1.zip        ...
  ...
  exams_part17.zip       -- 18 zip files total (parts 0–17)

HDF5 structure (inside each exams_part{i}.zip → exams_part{i}.h5):
  tracings  : float32 (N, 4096, 12)  -- ECG signals in millivolts
  exam_id   : int64   (N,)           -- exam IDs matching exams.csv

Signal specs:
  400 Hz, 4096 samples (signal is 10 s padded with zeros from 4000 real samples)
  Lead order: DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
  → resampled here to 500 Hz / 5000 samples to match our pipeline

exams.csv column names (label columns are binary 0/1):
  exam_id, patient_id, age, is_male, nn_predicted_age
  1dAVb, RBBB, LBBB, SB, AF, ST

Label distribution (approximate, from Ribeiro et al. 2020 Nat. Comms.):
  AF    :  ~17,000 ( 4.9%) → AFIB   ← PRIMARY FIX for our weakest class
  1dAVb :  ~20,000 ( 5.8%) → 1AVB   ← PRIMARY FIX for second weakest
  RBBB  :  ~22,000 ( 6.4%) → CRBBB
  LBBB  :  ~19,000 ( 5.5%) → CLBBB
  SB    :  ~51,000 (14.8%) → Brady
  ST    :  ~32,000 ( 9.3%) → STACH
  NORM  : ~195,000 (56.5%) → NORM   (records with all 6 labels = 0)

V3_CODES mapping:
  AF    → AFIB   (V3 index 1)
  1dAVb → 1AVB   (V3 index 9)
  RBBB  → CRBBB  (V3 index 7)
  LBBB  → CLBBB  (V3 index 6)
  SB    → Brady  (V3 index 15)
  ST    → STACH  (V3 index 13)
  NORM  → NORM   (V3 index 0)

Usage:
    # Step 1: Download (one-time, ~35 GB, ~60 min on 100 Mbps)
    python dataset_code15.py --download

    # Step 2: Build index (one-time, ~30 min, reads all H5 files)
    python dataset_code15.py --index

    # Step 3: Check stats
    python dataset_code15.py --stats

    # In multilabel_v3.py:
    from dataset_code15 import load_code15_multilabel, load_code15_signal, CODE15_PATH_PREFIX
"""

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CODE15_BASE   = Path("ekg_datasets/code15")
CODE15_RAW    = CODE15_BASE / "raw"      # downloaded zip + h5 files
CODE15_INDEX  = CODE15_BASE / "code15_index.csv"

ZENODO_BASE   = "https://zenodo.org/record/4916206/files"
N_PARTS       = 18   # exams_part0.zip ... exams_part17.zip

# Sentinel prefix used to distinguish CODE-15% paths in V3ECGDataset
CODE15_PATH_PREFIX = "code15::"

# ---------------------------------------------------------------------------
# Label mapping: CODE-15% column → V3 short code
# ---------------------------------------------------------------------------

CODE15_LABEL_MAP = {
    "AF":    "AFIB",
    "1dAVb": "1AVB",
    "RBBB":  "CRBBB",
    "LBBB":  "CLBBB",
    "SB":    "Brady",
    "ST":    "STACH",
}
CODE15_LABELS = list(CODE15_LABEL_MAP.keys())   # ["AF","1dAVb","RBBB","LBBB","SB","ST"]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_code15(base_path: Path = CODE15_BASE, skip_existing: bool = True):
    """
    Download CODE-15% from Zenodo.

    Downloads exams.csv and 18 zip archives (~35 GB total).
    Each zip is extracted to base_path/raw/ immediately after download.

    Estimated time: 30–90 minutes depending on connection speed.
    """
    import requests

    raw = Path(base_path) / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "code15-downloader/1.0"

    def _fetch(url: str, dest: Path, desc: str):
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            print(f"  [skip] {desc} already downloaded")
            return True
        print(f"  Downloading {desc} ...", flush=True)
        try:
            with session.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done  = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):   # 1 MB chunks
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = 100 * done / total
                            print(f"    {pct:5.1f}%  ({done>>20} / {total>>20} MB)\r",
                                  end="", flush=True)
            print(f"    Done — {done>>20} MB")
            return True
        except Exception as e:
            print(f"  ERROR downloading {desc}: {e}")
            return False

    # 1. exams.csv
    _fetch(f"{ZENODO_BASE}/exams.csv?download=1", raw / "exams.csv", "exams.csv")

    # 2. 18 zip archives
    for i in range(N_PARTS):
        part_name = f"exams_part{i}"
        zip_dest  = raw / f"{part_name}.zip"

        # Accept both .h5 and .hdf5 — Zenodo zips extract as .hdf5
        h5_dest   = raw / f"{part_name}.hdf5"
        if not h5_dest.exists():
            h5_dest = raw / f"{part_name}.h5"

        if skip_existing and h5_dest.exists() and h5_dest.stat().st_size > 100_000:
            print(f"  [skip] {h5_dest.name} already extracted")
            continue

        ok = _fetch(f"{ZENODO_BASE}/{part_name}.zip?download=1", zip_dest, f"{part_name}.zip")
        if not ok:
            continue

        # Extract
        print(f"  Extracting {part_name}.zip ...", flush=True)
        try:
            with zipfile.ZipFile(zip_dest, "r") as zf:
                extracted = zf.namelist()
                zf.extractall(raw)
            print(f"  Extracted → {extracted}")
            # Remove zip to save disk space (~half the footprint)
            zip_dest.unlink()
        except Exception as e:
            print(f"  ERROR extracting {part_name}.zip: {e}")

    print("\n  Download complete.")
    _print_download_status(raw)


def _h5_path(raw: Path, i: int) -> Path:
    """Return path for part i, preferring .hdf5 over .h5."""
    p = raw / f"exams_part{i}.hdf5"
    if p.exists():
        return p
    return raw / f"exams_part{i}.h5"


def _print_download_status(raw: Path):
    csv_ok = (raw / "exams.csv").exists()
    n_h5   = sum(1 for i in range(N_PARTS) if _h5_path(raw, i).exists())
    print(f"  exams.csv : {'OK' if csv_ok else 'MISSING'}")
    print(f"  H5 files  : {n_h5}/{N_PARTS}")


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_code15_index(base_path: Path = CODE15_BASE) -> pd.DataFrame:
    """
    Walk all exams_part{i}.h5 files, join with exams.csv labels, build index CSV.

    Index columns:
        path      : str  -- "code15::raw/exams_part0.h5::42"  (part file + row index)
        exam_id   : int
        age       : float
        is_male   : int  (1=male, 0=female)
        AF, 1dAVb, RBBB, LBBB, SB, ST : int (binary labels from CODE-15%)

    Returns: DataFrame saved to CODE15_INDEX
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required: pip install h5py --break-system-packages")

    base = Path(base_path)
    raw  = base / "raw"

    # Load metadata + labels
    csv_path = raw / "exams.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"exams.csv not found at {csv_path}. Run --download first.")

    print(f"\n  Loading exams.csv ...")
    meta = pd.read_csv(csv_path)
    # Normalise exam_id column name (may be 'exam_id' or 'id_exam')
    if "exam_id" not in meta.columns and "id_exam" in meta.columns:
        meta = meta.rename(columns={"id_exam": "exam_id"})

    meta = meta.set_index("exam_id")
    print(f"  exams.csv: {len(meta)} rows, columns: {list(meta.columns)}")

    rows = []
    missing_labels = 0
    missing_signal = 0

    for i in range(N_PARTS):
        h5_path = _h5_path(raw, i)
        if not h5_path.exists():
            print(f"  [skip] exams_part{i}.h5/.hdf5 not found")
            continue

        print(f"  Indexing exams_part{i}.h5 ...", flush=True)
        try:
            with h5py.File(h5_path, "r") as f:
                # Key may be 'exam_id' or 'id_exam' depending on version
                if "exam_id" in f:
                    exam_ids = np.array(f["exam_id"])
                elif "id_exam" in f:
                    exam_ids = np.array(f["id_exam"])
                else:
                    # Fall back: try to match by position if only one key group
                    available = list(f.keys())
                    id_key = next((k for k in available if "exam" in k.lower() or "id" in k.lower()), None)
                    if id_key:
                        exam_ids = np.array(f[id_key])
                    else:
                        print(f"  WARN: can't find exam_id key in {h5_path.name}. Keys: {available}")
                        continue

                n_in_file = len(exam_ids)
                print(f"    {n_in_file} exams in file", flush=True)

                for row_idx, eid in enumerate(exam_ids):
                    eid = int(eid)
                    if eid not in meta.index:
                        missing_labels += 1
                        continue

                    m = meta.loc[eid]

                    # Path token: code15::relative/path/to/h5::row_idx
                    rel_h5 = str(h5_path.relative_to(Path.cwd()) if h5_path.is_absolute()
                                 else h5_path)
                    path_token = f"{CODE15_PATH_PREFIX}{rel_h5}::{row_idx}"

                    row = {
                        "path":     path_token,
                        "exam_id":  eid,
                        "age":      float(m.get("age", 50.0) or 50.0),
                        "is_male":  int(m.get("is_male", 1) or 1),
                    }
                    for col in CODE15_LABELS:
                        row[col] = int(m.get(col, 0) or 0)

                    rows.append(row)

                    if (row_idx + 1) % 10000 == 0:
                        print(f"    {row_idx+1}/{n_in_file} processed ...", flush=True)

        except Exception as e:
            print(f"  ERROR reading {h5_path.name}: {e}")
            continue

    if not rows:
        print("  No records indexed! Check that H5 files and exams.csv are present.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    CODE15_INDEX.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CODE15_INDEX, index=False)
    print(f"\n  Indexed {len(df)} records -> {CODE15_INDEX}")
    print(f"  Skipped: {missing_labels} no-label, {missing_signal} no-signal")
    _print_stats(df)
    return df


# ---------------------------------------------------------------------------
# Signal loading
# ---------------------------------------------------------------------------

# Module-level cache of open h5py file handles (one per h5 path, per process)
_H5_CACHE: dict = {}


def load_code15_signal(path_token: str) -> np.ndarray:
    """
    Load a CODE-15% signal given a path token of the form:
        "code15::ekg_datasets/code15/raw/exams_part0.h5::42"

    Returns: (12, 5000) float32 in mV, resampled from 400 Hz to 500 Hz.
    Returns zeros on error.

    Thread-safe for single-process use; for DataLoader with num_workers > 0,
    each worker opens its own handles (forked from parent, no sharing).
    """
    from scipy.signal import resample as scipy_resample

    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required: pip install h5py --break-system-packages")

    _EMPTY = np.zeros((12, 5000), dtype=np.float32)

    # Parse token
    try:
        _, h5_part, row_str = path_token.split("::")
        row_idx = int(row_str)
    except Exception:
        return _EMPTY

    # Open h5 file (cached per process)
    h5_path = str(h5_part)
    if h5_path not in _H5_CACHE:
        try:
            _H5_CACHE[h5_path] = h5py.File(h5_path, "r")
        except Exception as e:
            print(f"  ERROR opening {h5_path}: {e}")
            return _EMPTY

    f = _H5_CACHE[h5_path]

    try:
        # tracings shape: (N, 4096, 12)
        raw = f["tracings"][row_idx]          # (4096, 12)
    except Exception as e:
        print(f"  ERROR reading row {row_idx} from {h5_path}: {e}")
        return _EMPTY

    # Transpose to (12, 4096) and cast
    sig = raw.T.astype(np.float32)            # (12, 4096)

    # The trailing zeros in CODE-15% come from padding 4000-sample signals to 4096.
    # Trim to actual signal length before resampling to avoid resampling the zero-pad.
    sig_trimmed = sig[:, :4000]               # (12, 4000) — real 10 s at 400 Hz

    # Resample 400 Hz → 500 Hz: 4000 → 5000 samples (exact ratio 5:4)
    sig_500 = scipy_resample(sig_trimmed, 5000, axis=1)  # (12, 5000)

    # Sanitise
    sig_500 = np.nan_to_num(sig_500, nan=0.0, posinf=0.0, neginf=0.0)
    sig_500 = np.clip(sig_500, -20.0, 20.0)

    return sig_500.astype(np.float32)


# ---------------------------------------------------------------------------
# Load for training (mirrors load_chapman_multilabel interface)
# ---------------------------------------------------------------------------

def load_code15_multilabel(
    codes: list,
    index_path: Path = CODE15_INDEX,
    include_norm: bool = True,
) -> tuple:
    """
    Load CODE-15% index and return (paths, label_matrix) aligned to `codes`.

    Parameters
    ----------
    codes        : list of str — label codes from V3_CODES (26-class list)
    index_path   : path to the index CSV (built by build_code15_index)
    include_norm : if True, records with all 6 pathology labels = 0 are given NORM=1

    Returns
    -------
    paths        : list of str  (path tokens in "code15::...::row" format)
    label_matrix : np.ndarray (N, len(codes))  float32 multi-hot
    """
    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"CODE-15% index not found: {index_path}\n"
            f"Run: python dataset_code15.py --index"
        )

    df = pd.read_csv(index_path)
    code_to_idx = {c: i for i, c in enumerate(codes)}
    n_classes   = len(codes)

    paths  = df["path"].tolist()
    labels = np.zeros((len(df), n_classes), dtype=np.float32)

    for c15_col, v3_code in CODE15_LABEL_MAP.items():
        if c15_col not in df.columns:
            continue
        if v3_code not in code_to_idx:
            continue
        labels[:, code_to_idx[v3_code]] = df[c15_col].values.astype(np.float32)

    # Infer NORM: rows with all pathology labels = 0 are presumed normal
    if include_norm and "NORM" in code_to_idx:
        pathology_cols = [c for c in CODE15_LABELS if c in df.columns]
        if pathology_cols:
            all_normal_mask = (df[pathology_cols].sum(axis=1) == 0).values
            norm_idx = code_to_idx["NORM"]
            labels[all_normal_mask, norm_idx] = 1.0

    # Drop rows with zero-vector labels (shouldn't happen but safety check)
    valid = labels.sum(axis=1) > 0
    n_drop = int((~valid).sum())
    if n_drop:
        paths  = [p for p, v in zip(paths, valid) if v]
        labels = labels[valid]

    n_total = len(paths)
    print(f"  CODE-15%: {n_total} records loaded  ({n_drop} zero-label dropped)")

    for i, code in enumerate(codes):
        n = int(labels[:, i].sum())
        if n > 0:
            print(f"    {code:<8}: {n:>7}")

    return paths, labels


# ---------------------------------------------------------------------------
# Auxiliary feature extraction for CODE-15% (demographics)
# ---------------------------------------------------------------------------

def build_code15_demo_cache(index_path: Path = CODE15_INDEX) -> dict:
    """
    Build a demographics cache {path_token: (sex_float, age_norm)} for CODE-15%.
    sex_float : 1.0 = male, 0.0 = female
    age_norm  : age / 80.0 (consistent with PTB-XL preprocessing)
    """
    if not Path(index_path).exists():
        return {}

    df = pd.read_csv(index_path)
    cache = {}
    for _, row in df.iterrows():
        sex = float(row.get("is_male", 1.0) or 1.0)
        age = float(row.get("age",     50.0) or 50.0)
        cache[row["path"]] = (sex, min(age / 80.0, 1.5))
    return cache


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _print_stats(df: pd.DataFrame):
    print("\n  Label distribution:")
    total = len(df)
    for col in CODE15_LABELS:
        if col in df.columns:
            n   = int(df[col].sum())
            pct = 100 * n / total if total else 0
            print(f"    {col:<8}: {n:>7}  ({pct:5.1f}%)")
    # NORM (inferred)
    if all(c in df.columns for c in CODE15_LABELS):
        n_norm = int((df[CODE15_LABELS].sum(axis=1) == 0).sum())
        pct    = 100 * n_norm / total if total else 0
        print(f"    NORM     : {n_norm:>7}  ({pct:5.1f}%)  [inferred: all pathology=0]")


def print_stats(index_path: Path = CODE15_INDEX):
    if not Path(index_path).exists():
        print(f"Index not found: {index_path}. Run --index first.")
        return
    df = pd.read_csv(index_path)
    print(f"\n  CODE-15% index: {len(df)} records")
    _print_stats(df)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CODE-15% ECG Dataset Integration")
    ap.add_argument("--download", action="store_true", help="Download from Zenodo (~35 GB)")
    ap.add_argument("--index",    action="store_true", help="Build index CSV from downloaded H5 files")
    ap.add_argument("--stats",    action="store_true", help="Print label distribution")
    ap.add_argument("--base",     default=str(CODE15_BASE), help="Local dataset directory")
    args = ap.parse_args()

    base = Path(args.base)

    if args.download:
        download_code15(base)
    if args.index:
        build_code15_index(base)
    if args.stats:
        print_stats(base / "code15_index.csv")
    if not any([args.download, args.index, args.stats]):
        ap.print_help()
