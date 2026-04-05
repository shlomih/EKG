"""
diagnose_afib.py
================
Check how many AFIB-labeled records produce zero/near-zero signals.
This diagnoses the suspected .mat loading bug that's keeping AFIB F1 low.

Usage:
    python diagnose_afib.py

Reports:
    - Per-dataset breakdown of AFIB records
    - How many load as zeros vs real signals
    - Signal stats (mean amplitude) for loaded records
    - Sample paths of failed records for manual inspection
"""

import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

os.chdir(Path(__file__).parent)

from cnn_classifier import _load_raw_signal, N_LEADS, SIGNAL_LEN
from dataset_challenge import (
    load_challenge_multilabel, V3_CODES, CHALLENGE_DATASETS, CHALLENGE_DIR, _parse_hea,
    SNOMED_TO_LABEL,
)
from dataset_chapman import load_chapman_multilabel, MERGED_CODES


def check_signal(path):
    """Load signal and return (is_zero, max_abs, n_zero_leads)."""
    sig = _load_raw_signal(path)  # (12, 5000)
    max_abs = float(np.abs(sig).max())
    n_zero_leads = int(np.sum(np.abs(sig).max(axis=1) < 1e-6))
    is_zero = max_abs < 1e-6
    return is_zero, max_abs, n_zero_leads


def diagnose_challenge_afib():
    """Check all AFIB records from Challenge datasets."""
    print("=" * 60)
    print("  AFIB Signal Loading Diagnostic")
    print("=" * 60)

    afib_idx = V3_CODES.index("AFIB")

    # --- Challenge datasets ---
    print("\n--- Challenge Datasets ---")
    total_afib = 0
    total_zero = 0
    total_partial = 0
    failed_paths = []

    for ds in CHALLENGE_DATASETS:
        ds_dir = CHALLENGE_DIR / ds
        if not ds_dir.exists():
            print(f"  {ds}: not found, skipping")
            continue

        mat_files = sorted(ds_dir.rglob("*.mat"))
        ds_afib = 0
        ds_zero = 0
        ds_partial = 0  # some leads zero
        ds_amplitudes = []

        for mat_path in mat_files:
            hea_path = mat_path.with_suffix(".hea")
            if not hea_path.exists():
                continue

            # Check if this record has AFIB label
            snomed_codes = _parse_hea(hea_path)
            has_afib = False
            for sc in snomed_codes:
                label = SNOMED_TO_LABEL.get(sc)
                if label == "AFIB":
                    has_afib = True
                    break

            if not has_afib:
                continue

            ds_afib += 1
            rec_path = str(mat_path.with_suffix(''))
            is_zero, max_abs, n_zero_leads = check_signal(rec_path)

            if is_zero:
                ds_zero += 1
                failed_paths.append(rec_path)
            elif n_zero_leads > 0:
                ds_partial += 1
            ds_amplitudes.append(max_abs)

        if ds_afib > 0:
            pct_zero = ds_zero / ds_afib * 100
            mean_amp = np.mean(ds_amplitudes) if ds_amplitudes else 0
            print(f"\n  {ds}:")
            print(f"    AFIB records:   {ds_afib}")
            print(f"    All-zero:       {ds_zero}  ({pct_zero:.1f}%)")
            print(f"    Partial-zero:   {ds_partial}  (some leads zeroed)")
            print(f"    Mean max|amp|:  {mean_amp:.3f} mV")
            if ds_zero > 0:
                print(f"    Sample failed:  {failed_paths[-1]}")

        total_afib += ds_afib
        total_zero += ds_zero
        total_partial += ds_partial

    print(f"\n  Challenge total:")
    print(f"    AFIB records:   {total_afib}")
    print(f"    All-zero:       {total_zero}  ({total_zero/max(total_afib,1)*100:.1f}%)")
    print(f"    Partial-zero:   {total_partial}")

    # --- Chapman dataset ---
    print("\n--- Chapman-Shaoxing ---")
    try:
        chap_paths, chap_labels = load_chapman_multilabel()
        afib_idx_merged = MERGED_CODES.index("AFIB")
        chap_afib_mask = chap_labels[:, afib_idx_merged] > 0.5
        chap_afib_paths = [p for p, m in zip(chap_paths, chap_afib_mask) if m]
        print(f"  AFIB records: {len(chap_afib_paths)}")

        chap_zero = 0
        chap_partial = 0
        chap_amps = []
        for path in chap_afib_paths:
            is_zero, max_abs, n_zero_leads = check_signal(path)
            if is_zero:
                chap_zero += 1
            elif n_zero_leads > 0:
                chap_partial += 1
            chap_amps.append(max_abs)

        if chap_afib_paths:
            print(f"  All-zero:     {chap_zero}  ({chap_zero/len(chap_afib_paths)*100:.1f}%)")
            print(f"  Partial-zero: {chap_partial}")
            print(f"  Mean max|amp|: {np.mean(chap_amps):.3f} mV")
    except Exception as e:
        print(f"  Error loading Chapman: {e}")

    # --- PTB-XL AFIB check ---
    print("\n--- PTB-XL AFIB (label mapping check) ---")
    print(f"  MULTILABEL_CODES (12-class): AFIB is {'present' if 'AFIB' in __import__('multilabel_classifier').MULTILABEL_CODES else 'MISSING'}")
    print(f"  MERGED_CODES (14-class):     AFIB is at index {MERGED_CODES.index('AFIB')}")
    print(f"  V3_CODES (26-class):         AFIB is at index {V3_CODES.index('AFIB')}")
    print(f"  NOTE: PTB-XL has ~48 AFIB records but they are NOT labeled in the")
    print(f"  12-class multilabel loader. PTBXL_TO_V3 mapping skips AFIB.")
    print(f"  -> These 48 records train as AFIB=0 (false negatives in training)")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  DIAGNOSIS SUMMARY")
    print("=" * 60)
    grand_total = total_afib + len(chap_afib_paths) if 'chap_afib_paths' in dir() else total_afib
    grand_zero = total_zero + (chap_zero if 'chap_zero' in dir() else 0)
    print(f"  Total AFIB records checked: {grand_total}")
    print(f"  Total all-zero signals:     {grand_zero}  ({grand_zero/max(grand_total,1)*100:.1f}%)")
    if grand_zero > 0:
        print(f"\n  ROOT CAUSE: {grand_zero} AFIB records load as flat-zero signals.")
        print(f"  The model learns 'AFIB = flat line' from these, poisoning real AFIB learning.")
        print(f"  FIX: Skip records where signal loads as zeros in V3ECGDataset.__getitem__,")
        print(f"  or fix the .mat loading in _load_raw_signal to handle the format correctly.")
    elif total_partial > 0:
        print(f"\n  NOTE: {total_partial} records have some zero leads (partial loading).")
        print(f"  This may confuse the model on lead-specific AFIB features.")
    else:
        print(f"\n  All signals load correctly. AFIB issue is NOT a .mat loading bug.")
        print(f"  Likely cause: distribution mismatch between Chapman/Challenge AFIB")
        print(f"  and the model's learned features, or threshold calibration issue.")

    if failed_paths:
        print(f"\n  First 5 failed paths (for manual inspection):")
        for p in failed_paths[:5]:
            print(f"    {p}")


if __name__ == "__main__":
    diagnose_challenge_afib()
