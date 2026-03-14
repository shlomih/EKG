"""
EKG Dataset Explorer
====================
After downloading, use this script to preview records, visualize strips,
and verify your data is clean and ready for the POC pipeline.

Usage:
    python explore_dataset.py                  # interactive summary
    python explore_dataset.py --plot 5         # plot 5 random EKG strips
    python explore_dataset.py --record 00001   # inspect a specific record
"""

import argparse
import ast
import random
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import wfdb.processing

BASE_DIR = Path("./ekg_datasets/ptbxl")

# ─────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────

def load_metadata():
    csv_path = BASE_DIR / "ptbxl_labeled.csv"
    if not csv_path.exists():
        print(f"✗ Metadata not found at {csv_path}")
        print("  Run download_ekg_datasets.py first.")
        return None
    df = pd.read_csv(csv_path, index_col="ecg_id")
    return df


def load_record(record_path: str):
    """Load a WFDB record and return signal + metadata."""
    full_path = str(BASE_DIR / record_path)
    record = wfdb.rdrecord(full_path)
    return record


# ─────────────────────────────────────────────
# Dataset summary
# ─────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("\n" + "═" * 60)
    print("  PTB-XL Dataset Summary")
    print("═" * 60)
    print(f"\n  Total records     : {len(df):,}")
    print(f"  Unique diagnoses  : {df['primary_diagnosis'].nunique()}")

    print(f"\n  Sex distribution:")
    for sex, count in df["sex"].value_counts().items():
        label = "Male" if sex == 0 else "Female"
        print(f"    {label:<10} {count:>6,}  ({count/len(df)*100:.1f}%)")

    print(f"\n  Age distribution:")
    print(f"    Min    : {df['age'].min():.0f}")
    print(f"    Max    : {df['age'].max():.0f}")
    print(f"    Mean   : {df['age'].mean():.1f}")
    print(f"    Median : {df['age'].median():.0f}")

    print(f"\n  Top 15 diagnoses:")
    counts = df["primary_diagnosis"].value_counts()
    for dx, n in counts.head(15).items():
        bar = "█" * int(n / counts.max() * 25)
        print(f"    {dx[:38]:<38} {n:>5}  {bar}")

    print(f"\n  Device types:")
    if "device" in df.columns:
        for device, count in df["device"].value_counts().head(5).items():
            print(f"    {str(device):<30} {count:>5}")

    print(f"\n  Validation split:")
    if "strat_fold" in df.columns:
        for fold, count in df["strat_fold"].value_counts().sort_index().items():
            role = "TEST" if fold == 10 else ("VAL" if fold == 9 else "TRAIN")
            print(f"    Fold {fold:>2}  ({role:<5})  {count:>5} records")


# ─────────────────────────────────────────────
# Visualize EKG strip
# ─────────────────────────────────────────────

def plot_strip(record_path: str, diagnosis: str = "", age: str = "", sex: str = ""):
    """Plot a 12-lead EKG strip in standard clinical layout."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  pip install matplotlib  to enable plotting")
        return

    record = load_record(record_path)
    signal = record.p_signal   # shape: (samples, 12)
    fs = record.fs
    leads = record.sig_name

    # Standard 12-lead layout: 3 rows × 4 columns
    lead_layout = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ]

    fig = plt.figure(figsize=(20, 9), facecolor="#0D1F1E")
    fig.suptitle(
        f"EKG Strip  |  {diagnosis or 'Unknown'}  |  Age: {age}  |  Sex: {sex}",
        color="#00E5B0", fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.3)

    for row_idx, row in enumerate(lead_layout):
        for col_idx, lead_name in enumerate(row):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            if lead_name in leads:
                lead_idx = leads.index(lead_name)
                # Plot 2.5 seconds of signal
                samples = min(int(2.5 * fs), signal.shape[0])
                t = np.arange(samples) / fs
                sig = signal[:samples, lead_idx]

                ax.plot(t, sig, color="#00E5B0", linewidth=0.8, alpha=0.9)
                ax.axhline(y=0, color="#1E3533", linewidth=0.5, linestyle="--")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        color="#3D6662", transform=ax.transAxes)

            ax.set_title(lead_name, color="#7AA8A4", fontsize=9, pad=3)
            ax.set_facecolor("#071312")
            ax.tick_params(colors="#3D6662", labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1E3533")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = Path(f"./ekg_preview_{Path(record_path).stem}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"  ✓ Saved to {out_path.resolve()}")


# ─────────────────────────────────────────────
# Inspect a single record
# ─────────────────────────────────────────────

def inspect_record(ecg_id: int, df: pd.DataFrame):
    row = df.loc[ecg_id]
    record = load_record(row["filename_hr"])

    print(f"\n  Record: ECG #{ecg_id}")
    print(f"  ─────────────────────────────")
    print(f"  Age           : {row['age']:.0f}")
    print(f"  Sex           : {'Male' if row['sex'] == 0 else 'Female'}")
    print(f"  Diagnosis     : {row['primary_diagnosis']}")
    print(f"  SCP codes     : {row['scp_codes']}")
    if "report" in row:
        print(f"  Clinical note : {row['report']}")
    print(f"  Signal shape  : {record.p_signal.shape}  (samples × leads)")
    print(f"  Sample rate   : {record.fs} Hz")
    print(f"  Duration      : {record.sig_len / record.fs:.1f} seconds")
    print(f"  Leads         : {record.sig_name}")

    # Basic interval estimates using NeuroKit2 if available
    try:
        import neurokit2 as nk
        lead_ii_idx = record.sig_name.index("II")
        signal_ii = record.p_signal[:, lead_ii_idx]
        _, info = nk.ecg_peaks(signal_ii, sampling_rate=record.fs)
        hr = nk.ecg_rate(info, sampling_rate=record.fs, desired_length=len(signal_ii))
        print(f"\n  NeuroKit2 measurements (Lead II):")
        print(f"  Heart rate    : {np.mean(hr):.0f} bpm  (mean)")
    except Exception:
        print("\n  (Install neurokit2 for interval measurements: pip install neurokit2)")

    # Plot it
    sex_label = "Male" if row["sex"] == 0 else "Female"
    plot_strip(
        row["filename_hr"],
        diagnosis=row["primary_diagnosis"],
        age=str(int(row["age"])),
        sex=sex_label
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Explore downloaded EKG datasets")
    parser.add_argument("--plot", type=int, default=0,
                        help="Plot N random EKG strips")
    parser.add_argument("--record", type=int, default=None,
                        help="Inspect a specific ECG record by ID")
    parser.add_argument("--diagnosis", type=str, default=None,
                        help="Filter random plots by diagnosis (e.g. 'STEMI')")
    args = parser.parse_args()

    df = load_metadata()
    if df is None:
        return

    if args.record:
        inspect_record(args.record, df)
        return

    print_summary(df)

    if args.plot > 0:
        print(f"\n  Plotting {args.plot} random strips...")
        subset = df
        if args.diagnosis:
            subset = df[df["primary_diagnosis"].str.contains(args.diagnosis, case=False, na=False)]
            print(f"  Filtered to '{args.diagnosis}': {len(subset)} records")
            if len(subset) == 0:
                print(f"\n  ✗ No records match '{args.diagnosis}' in your downloaded data.")
                print(f"\n  Available diagnoses in your dataset:")
                for dx, n in df["primary_diagnosis"].value_counts().items():
                    print(f"    {dx[:50]:<50} ({n} records)")
                print(f"\n  Tip: Download more records to get rare diagnoses like STEMI:")
                print(f"       python download_ekg_datasets.py --dataset ptbxl --small --count 500")
                return

        samples = subset.sample(min(args.plot, len(subset)))
        for ecg_id, row in samples.iterrows():
            sex_label = "Male" if row["sex"] == 0 else "Female"
            plot_strip(
                row["filename_hr"],
                diagnosis=row["primary_diagnosis"],
                age=str(int(row["age"])),
                sex=sex_label
            )

    print("\n  Done. Next step: run  python digitization_pipeline.py  to test strip scanning.\n")


if __name__ == "__main__":
    main()
