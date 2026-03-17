"""
EKG Dataset Downloader
======================
Downloads the three recommended datasets for the EKG Intelligence Platform POC:

  1. PTB-XL          — 21,837 labeled 12-lead clinical ECGs (PhysioNet)
  2. CPSC 2018       — 6,877 labeled ECGs from Chinese competition (PhysioNet)
  3. Georgia ECG     — 10,344 labeled ECGs (PhysioNet)

Requirements:
    pip install wfdb requests tqdm pandas numpy

Usage:
    python download_ekg_datasets.py                      # downloads all datasets
    python download_ekg_datasets.py --dataset ptbxl      # downloads PTB-XL only
    python download_ekg_datasets.py --dataset cpsc        # downloads CPSC 2018 only
    python download_ekg_datasets.py --dataset georgia     # downloads Georgia only
    python download_ekg_datasets.py --small               # downloads first 100 records only (quick test)
"""

import os
import argparse
import time
import requests
import pandas as pd
import numpy as np
import wfdb
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

BASE_DIR = Path("./ekg_datasets")

DATASETS = {
    "ptbxl": {
        "name": "PTB-XL",
        "description": "21,837 clinical 12-lead ECGs with expert labels. The gold standard benchmark.",
        "physionet_db": "ptb-xl",
        "output_dir": BASE_DIR / "ptbxl",
        "label_file": "ptbxl_database.csv",
        "sample_rate": 500,   # Hz — also available at 100Hz
        "leads": 12,
    },
    "cpsc": {
        "name": "CPSC 2018",
        "description": "6,877 12-lead ECGs from the Chinese Physiological Signal Challenge.",
        "physionet_db": "challenge-2021/1.0.3",
        "output_dir": BASE_DIR / "cpsc",
        "sample_rate": 500,
        "leads": 12,
    },
    "georgia": {
        "name": "Georgia ECG",
        "description": "10,344 12-lead ECGs with SNOMED-CT diagnostic labels.",
        "physionet_db": "challenge-2021/1.0.3",
        "output_dir": BASE_DIR / "georgia",
        "sample_rate": 500,
        "leads": 12,
    },
}

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def print_header():
    print("\n" + "═" * 60)
    print("  EKG Intelligence Platform — Dataset Downloader")
    print("═" * 60)

def print_section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def format_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"

# ─────────────────────────────────────────────
# PTB-XL Downloader (primary recommended dataset)
# ─────────────────────────────────────────────

def fetch_file(url: str, dest: Path) -> bool:
    """Download a single file via requests with retry logic."""
    import requests
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30, stream=True)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                return True
            elif r.status_code == 404:
                return False  # file genuinely missing, don't retry
        except Exception:
            time.sleep(2 ** attempt)
    return False


def download_ptbxl(small: bool = False, count: int = 100):
    """
    Downloads PTB-XL from PhysioNet using direct HTTPS requests.
    Uses https://physionet.org/files/ptb-xl/1.0.3/ directly,
    bypassing the wfdb.dl_files version-duplication bug.
    """
    PTBXL_BASE = "https://physionet.org/files/ptb-xl/1.0.3"

    cfg = DATASETS["ptbxl"]
    out_dir = cfg["output_dir"]
    ensure_dir(out_dir)

    print_section(f"Downloading {cfg['name']}")
    print(f"  Description : {cfg['description']}")
    print(f"  Output dir  : {out_dir.resolve()}")
    print(f"  Mode        : {'SMALL (' + str(count) + ' records)' if small else 'FULL dataset'}")

    # ── Step 1: Metadata CSVs ──────────────────────────────────────
    print("\n  [1/3] Fetching metadata CSV...")
    import requests
    try:
        for fname in ["ptbxl_database.csv", "scp_statements.csv"]:
            dest = out_dir / fname
            if dest.exists():
                print(f"  ↩ {fname} already exists, skipping")
                continue
            url = f"{PTBXL_BASE}/{fname}"
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"  ✓ {fname} downloaded ({len(r.content) / 1024:.0f} KB)")
    except Exception as e:
        print(f"  ✗ Metadata download failed: {e}")
        print(f"  URL tried: {PTBXL_BASE}/ptbxl_database.csv")
        print("  → Make sure you have internet access and try again.")
        return False

    # ── Step 2: Parse metadata ─────────────────────────────────────
    print("\n  [2/3] Parsing metadata...")
    try:
        df = pd.read_csv(out_dir / "ptbxl_database.csv", index_col="ecg_id")
        scp_df = pd.read_csv(out_dir / "scp_statements.csv", index_col=0)

        import ast
        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

        diagnostic_map = scp_df[scp_df["diagnostic"] == 1]["description"].to_dict()
        df["primary_diagnosis"] = df["scp_codes"].apply(
            lambda x: next(
                (diagnostic_map.get(k, k) for k, v in sorted(x.items(), key=lambda i: -i[1]) if k in diagnostic_map),
                "Unknown"
            )
        )

        record_list = df["filename_hr"].tolist()
        if small:
            record_list = record_list[:count]
            df_save = df.iloc[:count]
        else:
            df_save = df

        df_save.to_csv(out_dir / "ptbxl_labeled.csv")
        label_counts = df_save["primary_diagnosis"].value_counts()
        label_counts.to_csv(out_dir / "label_distribution.csv")

        print(f"  ✓ {len(df_save):,} records to download")
        print(f"  ✓ {len(label_counts)} unique diagnoses")
        print(f"\n  Top 10 diagnoses:")
        for diagnosis, count in label_counts.head(10).items():
            bar = "█" * int(count / label_counts.max() * 20)
            print(f"    {diagnosis[:35]:<35} {count:>5}  {bar}")

    except Exception as e:
        print(f"  ✗ Metadata parsing failed: {e}")
        return False

    # ── Step 3: Download ECG records ───────────────────────────────
    print(f"\n  [3/3] Downloading {len(record_list):,} ECG records...")
    est_size = "~130MB" if small else "~2.5GB"
    print(f"  Estimated size : {est_size}")

    success_count = 0
    fail_count = 0
    failed_records = []

    def download_record(record_path):
        record_dir = out_dir / Path(record_path).parent
        ensure_dir(record_dir)
        for ext in [".hea", ".dat"]:
            dest_file = out_dir / f"{record_path}{ext}"
            if dest_file.exists():
                continue
            url = f"{PTBXL_BASE}/{record_path}{ext}"
            if not fetch_file(url, dest_file):
                return record_path, False
        return record_path, True

    with tqdm(total=len(record_list), unit="rec", ncols=72, colour="green") as pbar:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(download_record, rp): rp for rp in record_list}
            for future in as_completed(futures):
                record_path, ok = future.result()
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
                    failed_records.append({"record": record_path})
                pbar.update(1)

    print(f"\n  ✓ Downloaded : {success_count:,} records")
    if fail_count > 0:
        print(f"  ✗ Failed     : {fail_count} records")
        pd.DataFrame(failed_records).to_csv(out_dir / "download_errors.csv", index=False)
        print(f"  → Retry by running the script again (already-downloaded files are skipped)")

    # ── Step 4: Verify a sample ────────────────────────────────────
    print("\n  Verifying a sample record...")
    try:
        sample_rel = df["filename_hr"].iloc[0]
        sample_path = str(out_dir / sample_rel)
        record = wfdb.rdrecord(sample_path)
        print(f"  ✓ Sample record loaded successfully")
        print(f"    Leads    : {record.sig_name}")
        print(f"    Duration : {record.sig_len / record.fs:.1f}s  |  Rate: {record.fs}Hz  |  Shape: {record.p_signal.shape}")
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")

    print(f"\n  ✅ PTB-XL download complete → {out_dir.resolve()}")
    return True


# ─────────────────────────────────────────────
# CPSC 2018 Downloader
# ─────────────────────────────────────────────

def download_cpsc(small: bool = False, count: int = 100):
    """
    Downloads CPSC 2018 subset from PhysioNet Challenge 2021.
    """
    cfg = DATASETS["cpsc"]
    out_dir = cfg["output_dir"]
    ensure_dir(out_dir)

    print_section(f"Downloading {cfg['name']}")
    print(f"  Description : {cfg['description']}")
    print(f"  Output dir  : {out_dir.resolve()}")

    try:
        print("\n  Fetching CPSC records from PhysioNet Challenge 2021...")
        # List available files in the CPSC subfolder
        records = wfdb.get_record_list(cfg["physionet_db"])
        cpsc_records = [r for r in records if "CPSC" in r.upper()]

        if small:
            cpsc_records = cpsc_records[:count]

        print(f"  Found {len(cpsc_records):,} CPSC records")

        success = 0
        with tqdm(total=len(cpsc_records), unit="record", ncols=70, colour="cyan") as pbar:
            for record in cpsc_records:
                try:
                    wfdb.dl_files(
                        db=cfg["physionet_db"],
                        dl_dir=str(out_dir),
                        files=[f"{record}.hea", f"{record}.mat"],
                    )
                    success += 1
                except Exception:
                    pass
                pbar.update(1)
                time.sleep(0.05)

        print(f"\n  ✅ CPSC 2018 download complete — {success:,} records → {out_dir.resolve()}")
        return True

    except Exception as e:
        print(f"  ✗ CPSC download failed: {e}")
        print("  Tip: Try accessing via https://physionet.org/content/challenge-2021/")
        return False


# ─────────────────────────────────────────────
# Georgia ECG Downloader
# ─────────────────────────────────────────────

def download_georgia(small: bool = False, count: int = 100):
    """
    Downloads Georgia ECG dataset from PhysioNet Challenge 2021.
    """
    cfg = DATASETS["georgia"]
    out_dir = cfg["output_dir"]
    ensure_dir(out_dir)

    print_section(f"Downloading {cfg['name']}")
    print(f"  Description : {cfg['description']}")
    print(f"  Output dir  : {out_dir.resolve()}")

    try:
        print("\n  Fetching Georgia records from PhysioNet Challenge 2021...")
        records = wfdb.get_record_list(cfg["physionet_db"])
        georgia_records = [r for r in records if "E" in r and r.startswith("E")]

        if small:
            georgia_records = georgia_records[:count]

        print(f"  Found {len(georgia_records):,} Georgia records")

        success = 0
        with tqdm(total=len(georgia_records), unit="record", ncols=70, colour="yellow") as pbar:
            for record in georgia_records:
                try:
                    wfdb.dl_files(
                        db=cfg["physionet_db"],
                        dl_dir=str(out_dir),
                        files=[f"{record}.hea", f"{record}.mat"],
                    )
                    success += 1
                except Exception:
                    pass
                pbar.update(1)
                time.sleep(0.05)

        print(f"\n  ✅ Georgia ECG download complete — {success:,} records → {out_dir.resolve()}")
        return True

    except Exception as e:
        print(f"  ✗ Georgia download failed: {e}")
        return False


# ─────────────────────────────────────────────
# Post-download summary
# ─────────────────────────────────────────────

def print_summary():
    print_section("Dataset Summary")
    total_size = 0
    for key, cfg in DATASETS.items():
        out_dir = cfg["output_dir"]
        if out_dir.exists():
            size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
            count = sum(1 for f in out_dir.rglob("*.hea"))
            total_size += size
            print(f"  {cfg['name']:<15} {count:>6} records   {format_size(size):>10}   → {out_dir}")
        else:
            print(f"  {cfg['name']:<15} not downloaded")

    print(f"\n  Total disk used: {format_size(total_size)}")
    print(f"\n  Next step: Run  python explore_dataset.py  to preview records")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download EKG datasets for the EKG Intelligence Platform POC"
    )
    parser.add_argument(
        "--dataset",
        choices=["ptbxl", "cpsc", "georgia", "all"],
        default="all",
        help="Which dataset to download (default: all)"
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Download only first N records per dataset (fast test run)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of records to download in --small mode (default: 100). Use 500+ to get rare diagnoses like STEMI."
    )
    args = parser.parse_args()

    print_header()

    if args.small:
        print(f"\n  ⚡ SMALL MODE: downloading {args.count} records per dataset")

    print("\n  Datasets to download:")
    datasets_to_run = [args.dataset] if args.dataset != "all" else list(DATASETS.keys())
    for key in datasets_to_run:
        cfg = DATASETS[key]
        print(f"  • {cfg['name']} — {cfg['description']}")

    print("\n  Requirements check...")
    try:
        import wfdb, pandas, numpy, tqdm, requests
        print("  ✓ All dependencies installed")
    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        print("  Run: pip install wfdb pandas numpy tqdm requests")
        return

    # Run downloads
    if "ptbxl" in datasets_to_run:
        download_ptbxl(small=args.small, count=args.count)

    if "cpsc" in datasets_to_run:
        download_cpsc(small=args.small, count=args.count)

    if "georgia" in datasets_to_run:
        download_georgia(small=args.small, count=args.count)

    print_summary()


if __name__ == "__main__":
    main()
