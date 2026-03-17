"""
dataset_pipeline.py
===================
Multi-dataset ECG integration pipeline.

Downloads, maps labels, and merges multiple public 12-lead ECG databases
into a unified training set compatible with our CNN classifier.

Supported datasets:
  1. PTB-XL (already local)          - 21,837 records
  2. CPSC 2018                       - ~6,877 records
  3. Chapman-Shaoxing (CSE)          - ~10,646 records
  4. Georgia 12-Lead (G12EC)         - ~10,344 records
  5. Ningbo                          - ~34,905 records (PhysioNet credentialed)
  6. PTB (original)                  - ~549 records

All datasets use WFDB format on PhysioNet and label with SNOMED-CT codes.
The CinC 2020/2021 challenge defined standard SNOMED -> diagnostic mappings
that we reuse here.

Usage:
    python dataset_pipeline.py --download    # Download all datasets
    python dataset_pipeline.py --status      # Show dataset status
    python dataset_pipeline.py --prepare     # Prepare unified index
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Base directory for all datasets
DATASETS_DIR = Path("ekg_datasets")

# Our 5 superclasses
SUPERCLASS_LABELS = ["NORM", "MI", "STTC", "HYP", "CD"]

# ----------------------------------------------------------------
# SNOMED-CT code -> Superclass mapping
# Based on PhysioNet/CinC 2020-2021 challenge + PTB-XL definitions
# ----------------------------------------------------------------

SNOMED_TO_SUPERCLASS = {
    # === NORM (Normal sinus rhythm / normal variants) ===
    "426783006": "NORM",     # Sinus rhythm
    "427393009": "NORM",     # Sinus arrhythmia (normal variant)

    # === MI (Myocardial Infarction / Ischemia) ===
    # Scored codes
    "164917005": "MI",       # Q wave abnormal (QAb) -- 2076 records
    "59931005":  "MI",       # T wave inversion (TInv) -- 3989 (ischemic context)
    # Unscored codes
    "164865005": "MI",       # Myocardial infarction -- 6144
    "164861001": "MI",       # Myocardial ischemia -- 2559
    "57054005":  "MI",       # Acute myocardial infarction -- 55
    "413444003": "MI",       # Acute myocardial ischemia -- 2
    "54329005":  "MI",       # Anterior myocardial infarction -- 473
    "164867002": "MI",       # Old myocardial infarction -- 1168
    "426434006": "MI",       # Anterior ischemia -- 325 (unscored name: inferior ischemia)
    "425419005": "MI",       # Inferior ischaemia -- 670
    "425623009": "MI",       # Lateral ischaemia -- 1045
    "428750005": "MI",       # Nonspecific ST-T abnormality -- 4712 (ischemic context)
    "413844008": "MI",       # Chronic myocardial ischemia -- 161
    "53741008":  "MI",       # Coronary heart disease -- 37
    "418818005": "MI",       # Brugada -- 5
    "75532003":  "MI",       # Ventricular escape beat (MI marker context)
    "70422006":  "MI",       # Acute subendocardial infarction

    # === STTC (ST/T Changes) ===
    # Scored codes
    "164934002": "STTC",     # T wave abnormal (TAb) -- 11716
    "428417006": "STTC",     # Early repolarization -- 506 (ST change)
    "251146004": "STTC",     # Low QRS voltages -- 1599 (LQRSV)
    "365413008": "STTC",     # Poor R wave progression (PRWP) -- 638
    "111975006": "STTC",     # Prolonged QT interval (LQT) -- 1907
    "164947007": "STTC",     # Prolonged PR interval (LPR) -- 392
    # Unscored codes
    "164930006": "STTC",     # ST interval abnormal -- 2276
    "164931005": "STTC",     # ST elevation -- 628
    "429622005": "STTC",     # ST depression -- 3645
    "55930002":  "STTC",     # ST changes -- 5009
    "164937009": "STTC",     # U wave abnormal -- 137
    "164942001": "STTC",     # FQRS wave -- 3
    "164951009": "STTC",     # Abnormal QRS -- 3389
    "164921003": "STTC",     # R wave abnormal -- 11
    "77867006":  "STTC",     # Decreased QT interval -- 3
    "251205003": "STTC",     # Prolonged P wave -- 106
    "164912004": "STTC",     # P wave change -- 142

    # === HYP (Hypertrophy) ===
    # Unscored codes (HYP has NO scored codes in CinC 2021, all unscored)
    "164873001": "HYP",      # Left ventricular hypertrophy -- 4406
    "55827005":  "HYP",      # Left ventricular high voltage -- 5401
    "89792004":  "HYP",      # Right ventricular hypertrophy -- 342
    "67741000119109": "HYP", # Left atrial enlargement -- 1299
    "446813000": "HYP",      # Left atrial hypertrophy -- 48
    "253352002": "HYP",      # Left atrial abnormality -- 72
    "446358003": "HYP",      # Right atrial hypertrophy -- 153
    "67751000119106": "HYP", # Right atrial high voltage -- 36
    "253339007": "HYP",      # Right atrial abnormality -- 14
    "195126007": "HYP",      # Atrial hypertrophy -- 62
    "266249003": "HYP",      # Ventricular hypertrophy -- 119

    # === CD (Conduction Disturbance) ===
    # Scored codes
    "164889003": "CD",       # Atrial fibrillation (AF) -- 5255
    "164890007": "CD",       # Atrial flutter (AFL) -- 8374
    "6374002":   "CD",       # Bundle branch block (BBB) -- 522
    "426627000": "CD",       # Bradycardia -- 295
    "733534002": "CD",       # Complete LBBB (CLBBB) -- 213
    "713427006": "CD",       # Complete RBBB (CRBBB) -- 1779
    "270492004": "CD",       # 1st degree AV block (IAVB) -- 3534
    "713426002": "CD",       # Incomplete RBBB (IRBBB) -- 1857
    "39732003":  "CD",       # Left axis deviation (LAD) -- 7631
    "445118002": "CD",       # Left anterior fascicular block (LAnFB) -- 2186
    "164909002": "CD",       # Left bundle branch block (LBBB) -- 1281
    "698252002": "CD",       # Nonspecific IVCD (NSIVCB) -- 1768
    "284470004": "CD",       # Premature atrial contraction (PAC) -- 3041
    "10370003":  "CD",       # Pacing rhythm (PR) -- 1481
    "427172004": "CD",       # Premature ventricular contractions (PVC) -- 1279
    "47665007":  "CD",       # Right axis deviation (RAD) -- 1280
    "59118001":  "CD",       # Right bundle branch block (RBBB) -- 3051
    "426177001": "CD",       # Sinus bradycardia (SB) -- 18918
    "427084000": "CD",       # Sinus tachycardia (STach) -- 9657
    "63593006":  "CD",       # Supraventricular premature beats (SVPB) -- 224
    "17338001":  "CD",       # Ventricular premature beats (VPB) -- 659
    # Unscored codes
    "233917008": "CD",       # AV block -- 323
    "27885002":  "CD",       # Complete heart block -- 127
    "195042002": "CD",       # 2nd degree AV block -- 124
    "426183003": "CD",       # Mobitz type II -- 7
    "54016002":  "CD",       # Mobitz type I (Wenckebach) -- 34
    "251120003": "CD",       # Incomplete LBBB -- 211
    "445211001": "CD",       # Left posterior fascicular block -- 207
    "164884008": "CD",       # Ventricular ectopics -- 1944
    "713422000": "CD",       # Atrial tachycardia -- 340
    "74390002":  "CD",       # Wolff-Parkinson-White -- 160
    "195060002": "CD",       # Ventricular pre-excitation -- 20
    "426761007": "CD",       # Supraventricular tachycardia -- 787
    "29320008":  "CD",       # AV junctional rhythm -- 145
    "426995002": "CD",       # Junctional escape -- 84
    "426648003": "CD",       # Junctional tachycardia -- 30
    "106068003": "CD",       # Atrial rhythm -- 215
    "195080001": "CD",       # AF and flutter -- 41
    "426664006": "CD",       # Accelerated junctional rhythm -- 31
    "49578007":  "CD",       # Shortened PR interval -- 28
    "195101003": "CD",       # Wandering atrial pacemaker -- 9
    "13640000":  "CD",       # Fusion beats -- 123
    "164896001": "CD",       # Ventricular fibrillation -- 97
    "164895002": "CD",       # Ventricular tachycardia -- 12
    "425856008": "CD",       # Paroxysmal ventricular tachycardia -- 124
    "81898007":  "CD",       # Ventricular escape rhythm -- 98
    "49260003":  "CD",       # Idioventricular rhythm -- 2
    "61277005":  "CD",       # Accelerated idioventricular rhythm -- 14
    "251268003": "CD",       # Atrial pacing pattern -- 52
    "251266004": "CD",       # Ventricular pacing pattern -- 46
    "74615001":  "CD",       # Brady-tachy syndrome -- 2
    "65778007":  "CD",       # Sinoatrial block -- 14
    "5609005":   "CD",       # Sinus arrest -- 33
    "60423000":  "CD",       # Sinus node dysfunction -- 2
    "11157007":  "CD",       # Ventricular bigeminy -- 101
    "251180001": "CD",       # Ventricular trigeminy -- 37
    "251182009": "CD",       # Paired ventricular premature complexes -- 23
    "251168009": "CD",       # Supraventricular bigeminy -- 1
    "251173003": "CD",       # Atrial bigeminy -- 6
    "251170000": "CD",       # Blocked PAC -- 67
    "251164006": "CD",       # Junctional premature complex -- 13
    "251187003": "CD",       # Atrial escape beat -- 17
    "233892002": "CD",       # Accelerated atrial escape rhythm -- 16
}

# Extended mapping: text-based labels -> superclass (for datasets without SNOMED)
TEXT_TO_SUPERCLASS = {
    # Normal
    "normal": "NORM",
    "normal ecg": "NORM",
    "sinus rhythm": "NORM",
    "normal sinus rhythm": "NORM",

    # MI
    "myocardial infarction": "MI",
    "anterior myocardial infarction": "MI",
    "inferior myocardial infarction": "MI",
    "lateral myocardial infarction": "MI",
    "old myocardial infarction": "MI",
    "acute myocardial infarction": "MI",
    "myocardial ischemia": "MI",
    "anterior ischemia": "MI",
    "inferior ischemia": "MI",

    # STTC
    "st-t change": "STTC",
    "st elevation": "STTC",
    "st depression": "STTC",
    "t-wave abnormality": "STTC",
    "t wave inversion": "STTC",
    "nonspecific st-t": "STTC",
    "st segment abnormal": "STTC",

    # HYP
    "left ventricular hypertrophy": "HYP",
    "right ventricular hypertrophy": "HYP",
    "lvh": "HYP",
    "rvh": "HYP",
    "left atrial enlargement": "HYP",
    "right atrial enlargement": "HYP",
    "biventricular hypertrophy": "HYP",
    "hypertrophy": "HYP",

    # CD
    "left bundle branch block": "CD",
    "right bundle branch block": "CD",
    "lbbb": "CD",
    "rbbb": "CD",
    "first degree av block": "CD",
    "second degree av block": "CD",
    "third degree av block": "CD",
    "complete heart block": "CD",
    "atrial fibrillation": "CD",
    "atrial flutter": "CD",
    "wpw": "CD",
    "wolff-parkinson-white": "CD",
    "sinus bradycardia": "CD",
    "sinus tachycardia": "CD",
    "conduction disturbance": "CD",
    "av block": "CD",
    "fascicular block": "CD",
    "left anterior fascicular block": "CD",
    "left posterior fascicular block": "CD",
    "incomplete rbbb": "CD",
    "incomplete lbbb": "CD",
    "intraventricular conduction delay": "CD",
}


# ----------------------------------------------------------------
# Dataset configurations
# ----------------------------------------------------------------

DATASET_CONFIGS = {
    "ptbxl": {
        "name": "PTB-XL",
        "description": "PTB-XL: 21,837 clinical 12-lead ECGs (PhysioNet)",
        "dir": DATASETS_DIR / "ptbxl",
        "cinc_base": None,  # Uses native format, not CinC 2020
        "expected_records": 21837,
        "fs": 500,
        "signal_len": 5000,
        "format": "ptbxl_native",
        "credentialed": False,
    },
    "cpsc2018": {
        "name": "CPSC 2018",
        "description": "China Physiological Signal Challenge 2018: ~6,877 12-lead ECGs",
        "dir": DATASETS_DIR / "cpsc2018",
        "cinc_base": "https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/",
        "expected_records": 6877,
        "fs": 500,
        "signal_len": None,
        "format": "cinc2020",
        "credentialed": False,
    },
    "cpsc2018_extra": {
        "name": "CPSC 2018 Extra",
        "description": "CPSC 2018 Extra training: ~3,453 12-lead ECGs",
        "dir": DATASETS_DIR / "cpsc2018_extra",
        "cinc_base": "https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018_extra/",
        "expected_records": 3453,
        "fs": 500,
        "signal_len": None,
        "format": "cinc2020",
        "credentialed": False,
    },
    "georgia": {
        "name": "Georgia 12-Lead",
        "description": "Georgia 12-Lead ECG Challenge Database: ~10,344 records",
        "dir": DATASETS_DIR / "georgia",
        "cinc_base": "https://physionet.org/files/challenge-2020/1.0.2/training/georgia/",
        "expected_records": 10344,
        "fs": 500,
        "signal_len": 5000,
        "format": "cinc2020",
        "credentialed": False,
    },
    "st_petersburg": {
        "name": "St Petersburg INCART",
        "description": "St Petersburg INCART 12-lead Arrhythmia DB: ~75 records",
        "dir": DATASETS_DIR / "st_petersburg",
        "cinc_base": "https://physionet.org/files/challenge-2020/1.0.2/training/st_petersburg_incart/",
        "expected_records": 75,
        "fs": 257,  # Needs resampling
        "signal_len": None,
        "format": "cinc2020",
        "credentialed": False,
    },
    "ptb_original": {
        "name": "PTB Diagnostic",
        "description": "Original PTB Diagnostic ECG Database: ~516 records",
        "dir": DATASETS_DIR / "ptb",
        "cinc_base": "https://physionet.org/files/challenge-2020/1.0.2/training/ptb/",
        "expected_records": 516,
        "fs": 1000,  # Needs resampling
        "signal_len": None,
        "format": "cinc2020",
        "credentialed": False,
    },
    "ningbo": {
        "name": "Ningbo",
        "description": "Ningbo First Hospital: ~34,905 12-lead ECGs",
        "dir": DATASETS_DIR / "ningbo",
        "cinc_base": None,  # Separate PhysioNet project, credentialed
        "expected_records": 34905,
        "fs": 500,
        "signal_len": 5000,
        "format": "cinc2020",
        "credentialed": True,
    },
    "chapman_shaoxing": {
        "name": "Chapman-Shaoxing",
        "description": "Chapman University & Shaoxing Hospital: ~10,646 12-lead ECGs",
        "dir": DATASETS_DIR / "chapman_shaoxing",
        "cinc_base": None,  # Separate PhysioNet project, may need credentialed
        "expected_records": 10646,
        "fs": 500,
        "signal_len": 5000,
        "format": "cinc2020",
        "credentialed": True,
    },
}


# ----------------------------------------------------------------
# Download functions
# ----------------------------------------------------------------

def _crawl_physionet_dir(base_url, session=None):
    """
    Recursively crawl a PhysioNet directory listing.
    Returns list of (relative_path, full_url) for all .hea and .mat/.dat files.
    """
    import re
    if session is None:
        import requests
        session = requests.Session()

    files = []
    try:
        r = session.get(base_url, timeout=30)
        if r.status_code != 200:
            return files
    except Exception:
        return files

    # Find all links in directory listing
    links = re.findall(r'href="([^"]+)"', r.text)

    for link in links:
        if link == "../":
            continue
        if link.endswith("/"):
            # Recurse into subdirectory
            sub_files = _crawl_physionet_dir(base_url + link, session)
            for rel_path, url in sub_files:
                files.append((link + rel_path, url))
        elif link.endswith((".hea", ".mat", ".dat")):
            files.append((link, base_url + link))

    return files


def _download_physionet_dataset(config, max_workers=16):
    """
    Download a CinC 2020 format dataset from PhysioNet.
    Crawls the directory structure and downloads .hea + .mat/.dat files.
    Uses parallel threads for speed.
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    cinc_base = config.get("cinc_base")
    if not cinc_base:
        print(f"  {config['name']}: no download URL configured")
        return False

    dest = config["dir"]
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading {config['name']}")
    print(f"  From: {cinc_base}")
    print(f"  To:   {dest}")

    # Crawl directory to find all files
    print(f"  Scanning directory structure...")
    session = requests.Session()
    all_files = _crawl_physionet_dir(cinc_base, session)

    if not all_files:
        print(f"  No files found at {cinc_base}")
        return False

    hea_count = sum(1 for f, _ in all_files if f.endswith(".hea"))
    print(f"  Found {hea_count} records ({len(all_files)} files total)")
    print(f"  Downloading with {max_workers} parallel threads...")

    # Filter out already-downloaded files
    to_download = []
    skipped = 0
    for rel_path, url in all_files:
        local_path = dest / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() and local_path.stat().st_size > 0:
            skipped += 1
        else:
            to_download.append((rel_path, url, local_path))

    if skipped > 0:
        print(f"  Skipping {skipped} already-downloaded files")

    if not to_download:
        print(f"  All files already downloaded!")
        return True

    # Thread-safe counters
    lock = threading.Lock()
    counters = {"downloaded": 0, "errors": 0}

    def download_one(args):
        rel_path, url, local_path = args
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                with lock:
                    counters["downloaded"] += 1
                    total = counters["downloaded"] + counters["errors"]
                    if total % 1000 == 0:
                        print(f"    Progress: {total + skipped}/{len(all_files)} "
                              f"(new: {counters['downloaded']}, errors: {counters['errors']})")
                return True
            else:
                with lock:
                    counters["errors"] += 1
                return False
        except Exception:
            with lock:
                counters["errors"] += 1
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_one, args) for args in to_download]
        for future in as_completed(futures):
            pass  # Results tracked in counters

    print(f"  Done: {counters['downloaded']} downloaded, "
          f"{skipped} cached, {counters['errors']} errors")
    return (counters["downloaded"] + skipped) > 0


# ----------------------------------------------------------------
# Label extraction functions (per dataset format)
# ----------------------------------------------------------------

def _extract_snomed_from_header(header_path):
    """
    Extract SNOMED-CT codes from a WFDB .hea file.
    CinC 2020/2021 format stores them as: #Dx: code1,code2,...
    """
    codes = []
    try:
        with open(header_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#Dx:") or line.startswith("# Dx:"):
                    # Extract codes after "Dx:"
                    dx_part = line.split(":", 1)[1].strip()
                    for code in dx_part.split(","):
                        code = code.strip()
                        if code:
                            codes.append(code)
    except Exception:
        pass
    return codes


def _snomed_codes_to_superclass(codes):
    """Map a list of SNOMED-CT codes to a primary superclass."""
    found = []
    for code in codes:
        sc = SNOMED_TO_SUPERCLASS.get(code)
        if sc:
            found.append(sc)

    if not found:
        return None

    # Priority: MI > STTC > HYP > CD > NORM
    # (If someone has MI + NORM, classify as MI)
    priority = {"MI": 0, "STTC": 1, "HYP": 2, "CD": 3, "NORM": 4}
    found.sort(key=lambda x: priority.get(x, 99))
    return found[0]


def _index_cinc2020_dataset(config):
    """
    Index a CinC 2020 format dataset.
    Returns list of (record_path, superclass_label, snomed_codes)
    """
    base = config["dir"]
    records = []

    # Find all .hea files
    hea_files = sorted(base.rglob("*.hea"))
    print(f"  Found {len(hea_files)} header files in {config['name']}")

    for hea_path in hea_files:
        # Extract SNOMED codes from header
        codes = _extract_snomed_from_header(str(hea_path))
        if not codes:
            continue

        superclass = _snomed_codes_to_superclass(codes)
        if superclass is None:
            continue

        # Record path (without extension) for wfdb
        rec_path = str(hea_path).replace(".hea", "")

        # Verify .dat file exists
        dat_path = rec_path + ".dat"
        if not os.path.exists(dat_path):
            # Some datasets use .mat instead
            mat_path = rec_path + ".mat"
            if not os.path.exists(mat_path):
                continue

        records.append({
            "path": rec_path,
            "superclass": superclass,
            "snomed_codes": codes,
            "dataset": config["name"],
            "fs": config["fs"],
        })

    return records


def _index_ptbxl_dataset(config):
    """
    Index PTB-XL using its native label format (scp_codes).
    Returns list of (record_path, superclass_label, snomed_codes)
    """
    import ast

    base = config["dir"]
    meta_path = base / "ptbxl_database.csv"
    scp_path = base / "scp_statements.csv"

    if not meta_path.exists() or not scp_path.exists():
        print(f"  PTB-XL metadata not found at {base}")
        return []

    # Build SCP -> superclass map
    scp_df = pd.read_csv(scp_path, index_col=0)
    diag = scp_df[scp_df["diagnostic"] == 1.0]
    scp_map = diag["diagnostic_class"].to_dict()

    meta = pd.read_csv(meta_path, index_col="ecg_id")
    records = []

    for ecg_id, row in meta.iterrows():
        try:
            codes = ast.literal_eval(row["scp_codes"])
        except Exception:
            continue

        # Find primary superclass
        best_class = None
        best_score = -1
        for code, likelihood in codes.items():
            sc = scp_map.get(code)
            if sc and likelihood > best_score:
                best_score = likelihood
                best_class = sc

        if best_class not in SUPERCLASS_LABELS:
            continue

        rec_path = str(base / row["filename_hr"])
        if not os.path.exists(rec_path + ".dat"):
            continue

        records.append({
            "path": rec_path,
            "superclass": best_class,
            "snomed_codes": list(codes.keys()),
            "dataset": "PTB-XL",
            "fs": 500,
            "strat_fold": int(row["strat_fold"]),
        })

    return records


# ----------------------------------------------------------------
# Unified index builder
# ----------------------------------------------------------------

def build_unified_index(datasets_to_include=None):
    """
    Build a unified index CSV with all available datasets.
    Each row: path, superclass, dataset, fs, snomed_codes

    Returns DataFrame and saves to ekg_datasets/unified_index.csv
    """
    if datasets_to_include is None:
        datasets_to_include = list(DATASET_CONFIGS.keys())

    all_records = []

    for ds_key in datasets_to_include:
        config = DATASET_CONFIGS[ds_key]
        if not config["dir"].exists():
            print(f"  [{ds_key}] Not downloaded, skipping")
            continue

        print(f"\n  Indexing {config['name']}...")

        if config["format"] == "ptbxl_native":
            records = _index_ptbxl_dataset(config)
        elif config["format"] == "cinc2020":
            records = _index_cinc2020_dataset(config)
        else:
            print(f"  [{ds_key}] Unknown format: {config['format']}")
            continue

        print(f"  [{ds_key}] {len(records)} records with valid superclass labels")
        all_records.extend(records)

    if not all_records:
        print("\n  No records found in any dataset!")
        return None

    df = pd.DataFrame(all_records)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  UNIFIED INDEX SUMMARY")
    print(f"{'='*60}")
    print(f"  Total records: {len(df)}")
    print(f"\n  By dataset:")
    for ds, count in df["dataset"].value_counts().items():
        print(f"    {ds}: {count}")
    print(f"\n  By superclass:")
    for sc in SUPERCLASS_LABELS:
        count = (df["superclass"] == sc).sum()
        print(f"    {sc}: {count}")

    # Save index
    index_path = DATASETS_DIR / "unified_index.csv"
    df.to_csv(index_path, index=False)
    print(f"\n  Saved to {index_path}")

    return df


# ----------------------------------------------------------------
# Signal reading with resampling
# ----------------------------------------------------------------

def read_signal(record_path, target_fs=500, target_len=5000):
    """
    Read a WFDB record and return (signal, actual_fs).
    Handles resampling to target_fs if needed.
    Pads/truncates to target_len samples.

    Returns: numpy array (target_len, 12) or None on failure
    """
    import wfdb
    from scipy.signal import resample

    try:
        rec = wfdb.rdrecord(record_path)
    except Exception:
        return None

    sig = rec.p_signal  # (N, n_channels)
    if sig is None:
        return None

    n_channels = sig.shape[1]
    actual_fs = rec.fs

    # Handle non-12-lead: pad with zeros or select first 12
    if n_channels < 12:
        pad = np.zeros((sig.shape[0], 12 - n_channels))
        sig = np.hstack([sig, pad])
    elif n_channels > 12:
        sig = sig[:, :12]

    # Resample if different sampling rate
    if actual_fs != target_fs:
        n_target = int(sig.shape[0] * target_fs / actual_fs)
        sig = resample(sig, n_target, axis=0)

    # Pad or truncate to target length
    if sig.shape[0] < target_len:
        pad = np.zeros((target_len - sig.shape[0], 12))
        sig = np.vstack([sig, pad])
    else:
        sig = sig[:target_len]

    return sig.astype(np.float32)


# ----------------------------------------------------------------
# Dataset status
# ----------------------------------------------------------------

def print_status():
    """Print status of all datasets."""
    print(f"\n{'='*60}")
    print(f"  ECG DATASET STATUS")
    print(f"{'='*60}")

    total = 0
    for ds_key, config in DATASET_CONFIGS.items():
        exists = config["dir"].exists()
        if exists:
            # Count .dat files
            dat_count = len(list(config["dir"].rglob("*.dat")))
            mat_count = len(list(config["dir"].rglob("*.mat")))
            n_files = dat_count + mat_count
            status = f"READY ({n_files} files)"
            total += n_files
        else:
            n_files = 0
            status = "NOT DOWNLOADED"
            if config.get("credentialed"):
                status += " (requires PhysioNet credentials)"

        cred = " [CREDENTIALED]" if config.get("credentialed") else ""
        print(f"\n  {config['name']}{cred}")
        print(f"    Dir:      {config['dir']}")
        print(f"    Expected: ~{config['expected_records']} records")
        print(f"    Status:   {status}")
        print(f"    Fs:       {config['fs']} Hz")

    print(f"\n  Total files on disk: {total}")
    print(f"{'='*60}\n")


# ----------------------------------------------------------------
# Download all datasets
# ----------------------------------------------------------------

def download_all(skip_credentialed=True):
    """Download all open-access datasets using wfdb or requests."""

    print(f"\n{'='*60}")
    print(f"  ECG DATASET DOWNLOADER")
    print(f"{'='*60}")

    results = {}
    for ds_key, config in DATASET_CONFIGS.items():
        if ds_key == "ptbxl":
            # PTB-XL should already be local
            if config["dir"].exists():
                print(f"\n  {config['name']} already present")
                results[ds_key] = True
            else:
                print(f"\n  {config['name']} not found — download from {config['url']}")
                results[ds_key] = False
            continue

        if skip_credentialed and config.get("credentialed"):
            print(f"\n  Skipping {config['name']} (requires PhysioNet credentialed access)")
            print(f"    Register at physionet.org to get access, then rerun with --credentials")
            results[ds_key] = False
            continue

        # Check if already downloaded
        if config["dir"].exists():
            dat_count = len(list(config["dir"].rglob("*.dat")))
            mat_count = len(list(config["dir"].rglob("*.mat")))
            if dat_count + mat_count > 100:
                print(f"\n  {config['name']} already downloaded ({dat_count + mat_count} files)")
                results[ds_key] = True
                continue

        results[ds_key] = _download_physionet_dataset(config)

    print(f"\n{'='*60}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for ds_key, success in results.items():
        name = DATASET_CONFIGS[ds_key]["name"]
        status = "OK" if success else "MISSING"
        print(f"    {name}: {status}")
    print()

    return results


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ECG Multi-Dataset Pipeline")
    parser.add_argument("--download", action="store_true", help="Download all open datasets")
    parser.add_argument("--status", action="store_true", help="Show dataset status")
    parser.add_argument("--prepare", action="store_true", help="Build unified index")
    parser.add_argument("--credentials", action="store_true", help="Include credentialed datasets")
    args = parser.parse_args()

    if args.status:
        print_status()

    elif args.download:
        skip_cred = not args.credentials
        download_all(skip_credentialed=skip_cred)

    elif args.prepare:
        build_unified_index()

    else:
        # Default: show status
        print_status()
        print("  Usage:")
        print("    python dataset_pipeline.py --download    # Download datasets")
        print("    python dataset_pipeline.py --status      # Show status")
        print("    python dataset_pipeline.py --prepare     # Build unified index")


if __name__ == "__main__":
    main()
