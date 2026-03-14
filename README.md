# EKG Intelligence Platform — POC Setup

## Quick Start

### Recommended Python version
Use Python 3.13 for this repo (3.14 can break pip packaging). Then create and activate venv:
```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### 1. Install dependencies
```bash
pip install wfdb pandas numpy tqdm matplotlib neurokit2
```

### 2. Download datasets

```bash
# Fast test — 100 records only (~50MB, takes ~2 min)
python download_ekg_datasets.py --dataset ptbxl --small

# Full PTB-XL dataset (~2.5GB, takes ~15–20 min)
python download_ekg_datasets.py --dataset ptbxl

# All three datasets (~6GB total)
python download_ekg_datasets.py
```

### 3. Explore what you downloaded

```bash
# Print summary statistics
python explore_dataset.py

# Plot 5 random EKG strips
python explore_dataset.py --plot 5

# Plot strips filtered by diagnosis
python explore_dataset.py --plot 3 --diagnosis STEMI

# Inspect a specific record
python explore_dataset.py --record 1
```

---

## What gets downloaded

| Dataset    | Records | Size   | Labels                              |
|------------|---------|--------|-------------------------------------|
| PTB-XL     | 21,837  | ~2.5GB | 71 SNOMED diagnoses, expert-labeled |
| CPSC 2018  | 6,877   | ~1.2GB | 9 rhythm classes                    |
| Georgia    | 10,344  | ~2.0GB | SNOMED-CT labels                    |

---

## Output structure

```
ekg_datasets/
├── ptbxl/
│   ├── ptbxl_database.csv       ← original PhysioNet metadata
│   ├── ptbxl_labeled.csv        ← enriched with human-readable diagnoses
│   ├── label_distribution.csv   ← diagnosis counts for inspection
│   ├── scp_statements.csv       ← SNOMED code descriptions
│   ├── records500/              ← 500Hz waveform files (.hea + .dat)
│   └── records100/              ← 100Hz waveform files (lighter)
├── cpsc/
│   └── ...
└── georgia/
    └── ...
```

---

## Key PTB-XL facts for your POC

- **Validation split**: Use `strat_fold` column. Folds 1–8 = train, fold 9 = validation, fold 10 = test. Never touch fold 10 until final evaluation.
- **Two sample rates**: 500Hz (`filename_hr`) for full fidelity, 100Hz (`filename_lr`) for lighter models.
- **Label format**: SCP codes in `scp_codes` column as a dict `{code: likelihood}`. Use `ptbxl_labeled.csv` for the pre-decoded human-readable `primary_diagnosis` column.
- **Demographics available**: age, sex, weight, height — ideal for testing patient-context integration.

---

## Next scripts (coming next)

- `digitization_pipeline.py` — OpenCV grid detection + trace extraction on printed strips
- `interval_calculator.py` — NeuroKit2-based QTc, PR, QRS measurement
- `poc_classifier.py` — Pre-trained model inference on your records
- `app.py` — Streamlit POC interface
