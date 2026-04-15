# CODE-15% Storage Plan — Training with Limited Colab Disk

## Problem
- Colab free tier: 112 GB disk, currently ~80 GB used
- CODE-15% full dataset: 18 HDF5 parts, ~35 GB as zips, ~30 GB extracted
- Current: 7 of 18 parts downloaded = 120K records (V3.2a)
- Goal: use more CODE-15% data without exceeding storage

## Storage Budget
- Available: ~32 GB free
- Each zip: ~2 GB, each extracted H5: ~1.5-2 GB
- Download deletes zip after extraction (already implemented in dataset_code15.py:173)
- So each part needs ~3.5 GB peak (zip + h5) but settles to ~2 GB (h5 only)

## Recommended: Option 3 — Incremental Download + Train

Instead of downloading all 18 parts (won't fit), download in rounds:

### Round 1: Download 5 more parts (total 12/18 = ~230K records)
```python
# On Colab — first check what we have:
!python dataset_code15.py --stats

# Download will skip existing parts, only get new ones
# Manually download parts 7-11 only (not all 18):
!python -c "
from dataset_code15 import download_code15, CODE15_BASE, N_PARTS, _h5_path, ZENODO_BASE
from pathlib import Path
import requests, zipfile

raw = CODE15_BASE / 'raw'
raw.mkdir(parents=True, exist_ok=True)
session = requests.Session()

# Only download parts 7-11 (5 new parts)
for i in range(7, 12):
    h5 = _h5_path(raw, i)
    if h5.exists() and h5.stat().st_size > 100_000:
        print(f'[skip] part {i} already exists')
        continue
    zip_dest = raw / f'exams_part{i}.zip'
    url = f'{ZENODO_BASE}/exams_part{i}.zip?download=1'
    print(f'Downloading part {i}...', flush=True)
    with session.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(zip_dest, 'wb') as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f'Extracting part {i}...')
    with zipfile.ZipFile(zip_dest, 'r') as zf:
        zf.extractall(raw)
    zip_dest.unlink()  # free ~2 GB immediately
    print(f'Part {i} done.')
"

# Rebuild index
!python dataset_code15.py --index
!python dataset_code15.py --stats
```

### Round 2: Train with ~230K CODE-15% + existing data
```python
!python multilabel_v3.py
```

### Round 3: Evaluate results
If 230K is sufficient (likely — diminishing returns after 200K), stop here.
If AFIB F1 still needs improvement, download 3 more parts (total 15/18).

## Why NOT download all 18
- 18 parts = ~30 GB H5 files alone
- Plus PTB-XL (~3 GB), Chapman (~8 GB), Challenge (~6 GB), models, code = exceeds 112 GB
- Risk of Colab session crash losing the download progress

## Expected Impact
- V3.2a used 120K CODE-15% records → AUROC=0.985
- Doubling to ~230K should improve:
  - AFIB generalization (more AF examples from different demographics)
  - 1AVB cross-domain F1 (more training signal)
  - NORM calibration (proportionally less dominant vs pathology)
- Diminishing returns: going from 230K to 345K probably adds <0.5% AUROC

## Alternative: Compact Format (Future, if needed)
If we ever need all 345K, we could:
1. Download each part, read signals, convert to float16, save to a single compact file
2. Delete original H5 after conversion
3. float16 saves ~50% vs float32 (negligible quality loss for ECG at mV scale)
4. Would require changes to `load_code15_signal()` to read the new format
5. Total: ~15 GB instead of ~30 GB for all 345K records

This is more work than it's worth unless Round 1+2 results are disappointing.

## Colab Disk Cleanup (before downloading)
Run this first to free space:
```python
# Check disk usage
!df -h /content
!du -sh /content/drive/MyDrive/EKG_models/* 2>/dev/null | sort -rh | head -20

# Clean Python caches
!find /content -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
!find /content -name "*.pyc" -delete 2>/dev/null

# Clean pip cache
!pip cache purge

# Check for old checkpoints that can be removed
!ls -lh /content/drive/MyDrive/EKG_models/models/ecg_multilabel_v3*.pt 2>/dev/null
```
