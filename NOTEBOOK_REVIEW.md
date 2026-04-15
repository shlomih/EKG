# multilabel_merged_colab.ipynb — Comprehensive Review

## Overview
**Purpose:** Train V3.2 ECG classifier (26-class multilabel model) on Colab GPU/TPU using 4 datasets: PTB-XL (18.5K), Chapman (42.4K), PhysioNet 2021 Challenge (50.8K), and CODE-15% (345.8K).

**Architecture:** ECGNetJoint CNN with SE attention (1.7M params) → 26 binary classifiers via BCEWithLogitsLoss  
**Target AUROC:** 0.97+ (V3.1 achieved 0.9682)  
**Data:** ~457K ECG records total when CODE-15% included  

---

## Notebook Structure (5 cells)

### **Cell 1: Setup + Data Restore** (~30-40 min runtime)
**Task:** Mount Drive, copy scripts, restore datasets from tar/HDF5 using Drive API streaming

**Key Operations:**
1. **Accelerator Check** — Detect TPU (v6e) vs GPU (T4) vs CPU
   - Raises error if none available
   - TPU: installs `torch_xla` via subprocess only (never imports it directly)

2. **Drive Mount + Auth**
   - Mounts `/content/drive` (FUSE mount for script/model access)
   - Authenticates Drive API separately for streaming large files
   - Reason: FUSE cache would fill 107 GB Colab disk with 96 GB EKG folder

3. **Script Copy from Drive** (Small files via FUSE)
   ```
   multilabel_v3.py, cnn_classifier.py, multilabel_classifier.py,
   dataset_chapman.py, dataset_challenge.py, dataset_code15.py
   ```

4. **Model Checkpoint Restore** (For transfer learning)
   - Priority: `ecg_multilabel_v3_best.pt` (AUROC=0.9682) → `ecg_multilabel_v3.pt`
   - Falls back to V2 if V3 unavailable

5. **Dataset Restore** (Large files via Drive API streaming)
   - **Challenge** (~7 GB): tar stream to `/content/ekg_datasets/challenge2021`
   - **Chapman** (~5.5 GB): tar stream to `/content/ekg_datasets/chapman`
   - **PTB-XL** (~2.7 GB): tar stream to `/content/ekg_datasets/ptbxl`
   - **CODE-15%** (HDF5 parts 0-5, ~21 GB): individual files via Drive API download
     - Requires pre-extracted `code15/raw/` folder synced to Drive
     - Builds index via `dataset_code15.py --index` if missing

**Critical Implementation Details:**
- Uses `subprocess.Popen` with `tar xf -` to stream tar directly (avoids FUSE cache)
- Drive API download with 32 MB chunks, progress reporting every 10%
- Temp files (`.tmp`) prevent partial results if download fails
- SSD free space checked at each step (needs ~36 GB available)
- Graceful fallback: if Chart/Challenge missing, continues without them

---

### **Cell 2: Training** (~40-90 min on TPU, ~2-4 hrs on GPU)

**Task:** Run `multilabel_v3.py` training script, save model to Drive

**Process:**
1. Verify accelerator (check `/dev/vfio/0` for TPU)
2. Run `python multilabel_v3.py` with stdout/stderr merged and flushed per line
3. Monitor output in real-time
4. Copy trained model `ecg_multilabel_v3.pt` back to Drive
5. Syncs to local PC via Google Drive for Desktop

**Note on XLA Compilation:** First TPU epoch is slow (30-60s) — normal for XLA graph tracing

---

### **Cell 3: Cleanup** (Optional)
```python
!rm -rf /content/*
```
Clears local SSD for next run (not usually needed if re-running).

---

### **Cell 4: Drive Remount** (Optional)
Quick Drive re-auth if Auth token expired.

---

## Dependency Graph

```
multilabel_merged_colab.ipynb
├── multilabel_v3.py (Main training script)
│   ├── cnn_classifier.py
│   │   └── (raw ECG loading, voltage features, augmentation)
│   ├── multilabel_classifier.py
│   │   └── (12-class PTB-XL label utilities)
│   ├── dataset_chapman.py
│   │   └── (14-class Chapman dataset loader)
│   ├── dataset_challenge.py
│   │   └── (PhysioNet 2021 Challenge loader, SNOMED-CT mapping)
│   └── dataset_code15.py
│       └── (CODE-15% HDF5 reader, label mapper, demographics)
```

---

## Data Loading Pipeline

### **PTB-XL (18,524 records)**
- Source: `ekg_datasets/ptbxl/` (PT-XL WFDB format)
- Labels: 14 conditions from `multilabel_classifier.py` (MULTILABEL_CODES)
- Folds: 1-8 = train, 9 = val, 10 = test
- **Preloaded to RAM** (~6-8 GB) for fast training access

### **Chapman (42,390 records)**
- Source: `ekg_datasets/chapman/` (WFDB format)
- Labels: 14 merged conditions (MERGED_CODES)
- Folds: 0 = train only (no val/test)
- **Lazy loading** (loads per-batch during training)

### **PhysioNet 2021 Challenge (50,842 records)**
- Source: `ekg_datasets/challenge2021/` (Georgia, CPSC-2018, CPSC-Extra, Ningbo)
- Labels: 26 conditions (V3_CODES) — includes 12 new beyond PTB-XL
- Folds: 0 = train (80%), 19 = val (5%), 20 = test (10%)
  - Created with seed=42 random split in `load_v3_data()`
- **Lazy loading**

### **CODE-15% (345,779 records — optional)**
- Source: `ekg_datasets/code15/raw/` (HDF5 binary format, Zenodo)
- Labels: 6 binary labels → mapped to V3_CODES (AFIB, 1AVB, CRBBB, CLBBB, Brady, STACH, NORM)
- Signal: 400 Hz, 4096 samples → resampled to 500 Hz, 5000 samples in memory
- Folds: 0 = train (85%), 29 = val (5%), 30 = test (10%)
  - Created with seed=1337 random split
- **Lazy HDF5 loading** (file handle cached per worker process)
- **Demographics cache** built for age/sex aux features
- If index missing: `python dataset_code15.py --index` creates it from all 18 HDF5 files

---

## Class Label Mapping (26 classes)

### **From multilabel_classifier.py (12 classes)**
```
0: NORM    — Normal ECG
1: AFIB    — Atrial Fibrillation  ← weak (48 samples in PTB-XL, fixed by CODE-15%)
2: PVC     — Premature ventricular complex
3: LVH     — Left ventricular hypertrophy
4: IMI     — Inferior MI
5: ASMI    — Anteroseptal MI
6: CLBBB   — Complete LBBB
7: CRBBB   — Complete RBBB
8: LAFB    — Left anterior fascicular block
9: 1AVB    — First-degree AV block
10: ISC_   — Non-specific ischemic ST changes
11: NDT    — Non-diagnostic T abnormalities
12: IRBBB  — Incomplete RBBB
13: STACH  — Sinus tachycardia
```

### **New from PhysioNet 2021 Challenge (12 classes)**
```
14: PAC    — Premature atrial contraction
15: Brady  — Bradycardia
16: SVT    — Supraventricular tachycardia
17: LQTP   — Prolonged QT interval
18: TAb    — T-wave abnormality
19: LAD    — Left axis deviation
20: RAD    — Right axis deviation
21: NSIVC  — Non-specific intraventricular conduction delay
22: AFL    — Atrial flutter
23: STc    — ST-T change
24: STD    — ST depression
25: LAE    — Left atrial enlargement
```

---

## Model Architecture (ECGNetJoint)

**Input:**
- Raw 12-lead signal: shape (B, 5000, 12) ← normalized by div 5.0 mV
- 18 auxiliary features: shape (B, 18)
  - Voltage: Sokolow index, Cornell voltage, etc. (8 features)
  - Demographics: sex (M/F), age normalized (2 features)
  - Morphology: QRS duration, axis angle (3 features)
  - **NEW (V3.1):** RR-interval features—SDNN, RMSSD, mean RR, irregularity (4 features)

**Architecture:**
```
Signal branch:
  Conv1D layers + SE-Attention blocks → 288-dim embedding

Aux branch:
  Linear(18) → ReLU → 288-dim embedding

Fusion:
  Concatenate (288 + 288 = 576)
  → FC layers → 26 binary classifiers

Loss: BCEWithLogitsLoss(pos_weight=[per-class])
  - AFIB weight boosted 1.5x (reduced from 2x, CODE-15% reduces imbalance)
```

**Parameters:** 1.7M total

---

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 256 (TPU) / 64 (GPU) | Colab T4 GPU: memory limit ~15 GB |
| Learning rate | 3e-4 base × (BS/64) | Linear scaling rule. Fine-tune from V3 ckpt: 1e-4 |
| Optimizer | AdamW | weight_decay=1e-4 |
| Scheduler | OneCycleLR | 5% warmup, pct_start=0.05 for fine-tuning |
| Epochs | 60 | Early stop if val AUROC no improvement × 12 epochs |
| Data augmentation | Noise, scaling, time shift, lead dropout | Train set only |
| Precison | float32 (GPU) / bfloat16 (TPU) | TPU native FP16 acceleration |

**Transfer Learning:** 
- Loads `ecg_multilabel_v3_best.pt` if exists (checkpoint from prior run)
- Handles N_AUX mismatch: copies old weights, Xavier-init new RR-feature columns

---

## Per-Class Performance (V3.1, Apr 5, Colab TPU, test fold 10+20)

| Class | AUROC | F1 | Issue |
|-------|-------|----|----|
| Brady | 0.994 | 0.952 | ✓ Excellent |
| STACH | 0.995 | 0.922 | ✓ Excellent |
| AFL | 0.990 | 0.888 | ✓ Excellent |
| **AFIB** | **0.904** | **0.268** | **← F1 ceiling at 3.7% prevalence, fixed by CODE-15%** |
| **1AVB** | **0.971** | **0.330** | **← Imbalanced, fixed by CODE-15%** |

**V3.2 Expected Improvement:**
- CODE-15% adds ~17K AFIB records (3x increase) → target AFIB F1 ≥0.50
- CODE-15% adds ~20K 1AVB records → target 1AVB F1 ≥0.50
- Combined AUROC target: 0.97+

---

## Validation Strategy

**Mixed validation:** Ensures early stopping sees all 26 classes across all data sources
```
val_mask = (folds == 9) | (folds == 19) | (folds == 29)
           ↓                ↓                ↓
        PTB-XL        Challenge        CODE-15%
```

**Per-epoch metrics:**
- Loss: BCEWithLogitsLoss on val set
- Macro AUROC: mean of 26 per-class AUROC scores
- Macro F1: mean of 26 per-class F1 scores

**Best model:** Saved when val_auroc improves; early stop after 12 no-improvement epochs

---

## Files Created / Modified by Notebook

### **Outputs to `/content/` (Colab local SSD)**
1. `multilabel_v3.pt` — Last checkpoint
2. `models/ecg_multilabel_v3_best.pt` — Best val AUROC checkpoint
3. `ekg_datasets/` — All restored datasets (persists across runs in same Colab session)

### **Outputs to Drive**
1. `/MyDrive/EKG/models/ecg_multilabel_v3.pt` — Syncs back after training via shutil.copy()
2. Automatically syncs to local PC via Google Drive for Desktop

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| "No GPU or TPU!" | Wrong runtime type | Runtime > Change runtime type > T4 GPU or TPU v6e |
| "/dev/vfio/0 locked" | torch_xla imported in Cell 1 | Check Cell 1 for `import torch_xla` — must NOT exist |
| "CODE-15% raw/ not found" | code15.tar not pre-extracted to Drive | Extract code15.tar locally, re-sync Drive |
| "SSD full (need 36 GB)" | Datasets too large for remaining space | Clear `/content/*`, restart Colab |
| "challeng2021.tar not found" | Not uploaded to Drive | Optional — continues without Challenge if missing |
| "First epoch very slow (30s)" | XLA compilation | Normal for TPU — subsequent epochs ~2-3s each |
| Model not saved to Drive | shutil.copy failed | Check Drive mount, model file exists at `/content/models/` |

---

## Key Design Decisions

### **Drive API Streaming (vs FUSE Mount)**
- **Why?** FUSE cache would consume 96 GB of 107 GB Colab disk
- **How?** Drive API → streaming tar via subprocess pipe → no disk cache
- **Fallback:** If Drive API fails, uses FUSE copy (slower but works)

### **Lazy Loading (Chapman, Challenge, CODE-15%)**
- **Why?** Only 6-8 GB PTB-XL preloaded; others follow
- **How?** DataLoader calls `__getitem__` per batch → loads signal on demand
- **CODE-15% HDF5:** Module-level file handle cache, fork-safe for multi-worker loaders

### **Lower Fine-Tune LR**
- **Why?** V3 checkpoint already strong (AUROC=0.9682)
- **How?** If V3 ckpt loaded: base_lr = 1e-4 (vs 3e-4 for scratch)

### **RR-Interval Features (N_AUX: 14 → 18)**
- **Why?** AFIB is rhythm disorder, not morphological
- **How?** Extract mean RR, SDNN, RMSSD, irregularity → aux indices 14-17
- **Checkpoint Compatibility:** Old checkpoints (N_AUX=14) extended with Xavier-init on new columns

### **AFIB Weight Boost: 4x → 2x → 1.5x**
- **v3.0:** 4x — too aggressive, destabilized training
- **v3.1:** 2x + RR features → AUROC=0.9682
- **v3.2:** 1.5x — CODE-15% triples AFIB prevalence, less boosting needed

---

## Next Steps (V3.2 Ready)

Before running on Colab:

1. **Verify CODE-15% locally** (or download on Colab):
   ```bash
   python dataset_code15.py --download  # ~35 GB, ~60 min on 100 Mbps
   python dataset_code15.py --index     # ~30 min, builds code15_index.csv
   python dataset_code15.py --stats     # Check record counts
   ```

2. **Run Cell 1 + 2** on Colab GPU/TPU
   - Cell 1: ~30-40 min (setup + restore data)
   - Cell 2: ~40-90 min (train on TPU), ~2-4 hrs (train on GPU)

3. **Post-training (local):**
   ```bash
   python tune_thresholds.py --model v3    # Optimize per-class thresholds
   python temperature_scaling.py             # Calibrate confidence scores
   python eval_v3_auroc.py                  # Full eval report
   ```

4. **Update [CLAUDE.md](CLAUDE.md) with new AUROC results**

---

## Summary

**multilabel_merged_colab.ipynb** is a production-ready Colab notebook that:
- Efficiently streams 4 large ECG datasets using Drive API (avoids FUSE caching)
- Trains a 26-class multilabel ECG classifier on TPU v6e (40-90 min) or GPU T4 (2-4 hrs)
- Integrates PTB-XL, Chapman, PhysioNet Challenge, and CODE-15% datasets seamlessly
- Implements transfer learning from V3 best checkpoint (AUROC=0.9682)
- Saves trained model back to Drive for auto-sync to local PC

**Status:** Ready for V3.2 training run once CODE-15% downloaded and indexed.
