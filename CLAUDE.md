# EKG Intelligence Platform — Claude Context

## Project Overview

Streamlit-based 12-lead ECG analysis POC (`app.py`), targeting a native mobile app later.
Training happens on **Google Colab** (GPU/TPU via TPU v6e). Local CPU is fallback only.
Python 3.14 at `C:\Users\osnat\AppData\Local\Programs\Python\Python314\python.exe`.
EKG dir syncs to Google Drive automatically — any file change is available in Colab immediately.

---

## Current Phase: V3.2b — 12/18 CODE-15% parts trained, need full 18/18

| File | Description |
|------|-------------|
| `multilabel_merged_colab.ipynb` | Main Colab notebook — 2 cells (setup + train), ready to run |
| `multilabel_v3.py` | Training script (TPU/GPU/CPU). CODE-15% integrated |
| `dataset_code15.py` | CODE-15% downloader, HDF5 reader, label mapper, index builder |
| `models/ecg_multilabel_v3_best.pt` | Best checkpoint — **ValAUROC=0.990** (Colab GPU, Apr 15) |
| `models/thresholds_v3.json` | Per-class thresholds (temp-scaled, per-class T, Apr 15) |
| `temperature_scaling.py` | Calibration script — run after each training to improve F1 |
| `tune_thresholds.py` | Threshold tuning — run with `--model v3` after training |
| `eval_v3_auroc.py` | Per-class AUROC + F1 report across PTB-XL + Challenge + CODE-15% test sets |
| `diagnose_afib.py` | AFIB signal quality diagnostic (confirmed: no data corruption) |
| `run_post_training.bat` | **One-click**: threshold tune + temp scale + eval in sequence |

**V3.2b result (Apr 15):** ValAUROC=0.990, 31 epochs on Colab GPU.
**Trained with 12/18 CODE-15% parts** (~240K records) — notebook has `N_CODE15_PARTS = 12`.
**Total training:** ~352K records (18.5K PTB-XL + 42.4K Chapman + 50.8K Challenge + ~240K CODE-15%).
**Post-training pipeline:** temp scaling (per-class T) improved MacroF1 0.575 → 0.601 on combined test.

**V3.2b vs V3.2a on combined test (fold 10+20, temp-scaled thresholds):**
- MacroAUROC: 0.962 (same), MacroF1: 0.601 (was 0.580), MicroF1: 0.710 (was 0.674)
- AFIB: AUROC=0.908, F1=0.271 (Challenge) — still domain-shifted
- AFIB on CODE-15% test: **F1=0.745, AUROC=0.994** — excellent
- ISC_: F1=0.496 (was 0.236) — **+110% improvement** (temp scaling unlocked this)
- IRBBB: F1=0.465 (was 0.329) — **+41% improvement**
- PVC: F1=0.753 (was 0.698) — **+8% improvement**
- Brady: F1=0.950, STACH: F1=0.928, AFL: F1=0.889 — all excellent

**Next step: V3.2c — train with all 18 CODE-15% parts (345K records):**
- Change `N_CODE15_PARTS = 12` → `18` in notebook
- Need ~10 GB more free space on Colab (parts 12-17 are ~20 GB additional)
- Alternative: stream parts directly from Zenodo via `dataset_code15.py --download`
- Expected: better generalization, especially for AFIB cross-domain

---

## Architecture

- **CNN backbone:** ECGNetJoint (1D CNN with SE attention), 1.7M params
- **Loss:** BCEWithLogitsLoss with per-class pos_weight + AFIB 1.5x boost (reduced from 2x as CODE-15% reduces AFIB imbalance)
- **26 classes:** PTB-XL (14) + Challenge 2021 (12 new: PAC, Brady, SVT, LQTP, TAb, LAD, RAD, NSIVC, AFL, STc, STD, LAE)
- **Datasets (V3.2b actual):** PTB-XL (18,524) + Chapman (42,390) + Challenge (50,842) + CODE-15% (~240K, 12/18 parts) = ~352K trained
- **Datasets (V3.2c target):** PTB-XL (18,524) + Chapman (42,390) + Challenge (50,842) + CODE-15% (345,779 full, 18/18 parts) = ~457,535 total
- **Folds:** PTB-XL fold 9=val, fold 10=test. Challenge fold 19=val, fold 20=test. CODE-15% fold 29=val, fold 30=test
- **ECG-FM verdict:** Frozen backbone AUROC=0.927 vs CNN AUROC=0.972 — stay on CNN

---

## app.py Status

- **V3 is active** — app loads `ecg_multilabel_v3_best.pt` + `thresholds_v3.json` on startup
- Falls back to 12-class multilabel if V3 unavailable
- Full 26-class clinical guidance, urgency colours, and patient context all wired up
- Model label shows "V3 Multilabel (26 conditions)" in UI

---

## Per-Class Performance (Apr 15, V3.2b — temp-scaled thresholds)

### Combined Test (fold 10+20, 6,950 records)
| Class | AUROC | F1 | Notes |
|-------|-------|----|-------|
| Brady | 0.993 | 0.950 | Excellent |
| STACH | 0.996 | 0.928 | Excellent |
| AFL | 0.989 | 0.889 | Excellent |
| CRBBB | 0.985 | 0.873 | Very good |
| NORM  | 0.934 | 0.825 | Good |
| CLBBB | 0.978 | 0.763 | Good |
| PVC | 0.979 | 0.753 | Good (was 0.698) |
| ASMI | 0.992 | 0.737 | Good |
| LVH | 0.947 | 0.707 | Good |
| IMI | 0.985 | 0.613 | Good |
| TAb | 0.912 | 0.620 | New class, good |
| LAD | 0.983 | 0.626 | New class, good |
| PAC | 0.958 | 0.591 | New class, ok |
| NDT | 0.982 | 0.588 | Improved from 0.522 |
| NSIVC | 0.962 | 0.533 | New class |
| LAE | 0.984 | 0.509 | New class |
| ISC_ | 0.942 | 0.496 | **+110% from 0.236** (temp scaling) |
| STD | 0.957 | 0.479 | New class |
| IRBBB | 0.948 | 0.465 | **+41% from 0.329** |
| LAFB | 0.931 | 0.449 | Improved from 0.403 |
| LQTP | 0.947 | 0.440 | New class |
| STc | 0.922 | 0.440 | New class |
| RAD | 0.964 | 0.432 | New class |
| 1AVB | 0.957 | 0.319 | Still weak on Challenge |
| **AFIB** | **0.908** | **0.271** | **Domain shift — F1=0.745 on CODE-15%** |

### CODE-15% Test Set (34,577 records)
| Class | AUROC | F1 | Notes |
|-------|-------|----|-------|
| NORM | 0.975 | 0.966 | Near-perfect |
| **AFIB** | **0.994** | **0.745** | **Excellent — proves model detects AFIB** |
| CLBBB | 0.996 | 0.780 | Excellent |
| CRBBB | 0.994 | 0.785 | Excellent |
| STACH | 0.995 | 0.742 | Good |
| Brady | 0.993 | 0.613 | Good |
| 1AVB | 0.987 | 0.503 | Good — 3x better than Challenge |

**Attention flags (F1 < 0.40 with N+ >= 10):**
- AFIB: F1=0.271 on Challenge (n=189) — domain shift, not model weakness
- LAFB: F1=0.364 (n=348) — poor on Challenge data
- 1AVB: F1=0.300 (n=83) — threshold too high
- ISC_: F1=0.232 raw, improved to 0.496 with temp scaling
- IRBBB: F1=0.338 (n=184)
- STc: F1=0.380 (n=311)

---

## Colab MCP Setup

**Goal:** Claude controls Colab notebooks directly via `colab-mcp`.
**Status:** `open_colab_browser_connection` available in Claude Code Desktop (Cowork) only — NOT in VS Code.

To enable in Cowork:
1. Run `run_setup_colab_mcp_desktop.bat` — adds colab-proxy to `~/.claude/mcp.json`
2. Fully quit and reopen Claude Desktop
3. Open `colab.new` in browser
4. Tell Claude to call `open_colab_browser_connection`

**Wrapper:** `C:\Users\osnat\.claude\colab_mcp_wrapper.py` — delays `tools/list` by 4s (working).
**Workaround:** EKG dir syncs to Drive — edit files here, run Colab manually.

---

## Open Problems (priority order)

1. **CODE-15% only 12/18 parts trained** — Notebook has `N_CODE15_PARTS = 12`, only downloads parts 0-11 (~240K records).
   - **Action:** Change to `N_CODE15_PARTS = 18` in notebook, need ~10 GB more disk on Colab
   - Alternative: download remaining parts via `dataset_code15.py --download` directly on Colab
   - Then rebuild index and retrain for V3.2c (target: 457K total records)

2. **AFIB domain shift** — F1=0.745 on CODE-15% but 0.271 on Challenge test.
   - Model detects AFIB well in CODE-15% recordings but not Challenge recordings
   - Root cause: different recording equipment, populations, and AFIB presentation styles
   - Ideas: domain-adversarial training, mixed-source augmentation, or separate AFIB head

3. **Threshold tuning doesn't include CODE-15% folds** — `tune_thresholds.py` and `temperature_scaling.py` use folds 9+19 (val) and 10+20 (test) only.
   - Missing folds 29 (CODE-15% val) and 30 (CODE-15% test)
   - Fix: add fold 29 to val_mask and fold 30 to test_mask for cross-domain threshold calibration
   - This could improve AFIB threshold — currently 0.749 may be too high for Challenge data

4. **Weak classes:** 1AVB (F1=0.319), LAFB (F1=0.449), STc (F1=0.440), IRBBB (F1=0.465)
   - Consider class-specific loss weighting or hard example mining
   - 1AVB has only 83 test samples — small sample variance may inflate the issue

---

## Key Model History

| Model | Classes | AUROC | Notes |
|-------|---------|-------|-------|
| v10 CNN (ECGNetJoint) | 5 superclass | HYP F1=0.442 | PTB-XL only |
| v9+v10 Ensemble | 5 superclass | HYP F1=0.456 | Per-class thresholds |
| ECG-FM Stage 2 | 5 superclass | HYP F1=0.478 | Full fine-tune, T4 GPU |
| V3.1 multilabel | 26 | 0.9682 | Colab TPU Apr 5, combined MacroF1=0.587 |
| V3.2a multilabel | 26 | 0.985 | Colab GPU Apr 14, +120K CODE-15% (7/18 parts) |
| V3.2b multilabel | 26 | **0.990** | **Current best** — Colab GPU Apr 15, +240K CODE-15% (12/18 parts), temp-scaled |
| V3.2c multilabel | 26 | TBD | Target: +345K CODE-15% (18/18 parts) — need more Colab disk space |

---

## Workflow: V3.2c Retrain (full 18/18 CODE-15%)

On Colab — need to download remaining 6 parts (12-17):
```
# Step 0 — Fix notebook: change N_CODE15_PARTS = 12 → 18
# Or download directly via dataset_code15.py:

# Step 1 — Download remaining parts from Zenodo
!pip install h5py -q
!python dataset_code15.py --download   # will skip already-downloaded parts 0-11

# Step 2 — Rebuild index with all 18 parts
!python dataset_code15.py --index

# Step 3 — Verify 345K+ records
!python dataset_code15.py --stats

# Step 4 — Train (will use full CODE-15% + existing PTB-XL/Chapman/Challenge)
!python multilabel_v3.py
```

**Disk space note:** Each H5 part is ~3.5 GB. Parts 12-17 need ~20 GB. Colab Pro has ~150 GB SSD.
If disk is tight, consider deleting downloaded zip files after extraction (dataset_code15.py does this automatically).

## Workflow: After Each Colab Training Run

1. New `ecg_multilabel_v3_best.pt` syncs to local via Drive
2. Run `run_tune_v3.bat` → updates `thresholds_v3.json`
3. Run `run_temperature_scaling.bat` → recalibrates thresholds
4. Run `run_eval_v3_auroc.bat` → check per-class AUROC
5. Update this CLAUDE.md with new AUROC + per-class highlights
6. `git commit` + `git push`
