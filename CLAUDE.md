# EKG Intelligence Platform — Claude Context

## Project Overview

Streamlit-based 12-lead ECG analysis POC (`app.py`), targeting a native mobile app later.
Training happens on **Google Colab** (GPU/TPU via TPU v6e). Local CPU is fallback only.
Python 3.14 at `C:\Users\osnat\AppData\Local\Programs\Python\Python314\python.exe`.
EKG dir syncs to Google Drive automatically — any file change is available in Colab immediately.

---

## Current Phase: V3.1 — RR rhythm features for AFIB fix

| File | Description |
|------|-------------|
| `multilabel_v3_colab.ipynb` | Main Colab notebook — 3 cells, ready to run |
| `multilabel_v3.py` | Training script (TPU/GPU/CPU). Also contains inference API for app.py |
| `models/ecg_multilabel_v3_best.pt` | Best checkpoint — AUROC=0.9682 (Colab TPU run, Apr 5) |
| `models/thresholds_v3.json` | Per-class thresholds (tuned Apr 5, temp-scaled T=1.256) |
| `temperature_scaling.py` | Calibration script — run after each training to improve F1 |
| `tune_thresholds.py` | Threshold tuning — run with `--model v3` after training |
| `eval_v3_auroc.py` | Per-class AUROC + F1 report across PTB-XL + Challenge test sets |
| `diagnose_afib.py` | AFIB signal quality diagnostic (confirmed: no data corruption) |
| `run_post_training.bat` | **One-click**: threshold tune + temp scale + eval in sequence |

**Best result:** AUROC=0.9682 (Colab TPU, Apr 5). Combined MacroF1=0.587, MicroF1=0.691.

**V3.1 code ready — needs next Colab run:** `cnn_classifier.py` N_AUX extended 14→18 with 4 RR-interval features. `multilabel_v3.py` checkpoint loader handles the shape mismatch (copies backbone, Xavier-inits new aux columns).

---

## Architecture

- **CNN backbone:** ECGNetJoint (1D CNN with SE attention), 1.7M params
- **Loss:** BCEWithLogitsLoss with per-class pos_weight + AFIB 4x boost
- **26 classes:** PTB-XL (14) + Challenge 2021 (12 new: PAC, Brady, SVT, LQTP, TAb, LAD, RAD, NSIVC, AFL, STc, STD, LAE)
- **Datasets:** PTB-XL (18,524) + Chapman (42,390) + Challenge (50,842) = ~111,756 total
- **Folds:** PTB-XL fold 9=val, fold 10=test. Challenge fold 19=val, fold 20=test (5%/10% split, seed=42)
- **ECG-FM verdict:** Frozen backbone AUROC=0.927 vs CNN AUROC=0.972 — stay on CNN

---

## app.py Status

- **V3 is active** — app loads `ecg_multilabel_v3_best.pt` + `thresholds_v3.json` on startup
- Falls back to 12-class multilabel if V3 unavailable
- Full 26-class clinical guidance, urgency colours, and patient context all wired up
- Model label shows "V3 Multilabel (26 conditions)" in UI

---

## Per-Class Performance (Apr 5, Colab TPU run, combined test fold 10+20)

| Class | AUROC | F1 | Notes |
|-------|-------|----|-------|
| Brady | 0.994 | 0.952 | Excellent |
| STACH | 0.995 | 0.922 | Excellent |
| AFL | 0.990 | 0.888 | Excellent |
| CRBBB | 0.989 | 0.867 | Very good |
| NORM  | 0.940 | 0.804 | Good |
| **AFIB** | **0.904** | **0.268** | **Hard ceiling — AUROC limits F1 at this prevalence** |
| 1AVB | 0.971 | 0.330 | High AUROC, low test count (83 positives) |
| LAFB | 0.915 | 0.409 | Domain shift PTB-XL→Challenge |

**AFIB root cause (confirmed Apr 5):**
- Signal loading is clean — 0/11,590 records load as zeros
- AFIB is a *rhythm* disorder; prior aux features (indices 0-13) were all morphology/voltage
- F1≈0.27 is the theoretical ceiling given AUROC=0.90 at 3.7% test prevalence
- Fix: added RR-interval features (mean RR, SDNN, RMSSD, irregularity) as aux indices 14-17
- N_AUX extended 14→18 in cnn_classifier.py; checkpoint loader handles size mismatch
- **Next Colab run will be V3.1 — expected AFIB AUROC improvement to 0.93+**

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

1. **AFIB F1=0.268** — Real cause identified: all-morphology aux features miss rhythm irregularity. Fix implemented (N_AUX 14→18, RR features at indices 14-17). Next Colab run = V3.1. AFIB weight boost reduced to 2x (4x was too aggressive, destabilised training).

2. **Colab MCP** — setup script exists (`run_setup_colab_mcp_desktop.bat`), needs Desktop restart to activate.

---

## Key Model History

| Model | Classes | AUROC | Notes |
|-------|---------|-------|-------|
| v10 CNN (ECGNetJoint) | 5 superclass | HYP F1=0.442 | PTB-XL only |
| v9+v10 Ensemble | 5 superclass | HYP F1=0.456 | Per-class thresholds |
| ECG-FM Stage 2 | 5 superclass | HYP F1=0.478 | Full fine-tune, T4 GPU |
| V3 multilabel | 26 | 0.9687 | **Current best** — CPU Apr 5 |

---

## Workflow: After Each Colab Training Run

1. New `ecg_multilabel_v3_best.pt` syncs to local via Drive
2. Run `run_tune_v3.bat` → updates `thresholds_v3.json`
3. Run `run_temperature_scaling.bat` → recalibrates thresholds
4. Run `run_eval_v3_auroc.bat` → check per-class AUROC
5. Update this CLAUDE.md with new AUROC + per-class highlights
6. `git commit` + `git push`
