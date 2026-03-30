# EKG Intelligence Platform — Development Plan
Last updated: 2026-03-30

---

## Phase 1 — Complete 14-Class Training  `[IN PROGRESS]`
**Goal:** Trained model with AFIB + STACH added to the 12-class baseline
**Runs:** Colab T4 (primary, epoch ~18+) + Local CPU (backup)
**Checkpoint:** Model saved to Drive on every AUROC improvement (`ecg_multilabel_v2_best.pt`)
**Done when:** Training completes, model file confirmed on Drive + local `models/ecg_multilabel_v2.pt`

Baseline to beat: MacroAUROC=0.972, MacroF1=0.699 (12-class model)

---

## Phase 2 — Per-Class Threshold Tuning  `[TODO]`
**Goal:** Boost F1 without retraining
**How:** Use validation fold 9 to find optimal threshold per class instead of fixed 0.5
**Expected gain:** F1 +0.05 to +0.10
**Checkpoint:** Save thresholds to `models/thresholds_v2.json`, run eval on test fold 10
**Done when:** Per-class F1 on test fold 10 exceeds current macro-F1 of ~0.60

---

## Phase 3 — Expand to 20+ Conditions  `[TODO]`
**Goal:** Add conditions from PhysioNet 2021 Challenge datasets
**New data:** Georgia (10,344) + CPSC-2018 (6,877) + CPSC-Extra (3,453) + Ningbo (34,905) = ~55K records
**Download:** `ekg_datasets/challenge2021/` — downloading now in background
**New conditions to add:** STEMI, QT prolongation, hyperkalemia, WPW, RAE, LAE, PAC, bradycardia
**Architecture:** Add output neurons, same CNN
**Checkpoint:** New `models/ecg_multilabel_v3.pt`, per-class results logged, AUROC >= 0.96
**Done when:** 20+ conditions trained and evaluated on PTB-XL fold 10

---

## Phase 4 — LVEF Binary Classification  `[TODO]`
**Goal:** Detect reduced ejection fraction (EF < 35%) from ECG — matches Anumana's FDA product
**Why binary not regression:** Same CNN architecture, no change needed, still high clinical value
**Data needed:** Paired ECG + echo reports with EF values
**Source:** MIMIC-IV-ECG (400k ECGs) + MIMIC-IV clinical notes (echo reports)
**Requires:** PhysioNet credentialing — see registration steps below
**Steps:**
  1. Register at physionet.org (free, 1-2 days)
  2. Complete CITI training course (free, ~2-3 hours online)
  3. Download MIMIC-IV-ECG + notes (~50 GB)
  4. Extract EF < 35% labels from echo notes via regex
  5. Fine-tune current model with EF label
**Checkpoint:** `models/ecg_lvef.pt`, AUROC >= 0.85 on held-out echo records
**Done when:** LVEF flag working in app, AUROC >= 0.85 (Attia et al. benchmark)

---

## Phase 5 — App Integration  `[TODO]`
**Goal:** Merge all models into app, ship updated version
**Steps:** Apply per-class thresholds, add LVEF flag to report, update UI
**Done when:** App shows 20+ conditions + LVEF risk flag

---

## PhysioNet Registration (for MIMIC/LVEF data)
User must do this — I cannot register on your behalf:
1. Go to https://physionet.org/register/
2. Fill name, institution, email — verify email
3. Go to https://physionet.org/settings/credentialing/
4. Complete CITI "Data or Specimens Only Research" course at citiprogram.org (free, ~2 hrs)
5. Upload CITI certificate to PhysioNet credentialing page
6. Sign MIMIC-IV data use agreement
7. Access granted within 1-2 days

Once credentialed, share your PhysioNet username so I can prepare the MIMIC download script.

---

## What I Can Do Without You
- Download open-access datasets (PhysioNet 2021 Challenge — no credentials)
- Write data loading + label mapping code for new datasets
- Run training locally
- Write per-class threshold tuning script
- Prepare MIMIC download + EF extraction scripts (ready to run once you have credentials)
- Run tests and evaluations

## What Requires You
- PhysioNet account registration + CITI training (for MIMIC/LVEF)
- Upload files to Colab/Drive
- Stop Colab run when local finishes (or vice versa)
- App UI decisions
