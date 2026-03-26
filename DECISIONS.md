# EKG Intelligence Platform — Decision Log

This document captures the key technical decisions made during development, the rationale behind them, and the outcomes. It is intended as a living document — update it whenever a significant decision is made or a result changes the direction.

---

## Project Goal

Build an AI-powered 12-lead ECG interpretation system, starting as a Streamlit POC and targeting a native mobile app (Android/iOS). The system must be deployable without a GPU (CPU-only constraint during development).

**Target users:** Healthcare professionals in clinical settings.
**Competitive benchmark:** PMcardio (38+ conditions), ECG Buddy (arrhythmia-focused), Tricog (STEMI-focused).

---

## Decision Log

---

### D-001 · Architecture: 1D CNN with auxiliary voltage features
**Date:** Early development
**Decision:** Use a 1D ResNet-style CNN operating on raw 5000-sample 12-lead signals, with a separate auxiliary branch for hand-crafted voltage/demographic features. Not a pure end-to-end model.

**Rationale:**
- Raw signal + hand-crafted features outperforms either alone for HYP detection
- Voltage criteria (Sokolow-Lyon, Cornell) are well-validated clinically — encoding them explicitly rather than hoping the CNN rediscovers them gives a reliable floor
- 1D CNN is fast enough for CPU-only inference (<100ms per prediction)

**Outcome:** v10 CNN reached 72.6% accuracy, HYP F1=0.442 on PTB-XL test fold 10 (10-fold stratified CV).

---

### D-002 · Training data: PTB-XL as primary dataset
**Date:** Early development
**Decision:** Use PTB-XL (21,837 12-lead ECGs from German hospital) as the primary training dataset. Use the standard 10-fold stratified split; fold 10 = held-out test, fold 9 = validation, folds 1–8 = train.

**Rationale:**
- Largest open ECG dataset with standardized SCP diagnostic codes and fold assignments
- 500Hz, 10s recordings — matches clinical standard
- 10-fold split is widely used in ECG literature, enables fair comparison

**Alternative considered:** PhysioNet CinC datasets. Rejected because they use different label schemes, require significant harmonization, and are smaller.

**Outcome:** All models trained and evaluated on this split. Allows reproducible comparison.

---

### D-003 · Loss function: Asymmetric Loss (not CrossEntropy)
**Date:** v4 training
**Decision:** Replace standard CrossEntropyLoss with AsymmetricLoss (γ_neg=4, γ_pos=0, margin=0.05) + label smoothing.

**Rationale:**
- HYP is the hardest class (rare, subtle voltage changes). Standard CE underweights it.
- AsymmetricLoss applies heavier penalty to false negatives than false positives — improves minority class recall
- γ_neg=4 suppresses easy negatives from dominating the gradient
- Focal loss was tried first (D-004 below) but asymmetric gave better HYP F1

**Outcome:** HYP F1 improved from ~0.30 (CE) to ~0.39 (focal) to ~0.44 (asymmetric).

---

### D-004 · HYP class: ensemble of v9 + v10 models
**Date:** v9/v10 training
**Decision:** Run two independently trained models (v9 and v10) and ensemble their outputs with per-class optimized thresholds. HYP threshold = 0.45.

**Rationale:**
- v9 (focal loss) and v10 (asymmetric loss + amplitude-preserving normalization) have uncorrelated errors — ensemble reduces variance
- Per-class threshold tuning on validation fold 9 is principled and avoids leaking test fold
- HYP threshold 0.45 (vs default 0.5) trades precision for recall — appropriate for a screening tool where missing HYP is more costly than a false positive

**Outcome:** Ensemble HYP F1=0.456, MacroF1=0.675 on test fold 10. Best result with the 5-superclass scheme.

---

### D-005 · Foundation model: ECG-FM (MIMIC-IV pretrained wav2vec2)
**Date:** Phase 2 start
**Decision:** Integrate ECG-FM — a 90.4M parameter wav2vec2-style transformer pretrained on 800k ECGs from MIMIC-IV — as an alternative to the CNN backbone.

**Rationale:**
- Self-supervised pretraining on orders-of-magnitude more data should yield richer representations
- Published results showed ECG-FM achieving state-of-the-art on PTB-XL with fine-tuning
- A foundation model approach is closer to what leading academic groups are doing

**Implementation note:** Rebuilt ECG-FM as a standalone PyTorch module (`ecgfm_encoder.py`) — no dependency on fairseq. Loads 207 encoder tensors from the pretrained checkpoint.

**Stage 1 result (frozen encoder, linear probe):** HYP F1=0.375 test / 0.500 val.
**Stage 2 result:** See D-006.

---

### D-006 · ECG-FM Stage 2: Kill after epoch 1
**Date:** 2026-03-26
**Decision:** Terminate ECG-FM Stage 2 fine-tuning after a single epoch.

**Rationale:**
- Stage 2 epoch 1 on CPU took ~9.5 hours. With 20 epochs max and patience=8, full training = potentially 9+ days. Not feasible on CPU.
- Partial-freeze (top 4 of 12 transformer layers) reduced trainable params from 90M→28.6M but only saved ~14% wall-clock time: the bottleneck is the forward pass through all 12 layers, not the backward pass.
- Epoch 1 MacroF1=0.398 — worse than Stage 1 baseline (0.550). Suggests model needs many more epochs to recover, which is impractical.
- The multi-label CNN (D-007) completed overnight with MacroAUROC=0.972, making ECG-FM Stage 2 a lower priority.

**When to revisit:** If a GPU becomes available (even a modest one — T4 on Colab would reduce epoch time from 9h to ~15min, making Stage 2 viable in an afternoon).

**ECG-FM status:** Remains in the codebase as `ecgfm_encoder.py` and `ecgfm_finetune.py` with full resume capability. Stage 1 checkpoint saved at `models/ecgfm_stage1.pt`.

---

### D-007 · Expand from 5 superclasses → 12 specific conditions (multi-label)
**Date:** 2026-03-26
**Decision:** Replace the 5 PTB-XL superclass model with a 12-condition multi-label classifier (`multilabel_classifier.py`).

**Rationale:**
- The 5 superclasses (NORM, MI, STTC, HYP, CD) are coarse aggregations — multiple clinically distinct conditions map to the same class (e.g., LBBB and RBBB both → CD; LVH and RVH both → HYP)
- A single ECG can have multiple simultaneous conditions; multi-label (sigmoid + BCE) is the correct framing
- Clinicians need specific diagnoses, not superclasses — "Complete LBBB" is actionable; "Conduction Disturbance" is not
- PTB-XL SCP codes give fine-grained labels for free; no new data needed

**The 12 conditions chosen** (all with ≥536 samples at ≥50% confidence in PTB-XL):
NORM, PVC, LVH, IMI, ASMI, CLBBB, CRBBB, LAFB, 1AVB, ISC_, NDT, IRBBB

**Why not AFIB?** PTB-XL only has 48 confirmed AFIB cases at ≥50% confidence (1,466 are annotated at 0% — i.e., the annotators considered and rejected AFIB). 48 samples is insufficient for training. AFIB is on the roadmap for Phase 3 using the Chapman-Shaoxing dataset (~10k arrhythmia-labeled ECGs including many AF cases).

**Why not STACH (sinus tachycardia)?** Same issue: only 4 confirmed cases at ≥50% confidence in PTB-XL.

**Architecture:** Same ECGNetJoint backbone (1D ResNet + voltage aux branch) from v10, with the final Linear(288→5) replaced by Linear(288→12) + sigmoid. BCEWithLogitsLoss with per-class positive-frequency weighting. No structural change to the CNN — just a different output head and loss.

**Result:** MacroAUROC=0.972, MacroF1=0.699 on test fold 10.

| Condition | AUROC | F1 | Note |
|---|---|---|---|
| CLBBB | 0.997 | 0.936 | Near-perfect — distinctive QRS morphology |
| CRBBB | 0.997 | 0.824 | Near-perfect |
| LAFB | 0.989 | 0.711 | |
| PVC | 0.987 | 0.786 | |
| ASMI | 0.980 | 0.702 | |
| IRBBB | 0.977 | 0.593 | |
| 1AVB | 0.972 | 0.465 | F1 low — PR interval detection is subtle |
| ISC_ | 0.969 | 0.657 | |
| NORM | 0.964 | 0.896 | |
| LVH | 0.955 | 0.674 | |
| IMI | 0.947 | 0.581 | |
| NDT | 0.936 | 0.562 | Non-diagnostic T changes — inherently noisy label |

**Training:** 35 epochs (early stop), batch=64, AdamW lr=3e-4, OneCycleLR, ~2 min/epoch on CPU.

---

## Current Production Stack (as of 2026-03-26)

```
app.py  →  multilabel_classifier.py  →  ECGNetJoint (v10 backbone, 12-class head)
                                          + extract_voltage_features (14-dim aux)
        →  [fallback] ensemble_classifier.py (v9+v10, 5-class)
        →  [fallback] cnn_classifier.py (v10, 5-class)
```

Supporting:
- `clinical_rules.py` — rule-based axis, ST, T-wave findings (independent of ML)
- `digitization_pipeline.py` — paper ECG scan → digital signal
- `report_generator.py` — PDF reports
- `database_setup.py` — SQLite patient/records DB
- `ecgfm_encoder.py` + `ecgfm_finetune.py` — ECG-FM integration (GPU path, dormant)

---

## Roadmap

### Phase 3A — AFIB detection (highest clinical value gap)
Add Chapman-Shaoxing 12-lead dataset (~10k ECGs, detailed arrhythmia labels including AF).
Train a dedicated AFIB detector or retrain the multi-label model with the merged dataset.
Target: AFIB AUROC > 0.95.

### Phase 3B — ECG-FM on GPU
If a GPU is available, resume ECG-FM Stage 2 fine-tuning:
```bash
python -u ecgfm_finetune.py --stage 2
```
Stage 1 checkpoint is saved. Stage 2 is resumable. Expected ~15 min/epoch on a T4.
Target: HYP F1 > 0.52, MacroF1 > 0.72 — would justify replacing the CNN backbone.

### Phase 3C — App UX: condition-specific clinical guidance
For each detected condition, display:
- Brief explanation (what it means clinically)
- Key clinical actions (e.g., "CLBBB: rule out acute MI — check prior ECGs")
- Severity tier (Critical / Abnormal / Mild)

### Phase 4 — Mobile app
Convert Streamlit POC → FastAPI backend + React Native or Flutter frontend.
PWA as an intermediate step.

### Phase 5 — Signal quality & robustness
- Noise level detection (SNR per lead)
- Lead-off / poor contact detection
- Baseline wander correction
- Confidence calibration (Platt scaling per condition)

---

## Key Metrics History

| Model | HYP F1 | MacroF1 | Notes |
|---|---|---|---|
| v4 CNN (focal loss) | 0.39 | ~0.62 | First working model |
| v9 CNN (focal + HYP OS) | 0.442 | 0.654 | Aggressive HYP oversampling |
| v10 CNN (asymmetric loss) | 0.442 | 0.654 | Amplitude-preserving norm |
| v9+v10 Ensemble | 0.456 | 0.675 | Per-class threshold tuning |
| ECG-FM Stage 1 (frozen) | 0.375 | 0.492 | Linear probe only |
| **Multi-label CNN (12 class)** | **—** | **0.699** | **MacroAUROC=0.972** |

Note: Multi-label F1 is not directly comparable to 5-class F1 (different task). The multi-label model is the current production model.
