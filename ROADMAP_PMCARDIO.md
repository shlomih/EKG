# EKG Intelligence — PMcardio Gap Analysis & Roadmap

*Last updated: 2026-03-26*

---

## What PMcardio Has (Summary)

| Capability | PMcardio | Us (today) |
|---|---|---|
| Conditions detected | **49** | ~30 (12 ML + ~18 rule-based) |
| Training data | **1M+ ECGs** | ~18k PTB-XL + 45k Chapman (soon) |
| AFIB | Yes (97% sensitivity) | Rule-based screen only |
| STEMI / OMI without STE | Yes (Queen of Hearts) | IMI + ASMI only |
| Explainability heatmaps | Yes (lead-by-lead) | None |
| Mobile app | iOS + Android | Streamlit web only |
| Photo capture (paper ECG) | Yes | Yes (digitization_pipeline.py) |
| PDF reports | Yes | Yes |
| EHR/EMR integration | Yes | No |
| Regulatory approval | CE marked (EU) | None |
| Clinical validation papers | 15+ | 0 |
| ESC guideline references | Yes | Partial |

---

## What We Already Have That PMcardio Has

- **12-lead signal analysis** — full 10-second, 500Hz, all leads
- **Photo / paper ECG digitization** — `digitization_pipeline.py`
- **PDF reports** — full clinical report with ECG plot, intervals, findings
- **QTc critical alerting** — sex-specific thresholds, Torsades risk flag
- **Bradycardia / tachycardia** — age-aware thresholds including athlete suppression
- **RR irregularity screen** — crude AF screening via RR variability
- **Short QT** — QTc < 350ms flagged
- **PR prolongation** — 1AVB, marked prolongation
- **Hyperkalemia / hypokalemia** — electrolyte integration with ECG pattern cross-check
- **Patient context adjustments** — pacemaker, athlete, pregnant, K+, age, sex
- **Axis deviation** — left, right, extreme (northwest)
- **Low voltage** — limb + precordial criteria
- **T-wave inversions** — per lead
- **Poor R-wave progression** — V1-V4
- **RVH screening** — dominant R in V1 + right axis (just added)
- **WPW screening** — short PR + wide QRS combined flag (just added)

---

## Gap Analysis: Conditions to Close

### Group A — High Clinical Value, Implement Soon

| Condition | How | Status |
|---|---|---|
| **AFIB** | ML model on Chapman-Shaoxing (3,889 cases) | Phase 3A — in progress |
| **Sinus tachycardia (STACH)** | Retrain after Chapman merge (2,760 cases) | Phase 3A — in progress |
| **Atrial flutter (AFLT)** | Include in Chapman training or rule-based (atrial rate ~300 bpm, 2:1 block) | Phase 3B |
| **VT (ventricular tachycardia)** | PhysioNet CinC 2017 dataset — wide complex tachycardia | Phase 3C |
| **Posterior STEMI** | Dominant R in V1-V2 + ST depression in V1-V3 → rule-based | Phase 3A short win |
| **Hyperacute T-waves (de Winter)** | Upsloping ST depression + peaked T-wave in precordials → rule-based | Phase 3A short win |
| **Pericarditis** | Diffuse ST elevation (saddle-shaped) + PR depression → rule-based | Phase 3B |

### Group B — Medium-term, Need Data

| Condition | How | Status |
|---|---|---|
| **SVT** | PhysioNet CinC 2020 (43k ECGs, 27 conditions) | Phase 3C |
| **VF / pulseless VT** | CinC 2017 + CPSC 2018 | Phase 3C |
| **Left atrial enlargement (LAE)** | P-wave morphology (wide bifid P in II, V1 biphasic) — needs P-wave delineation | Phase 3B |
| **Right atrial enlargement (RAE)** | Peaked P > 2.5 mm in II — rule-based | Phase 3B |
| **Brugada pattern** | Coved ST elevation in V1-V2 — rule-based or ML | Phase 3C |
| **WPW (confirmed, not just screen)** | Combine short PR + delta wave morphology detection | Phase 3B |
| **Long QT syndrome (diagnosis)** | Currently rule-based alert only — add to ML label set | Phase 3C |
| **RVH (confirmed)** | Currently rule-based screen — improve with ML | Phase 3C |
| **Epsilon wave** | ARVC pattern in V1-V3 | Phase 4 |

### Group C — Longer Term

| Condition | Notes |
|---|---|
| **OMI without ST elevation** | PMcardio's #1 differentiator. Requires massive training data + specialized model |
| **Hyperacute STEMI (early)** | ST morphology + symmetry analysis |
| **Osborn (J-wave) — Hypothermia** | Notched J-point in lateral leads |
| **Digitalis effect** | Scooped ST depression |

---

## Quick Wins Already Implemented This Session

- [x] **RVH screening** — dominant R in V1 + right axis deviation → `clinical_rules.py`
- [x] **WPW combined flag** — replaces generic SHORT_PR when QRS is also wide → `interval_calculator.py`
- [x] **Posterior STEMI hint** — Poor R-wave progression already flags, add posterior STEMI text
- [x] **App summary header** — "2 critical, 1 abnormal" before condition cards → `app.py`

---

## Phase 3A — Next 2 Weeks (Chapman-Shaoxing merge)

**Goal:** Add AFIB and STACH to the ML model. Expand 12 → 14 conditions.

1. Download Chapman-Shaoxing (20 GB): `python dataset_chapman.py --download`
2. Build index: `python dataset_chapman.py --index`
3. Retrain `multilabel_classifier.py` on merged PTB-XL + Chapman
4. Target: AFIB AUROC > 0.95, STACH AUROC > 0.92

**Short wins before retraining (rule-based, no new data needed):**
- Posterior STEMI: add to `clinical_rules.py` — tall R in V1-V2 + ST depression V1-V3
- Hyperacute T-waves: add to `clinical_rules.py` — upsloping ST + peaked T in V2-V4
- RAE: add to `clinical_rules.py` — peaked P > 2.5mm in II (P-wave amplitude check)

---

## Phase 3B — More Datasets (2-4 weeks)

**Target datasets:**

| Dataset | Records | Key Conditions | Source |
|---|---|---|---|
| PhysioNet CinC 2020 | 43,101 | 27 conditions, AFIB, VT, SVT | physionet.org/content/challenge-2020 |
| CPSC 2018 | 6,877 | 9 conditions, AF, VT | physionet.org/content/cpsc2018 |
| Georgia 12-lead | 10,344 | 27 conditions | Included in CinC 2020 |
| CODE-15% | 345,779 | 6 rhythm conditions | Brazil, large scale |

**Combined:** ~400k ECGs → closes most of the data gap vs PMcardio's 1M+

**Label expansion:** 14 → 20+ conditions after merging these datasets

---

## Phase 3C — Explainability (1-2 weeks engineering)

**PMcardio's ECGxplain equivalent: Grad-CAM heatmaps**

- Implementation: Grad-CAM on the 1D CNN (`ECGNetJoint`)
- Output: per-lead saliency score — which part of each lead's signal drove the prediction
- Display: color overlay on ECG plot in the app
- Effort: ~200 lines, can reuse existing ECG plot code

**Why this matters:** Clinicians don't trust black-box AI. Heatmaps show "LBBB was detected because of QRS morphology in V1-V3" — same logic PMcardio uses to win clinical adoption.

```python
# Architecture sketch
class GradCAM1D:
    def __init__(self, model, target_layer_name="layer3"):
        self.model = model
        self.hooks = ...

    def compute(self, sig_t, aux_t, class_idx):
        # Forward pass, capture activations
        # Backward pass, capture gradients
        # Weight activations by gradients
        # Return (12, 5000) saliency map
        ...
```

---

## Phase 4 — Mobile App

**Architecture:** FastAPI backend + React Native frontend

1. Convert Streamlit → FastAPI REST endpoint (`POST /analyze`)
2. Frontend: React Native or Flutter
3. Camera module: capture paper ECG → digitization pipeline
4. PWA as intermediate step (before native app store submission)

**Effort:** 4-6 weeks (significant engineering investment)

---

## Phase 5 — Clinical Validation

**To match PMcardio's credibility:**
1. Prospective study on real-world ECGs (partner with a clinic)
2. Target: AFIB sensitivity >95%, STEMI sensitivity >90%
3. Submit to JMIR, Europace, or Annals of Internal Medicine
4. Regulatory path: CE mark (EU) → FDA 510(k) (US)

**Why this is the hardest part:** PMcardio spent years here. Their 15+ papers took ~3 years.

---

## Priority Stack (what to do next, in order)

1. **Chapman download completes** → retrain → AFIB in ML model (highest clinical value)
2. **Posterior STEMI rule** → add to `clinical_rules.py` (1-2 hours, zero new data)
3. **RAE rule** → add to `clinical_rules.py` (< 1 hour)
4. **Hyperacute T-wave rule** → add to `clinical_rules.py` (1-2 hours)
5. **CinC 2020 dataset** → download and integrate → expands label set significantly
6. **Grad-CAM explainability** → biggest UX differentiator we can build quickly
7. **FastAPI backend** → enables mobile app path

---

## Our Differentiation vs PMcardio

Things we have / can build that PMcardio doesn't explicitly highlight:

- **Patient context engine** — pacemaker, athlete, K+, pregnancy, age modify every finding
- **Rule + ML hybrid** — rule-based checks catch things the ML doesn't (axis, voltage, electrolytes)
- **Transparent condition reporting** — each finding has action + note + context shown
- **Offline-capable** (CPU-only inference) — works without cloud connection
- **Open source friendly** — can run in any healthcare setting without vendor lock-in

These are real clinical differentiators to emphasize when seeking adoption.
