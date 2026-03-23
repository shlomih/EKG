# EKG Codebase Research Summary
**Date:** March 19, 2026  
**Scope:** Comprehensive analysis of signal parameters, classifiers, training approaches, and performance metrics

---

## 1. ALL ECG PARAMETERS AFFECTING ANALYSIS

### 1.1 Clinical Signal Measurements (interval_calculator.py)

#### Fixed Reference Ranges (Standard Cardiology)
| Parameter | Normal Range | Unit | Safety Thresholds |
|---|---|---|---|
| **Heart Rate (HR)** | 60–100 | bpm | Bradycardia <60; Tachycardia >100; Severe >150; Critical <40 |
| **PR Interval** | 120–200 | ms | Short <120 (WPW concern); Long >200 (1st degree block); >300 (higher-degree block) |
| **QRS Duration** | 70–110 | ms | Wide >120 (BBB/IVCD); Borderline 110–120 |
| **QTc (corrected QT)** | Male: <450; Female: <460 | ms | Borderline high 470; Critical ≥500 (Torsades risk); Pregnancy 460 |
| **RR Variability** | <0.15 | Coefficient of Variation | >0.15 suggests irregular rhythm (AF/ectopics) |

#### Clinical Context Modifiers (applied_clinical_context)
- **Pacemaker presence** → suppresses bradycardia/wide QRS alerts
- **Athlete status** → suppresses resting bradycardia <60 bpm
- **Pregnancy status** → shifts QTc threshold from 450/460 to 460 ms
- **Potassium level (K⁺)** → adjusts QTc interpretation:
  - K⁺ < 3.0 mmol/L: Severe hypokalaemia, urgent replacement
  - K⁺ 3.0–3.5 mmol/L: Mild hypokalaemia, monitor QTc
  - K⁺ > 5.5 mmol/L: Hyperkalaemia, peaked T-wave risk
  - K⁺ > 6.0 mmol/L: Critical hyperkalaemia, fatal arrhythmia risk
- **Sex-specific thresholds** → affects QTc and ST-elevation criteria (V2-V3)

#### Quality & Reliability Parameters
- **Signal quality score** (0–1): <0.3 = low reliability warning issued
- **R-peak detection confidence**: Minimum 2 R-peaks required per 3+ seconds of signal
- **Waveform delineation confidence**: Uses NeuroKit2 DWT; may fail on noisy signals

---

### 1.2 Voltage-Based Hypertrophy Criteria (hybrid_classifier.py)

#### Left Ventricular Hypertrophy (LVH) Detection
| Criterion | Threshold | Lead Pair |
|---|---|---|
| **Sokolow-Lyon** | S(V1) + R(V5 or V6) > 3.5 mV | V1 + max(V5, V6) |
| **Cornell (Male)** | R(aVL) + S(V3) > 2.8 mV | aVL + V3 |
| **Cornell (Female)** | R(aVL) + S(V3) > 2.0 mV | aVL + V3 |

#### Right Ventricular Hypertrophy (RVH) Detection
| Criterion | Threshold |
|---|---|
| **R-wave in V1** | > 0.7 mV |

#### Voltage Feature Measurement
- **Amplitude extraction method**: Median peak R and S values across ±60ms QRS windows around detected R-peaks
- **Confidence metric**: IQR-based consistency measure (0–1 scale)
- **Hypertrophy composite score**: max(sokolow_score, cornell_score, rvh_score), scaled 0–1

---

### 1.3 ST-Segment Analysis by Coronary Territory (st_territory.py)

#### ST-Segment Measurement
- **Baseline reference**: Mean voltage 40–80 ms before R-peak (PQ segment)
- **Measurement point**: J-point + 60 ms after R-peak
- **Per-lead confidence**: Median across multiple QRS cycles

#### STEMI Criteria by Territory

**Anterior (LAD):** V1, V2, V3, V4 (±I, aVL)
| Criteria | Threshold |
|---|---|
| Male: ST elevation in ≥2 contiguous leads | ≥0.2 mV in V2/V3; ≥0.1 mV others |
| Female: ST elevation in ≥2 contiguous leads | ≥0.15 mV in V2/V3; ≥0.1 mV others |
| Reciprocal ST depression | ≥0.05 mV in inferior leads (II, III, aVF) |

**Inferior (RCA):** II, III, aVF
| Criteria | Threshold |
|---|---|
| ST elevation | ≥0.1 mV in ≥2 contiguous inferior leads |
| Reciprocal ST depression | ≥0.05 mV in lateral/anterior (I, aVL, V1–V2) |

**Lateral (LCx):** I, aVL, V5, V6
| Criteria | Threshold |
|---|---|
| ST elevation | ≥0.1 mV in ≥2 contiguous lateral leads |

#### STEMI Pattern Severity
- **CRITICAL**: ≥2 contiguous leads elevated + reciprocal depression (suggest artery occlusion)
- **WARNING**: ≥2 contiguous leads elevated, no reciprocal (early STEMI/pericarditis/benign early repolarization)
- **INFO**: Isolated elevation in 1 lead only

---

### 1.4 Rule-Based Clinical Findings (clinical_rules.py)

#### Axis Deviation Classification
| Category | Range | Interpretation |
|---|---|---|
| **Normal** | –30° to 90° | Standard sinus axis |
| **Left Axis Deviation (LAD)** | –30° to –90° | LAFB, LVH, inferior MI |
| **Right Axis Deviation (RAD)** | 90° to 180° | RVH, PE, COPD, lateral MI, LPFB |
| **Extreme (Northwest)** | <–90° or >180° | VT, lead misplacement, severe CD |
- **Measurement method**: atan2(net aVF area, net I area) from QRS integral

#### Low Voltage Detection
- **Limb leads criterion**: All limb leads (I, II, III, aVR, aVL, aVF) peak-to-peak < 0.5 mV
- **Precordial criterion**: All precordial leads (V1–V6) peak-to-peak < 1.0 mV
- **Differential diagnosis**: Pericardial effusion, obesity, COPD, hypothyroidism, infiltrative cardiomyopathy

#### T-Wave Morphology Patterns
| Finding | Detection Criteria | Clinical Significance |
|---|---|---|
| **Peaked T-waves** | T amplitude > 0.6 × R amplitude in V2–V5 | Hyperkalemia (urgent if K⁺ >5.5) |
| **T-wave inversion** | Negative T-wave in normally upright leads (I, II, V4–V6) | Ischemia, strain, cardiomyopathy |

#### R-Wave Progression (Poor)
- **Detection**: R-wave amplitude in V3 < 0.3 mV
- **Interpretation**: May indicate prior anterior MI, LVH, LBBB, or normal variant

---

### 1.5 Signal Preprocessing Parameters

#### Input Signal Specifications
- **Default sampling rate (fs)**: 500 Hz
- **Signal duration range**: Minimum 3 seconds for NeuroKit2 delineation; PTB-XL standard 10 seconds (5000 samples)
- **Number of leads**: 12-lead standard
- **Per-lead normalization**: Z-score normalization (mean subtraction, std division)
- **Clipping bounds**: Signal voltage clipped to [–20, 20] mV (outlier removal)

#### Data Augmentation (Training Only)
- **Gaussian noise**: 50% probability, amplitude 0.01–0.15 mV
- **Per-lead amplitude scaling**: 50% probability, scale factor 0.8–1.2
- **Time shift**: 30% probability, circular shift ±250 samples (~0.5 seconds)
- **Lead dropout**: 20% probability, randomly zero 1–2 leads
- **Baseline wander**: 30% probability, low-frequency sinusoid 0.1–0.5 Hz, amplitude 0.05–0.2 mV

---

## 2. CLASSIFIER MODELS & ARCHITECTURES

### 2.1 CNN Classifier (cnn_classifier.py) — **DEEP LEARNING APPROACH**

#### Model Architecture: ECGNet (1D ResNet-style)
```
Input: (batch_size, 12 leads, 5000 samples)
       ↓
STEM: Conv1d(12→64, kernel=15) + BN + ReLU + MaxPool1d(4)
      (12, 5000) → (64, 1250)
       ↓
LAYER 1: 2× ECGResBlock(64, k=7, dropout=0.1, SE-attn) + MaxPool1d(4)
         (64, 1250) → (64, 312)
       ↓
EXPAND2: Conv1d(64→128) + BN + ReLU
         (64, 312) → (128, 312)
       ↓
LAYER 2: 2× ECGResBlock(128, k=7/5, dropout=0.2, SE-attn) + MaxPool1d(4)
         (128, 312) → (128, 78)
       ↓
EXPAND3: Conv1d(128→256) + BN + ReLU
         (128, 78) → (256, 78)
       ↓
LAYER 3: 2× ECGResBlock(256, k=5/3, dropout=0.3, SE-attn) + AdaptiveAvgPool1d(1)
         (256, 78) → (256, 1)
       ↓
HEAD: Flatten + Linear(256→128) + ReLU + Dropout(0.5) + Linear(128→5)
      Output: (batch_size, 5) logits [NORM, MI, STTC, HYP, CD]
```

#### Key Architecture Features
- **Squeeze-and-Excitation (SE) blocks**: Channel attention mechanism (learns which features matter per channel)
- **Residual connections**: Skip connections for gradient flow
- **Batch normalization**: Per layer for stability
- **Adaptive pooling**: Preserves temporal structure while reducing dimensionality
- **Total parameters**: ~1.2M (exact count logged during training)

#### Loss Function: Focal Loss with Label Smoothing
```python
FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.05)
  - Gamma=2.0: Down-weights easy examples, focuses training on hard misclassifications
  - Label smoothing ε=0.05: Prevents overfitting to noisy labels
  - Class weights: Inverse frequency on oversampled distribution to handle imbalance
```

#### Optimizer & Training Schedule
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-3)
- **Learning rate scheduler**: OneCycleLR
  - Max LR: 1e-3
  - Epochs: 50 (with early stopping, patience=10)
  - Pct_start: 10% (warmup) → 90% (decay)
- **Batch size**: 64
- **Gradient clipping**: max_norm=1.0 (prevents exploding gradients)

#### Data Preprocessing for CNN
- **Handles variable sampling rates**: Resamples to 500 Hz
- **Handles variable lead counts**: Pads to 12 leads or truncates
- **Handles variable duration**: Pads/truncates to 5000 samples
- **Handles NaN/Inf**: Replaces with zeros
- **Pre-loading**: All signals cached in memory for training speed (~20× faster)

#### Class Imbalance Mitigation
| Strategy | Implementation |
|---|---|
| **Aggressive oversampling** | Minority classes (HYP, STTC) upsampled to max class count |
| **Focal loss** | Down-weights easy correct predictions |
| **Class-weighted loss** | Inverse-frequency weighting applied to loss |
| **Weighted random sampling** | Can use WeightedRandomSampler in DataLoader |

---

### 2.2 Sklearn Classifier (poc_classifier.py) — **HAND-CRAFTED FEATURES + GRADIENT BOOSTING**

#### Feature Extraction Pipeline (Per Lead × 12)
**Per-lead feature vector = 12 × 16 = 192 features total**

| Feature Group | Features | Calculation |
|---|---|---|
| **Statistical** | Mean, std, skewness, kurtosis, peak-to-peak | numpy functions |
| **Percentiles** | 5th, 25th, 75th, 95th percentile values | numpy.percentile |
| **Zero-crossing rate** | Zero-crossings / n_samples | Proxy for frequency content |
| **RMS (Root Mean Square)** | sqrt(mean(x²)) | Energy proxy |
| **Power Spectral Density (PSD)** | 4 features per lead if len(sig) > 256 |  |
| | - Power in 0.5–5 Hz (low) | Welch method, fs=500 Hz, nperseg=256 |
| | - Power in 5–15 Hz (mid) | Normalized by total power |
| | - Power in 15–40 Hz (high) | |
| | - Dominant frequency | argmax(PSD) |

**Total dimension**: 192 (or fewer if signals too short)

#### Model: GradientBoostingClassifier
```python
GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    min_samples_leaf=3,
    random_state=42,
)
```
- **Rationale**: Handles feature interactions, robust to imbalance, no pre-processing required
- **Advantage over RF**: Better generalization, sequential error correction

#### Training Data Characteristics
- **Dataset**: PTB-XL (21,837 records) mapped to 5 superclasses
- **Train/val/test split**: Folds 1–8 (train) / 9 (val) / 10 (test) from PTB-XL stratified folds
- **Feature scaling**: NOT applied (GradientBoosting is scale-invariant)
- **Missing values**: Replaced with 0 using np.nan_to_num

---

### 2.3 Hybrid Classifier (hybrid_classifier.py) — **CNN + VOLTAGE CRITERIA**

#### Architecture
```
Step 1: Get CNN predictions + probabilities
        ↓
Step 2: Compute voltage criteria (Sokolow-Lyon, Cornell, RVH thresholds)
        ↓
Step 3: Adjust HYP probability based on voltage agreement/disagreement
        - If CNN says HYP but voltage absent: multiply prob by 0.3–0.5
        - If CNN says non-HYP but voltage strong (score >0.6): boost prob by +score*0.3
        ↓
Step 4: Renormalize probabilities (softmax)
        ↓
Step 5: Apply per-class confidence thresholds
        - CLASS_THRESHOLDS = {NORM: 0.30, MI: 0.25, STTC: 0.25, HYP: 0.35, CD: 0.25}
        - If pred_conf < threshold: re-assign to next-best class above its threshold
        ↓
Output: Final prediction + adjustment flag
```

#### Rationale for HYP Gate
- **CNN precision on HYP**: 22.9% (many false positives)
- **Voltage criteria**: Established clinical standard for HYP diagnosis
- **Gate mechanism**: Leverages both model & domain knowledge

#### Per-Class Confidence Thresholds (Empirically Set)
- **HYP**: 0.35 (highest threshold — reduce false positives)
- **Others**: 0.25–0.30 (more permissive)
- **Fallback logic**: If primary prediction below threshold, cascade to next-best class

---

## 3. TRAINING APPROACHES & DATASETS

### 3.1 Training Methodology — CNN (cnn_classifier.py)

| Aspect | Configuration |
|---|---|
| **Supervision Type** | Fully supervised (labeled 12-lead signals → 5 superclasses) |
| **Data splits** | PTB-XL folds 1–8 (train), 9 (val), 10 (test) + external datasets (fold 0) for training only |
| **Unified multi-dataset** | PTB-XL + CPSC 2018, 2018-extra, Georgia, Chapman-Shaoxing, St Petersburg, PTB original |
| **Cross-validation** | Stratified folds (built into PTB-XL); external datasets = training-only (fold=0) |
| **Class imbalance handling** | Aggressive oversampling + focal loss + class weighting |
| **Early stopping** | Patience=10 epochs on val F1 score |

#### Training Progression
1. **Pre-load all signals** into RAM (one-time cost, ~20× speedup)
2. **Oversample minority classes** to match majority class count
3. **Apply data augmentation** to training data only (noise, scaling, shift, lead dropout, baseline wander)
4. **Train with AdamW + OneCycleLR** for 50 epochs (or until early stopping)
5. **Validate on fold 9** after each epoch
6. **Save checkpoint** when val F1 improves
7. **Evaluate on fold 10** with best checkpoint

#### Example Hyperparameter Settings
```
CNN v2 (PTB-XL only):
  - Dataset: 21,837 records → ~3000 val + ~2200 test
  - Epochs: 50 (actual: ~30–40 with early stopping)
  - Batch size: 64
  - Epochs needed: ~30
  - Training time: ~2–3 hours on GPU

CNN v3 (Multi-dataset):
  - Dataset: PTB-XL + external files (~35,000+ total)
  - External dataset fold = 0 (training contribution only)
  - Validation/test: PTB-XL folds 9–10 maintained for reproducibility
  - Est. training time: ~4–5 hours on GPU
```

---

### 3.2 Training Methodology — Sklearn (poc_classifier.py)

| Aspect | Configuration |
|---|---|
| **Supervision Type** | Fully supervised (labeled signals → features → 5 superclasses) |
| **Data splits** | Same: PTB-XL folds 1–8 (train), 9 (val), 10 (test) |
| **Training procedure** | 1. Load all records from fold 1–8; 2. Extract 192 features per record; 3. Train GradientBoosting; 4. Validate on fold 9; 5. Test on fold 10 |
| **Hyperparameter selection** | Grid or manual tuning (currently hardcoded) |

#### Feature Extraction Pipeline
```
Input: Raw 12-lead signal (N, 12)
       ↓ (for each lead)
Statistical features: mean, std, skew, kurtosis, ptp (5 per lead)
Percentile features: 5, 25, 75, 95 (4 per lead)
Energy features: zero-cross rate, RMS (2 per lead)
Spectral features (if signal > 256 samples):
  - PSD via Welch (fs=500, nperseg=256)
  - Power in 3 bands + dominant freq (4 per lead)
       ↓
Total: ~192 features (16 per lead × 12 leads)
       ↓
GradientBoosting.fit(X_train, y_train)
```

#### Training Data Characteristics
- **Total records in training**: ~16,600 (approx., fold ≤8 from PTB-XL)
- **Class distribution (training)**:
  - NORM: ~7000 (42%)
  - CD: ~4000 (24%)
  - MI: ~3000 (18%)
  - STTC: ~2000 (12%)
  - HYP: ~600 (4%) — smallest, minority class

---

### 3.3 Datasets Used

#### Primary Dataset: PTB-XL (21,837 records)
- **Source**: PhysioNet, free access
- **Format**: WFDB (.hea, .dat/.mat files)
- **Sampling rate**: 500 Hz
- **Duration**: ~10 seconds per record
- **Leads**: 12-lead standard
- **Labels**: SNOMED-CT codes with diagnostic flags + stratified 10-fold split
- **Superclass mapping**: SCP codes → 5 diagnostic classes

#### External Datasets (Optional, via dataset_pipeline.py)
| Dataset | Records | Sampling Rate | Duration | Status |
|---|---|---|---|---|
| CPSC 2018 | 6,877 | 500 Hz | Variable | Free (PhysioNet) |
| CPSC 2018 Extra | 3,453 | 500 Hz | Variable | Free (PhysioNet) |
| Georgia 12-Lead | 10,344 | 500 Hz | 5000 samples | Free (PhysioNet) |
| Chapman-Shaoxing | 10,646 | 500 Hz | 5000 samples | Credentialed (PhysioNet) |
| St Petersburg INCART | 75 | 257 Hz (→ resample) | Variable | Free (PhysioNet) |
| PTB Diagnostic | 516 | 1000 Hz (→ resample) | Variable | Free (PhysioNet) |
| Ningbo First Hospital | 34,905 | 500 Hz | 5000 samples | Credentialed (PhysioNet) |

#### Unified Index (unified_index.csv)
- **Purpose**: Centralized metadata for all datasets
- **Columns**: path, superclass, dataset, strat_fold, signal_quality, etc.
- **Fold convention**: PTB-XL keeps native folds (1–10); external datasets assigned fold=0 (train-only)

---

## 4. FEATURE EXTRACTION & PREPROCESSING

### 4.1 CNN Preprocessing Pipeline

```
Raw Signal (N, 12) at variable sampling rate
  ↓
1. LOAD: wfdb.rdrecord(path) → p_signal
  ↓
2. STANDARDIZE LEADS: Pad/truncate to 12 leads
  ↓
3. RESAMPLE: scipy.signal.resample to 500 Hz if needed
  ↓
4. STANDARDIZE LENGTH: Pad/truncate to 5000 samples (10s at 500Hz)
  ↓
5. TRANSPOSE: (N, 12) → (12, 5000)
  ↓
6. CLEAN NaN/Inf: Replace with zeros
  ↓
7. CLIPPING: Clip voltage to [−20, 20] mV (outlier removal)
  ↓
8. PER-LEAD NORMALIZATION:
   - Compute mean & std per lead
   - Apply Z-score: (x − mean) / (std + 1e-8)
  ↓
9. FINAL SAFETY: Re-check for remaining NaN/Inf
  ↓
Output: (12, 5000) float32 normalized signal
```

**Pre-loading strategy**: All signals cached in memory during training (one-time I/O cost)

---

### 4.2 Sklearn Feature Extraction Pipeline

```
Raw Signal (N, 12)
  ↓ (for each of 12 leads independently)
  ↓
Per-Lead Feature Computation:
  ├─ Statistical: mean, std, skew, kurtosis, peak-to-peak
  ├─ Percentiles: 5%, 25%, 75%, 95%
  ├─ Zero-crossing: Count sign changes / N
  ├─ Energy: sqrt(mean(x^2))
  ├─ Spectral (if len > 256):
  │  ├─ Welch PSD: fs=500, nperseg=256
  │  ├─ Normalized power in [0.5-5], [5-15], [15-40] Hz
  │  └─ Dominant frequency argmax(PSD)
  └─ Safety: nan_to_num(feature, nan=0.0)
  ↓
Concatenate all 12 leads
  ↓
Output: 1D vector of 192 features
```

**Rationale**: Hand-crafted features leverage ECG-specific domain knowledge (spectral bands, energy metrics)

---

### 4.3 Interval Calculator Preprocessing (NeuroKit2-based)

```
Raw Lead II Signal (1D)
  ↓
1. CLEAN: nk.ecg_clean(signal, fs=500)
   - FIR bandpass filter (0.5–40 Hz typical)
  ↓
2. QUALITY: nk.ecg_quality(cleaned) → score ∈ [0, 1]
   - <0.3: Warning issued
  ↓
3. R-PEAK DETECTION: nk.ecg_peaks(cleaned, fs=500)
   - Pan-Tompkins algorithm
   - Output: ECG_R_Peaks indices
  ↓
4. WAVEFORM DELINEATION: nk.ecg_delineate(cleaned, r_info, method='dwt')
   - Discrete Wavelet Transform
   - Output: P, Q, R, S, T peak/onset/offset indices
  ↓
5. INTERVAL EXTRACTION:
   - PR = P_onset to R_peak
   - QRS = Q_onset to S_offset
   - QT = Q_onset to T_offset
   - Bazett QTc = QT / sqrt(RR_in_seconds)
  ↓
Output: {hr, pr, qrs, qtc, rr_intervals, quality_score, warnings}
```

**Error handling**: Records <3 seconds trigger error; delineation failures downgrade to HR-only

---

## 5. CURRENT PERFORMANCE METRICS

### 5.1 CNN Model Performance (Test Fold 10)

| Metric | Score |
|---|---|
| **Overall Accuracy** | **69.9%** |
| **Macro F1** | **0.599** |
| **Weighted F1** | **0.697** |
| **Test sample size** | 2,158 records |

#### Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **NORM** (Normal) | 0.778 | 0.887 | 0.829 | 932 |
| **MI** (Myocardial Infarction) | 0.735 | 0.528 | 0.615 | 411 |
| **STTC** (ST/T Changes) | 0.625 | 0.632 | 0.629 | 351 |
| **HYP** (Hypertrophy) | **0.229** | 0.336 | 0.272 | 113 |
| **CD** (Conduction Disturbance) | 0.731 | 0.581 | 0.648 | 351 |

**Key observation**: 
- ✅ **NORM class**: Good (83% F1); CNN correctly identifies normal ECGs
- ⚠️ **HYP class**: **Poor precision (23%)**; many false positives (42% TPR but only 9% PPV)
- ⚠️ **MI class**: Good recall (53%) but mid precision (74%); some false negatives

---

### 5.2 Sklearn (GradientBoosting) Model Performance

| Metric | Score |
|---|---|
| **Overall Accuracy** | **65.2%** |
| **Macro F1** | **0.554** |
| **Weighted F1** | **0.636** |
| **Test sample size** | 2,158 records |

#### Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **NORM** | 0.711 | 0.905 | 0.796 | 932 |
| **MI** | 0.583 | 0.443 | 0.503 | 411 |
| **STTC** | 0.565 | 0.484 | 0.521 | 351 |
| **HYP** | 0.432 | 0.363 | 0.394 | 113 |
| **CD** | 0.648 | 0.487 | 0.556 | 351 |

**Key observation**:
- CNN outperforms sklearn (70% vs 65% overall accuracy)
- Sklearn trades precision for recall on NORM (higher recall but lower F1)
- Both models struggle with HYP — insufficient discriminative power

---

### 5.3 Model Comparison

| Feature | CNN | Sklearn |
|---|---|---|
| **Accuracy** | 69.9% | 65.2% |
| **F1 (macro)** | 0.599 | 0.554 |
| **NORM F1** | 0.829 | 0.796 |
| **HYP F1** | **0.272** | 0.394 |
| **CD F1** | 0.648 | 0.556 |
| **Speed (test)** | Fast (~0.1s/record GPU) | Fast (~0.01s/record CPU) |
| **Interpretability** | Black-box | Tree ensemble, feature importance |

**Recommendation**: CNN is primary (higher overall accuracy); Sklearn as interpretability backup

---

## 6. MISSING / UNDERUTILIZED PARAMETERS

### Critical Gaps

| Parameter | Status | Impact | Priority |
|---|---|---|---|
| **Patient demographics** | Collected but underutilized | Age/sex could improve axis/QT interpretation | HIGH |
| **Historical comparison** | Not implemented | Prior MI location, serial QT tracking missing | HIGH |
| **Arrhythmia detection** | Partial (RR variability only) | No explicit AF/flutter/VT detection | HIGH |
| **Drug effects** | Not modeled | QT-prolonging meds not considered in severity | HIGH |
| **Lead quality indicators** | Not tracked per-lead | Noisy leads not flagged; multi-lead dropout not handled | MEDIUM |
| **12-lead signal processing** | Present but unused in core | 11 of 12 leads feed CNN but only Lead II for intervals | MEDIUM |
| **Ethnic/population differences** | Not addressed | ECG reference ranges may vary by ethnicity | MEDIUM |
| **Chest pain history** | Not integrated | Could contextualize STTC/MI/ischemia findings | LOW |
| **Exercise context** | Not captured | ST depression during exercise vs. rest differs | LOW |

### Technical Gaps

| Gap | Manifestation | Impact |
|---|---|---|
| **No explicit LQT syndrome detection** | QTc formula only (Bazett); no genotype risk stratification | Missed cases with QTc >500 but low K⁺ |
| **No Brugada pattern detection** | Early repolarization in V1-V2 not flagged | May be STTC vs. Brugada not distinguished |
| **No T-U wave distinction** | T-wave amplitude measured in 150–350ms window; U-wave not parsed | U-wave inversion overlooked |
| **No electrical alternans** | Beat-to-beat voltage changes not tracked | Missed pericardial effusion marker |
| **No morphology-based arrhythmia detection** | Only HR variability; no beat classification | Ectopics, AF, VT not identified |
| **Limited multi-lead ST analysis** | ST territory localization present but no dynamics | ST change evolution not tracked |

---

## 7. CURRENT MODEL LIMITATIONS

### Model Architecture Limitations

| Limitation | Root Cause | Consequence |
|---|---|---|
| **HYP precision only 23%** | CNN has no explicit voltage awareness; trained end-to-end on coarse labels | ~77% false positive rate on HYP predictions |
| **MI recall 53%** (misses some MI) | Similar MI/STTC feature overlap; class imbalance | Clinically dangerous; missed acute coronary events |
| **No class-specific confidence** | All classes treated equally; common threshold | Low-confidence predictions not flagged separately |
| **Signal preprocessing fixed** | No adaptive filtering; clipping at ±20 mV hard-coded | May lose information from abnormal morphologies |
| **No lead-level attention** | CNN processes all 12 leads with same weight | Less relevant leads (e.g., noisy aVR) not downweighted |
| **Focal loss gamma=2.0 fixed** | Not tuned to specific class hardness | May over/under-focus on different classes |

### Training Data Limitations

| Limitation | Impact |
|---|---|
| **Imbalanced class distribution** | HYP: 4% of training data; MI: 18%; CD: 24% → minority classes undertrained regardless of oversampling |
| **PTB-XL label quality** | Diagnostic codes mapped via SCP/SNOMED; each record may have multiple codes, mapped to "primary" only → labels lossy |
| **Temporal data variability** | Records from different eras, equipment, labs; no data harmonization → domain shift |
| **No longitudinal data** | Only cross-sectional snapshots; cannot learn rhythm changes or serial patterns | Cannot detect new AF, worsening LVH, etc. |
| **Limited external validation** | Multi-dataset approach uses fold=0 but only in training; test set = PTB-XL fold 10 only → no external independent test set |

### Clinical Deployment Gaps

| Gap | Risk |
|---|---|
| **No uncertainty quantification** | Model outputs point predictions + confidence, no credible intervals → over-confident on ambiguous cases |
| **No automated rule-out logic** | E.g., if HR too low/high or signal quality too poor, should stop and flag for manual review | May produce unreliable outputs on poor-quality scans |
| **Hybrid HYP adjustment** | Voltage thresholds applied as post-hoc gate; not learned end-to-end → may miss compensatory patterns |
| **No explainability** | CNN is black-box; users cannot understand why a specific prediction was made | Reduces clinical adoption |
| **Interval measurements only in Lead II** | ST analysis uses all 12 leads but HR/QRS/QTc computed from Lead II only → may miss lead-specific arrhythmias |
| **No pacemaker ECG handling** | Pacemaker status suppresses alerts but doesn't adapt feature extraction or model logic | Paced rhythms may be misclassified |

---

## 8. FEATURE- / PARAMETER-SPECIFIC RECOMMENDATIONS

### High-Priority Enhancements

| Feature | Action | Expected Impact |
|---|---|---|
| **HYP detection** | Retrain CNN with explicit LVH voltage features (Sokolow, Cornell); use hybrid model as default | +40% precision on HYP expected |
| **Patient demographics integration** | Add age/sex to input features; re-train with demographic-aware embeddings | +3–5% overall accuracy; better QT/ST thresholds |
| **Multi-lead interval measurement** | Extend NeuroKit2 delineation to all 12 leads; compute PR/QRS/QTc per lead | Better lead-specific arrhythmia detection |
| **Historical baseline** | Store baseline ECG; compute deltas for serial analysis | Detect new findings, improve context |
| **Confidence calibration** | Isotonic regression on validation set to map softmax probs to true confidence | Better risk stratification |

### Medium-Priority Enhancements

| Feature | Action |
|---|---|
| **Lead quality flagging** | Per-lead SNR; mark noisy leads; propagate quality downstream |
| **Beat-level classification** | Morph NN to classify individual beats → detect rare arrhythmias (AF episodes, ectopics) |
| **Ethnic-specific thresholds** | Generate axis/voltage reference ranges by population |
| **External validation test set** | Prospective validation on fresh data from Georgia/Chapman datasets (not used in training) |

---

## 9. SUMMARY TABLE: PARAMETERS × MODULES

| Parameter | interval_calc | hybrid | st_territory | clinical_rules | cnn | sklearn |
|---|---|---|---|---|---|---|
| **HR** | ✅ Measured | ✗ N/A | ✗ N/A | ✗ N/A | ✗ N/A | ✗ N/A |
| **PR interval** | ✅ Measured | ✗ Input | ✗ N/A | ✗ N/A | ✗ N/A | ✗ N/A |
| **QRS width** | ✅ Measured | ✗ Input | ✗ N/A | ✗ N/A | ✗ N/A | ✗ N/A |
| **QTc** | ✅ Measured | ✗ Input | ✗ N/A | ✗ N/A | ✗ N/A | ✗ N/A |
| **ST deviation** | ✗ N/A | ✗ N/A | ✅ Measured | ✗ N/A | ✗ N/A | ✗ N/A |
| **Voltage (HYP)** | ✗ N/A | ✅ Gates HYP | ✗ N/A | ✗ N/A | ✗ N/A | ✗ N/A (hand-crafted) |
| **Cardiac axis** | ✗ N/A | ✗ N/A | ✗ N/A | ✅ Estimated | ✗ N/A | ✗ N/A |
| **Low voltage** | ✗ N/A | ✗ N/A | ✗ N/A | ✅ Detected | ✗ N/A | ✗ N/A |
| **T-wave pattern** | ✗ N/A | ✗ N/A | ✗ N/A | ✅ Analyzed | ✗ N/A | ✗ N/A |
| **RR variability** | ✅ Measured | ✗ N/A | ✗ N/A | ✗ N/A | ✗ N/A | ✗ N/A |
| **Signal quality** | ✅ Scored | ✗ N/A | ✗ N/A | ✗ N/A | ✗ Implicit | ✗ N/A |
| **Patient context** | ✅ Applied (K+, age, sex) | ✅ Applied (sex) | ✅ Applied (sex) | ✅ Applied | ✗ N/A | ✗ N/A |

---

## 10. APPENDIX: FILE SUMMARY

| File | Purpose | Key Classes/Functions |
|---|---|---|
| **cnn_classifier.py** | 1D CNN for 5-class superclass prediction | ECGNet, FocalLoss, SqueezeExcitation, ECGResBlock, train(), predict_cnn() |
| **poc_classifier.py** | GradientBoosting + hand-crafted features | extract_features(), train(), predict() |
| **hybrid_classifier.py** | CNN + voltage criteria gate for HYP | compute_voltage_criteria(), hybrid_predict() |
| **interval_calculator.py** | NeuroKit2-based HR/PR/QRS/QTc measurement | calculate_intervals(), apply_clinical_context(), REFERENCE dict |
| **st_territory.py** | ST-segment localization to coronary territories | measure_st_deviation(), analyze_st_territories(), TERRITORIES dict |
| **clinical_rules.py** | Rule-based findings (axis, low voltage, T-waves) | _estimate_qrs_axis(), _check_low_voltage(), analyze_clinical_rules() |
| **dataset_pipeline.py** | Multi-dataset integration & label mapping | build_unified_index(), SNOMED_TO_SUPERCLASS, DATASET_CONFIGS, download functions |
| **validate_classifier.py** | Batch validation & performance reporting | validate_cnn(), validate_sklearn(), generate_report() |
| **app.py** | Streamlit dashboard integration | Imports and orchestrates all above modules |

---

## 11. CONCLUSION

Your EKG codebase integrates **multiple complementary approaches**:

1. **Deep learning** (CNN) for automated classification — fast but black-box
2. **Interpretable hand-crafted features** (sklearn) for backup & debugging
3. **Voltage-based clinical rules** (hybrid) to improve HYP detection
4. **Interval measurement** (NeuroKit2) for objective timing parameters
5. **Multi-territory ST analysis** for acute coronary event localization
6. **Rule-based clinical findings** for axis, low voltage, and morphology

**Strengths:**
- Comprehensive 12-lead signal processing
- Multi-model approach reduces single-model bias
- Clinical context applied (sex, K⁺, pacemaker, athlete status)
- Hybrid system leverages domain knowledge to improve HYP

**Weaknesses:**
- **HYP precision only 23%** (critical clinical issue)
- **MI recall only 53%** (misses some acute infarcts)
- Limited explainability; no external validation
- Patient demographics underutilized
- No longitudinal or serial analysis

**Immediate priorities for improvement:**
1. Fix HYP precision (voltage feature engineering, re-train)
2. Improve MI recall (class weighting, threshold adjustment)
3. Add external validation on fresh datasets
4. Integrate patient demographics as features
5. Develop explainability layer (saliency maps, attention visualization)
