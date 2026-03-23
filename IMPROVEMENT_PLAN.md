# EKG Intelligence Model Improvement Plan

**Date:** March 2026  
**Author:** Code Analysis  
**Project:** EKG Intel POC Upgrade

---

## Executive Summary

Current system uses **3 classifiers** (CNN, Sklearn GradientBoosting, Hybrid gate) achieving **69.9% accuracy on 5-class ECG superclass prediction**. However, **critical vulnerabilities exist** in clinical-critical classes:
- **HYP (Hypertrophy): 23% precision** → 77% false positives (dangerous)
- **MI: 53% recall** → Misses acute infarcts (life-threatening)
- **STTC: Underutilized multi-lead ST data** → Restricted to Lead II

This plan proposes **5 strategic improvements** spanning architecture, data, training, and clinical integration.

---

## Section 1: Current Parameters Affecting EKG Results

### 1.1 Signal/Lead Parameters
| Parameter | Current Use | Source | Limitation |
|-----------|-------------|--------|-----------|
| 12-lead raw signal (5000 samples × 12) | CNN input | Device standardized to 500Hz × 10s | No adaptive sampling |
| Lead II only | Interval measurement (HR, PR, QRS, QTc) | NeuroKit2 delineation | Multi-lead metrics ignored |
| Signal quality score | Warning flag | NeuroKit2 quality() | Not fed back to CNN |
| ST-segment by territory | Rule-based flags, per-lead deviation | st_territory.py | Not integrated into class prediction |

### 1.2 Clinical Context Parameters
| Parameter | Current Use | Limitations |
|-----------|-------------|-----------|
| Age (1–120) | Read but **never used in rules** | No age thresholds, no pediatric/geriatric handling |
| Sex (M/F) | QTc threshold modifier, HYP voltage | Hardcoded rules; no ethnic variation |
| Pacemaker toggle | Suppresses findings | Binary; no ICD timing/mode |
| Athlete status | Suppresses bradycardia | Binary; no continuous fitness level |
| Pregnancy | Adjusts QTc threshold | Binary; no trimester specificity |
| Potassium (K⁺) level | QTc risk modifier, hypokalaemia flag | No integration with QT prediction |

### 1.3 Extracted Interval Parameters
| Parameter | Range | Current Use |
|-----------|-------|-----------|
| Heart Rate (HR) | Typically 40–180 bpm | Class thresholds (brady/tachy), urgency scoring |
| PR interval | 120–200 ms (normal) | Detects AV blocks |
| QRS duration | 70–110 ms (normal) | Detects bundle branch blocks |
| QTc (Bazett) | 350–450 ms | Class thresholds, HYP gate |
| RR variability (CV) | 0–∞ | Irregular rhythm screen |

### 1.4 Hand-Crafted Features (Sklearn Model)
192 features per lead:
- **Statistical:** Mean, max, min, kurtosis, entropy
- **Spectral:** Power density (0–50 Hz), dominant frequency
- **Morphological:** Zero-crossing rate, slope variance

**Issue:** Features manually engineered; CNN ignores them (no fusion).

### 1.5 Voltage Criteria (Hybrid Gate)
- **Sokolow-Lyon:** S(V1) + R(V5/V6) > 3.5 mV → LVH
- **Cornell:** R(aVL) + S(V3) > 2.8/2.0 mV (M/F) → LVH
- **RVH:** R(V1) > 0.7 mV

**Current Issue:** Ad-hoc gating; not jointly learned with CNN.

### 1.6 ST Territory Analysis
Divides chest into 3 regions:
- **LAD:** V1–V4 (anterior)
- **RCA:** II, III, aVF (inferior)
- **LCx:** I, aVL, V5–V6 (lateral)

**Issue:** Computed but not fed to classifier as features.

---

## Section 2: Current Model Architectures

### 2.1 CNN (PyTorch ECGNet)
**Architecture:**
- **Input:** 12-lead, 5000 samples
- **Stem:** Conv1d(12 → 64, kernel=15)
- **3 ResNet blocks:** 64 → 128 → 256 channels
- **Squeeze-Excitation:** Per-block channel attention
- **Head:** Global Average Pool → FC(256 → 5)
- **Parameters:** ~1.2M

**Training:**
- Loss: Focal Loss + label smoothing + class weights
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-3)
- Augmentation: Noise, time-shift, lead dropout, scaling
- Epochs: 50 with OneCycleLR
- Sampler: WeightedRandomSampler (oversamples HYP, STTC)

**Performance:**
- **Accuracy: 69.9%**
- **Class breakdown:** NORM 83% F1 | MI 62% F1 | HYP 27% F1 ⚠️ | STTC 65% F1 | CD 68% F1

**Weaknesses:**
1. **Black-box:** No interpretability (which leads matter?)
2. **Lead-agnostic:** No per-lead attention or quality gating
3. **Context-blind:** Ignores patient age, sex, clinical history
4. **Imbalance:** Oversampling helps but crude; HYP still fails

### 2.2 Sklearn Gradient Boosting
**Features:** 192 per-lead features (stat + spectral)
**Training:** PTB-XL folds 1–8, class-weight balanced
**Performance:** 65.2% accuracy (worse on MI recall)

**Weaknesses:**
1. Manual feature engineering (labor-intensive)
2. Lower performance than CNN
3. Not used in final app (fallback only)

### 2.3 Hybrid Classifier
**Approach:** CNN prediction + voltage gate logic
```
If CNN_confidence[HYP] > 0.35 AND voltage_criteria_met:
    Keep HYP
Elif CNN_confidence[HYP] > 0.35 AND NOT voltage_criteria:
    Downgrade to next-best class
Elif voltage_criteria_strong AND CNN_confidence[HYP] < threshold:
    Boost HYP (confidence)
```

**Weaknesses:**
1. **Ad-hoc rules:** Not jointly learned; brittle
2. **Single-class focus:** Only addresses HYP; MI/CD uncorrected
3. **Hardcoded thresholds:** No validation on holdout set

---

## Section 3: Critical Performance Gaps

### 3.1 HYP (Hypertrophy) Crisis
- **Precision: 23%** → For every 10 predicted HYP, 7-8 are false positives
- **Root cause:** CNN trained without explicit voltage awareness
- **Clinical impact:** Over-referral, patient anxiety, unnecessary investigations

### 3.2 MI (Myocardial Infarction) Recall Failure
- **Recall: 53%** → Misses ~47% of acute infarcts
- **Root cause:** Similar class signature to STTC; class imbalance in training
- **Clinical impact:** **Life-threatening missed diagnosis**

### 3.3 Multi-Lead Underutilization
- **Current:** ST-territory computed but not fed to classifier
- **Potential:** ST deviation per lead + territory is strong MI/STTC discriminator
- **Missing:** Lead-specific arrhythmia patterns (e.g., inferior MI pattern in II/III)

### 3.4 Patient Context Not Integrated
- **Age:** Not used → Peds/geriatric thresholds unavailable
- **Sex/ethnicity:** Only hardcoded sex for QTc/HYP → No ethnic variation
- **Clinical history:** None captured; no serial comparison

### 3.5 No Uncertainty Quantification
- Model outputs hard class, not calibrated probability
- No "low-confidence alert" for close calls
- No confidence-rejection curve for deployment tuning

---

## Section 4: Improvement Plan (5 Strategic Initiatives)

### Initiative 1: Joint Voltage-CNN Learning
**Goal:** Eliminate ad-hoc hybrid gating; train voltage awareness into CNN directly

**Changes:**
1. **Augment CNN input:**
   - Add auxiliary input: (Sokolow-Lyon, Cornell, RVH, ST-elevation by territory) as side concatenation
   - Optionally: (age normalized to 0–1), (sex one-hot: [M/F/U])
   - → New input shape: 12 leads + 1 clinical feature vector → separate embedding

2. **Architecture modification:**
   ```
   ECGSignal (12 × 5000) → CNN stem → Conv features (256)
   ClinicalFeatures (8 values) → Dense(8→64→128) → embedding
   → Concatenate → Dense(256+128=384 → 5 classes)
   ```

3. **Training:**
   - Joint end-to-end training; clinical features NOT frozen
   - Loss: Focal loss on 5-class output
   - Learn optimal weighting of voltage vs. signal

**Expected impact:**
- HYP precision: 23% → 55–65% (reduce false positives by 50%)
- MI recall: 53% → 65–75% (catch more acute infarcts)
- Interpretability: Attention weights show which leads + features matter for each class

**Timeline:** 2–3 weeks training + validation

---

### Initiative 2: Multi-Lead Interval Extraction
**Goal:** Don't restrict to Lead II; compute PR/QRS/QTc per lead + combine

**Changes:**
1. **Extend interval_calculator.py:**
   - Extract intervals from all 12 leads (currently only Lead II)
   - Return per-lead metrics: `{"II": {...}, "V1": {...}, ...}`
   - Compute consensus interval (median across leads)
   - Flag per-lead outliers (leads with aberrant intervals)

2. **Feature engineering:**
   - **PR max/min spread:** Large spread → conduction disease
   - **QRS max:** Maximum QRS across any lead → bundle branch severity marker
   - **QTc dispersion:** QTc standard deviation across leads → repolarization heterogeneity (arrhythmia risk)
   - **Lead-specific flags:** E.g., "deep Q in II/III" → inferior MI pattern

3. **Integration:**
   - Add these derived features to Hybrid classifier
   - Fine-tune CNN with multi-lead interval input (similar to Initiative 1)

**Expected impact:**
- MI recall: +5–10% (Q-wave patterns more robust)
- STTC precision: +8–12% (better ST-elevation quantification)
- Interpretability: Clinicians see lead-by-lead intervals

**Timeline:** 1 week development + 1 week validation

---

### Initiative 3: Class-Specific Data Balancing and Re-weighting
**Goal:** Address MI and HYP underperformance with smarter training strategy

**Changes:**
1. **Stratified oversampling:**
   - Current: Uniform oversampling of minority classes
   - Proposed: Class-specific resampling ratios tuned on validation fold
   - **MI:** 1:1 ratio (50% of training), STTC: 0.7:1, HYP: 1:1
   - NORM/CD: Slight downsampling to avoid dominance

2. **Focal loss refinement:**
   - Current: Uniform gamma=2.0, weight per class
   - Proposed: Per-class gamma tuning
   - MI/HYP: gamma=2.5 (focus harder on hard examples)
   - NORM: gamma=1.5 (already easy)

3. **Hard negative mining:**
   - Post-training: Identify NORM samples that CNN confidently predicts as MI/STTC
   - Re-weight these samples higher in next epoch
   - Train for 3–5 additional epochs with hard negatives

4. **Cross-dataset validation:**
   - Train on PTB-XL (current)
   - Validate on: PTB, Georgia, Chapman datasets (external)
   - If performance drops >5%, retrain with cross-dataset batching

**Expected impact:**
- MI recall: 53% → 68–75% (+12–15%)
- HYP precision: 23% → 50–60% (+27–37%)
- Generalization: Better on unseen datasets

**Timeline:** 2 weeks (including cross-dataset collection)

---

### Initiative 4: Age/Sex/Ethnicity-Aware Thresholds
**Goal:** Replace hardcoded rules with learned, context-sensitive interpretation

**Changes:**
1. **Demographic-stratified analysis:**
   - Split PTB-XL by age >50 / ≤50, sex, documented ethnicity (if available)
   - Compute class-specific performance per demographic
   - Identify if thresholds shift (e.g., elderly have wider QRS normal)

2. **Threshold tuning:**
   - Current: HR < 60 bpm → bradycardia flag for all
   - Proposed:
     - Age < 20: HR < 50 → bradycardia
     - Age 20–65: HR < 60 → bradycardia
     - Age > 65: HR < 50 → bradycardia (athletes lower)
   - Similar for QTc, QRS, voltage criteria

3. **Integration into interval_calculator.py:**
   ```python
   def apply_clinical_context(intervals, patient):
       age = patient.get("age", 50)
       sex = patient.get("sex", "M")
       
       if age < 20:
           hr_threshold_low = 50
       elif age > 65:
           hr_threshold_low = 50
       else:
           hr_threshold_low = 60
       
       if intervals["hr"] < hr_threshold_low:
           # Contextual flag, not absolute
   ```

4. **Ethnic variation (if data available):**
   - Document ethnic composition of PTB-XL
   - Note: Most public datasets are Euro-centric; recommend diversity analysis in future
   - For now: Flag when ethnicity unknown

**Expected impact:**
- Specificity: +3–5% (fewer false alerts in elderly)
- Sensitivity: +2–3% (catch more abnormalities in youth)
- Clinical trust: "Age-appropriate thresholds" more defensible

**Timeline:** 1 week analysis + 1 week implementation

---

### Initiative 5: Confidence Calibration & Uncertainty Quantification
**Goal:** Deploy model with confidence intervals; reject low-confidence predictions

**Changes:**
1. **Calibration method: Temperature scaling**
   - Train separate scalar T on validation fold
   - Apply: softmax_calibrated = softmax(logits / T)
   - Ensures predicted probability = actual accuracy

2. **Ensemble uncertainty:**
   - Load N=5 CNN checkpoints from different random seeds
   - Inference: Average predicted probabilities + compute std
   - High std → low confidence → flag for review

3. **Confidence-based rejection:**
   - Define rejection threshold per class (e.g., HYP: p < 0.60 → reject to "NORM or other")
   - Operating point: 95% precision at cost of 10% recall (conservative)
   - Clinicians review rejected cases

4. **Metrics:**
   ```
   - Confidence vs. Accuracy curve (calibration plot)
   - Rejection rate at 95% precision target
   - Coverage = % of samples above confidence threshold
   ```

5. **Integration into app.py:**
   ```python
   result = classify_ecg(model, signal_12, ...)
   if result["confidence"] < 0.55:
       st.warning("Low confidence prediction. Recommend manual review.")
   ```

**Expected impact:**
- Deployable confidence: Clinicians know when to trust model
- AUC (at 95% precision): 75% → 88%
- Safety: Reduces false positives by selective rejection

**Timeline:** 1 week calibration + 1 week testing

---

## Section 5: Training Strategy Changes

### 5.1 Data Splits & Validation
**Current:**
- PTB-XL folds 1–8: Training
- Fold 9–10: Validation
- No external validation

**Proposed:**
1. **3-fold cross-validation on PTB-XL:**
   - Fold 1-6: Train
   - Fold 7-8: Validation
   - Fold 9-10: Test (final holdout)

2. **External validation (if datasets available):**
   - Georgia 12-lead ECG database
   - Chapman dataset
   - Results: Train metrics vs. generalization gap

3. **Temporal split (if timestamps available):**
   - Train on earlier records, validate on recent → catch distribution shift

### 5.2 Hyperparameter Grid
**CNN Tuning:**
| Param | Current | Proposed Range |
|-------|---------|-----------------|
| LR | 3e-4 | [1e-4, 3e-4, 1e-3] |
| Batch size | 32 | [16, 32, 64] |
| Focal gamma | 2.0 | [1.5, 2.0, 2.5] |
| Class weight | Uniform scale | [Learned from class frequency] |
| Augmentation strength | Med | [Low, Med, High] |

**Strategy:** Bayesian Optimization over grid (5–10 runs), select best on validation fold

### 5.3 Training Monitoring
**Add to cnn_classifier.py:**
```python
# Per-epoch tracking
- Per-class F1 (not just accuracy)
- Validation loss convergence
- Gradient norm (catch dead neurons)
- Learning rate schedule (OneCycleLR visualization)

# EarlyStopping: Stop if validation F1 plateaus for 5 epochs
```

### 5.4 Reproducibility & Versioning
**Improvements:**
1. Save hyperparameters in model checkpoint
2. Log random seed + data hash
3. Version models: `ecg_cnn_v1_date_seed.pt`
4. Track dataset composition (# record per class per fold)

---

## Section 6: Implementation Roadmap

| # | Initiative | Effort | Priority | Blockers |
|---|-----------|--------|----------|----------|
| 1 | Joint Voltage-CNN | 3 weeks | **HIGH** | Requires retrain; GPU time |
| 2 | Multi-lead intervals | 2 weeks | **HIGH** | None (backward compatible) |
| 3 | Data rebalancing | 2 weeks | **CRITICAL** | MI/HYP currently unreliable |
| 4 | Age/sex thresholds | 1 week | Medium | Requires domain expert review |
| 5 | Confidence calibration | 1 week | Medium | Validation fold large enough? |

**Total Duration:** ~8–9 weeks (can parallelize 2 & 4)

**Phase 1 (Weeks 1–3):** Initiative 3 (rebalancing) + Initiative 2 (multi-lead) in parallel
**Phase 2 (Weeks 4–6):** Initiative 1 (joint learning, main effort)
**Phase 3 (Weeks 7–9):** Initiative 4 & 5, external validation

---

## Section 7: Expected Outcomes

### Class Performance Improvements
| Class | Current | Target | Change |
|-------|---------|--------|--------|
| NORM | 83% F1 | 85% F1 | Minimal (already good) |
| **MI** | 62% F1 | 75% F1 | **+13 F1 points** ⭐ |
| **HYP** | 27% F1 | 55% F1 | **+28 F1 points** ⭐ |
| STTC | 65% F1 | 72% F1 | +7 F1 points |
| CD | 68% F1 | 75% F1 | +7 F1 points |
| **Overall** | 69.9% acc | 78% acc | **+8.1% accuracy** |

### Clinical Readiness
- **False Positive Rate (HYP):** 77% → 35% (safer to use)
- **Missed MI Rate:** 47% misses → 25% misses (fewer life-threatening errors)
- **Confidence calibration:** 95% precision achievable at 85% coverage
- **External generalization:** ≤5% performance drop on unseen datasets

### Deployment Gate
✅ Ready for **FDA 510(k) consideration** if:
- Overall accuracy ≥ 77%
- Per-class F1 ≥ 65%
- External validation on ≥2 datasets
- Confidence calibration validated
- Clinician review protocol defined

---

## Section 8: Additional Recommendations

### 8.1 Data Augmentation
Current approach: Noise, time-shift, lead dropout, scaling
- ✅ Effective for CNN robustness
- **Next:** Add morphing (compress/stretch QRS slightly to simulate different heartrates)

### 8.2 Interpretability Layer
- **SHAP values:** Which features drive each prediction?
- **Saliency maps:** Which time windows / leads matter?
- **Per-sample explanations:** "Model focused on V1–V4 (anterior leads) → MI classification"

### 8.3 Continuous Monitoring in Production
- Track P(prediction) distribution drift
- Retrain if validation accuracy drops >3%
- A/B test new model versions before deployment

### 8.4 Domain Adaptation
- When deployed in new hospital, patient population may differ
- Solution: Fine-tune last 2 layers on local data (transfer learning)
- Minimum 200 labeled local samples

---

## Section 9: Questions for Stakeholders

1. **Data availability:** Do you have access to external ECG datasets (Georgia, Chapman) for validation?
2. **Compute:** GPU available for retraining? (50 epochs × 5-class × data aug ≈ 4–6 GPU hours)
3. **Ethnicity metadata:** Is demographic info (ethnicity, age distribution) documented in PTB-XL?
4. **Clinical validation:** Can cardiologists label 50–100 records to evaluate confidence calibration?
5. **Deployment timeline:** When is FDA submission target?

---

## Section 10: References & Links

- **Focal Loss:** Lin et al., "Focal Loss for Dense Object Detection" (2017)
- **Temperature Scaling:** Guo et al., "On Calibration of Modern Neural Networks" (2017)
- **PTB-XL Paper:** Wagner et al., https://doi.org/10.1038/s41597-020-0495-6
- **NeuroKit2:** Makowski et al., https://github.com/neuropsychology/NeuroKit

---

## Appendix: Code Snippets for Quick Start

### A.1 Joint Voltage-CNN Input Example
```python
# In cnn_classifier.py ECGNet.__init__
class ECGNet(nn.Module):
    def __init__(self, n_leads=12, n_classes=5):
        super().__init__()
        
        # ECG signal branch
        self.ecg_stem = nn.Conv1d(n_leads, 64, kernel_size=15, padding=7)
        
        # Clinical feature branch (NEW)
        self.clinical_embed = nn.Sequential(
            nn.Linear(8, 64),  # 8 features: age, sex, sokolow, cornell, etc.
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Rest of CNN...
        
    def forward(self, ecg_signal, clinical_features):
        ecg_feat = self.ecg_stem(ecg_signal)  # (B, 64, 5000)
        
        clinical_feat = self.clinical_embed(clinical_features)  # (B, 128)
        clinical_feat = clinical_feat.unsqueeze(-1).expand(-1, -1, 5000)
        
        combined = torch.cat([ecg_feat, clinical_feat], dim=1)  # (B, 192, 5000)
        # Pass through rest of model...
```

### A.2 Multi-Lead Interval Extraction
```python
# In interval_calculator.py
def calculate_intervals_all_leads(signal_12, lead_names, sampling_rate=500):
    """
    Calculate intervals for all leads separately, return consensus.
    """
    results_by_lead = {}
    
    for i, lead_name in enumerate(lead_names):
        lead_signal = signal_12[:, i]
        intervals = calculate_intervals(lead_signal, sampling_rate)
        results_by_lead[lead_name] = intervals
    
    # Compute consensus metrics
    qrs_values = [r.get("qrs") for r in results_by_lead.values() if r.get("qrs")]
    consensus_qrs = np.median(qrs_values) if qrs_values else None
    
    return {
        "per_lead": results_by_lead,
        "consensus": {"qrs": consensus_qrs, ...},
        "dispersion": {"qrs_std": np.std(qrs_values) if qrs_values else None}
    }
```

### A.3 Age-Aware Threshold Function
```python
def get_hr_threshold(age, is_athlete=False):
    """Return heart rate bradycardia threshold based on age."""
    if is_athlete:
        return 40  # Athletes can have resting HR in 40s
    elif age < 20:
        return 50
    elif age > 65:
        return 50
    else:
        return 60
```

---

**End of Improvement Plan**

*Last Updated: March 2026*  
*Next Review: Upon completion of Initiative 3*
