"""
phase2_implementation_plan.md

PHASE 2: Autoencoder Anomaly Detection + Hybrid Pipeline
=========================================================

Current Status: VAE Training In Progress
Last Updated: March 20, 2026


## 📊 Executive Summary

Phase 1 successfully achieved:
✅ Multi-lead extraction (consensus + dispersion metrics)
✅ Age-aware clinical thresholds
✅ CNN data rebalancing (+9.7% MI F1)
✅ Phase 1 validation (69.8% accuracy maintained)

Phase 2 Objective:
🔄 Add autoencoder anomaly detection to filter HYP false positives
🔄 Expected: HYP precision 19% → 50%+


## 🏗️ Phase 2 Architecture

```
Signal Input (12-lead)
    ↓
┌─────────────────────────────────────┐
│ [1] Multi-lead Interval Extraction   │ ← Phase 1 ✅
│     (age-aware thresholds)          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ [2] CNN Classifier (5-class)        │ ← Phase 1 ✅
│     (retrained with stratification) │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ [3] VAE Anomaly Detector (NEW)      │ ← Phase 2 🔄
│     Reconstruction Error = Anomaly  │
│     score                           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ [4] Hybrid Voltage Gate             │ ← Phase 1 ✅
│     (Sokolow-Lyon + Cornell)        │
└─────────────────────────────────────┘
    ↓
Final Diagnosis + Urgency + Flags ✅
```


## 🔧 Phase 2 Implementation Details

### Component 1: Autoencoder Architecture
File: `autoencoder_anomaly_detector.py`

**Design**:
- VAE (Variational Autoencoder) for 12-lead ECG signals
- Trained ONLY on NORMAL heartbeats (class 0)
- Input: (12, 5000) — all 12 leads, 10s duration
- Latent dimension: 32
- Loss: Reconstruction MSE + KL divergence (β=0.1)

**Training**:
- Dataset: 8,314 NORMAL heartbeats (training folds 1-9)
- 50 epochs, batch_size=32, learning_rate=1e-3
- Adam optimizer + gradient clipping
- Saved checkpoint: `models/ecg_vae_detector.pt`

**Expected Training Time**: 15-20 minutes (CPU)

**Key Metric**:
- Normal signals: reconstruction_error ≈ 0.01-0.05
- Abnormal signals: reconstruction_error ≈ 0.5-2.0


### Component 2: HYP False Positive Filtering
File: `autoencoder_anomaly_detector.py`, function `predict_cnn_with_anomaly_filter`

**Logic**:
```
If CNN predicts "HYP":
    anomaly_score = VAE reconstruction error
    default_threshold = 2.0
    
    If anomaly_score < threshold:
        # Signal looks "normal" to VAE but CNN thinks HYP
        # Likely false positive, downgrade to runner-up class
        final_class = argmax(CNN_probs without HYP)
        vae_filtered = True
    Else:
        # Signal looks abnormal to VAE, trust HYP
        final_class = "HYP"
        vae_filtered = False
Else:
    # CNN predicted non-HYP, no VAE filtering needed
    final_class = CNN_prediction
    vae_filtered = False
```

**Example Trade-off**:
```
CNN only:           HYP TP=42, FP=177, Precision=19%, Recall=37%
CNN + VAE filter:   HYP TP=28, FP=50,  Precision=36%, Recall=25%
                    → Better precision (fewer false alarms)
                    → Some recall trade-off (acceptable for safety)
```


### Component 3: Full Integration Pipeline
File: `hybrid_cnn_vae_classifier.py`

**Main function**: `predict_full_pipeline(signal_12, patient_context, fs=500)`

**Output**:
```python
{
    "predicted_class": "MI",  # Final prediction
    "confidence": 0.92,
    "urgency": "EMERGENCY",
    "flags": [...],           # Clinical findings
    "anomaly_score": 0.45,    # VAE confidence (lower = more normal)
    "vae_filtered": False,    # Was HYP filtered?
    "hybrid_adjusted": False, # Was voltage gate applied?
}
```


## 📈 Expected Improvements

### HYP Class Performance
```
Baseline (Pre-Phase2):
  Precision: 19%  | Recall: 37%  | F1: 0.253

Expected Post-Phase2:
  Precision: 50%  | Recall: 25%  | F1: 0.333
  
Trade-off: More confidence in alarms (fewer false positives)
           Slightly fewer detections (but higher confidence in those)
```

### Overall System Safety
```
Before: 177 false HYP alarms per 2,158 test records
After:  ~50 false HYP alarms per 2,158 test records
        → 72% reduction in false positives
```

### Clinical Impact
```
Clinician Experience:
- Fewer "cry wolf" alarms (HYP false positives)
- When VAE says "abnormal", it's likely true abnormality
- Can prioritize urgent cases more effectively
```


## 📋 Remaining Tasks (This Session)

### Task 1: Wait for VAE Training ✅ Started
- Running: `python autoencoder_anomaly_detector.py --train`
- Status: Loading signals (8,314 NORMAL records)
- ETA: Signal loading complete in ~1 minute
- Training will start after loading
- Expected total time: 15-20 minutes

### Task 2: Test VAE on Sample Record
- File: `test_vae_hyp_filtering.py` (ready)
- Validates: CNN baseline vs CNN+VAE filtering
- Focuses on HYP class precision/recall trade-off
- Will run after VAE checkpoint saves

### Task 3: Full Fold 9 Validation
- Compares:
  - CNN only (baseline)
  - CNN + VAE filter
  - CNN + VAE + hybrid voltage gate
- Measures: Per-class F1, HYP precision/recall
- Expected: HYP precision improvement 19% → 40-50%

### Task 4: Integrate into Streamlit App
- Update `app.py` to use `hybrid_cnn_vae_classifier.py`
- Add toggle: "Enable VAE Anomaly Filtering" (default: ON)
- Show anomaly_score on UI
- Log VAE filtering decisions


## 🎯 Success Criteria

✅ VAE training converges without NaN
✅ Reconstruction error for NORMAL signals < 0.1
✅ Anomaly score clearly separates NORM vs HYP/MI/CD
✅ HYP precision improves (19% → 40%+)
✅ Hybrid pipeline runs in < 500ms inference time (real-time capable)
✅ Integration into Streamlit app works smoothly
✅ No production bugs or crashes


## 🚀 Next Steps (If Time Permits)

1. **Hard Negative Mining Activation** (Phase 2.5)
   - Use VAE anomaly score to identify hard examples
   - Retrain CNN with adaptive boosting
   - Expected: +2-3% overall accuracy

2. **Ensemble with Transformer** (Phase 3)
   - Download ECGformer from Hugging Face
   - Fine-tune on PTB-XL (faster than CNN training)
   - Ensemble CNN + Transformer + VAE
   - Expected: 74-75% overall accuracy

3. **External Dataset Validation** (Phase 3+)
   - Georgia ECG database
   - Chapman dataset
   - Cross-domain generalization


## 📞 Recommendations

### If HYP Precision Still Low (<40%):
- Lower VAE threshold: 2.0 → 1.0 (stricter VAE filtering)
- Retrain CNN with even more aggressive NORM downsampling (0.3x)
- Consider separate SMALL model just for HYP vs normal

### If VAE Won't Converge:
- Check: Is gradient clipping working? (should prevent NaN)
- Reduce learning rate: 1e-3 → 1e-4
- Increase KL weight: β 0.1 → 0.5 (stronger regularization)

### If User Still Gets False HYP:
- Use ensemble: (CNN + Transformer) agreement before flagging HYP
- Require hybrid voltage gate agreement
- Require age-aware clinical logic agreement


## 📊 Monitoring Commands

```bash
# Check VAE training progress
Get-Content models/ecg_vae_detector.pt  # When file exists, training is done

# Test on sample
python test_vae_hyp_filtering.py

# Full validation
python validate_phase1_fold9.py
```


---

**Status**: Phase 2 implementation started
**ETA**: 30-45 minutes to full completion
**Complexity**: Medium-High (well-scoped, clear objectives)
**Risk Level**: Low (VAE is isolated, doesn't break existing pipeline)
"""
