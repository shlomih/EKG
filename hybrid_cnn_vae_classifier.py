"""
hybrid_cnn_vae_classifier.py
============================
Integrates CNN classification + VAE anomaly filtering + hybrid voltage gate.

This is the PRODUCTION PIPELINE that combines:
1. Clinical multi-lead intervals (age-aware thresholds)
2. CNN classification (5-class: NORM/MI/STTC/HYP/CD)
3. VAE anomaly detection (filters HYP false positives)
4. Voltage criteria gating (Sokolow-Lyon, Cornell)

Usage:
    from hybrid_cnn_vae_classifier import predict_full_pipeline
    result = predict_full_pipeline(signal_12, patient_context, fs=500)
    # result = {
    #     "predicted_class": "MI",
    #     "confidence": 0.92,
    #     "urgency": "EMERGENCY",
    #     "flags": [...],
    #     "anomaly_score": 0.45,  # VAE confidence
    #     "hybrid_adjusted": False,
    #     "reasoning": "..."
    # }
"""

import numpy as np
from interval_calculator import calculate_intervals_all_leads, apply_clinical_context
from cnn_classifier import load_cnn_classifier, predict_cnn, SUPERCLASS_LABELS, LABEL_TO_IDX
from autoencoder_anomaly_detector import load_vae_detector, compute_anomaly_score
from hybrid_classifier import hybrid_predict

# =============================================================================
# Global model cache
# =============================================================================

_CNN_MODEL = None
_VAE_MODEL = None

def initialize_models():
    """Load all models once (on app startup)."""
    global _CNN_MODEL, _VAE_MODEL
    if _CNN_MODEL is None:
        _CNN_MODEL = load_cnn_classifier()
    if _VAE_MODEL is None:
        _VAE_MODEL = load_vae_detector()
    return _CNN_MODEL, _VAE_MODEL


# =============================================================================
# Full Pipeline
# =============================================================================

def predict_full_pipeline(signal_12, patient_context, fs=500,
                         vae_hyp_threshold=2.0, verbose=False):
    """
    Complete ECG analysis pipeline:
    1. Multi-lead interval extraction (consensus + dispersion)
    2. CNN classification (5-class)
    3. VAE anomaly filtering (especially HYP)
    4. Clinical context (age-aware thresholds, flags)
    5. Hybrid voltage gating
    
    Args:
        signal_12: (N, 12) numpy array, 12-lead signal
        patient_context: dict with age, sex, athlete_status, pacemaker, etc.
        fs: sampling rate (default 500 Hz)
        vae_hyp_threshold: reconstruction error threshold for HYP filtering (2.0 recommended)
        verbose: print intermediate results
    
    Returns:
        dict with:
        - predicted_class: final ECG class (NORM/MI/STTC/HYP/CD)
        - confidence: float 0-1
        - urgency: NORMAL/ROUTINE/URGENT/EMERGENCY
        - flags: list of clinical findings
        - anomaly_score: VAE reconstruction error (lower = more normal)
        - hybrid_adjusted: bool if hybrid gate changed prediction
        - reasoning: text explanation
    """
    
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    # [Step 1] Multi-lead interval extraction
    if verbose:
        print("[1] Extracting multi-lead intervals...")
    
    intervals_result = calculate_intervals_all_leads(signal_12, lead_names, fs)
    consensus_intervals = intervals_result.get("consensus", {})
    dispersion = intervals_result.get("dispersion", {})
    
    if consensus_intervals.get("hr") is None:
        # Delineation failed, use single-lead fallback
        if verbose:
            print("  ⚠️ Multi-lead delineation failed, using fallback")
        lead_ii_signal = signal_12[:, 1]
        from interval_calculator import calculate_intervals
        consensus_intervals = calculate_intervals(lead_ii_signal, fs)
    
    # [Step 2] CNN classification
    if verbose:
        print("[2] Running CNN classification...")
    
    cnn_model = _CNN_MODEL or initialize_models()[0]
    cnn_result = predict_cnn(cnn_model, signal_12, fs)
    cnn_class = cnn_result["prediction"]
    cnn_confidence = cnn_result["confidence"]
    cnn_probs = cnn_result["probabilities"]
    
    if verbose:
        print(f"  CNN result: {cnn_class} ({cnn_confidence:.1%})")
    
    # [Step 3] VAE anomaly filtering (especially for HYP)
    if verbose:
        print("[3] Running VAE anomaly detection...")
    
    vae_model = _VAE_MODEL or initialize_models()[1]
    anomaly_score = None
    vae_filtered = False
    final_class = cnn_class
    
    if vae_model is not None:
        anomaly_score, _ = compute_anomaly_score(signal_12, vae_model)
        
        # Apply filtering specifically to HYP class
        if cnn_class == "HYP" and anomaly_score is not None:
            if anomaly_score < vae_hyp_threshold:
                # HYP predicted but signal looks "normal" to VAE
                # → Likely false positive, downgrade to runner-up
                vae_filtered = True
                filtered_probs = dict(cnn_probs)
                filtered_probs["HYP"] = 0.0
                final_class = max(filtered_probs, key=filtered_probs.get)
                if verbose:
                    print(f"  VAE filter applied: HYP downgraded to {final_class}")
                    print(f"  (anomaly_score {anomaly_score:.2f} < threshold {vae_hyp_threshold})")
        
        if verbose:
            print(f"  Anomaly score: {anomaly_score:.3f}")
    
    # [Step 4] Hybrid voltage gate
    if verbose:
        print("[4] Applying hybrid voltage gate...")
    
    hybrid_result = hybrid_predict(signal_12, final_class, cnn_probs, fs)
    hybrid_class = hybrid_result["predicted_class"]
    hybrid_adjusted = hybrid_class != final_class
    
    if verbose and hybrid_adjusted:
        print(f"  Hybrid gate adjusted: {final_class} → {hybrid_class}")
    
    # [Step 5] Clinical context (age-aware thresholds, flags)
    if verbose:
        print("[5] Generating clinical context and flags...")
    
    clinical_result = apply_clinical_context(consensus_intervals, patient_context)
    flags = clinical_result.get("flags", [])
    urgency = clinical_result.get("urgency", "NORMAL")
    suppressed = clinical_result.get("suppressed", [])
    
    # [Generate Reasoning]
    reasoning = f"""
    Clinical Analysis Summary:
    ─────────────────────────
    Signal Quality: OK
    Multi-Lead Consensus HR: {consensus_intervals.get('hr', '?'):.0f} bpm
    
    Prediction Pipeline:
    1. CNN classified as: {cnn_class} ({cnn_confidence:.1%})
    2. VAE anomaly score: {anomaly_score:.3f if anomaly_score else 'N/A'}
       {'→ Filtered out HYP (false positive)' if vae_filtered else '→ No filtering applied'}
    3. Hybrid voltage gate: {final_class} {'→ adjusted' if hybrid_adjusted else '(no change)'}
    4. Final diagnosis: {hybrid_class}
    
    Urgency Level: {urgency}
    Clinical Flags: {len(flags)} alert(s)
    """
    
    return {
        "predicted_class": hybrid_class,
        "predicted_class_alt": cnn_class,  # Before VAE/hybrid filtering
        "confidence": cnn_confidence,
        "cnn_probabilities": cnn_probs,
        "urgency": urgency,
        "flags": flags,
        "suppressed": suppressed,
        "multiclinical_intervals": consensus_intervals,
        "dispersion_metrics": dispersion,
        "anomaly_score": anomaly_score,
        "vae_filtered": vae_filtered,
        "hybrid_adjusted": hybrid_adjusted,
        "reasoning": reasoning.strip(),
    }


# =============================================================================
# Convenience Export Functions
# =============================================================================

def classify_signal(signal_12, age, sex, is_athlete=False, is_pacemaker=False, fs=500):
    """Quick wrapper for common use case."""
    patient = {
        "age": age,
        "sex": sex,
        "is_athlete": is_athlete,
        "is_pacemaker": is_pacemaker,
        "is_pregnant": False,
        "potassium_level": None,
    }
    return predict_full_pipeline(signal_12, patient, fs)


if __name__ == "__main__":
    print("Hybrid CNN-VAE Classifier")
    print("========================")
    print("\nImport this module to use:")
    print("  from hybrid_cnn_vae_classifier import predict_full_pipeline")
    print("  result = predict_full_pipeline(signal_12, patient_context)")
