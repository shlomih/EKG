"""
hybrid_classifier.py
====================
Hybrid ECG classifier combining CNN predictions with hand-crafted
voltage criteria for improved precision, especially for HYP.

Hypertrophy is diagnosed by voltage criteria in clinical practice:
  - Sokolow-Lyon:  S(V1) + R(V5 or V6) > 3.5 mV  -> LVH
  - Cornell:       R(aVL) + S(V3) > 2.8 mV (male) / 2.0 mV (female)  -> LVH
  - RVH:           R(V1) > 0.7 mV  or  R(V1)/S(V1) > 1

The CNN alone gets 19% precision on HYP because it lacks explicit
voltage awareness. This module adds voltage features as a gate:
  - If CNN says HYP but voltage criteria disagree -> downgrade confidence
  - If CNN says non-HYP but voltage criteria strongly positive -> boost HYP

Also applies per-class confidence thresholds to eliminate low-confidence
false positives across all classes.

Usage (from app.py):
    from hybrid_classifier import hybrid_predict
    result = hybrid_predict(model_data, signal_12, fs, lead_names, sex="M")
"""

import numpy as np

SUPERCLASS_LABELS = ["NORM", "MI", "STTC", "HYP", "CD"]
SUPERCLASS_DESCRIPTIONS = {
    "NORM": "Normal ECG",
    "MI":   "Myocardial Infarction",
    "STTC": "ST/T Change",
    "HYP":  "Hypertrophy",
    "CD":   "Conduction Disturbance",
}

LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# Per-class minimum confidence thresholds
# Below these, the prediction falls back to the next-best class
CLASS_THRESHOLDS = {
    "NORM": 0.30,
    "MI":   0.25,
    "STTC": 0.25,
    "HYP":  0.35,  # Higher threshold — reduce false positives
    "CD":   0.25,
}


# ---------------------------------------------------------
# Voltage Feature Extraction
# ---------------------------------------------------------

def _get_lead_amplitudes(signal_12, fs, lead_names):
    """
    Extract peak R-wave and S-wave amplitudes per lead.
    Uses a simple approach: R = max positive peak, S = max negative peak
    in the QRS complex region around detected R-peaks.

    Returns dict: lead_name -> {"r_amp": float, "s_amp": float} in mV
    """
    name_to_idx = {name.upper(): i for i, name in enumerate(lead_names)}
    amplitudes = {}

    for lead_name in LEAD_ORDER:
        idx = name_to_idx.get(lead_name.upper())
        if idx is None:
            amplitudes[lead_name] = {"r_amp": 0.0, "s_amp": 0.0}
            continue

        sig = signal_12[:, idx].copy()
        sig = sig - np.mean(sig)

        # Simple R-peak detection
        std_val = np.std(sig)
        if std_val < 1e-6:
            amplitudes[lead_name] = {"r_amp": 0.0, "s_amp": 0.0}
            continue

        threshold = 2.0 * std_val
        above = np.where(sig > threshold)[0]

        peaks = []
        if len(above) > 0:
            peaks.append(above[0])
            for i in range(1, len(above)):
                if above[i] - above[i - 1] > fs // 2:
                    peaks.append(above[i])

        if len(peaks) < 2:
            # Fallback: use overall max/min
            amplitudes[lead_name] = {
                "r_amp": float(np.max(sig)),
                "s_amp": float(np.min(sig)),
            }
            continue

        # Measure R and S amplitudes around each peak
        r_values = []
        s_values = []
        qrs_window = int(0.06 * fs)  # 60ms each side of peak

        for peak in peaks:
            start = max(0, peak - qrs_window)
            end = min(len(sig), peak + qrs_window)
            segment = sig[start:end]
            if len(segment) > 0:
                r_values.append(float(np.max(segment)))
                s_values.append(float(np.min(segment)))

        amplitudes[lead_name] = {
            "r_amp": float(np.median(r_values)) if r_values else 0.0,
            "s_amp": float(np.median(s_values)) if s_values else 0.0,
        }

    return amplitudes


def compute_voltage_criteria(signal_12, fs, lead_names, sex="M"):
    """
    Compute LVH/RVH voltage criteria from 12-lead signal.

    Returns dict with:
      - sokolow_lyon: S(V1) + R(V5/V6) value and bool
      - cornell: R(aVL) + S(V3) value and bool
      - rvh_r_v1: R(V1) value and bool
      - lvh_any: True if any LVH criterion met
      - rvh_any: True if any RVH criterion met
      - hypertrophy_score: 0-1 composite score
    """
    amps = _get_lead_amplitudes(signal_12, fs, lead_names)

    # Sokolow-Lyon: S(V1) + R(V5 or V6) > 3.5 mV
    s_v1 = abs(amps.get("V1", {}).get("s_amp", 0.0))
    r_v5 = amps.get("V5", {}).get("r_amp", 0.0)
    r_v6 = amps.get("V6", {}).get("r_amp", 0.0)
    sokolow_value = s_v1 + max(r_v5, r_v6)
    sokolow_met = sokolow_value > 3.5

    # Cornell: R(aVL) + S(V3) > 2.8mV (male) / 2.0mV (female)
    r_avl = amps.get("AVL", {}).get("r_amp", 0.0)
    s_v3 = abs(amps.get("V3", {}).get("s_amp", 0.0))
    cornell_value = r_avl + s_v3
    cornell_threshold = 2.0 if sex == "F" else 2.8
    cornell_met = cornell_value > cornell_threshold

    # RVH: R(V1) > 0.7 mV
    r_v1 = amps.get("V1", {}).get("r_amp", 0.0)
    rvh_met = r_v1 > 0.7

    # Composite score
    lvh_any = sokolow_met or cornell_met
    rvh_any = rvh_met
    hyp_any = lvh_any or rvh_any

    # Score: how far above thresholds (0-1 range)
    sokolow_score = max(0.0, min(1.0, (sokolow_value - 2.5) / 2.0))  # ramps from 2.5 to 4.5
    cornell_score = max(0.0, min(1.0, (cornell_value - 1.5) / 2.0))
    rvh_score = max(0.0, min(1.0, (r_v1 - 0.4) / 0.6))
    hypertrophy_score = max(sokolow_score, cornell_score, rvh_score)

    return {
        "sokolow_lyon": {"value": round(sokolow_value, 2), "met": sokolow_met},
        "cornell": {"value": round(cornell_value, 2), "threshold": cornell_threshold, "met": cornell_met},
        "rvh_r_v1": {"value": round(r_v1, 2), "met": rvh_met},
        "lvh_any": lvh_any,
        "rvh_any": rvh_any,
        "hypertrophy_any": hyp_any,
        "hypertrophy_score": round(hypertrophy_score, 3),
        "amplitudes": amps,
    }


# ---------------------------------------------------------
# Hybrid Prediction
# ---------------------------------------------------------

def hybrid_predict(model_data, signal_12, fs, lead_names, sex="M"):
    """
    Hybrid prediction: CNN probabilities + voltage criteria + thresholds.

    Args:
        model_data: dict from load_cnn_classifier()
        signal_12: (N, 12) numpy array
        fs: sampling rate
        lead_names: list of 12 lead name strings
        sex: "M" or "F" for Cornell threshold

    Returns:
        dict: prediction, description, confidence, probabilities,
              voltage_criteria, adjustment_applied
    """
    from cnn_classifier import predict_cnn, SUPERCLASS_LABELS as CNN_LABELS

    # Step 1: Get CNN predictions
    cnn_result = predict_cnn(model_data, signal_12, fs)
    probs = cnn_result["probabilities"]

    # Convert to array for manipulation
    prob_array = np.array([probs.get(label, 0.0) for label in SUPERCLASS_LABELS])

    # Step 2: Compute voltage criteria
    voltage = compute_voltage_criteria(signal_12, fs, lead_names, sex=sex)
    hyp_score = voltage["hypertrophy_score"]
    hyp_idx = SUPERCLASS_LABELS.index("HYP")

    adjustment = None

    # Step 3: Adjust HYP probability based on voltage criteria
    cnn_hyp_prob = prob_array[hyp_idx]

    if cnn_result["prediction"] == "HYP":
        if not voltage["hypertrophy_any"] and hyp_score < 0.3:
            # CNN says HYP but voltage criteria disagree -> penalize
            prob_array[hyp_idx] *= 0.3
            adjustment = "HYP downgraded: voltage criteria not met"
        elif hyp_score < 0.2:
            prob_array[hyp_idx] *= 0.5
            adjustment = "HYP reduced: weak voltage evidence"
    else:
        if voltage["hypertrophy_any"] and hyp_score > 0.6:
            # CNN missed HYP but voltage criteria are strong -> boost
            boost = hyp_score * 0.3
            prob_array[hyp_idx] = min(0.9, prob_array[hyp_idx] + boost)
            adjustment = f"HYP boosted: strong voltage criteria (score={hyp_score:.2f})"

    # Renormalize
    prob_sum = prob_array.sum()
    if prob_sum > 0:
        prob_array = prob_array / prob_sum

    # Step 4: Apply per-class confidence thresholds
    pred_idx = int(np.argmax(prob_array))
    pred_label = SUPERCLASS_LABELS[pred_idx]
    pred_conf = float(prob_array[pred_idx])

    threshold = CLASS_THRESHOLDS.get(pred_label, 0.25)
    if pred_conf < threshold:
        # Fall back to next-best class above its threshold
        sorted_indices = np.argsort(prob_array)[::-1]
        for idx in sorted_indices:
            label = SUPERCLASS_LABELS[idx]
            conf = float(prob_array[idx])
            if conf >= CLASS_THRESHOLDS.get(label, 0.25):
                pred_idx = idx
                pred_label = label
                pred_conf = conf
                if adjustment:
                    adjustment += f"; threshold fallback to {label}"
                else:
                    adjustment = f"Threshold fallback: {SUPERCLASS_LABELS[int(np.argmax(prob_array))]} below {threshold:.0%}, using {label}"
                break
        else:
            # All below threshold — use highest confidence anyway
            pred_idx = int(np.argmax(prob_array))
            pred_label = SUPERCLASS_LABELS[pred_idx]
            pred_conf = float(prob_array[pred_idx])

    # Build result
    prob_dict = {SUPERCLASS_LABELS[i]: round(float(p), 3) for i, p in enumerate(prob_array)}

    return {
        "prediction": pred_label,
        "description": SUPERCLASS_DESCRIPTIONS.get(pred_label, pred_label),
        "confidence": pred_conf,
        "probabilities": prob_dict,
        "voltage_criteria": voltage,
        "adjustment_applied": adjustment,
        "cnn_raw_prediction": cnn_result["prediction"],
        "cnn_raw_confidence": cnn_result["confidence"],
    }
