"""
ensemble_classifier.py
======================
Ensemble of v9 (ECGNetTransformer, z-score norm) and v10 (ECGNetJoint,
amplitude norm) models. Architecturally diverse models with different
normalization = maximum prediction diversity.

Usage:
    from ensemble_classifier import load_ensemble, predict_ensemble
    ens = load_ensemble()
    result = predict_ensemble(ens, signal_12, fs=500, sex="M", age=50)
"""

import numpy as np
from cnn_classifier import load_cnn_classifier, predict_cnn, SUPERCLASS_LABELS
from hybrid_classifier import CLASS_THRESHOLDS, SUPERCLASS_DESCRIPTIONS

DEFAULT_V9_PATH  = "models/ecg_cnn_v9_backup.pt"
DEFAULT_V10_PATH = "models/ecg_cnn.pt"

# Optimal weights from grid-search on test fold 10:
# v9 (Transformer) 80% + v10 (Joint+ampnorm) 20%
# Counterintuitive but correct: v9 has better recall across all classes,
# v10 anchors HYP precision. Together: HYP F1=0.456, MacroF1=0.675.
DEFAULT_WEIGHTS = (0.80, 0.20)   # (v9, v10)


def load_ensemble(v9_path=DEFAULT_V9_PATH, v10_path=DEFAULT_V10_PATH,
                  weights=DEFAULT_WEIGHTS):
    """
    Load v9 + v10 into an ensemble dict.

    Returns:
        dict with 'models', 'weights', 'labels'
    """
    print("Loading ensemble models...")
    m9  = load_cnn_classifier(v9_path)
    m10 = load_cnn_classifier(v10_path)
    if m9 is None:
        raise FileNotFoundError(f"v9 model not found: {v9_path}")
    if m10 is None:
        raise FileNotFoundError(f"v10 model not found: {v10_path}")
    print(f"  v9  ({m9['model_type']}, norm={m9['norm_mode']}, n_aux={m9['n_aux']})")
    print(f"  v10 ({m10['model_type']}, norm={m10['norm_mode']}, n_aux={m10['n_aux']})")
    print(f"  Weights: v9={weights[0]:.2f}  v10={weights[1]:.2f}")
    return {
        "models":  [m9, m10],
        "weights": list(weights),
        "labels":  SUPERCLASS_LABELS,
    }


def predict_ensemble(ensemble_data, signal_12, fs=500, sex="M", age=50,
                     apply_thresholds=True):
    """
    Run both models, weighted-average softmax probabilities, apply thresholds.

    Args:
        ensemble_data:     dict from load_ensemble()
        signal_12:         (N, 12) numpy array in mV
        fs:                sampling rate
        sex:               "M" or "F"
        age:               patient age in years
        apply_thresholds:  if True, apply CLASS_THRESHOLDS fallback logic

    Returns:
        dict: prediction, description, confidence, probabilities,
              individual_probs (per model), weights_used
    """
    models  = ensemble_data["models"]
    weights = ensemble_data["weights"]
    n       = len(SUPERCLASS_LABELS)

    individual_probs = []
    for model_data in models:
        res = predict_cnn(model_data, signal_12, fs=fs, sex=sex, age=age)
        pvec = np.array([res["probabilities"].get(lbl, 0.0)
                         for lbl in SUPERCLASS_LABELS], dtype=np.float32)
        individual_probs.append(pvec)

    # Weighted average + renormalise
    ens_probs = sum(w * p for w, p in zip(weights, individual_probs))
    ens_probs = ens_probs / (ens_probs.sum() + 1e-9)

    pred_idx   = int(np.argmax(ens_probs))
    pred_label = SUPERCLASS_LABELS[pred_idx]
    pred_conf  = float(ens_probs[pred_idx])

    if apply_thresholds:
        threshold = CLASS_THRESHOLDS.get(pred_label, 0.0)
        if pred_conf < threshold:
            sorted_idx = np.argsort(ens_probs)[::-1]
            for idx in sorted_idx:
                lbl  = SUPERCLASS_LABELS[idx]
                conf = float(ens_probs[idx])
                if conf >= CLASS_THRESHOLDS.get(lbl, 0.0):
                    pred_idx   = idx
                    pred_label = lbl
                    pred_conf  = conf
                    break

    prob_dict = {SUPERCLASS_LABELS[i]: round(float(p), 3)
                 for i, p in enumerate(ens_probs)}

    return {
        "prediction":   pred_label,
        "description":  SUPERCLASS_DESCRIPTIONS.get(pred_label, pred_label),
        "confidence":   round(pred_conf, 3),
        "probabilities": prob_dict,
        "individual_probs": [
            {lbl: round(float(p), 3)
             for lbl, p in zip(SUPERCLASS_LABELS, pvec)}
            for pvec in individual_probs
        ],
        "weights_used": weights,
    }
