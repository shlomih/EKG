"""
test_multilabel.py
==================
Unit tests for the multi-label ECG classifier.

Run:
    python -m pytest test_multilabel.py -v
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# extract_multilabel_vector
# ---------------------------------------------------------------------------

def test_extract_norm():
    from multilabel_classifier import extract_multilabel_vector, MULTILABEL_CODES
    vec = extract_multilabel_vector({"NORM": 100.0})
    assert vec[MULTILABEL_CODES.index("NORM")] == 1.0
    assert vec.sum() == 1.0


def test_extract_multilabel():
    from multilabel_classifier import extract_multilabel_vector, MULTILABEL_CODES
    vec = extract_multilabel_vector({"NORM": 100.0, "LVH": 75.0, "CLBBB": 100.0})
    assert vec[MULTILABEL_CODES.index("NORM")]  == 1.0
    assert vec[MULTILABEL_CODES.index("LVH")]   == 1.0
    assert vec[MULTILABEL_CODES.index("CLBBB")] == 1.0
    assert vec.sum() == 3.0


def test_extract_below_threshold():
    from multilabel_classifier import extract_multilabel_vector, MULTILABEL_CODES
    # likelihood 40 < default threshold 50 — should not fire
    vec = extract_multilabel_vector({"IMI": 40.0, "NORM": 100.0})
    assert vec[MULTILABEL_CODES.index("IMI")]  == 0.0
    assert vec[MULTILABEL_CODES.index("NORM")] == 1.0


def test_extract_unknown_code():
    from multilabel_classifier import extract_multilabel_vector
    # Unknown code should not crash and should produce zero vector
    vec = extract_multilabel_vector({"AFIB": 100.0, "STACH": 100.0})
    assert vec.sum() == 0.0


def test_extract_empty():
    from multilabel_classifier import extract_multilabel_vector
    vec = extract_multilabel_vector({})
    assert vec.sum() == 0.0


def test_extract_custom_threshold():
    from multilabel_classifier import extract_multilabel_vector, MULTILABEL_CODES
    vec = extract_multilabel_vector({"LVH": 30.0}, conf_threshold=25.0)
    assert vec[MULTILABEL_CODES.index("LVH")] == 1.0


# ---------------------------------------------------------------------------
# predict_multilabel — input validation
# ---------------------------------------------------------------------------

class _FakeModel:
    """Minimal stand-in for ECGNetJoint that returns fixed logits."""
    def __call__(self, sig_t, aux_t):
        import torch
        batch = sig_t.shape[0]
        from multilabel_classifier import N_ML_CLASSES
        return torch.zeros(batch, N_ML_CLASSES)

    def eval(self):
        return self


def test_predict_none_raises():
    from multilabel_classifier import predict_multilabel
    model = _FakeModel()
    with pytest.raises(ValueError, match="2D numpy array"):
        predict_multilabel(model, None)


def test_predict_1d_raises():
    from multilabel_classifier import predict_multilabel
    model = _FakeModel()
    with pytest.raises(ValueError):
        predict_multilabel(model, np.zeros(5000))


def test_predict_output_keys():
    from multilabel_classifier import predict_multilabel, MULTILABEL_CODES
    model = _FakeModel()
    sig = np.zeros((12, 5000), dtype=np.float32)
    result = predict_multilabel(model, sig)
    for key in ("primary", "description", "confidence", "conditions", "scores", "per_class"):
        assert key in result, f"Missing key: {key}"
    # All codes present in scores and per_class
    for code in MULTILABEL_CODES:
        assert code in result["scores"]
        assert code in result["per_class"]


def test_predict_primary_in_multilabel_codes():
    from multilabel_classifier import predict_multilabel, MULTILABEL_CODES
    model = _FakeModel()
    sig = np.zeros((12, 5000), dtype=np.float32)
    result = predict_multilabel(model, sig)
    assert result["primary"] in MULTILABEL_CODES


def test_predict_confidence_in_range():
    from multilabel_classifier import predict_multilabel
    model = _FakeModel()
    sig = np.zeros((12, 5000), dtype=np.float32)
    result = predict_multilabel(model, sig)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_threshold_behavior():
    """All logits=0 → all probs=0.5. With default threshold 0.40, all should be detected."""
    from multilabel_classifier import predict_multilabel, MULTILABEL_CODES
    model = _FakeModel()   # returns logits=0 → prob=0.5 for all classes
    sig = np.zeros((12, 5000), dtype=np.float32)
    result = predict_multilabel(model, sig, threshold=0.40)
    # prob=0.5 >= 0.40 → all 12 conditions detected
    assert set(result["conditions"]) == set(MULTILABEL_CODES)


def test_predict_high_threshold_empty():
    """threshold=1.0 should give empty conditions list (prob can't reach 1.0 from logit=0)."""
    from multilabel_classifier import predict_multilabel
    model = _FakeModel()
    sig = np.zeros((12, 5000), dtype=np.float32)
    result = predict_multilabel(model, sig, threshold=1.0)
    assert result["conditions"] == []


# ---------------------------------------------------------------------------
# apply_patient_context
# ---------------------------------------------------------------------------

def _make_result(detected_codes=None):
    """Build a minimal result dict for apply_patient_context testing."""
    from multilabel_classifier import (
        MULTILABEL_CODES, CLINICAL_GUIDANCE, URGENCY, CONDITION_DESCRIPTIONS
    )
    detected = set(detected_codes or [])
    per_class = {}
    for code in MULTILABEL_CODES:
        per_class[code] = {
            "prob":        1.0 if code in detected else 0.1,
            "detected":    code in detected,
            "description": CONDITION_DESCRIPTIONS.get(code, code),
            "urgency":     URGENCY.get(code, 0),
            "action":      CLINICAL_GUIDANCE.get(code, {}).get("action", ""),
            "note":        CLINICAL_GUIDANCE.get(code, {}).get("note", ""),
        }
    return {
        "primary":     next(iter(detected), MULTILABEL_CODES[0]),
        "description": "",
        "confidence":  1.0,
        "conditions":  list(detected),
        "scores":      {c: (1.0 if c in detected else 0.1) for c in MULTILABEL_CODES},
        "per_class":   per_class,
    }


def test_pacemaker_clbbb_note():
    from multilabel_classifier import apply_patient_context
    result = _make_result(["CLBBB"])
    out = apply_patient_context(result, {"has_pacemaker": True})
    note = out["per_class"]["CLBBB"]["note"]
    assert "Pacemaker" in note


def test_athlete_lvh_note():
    from multilabel_classifier import apply_patient_context
    result = _make_result(["LVH"])
    out = apply_patient_context(result, {"is_athlete": True})
    note = out["per_class"]["LVH"]["note"]
    assert "Athlete" in note or "athletic" in note.lower()


def test_athlete_irbbb_note():
    from multilabel_classifier import apply_patient_context
    result = _make_result(["IRBBB"])
    out = apply_patient_context(result, {"is_athlete": True})
    note = out["per_class"]["IRBBB"]["note"]
    assert "Athlete" in note or "benign" in note.lower()


def test_hypokalemia_pvc_note():
    from multilabel_classifier import apply_patient_context
    result = _make_result(["PVC"])
    out = apply_patient_context(result, {"k_level": 3.0})
    note = out["per_class"]["PVC"]["note"]
    assert "3.0" in note or "Hypokalemia" in note or "potassium" in note.lower()


def test_hyperkalemia_wide_qrs_note():
    from multilabel_classifier import apply_patient_context
    result = _make_result(["CLBBB"])
    out = apply_patient_context(result, {"k_level": 6.0})
    note = out["per_class"]["CLBBB"]["note"]
    assert "Hyperkalemia" in note or "6.0" in note


def test_invalid_k_level_does_not_crash():
    from multilabel_classifier import apply_patient_context
    result = _make_result(["PVC"])
    # "N/A" string should not crash — falls back to 4.0 (normal)
    out = apply_patient_context(result, {"k_level": "N/A"})
    # Note should NOT contain hypokalemia/hyperkalemia tags
    note = out["per_class"]["PVC"]["note"]
    assert "Hypokalemia" not in note
    assert "Hyperkalemia" not in note


def test_young_lvh_hcm_note():
    from multilabel_classifier import apply_patient_context
    result = _make_result(["LVH"])
    out = apply_patient_context(result, {"age": 28, "k_level": 4.0})
    note = out["per_class"]["LVH"]["note"]
    assert "HCM" in note or "hypertrophic" in note.lower()


def test_elderly_clbbb_note():
    from multilabel_classifier import apply_patient_context
    result = _make_result(["CLBBB"])
    out = apply_patient_context(result, {"age": 80, "k_level": 4.0})
    note = out["per_class"]["CLBBB"]["note"]
    assert "80" in note or "lderly" in note or "ischemic" in note.lower()


def test_pregnant_stach_action():
    """STACH action should be replaced for pregnant patients."""
    from multilabel_classifier import apply_patient_context, CLINICAL_GUIDANCE
    # STACH is not in MULTILABEL_CODES but IS in CLINICAL_GUIDANCE — test shows
    # that apply_patient_context only modifies codes present in per_class (safe).
    result = _make_result([])
    original_action = result["per_class"].get("STACH", {}).get("action", None)
    out = apply_patient_context(result, {"is_pregnant": True, "k_level": 4.0})
    # STACH is not in per_class (not a MULTILABEL_CODE) so no crash expected
    assert out is result  # returns same dict


def test_female_lvh_cornell_note():
    from multilabel_classifier import apply_patient_context
    result = _make_result(["LVH"])
    out = apply_patient_context(result, {"sex": "F", "k_level": 4.0})
    note = out["per_class"]["LVH"]["note"]
    assert "Cornell" in note or "Female" in note or "female" in note.lower()


def test_no_context_no_crash():
    from multilabel_classifier import apply_patient_context
    result = _make_result(["NORM", "LVH", "CLBBB"])
    # Empty profile should use all defaults without crashing
    out = apply_patient_context(result, {})
    assert out is result


# ---------------------------------------------------------------------------
# dataset_chapman — parse_snomed_codes
# ---------------------------------------------------------------------------

def test_parse_snomed_codes(tmp_path):
    from dataset_chapman import parse_snomed_codes
    hea = tmp_path / "test.hea"
    hea.write_text("JS00001 12 500 5000\n#Dx: 164889003,426783006\n")
    codes = parse_snomed_codes(str(hea))
    assert "164889003" in codes
    assert "426783006" in codes
    assert len(codes) == 2


def test_parse_snomed_codes_missing_field(tmp_path):
    from dataset_chapman import parse_snomed_codes
    hea = tmp_path / "nodx.hea"
    hea.write_text("JS00001 12 500 5000\n# no Dx field here\n")
    codes = parse_snomed_codes(str(hea))
    assert codes == []


def test_parse_snomed_codes_nonexistent():
    from dataset_chapman import parse_snomed_codes
    codes = parse_snomed_codes("/nonexistent/path.hea")
    assert codes == []


# ---------------------------------------------------------------------------
# dataset_chapman — snomed_to_multilabel (no duplicate mapping)
# ---------------------------------------------------------------------------

def test_no_duplicate_snomed_keys():
    """Verify the SNOMED_TO_LABEL dict has no duplicate keys (Python silently keeps last)."""
    import ast, inspect
    import dataset_chapman
    src = inspect.getsource(dataset_chapman)
    # Find the dict literal and count each key
    start = src.index("SNOMED_TO_LABEL = {")
    end   = src.index("}", start) + 1
    dict_src = src[start + len("SNOMED_TO_LABEL = "):end]
    # Extract quoted keys
    import re
    keys = re.findall(r'"(\d+)"', dict_src)
    assert len(keys) == len(set(keys)), f"Duplicate SNOMED keys: {[k for k in keys if keys.count(k) > 1]}"


def test_snomed_afib_maps_correctly():
    from dataset_chapman import snomed_to_multilabel, MERGED_CODE_TO_IDX
    vec = snomed_to_multilabel(["164889003"])   # AFIB code
    assert vec[MERGED_CODE_TO_IDX["AFIB"]] == 1.0


def test_snomed_unknown_code_zero():
    from dataset_chapman import snomed_to_multilabel
    vec = snomed_to_multilabel(["999999999"])
    assert vec.sum() == 0.0
