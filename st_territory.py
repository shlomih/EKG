"""
st_territory.py
===============
Multi-lead ST-segment analysis with coronary territory localization.

Maps ST elevation/depression patterns across leads to the affected
coronary artery territory:

    LAD  (Left Anterior Descending) — Anterior wall
         Leads: V1, V2, V3, V4 (± I, aVL)

    RCA  (Right Coronary Artery)    — Inferior wall
         Leads: II, III, aVF (± V5, V6)

    LCx  (Left Circumflex)          — Lateral wall
         Leads: I, aVL, V5, V6

Reciprocal changes (depression in opposite territory) increase
specificity and urgency.

Usage (from app.py):
    from st_territory import analyze_st_territories
    result = analyze_st_territories(signals_12, fs, lead_names)
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# Lead → Territory Mapping
# ─────────────────────────────────────────────────────────────

TERRITORIES = {
    "Anterior (LAD)": {
        "primary_leads": ["V1", "V2", "V3", "V4"],
        "extended_leads": ["I", "AVL"],
        "reciprocal_leads": ["II", "III", "AVF"],
        "artery": "LAD",
        "wall": "Anterior",
        "description": "Left Anterior Descending — anterior wall, septum",
    },
    "Inferior (RCA)": {
        "primary_leads": ["II", "III", "AVF"],
        "extended_leads": [],
        "reciprocal_leads": ["I", "AVL"],
        "artery": "RCA",
        "wall": "Inferior",
        "description": "Right Coronary Artery — inferior wall, RV",
    },
    "Lateral (LCx)": {
        "primary_leads": ["I", "AVL", "V5", "V6"],
        "extended_leads": [],
        "reciprocal_leads": ["III", "AVF"],
        "artery": "LCx",
        "wall": "Lateral",
        "description": "Left Circumflex — lateral wall",
    },
}

# ST thresholds (in mV) — standard clinical criteria
# Men: ≥0.2mV in V2-V3, ≥0.1mV in other leads
# Women: ≥0.15mV in V2-V3, ≥0.1mV in other leads
ELEVATION_THRESHOLD_V2V3_MALE = 0.2
ELEVATION_THRESHOLD_V2V3_FEMALE = 0.15
ELEVATION_THRESHOLD_OTHER = 0.1
DEPRESSION_THRESHOLD = -0.05  # ≥0.05mV depression = significant


# ─────────────────────────────────────────────────────────────
# Per-lead ST measurement
# ─────────────────────────────────────────────────────────────

def measure_st_deviation(signal, fs=500):
    """
    Measure ST deviation for a single lead.

    Uses a simple approach:
    1. Find R-peaks via threshold
    2. Measure voltage at J-point + 60ms relative to PQ baseline

    Returns: dict with st_mv (mV), n_beats, confidence
    """
    signal = np.nan_to_num(signal, nan=0.0)
    signal_clean = signal - np.mean(signal)
    std_val = np.std(signal_clean)

    if std_val < 1e-6:
        return {"st_mv": 0.0, "n_beats": 0, "confidence": 0.0}

    # R-peak detection via threshold
    threshold = np.mean(signal_clean) + 2.5 * std_val
    peaks = np.where(signal_clean > threshold)[0]

    real_peaks = []
    if len(peaks) > 0:
        real_peaks.append(peaks[0])
        for i in range(1, len(peaks)):
            if peaks[i] - peaks[i - 1] > fs // 2:
                real_peaks.append(peaks[i])

    if len(real_peaks) < 2:
        return {"st_mv": 0.0, "n_beats": 0, "confidence": 0.0}

    # Measure ST deviation per beat
    st_values = []
    for peak in real_peaks:
        # PQ baseline: 80-40ms before R peak
        bl_start = max(0, peak - int(0.08 * fs))
        bl_end = max(0, peak - int(0.04 * fs))
        if bl_end <= bl_start:
            continue
        baseline = np.mean(signal_clean[bl_start:bl_end])

        # J-point + 60ms after R peak (ST segment measurement point)
        j_point = peak + int(0.08 * fs)  # ~80ms after R = J-point approx
        st_point = j_point + int(0.06 * fs)  # +60ms into ST segment

        if st_point < len(signal_clean):
            st_voltage = signal_clean[st_point] - baseline
            st_values.append(st_voltage)

    if not st_values:
        return {"st_mv": 0.0, "n_beats": 0, "confidence": 0.0}

    median_st = float(np.median(st_values))
    # Confidence based on consistency across beats
    if len(st_values) >= 3:
        iqr = float(np.percentile(st_values, 75) - np.percentile(st_values, 25))
        confidence = max(0.0, min(1.0, 1.0 - iqr / (abs(median_st) + 0.05)))
    else:
        confidence = 0.5

    return {
        "st_mv": median_st,
        "n_beats": len(st_values),
        "confidence": confidence,
    }


# ─────────────────────────────────────────────────────────────
# Multi-lead territory analysis
# ─────────────────────────────────────────────────────────────

def analyze_st_territories(signals_12, fs, lead_names, patient_sex="M"):
    """
    Analyze ST changes across all 12 leads and map to coronary territories.

    Args:
        signals_12: (N, 12) numpy array
        fs: sampling rate
        lead_names: list of 12 lead name strings
        patient_sex: "M" or "F" (affects V2-V3 threshold)

    Returns:
        dict with:
          - lead_results: per-lead ST measurements
          - territories: per-territory findings
          - summary: overall interpretation
          - stemi_criteria_met: bool
          - affected_territory: str or None
    """
    name_to_idx = {name: i for i, name in enumerate(lead_names)}

    # Measure ST in every lead
    lead_results = {}
    for lead_name in lead_names:
        idx = name_to_idx.get(lead_name)
        if idx is not None:
            result = measure_st_deviation(signals_12[:, idx], fs)
            result["lead"] = lead_name
            lead_results[lead_name] = result

    # Analyze each territory
    territory_findings = {}

    for terr_name, terr_config in TERRITORIES.items():
        primary = terr_config["primary_leads"]
        reciprocal = terr_config["reciprocal_leads"]

        # Count leads with significant elevation
        elevated_leads = []
        for lead in primary:
            if lead not in lead_results:
                continue
            st = lead_results[lead]["st_mv"]

            # Apply sex-specific V2/V3 threshold
            if lead in ("V2", "V3"):
                thresh = (ELEVATION_THRESHOLD_V2V3_FEMALE
                          if patient_sex == "F"
                          else ELEVATION_THRESHOLD_V2V3_MALE)
            else:
                thresh = ELEVATION_THRESHOLD_OTHER

            if st >= thresh:
                elevated_leads.append({"lead": lead, "st_mv": st})

        # Count reciprocal depression
        depressed_leads = []
        for lead in reciprocal:
            if lead not in lead_results:
                continue
            st = lead_results[lead]["st_mv"]
            if st <= DEPRESSION_THRESHOLD:
                depressed_leads.append({"lead": lead, "st_mv": st})

        # Max elevation in this territory
        max_elev = max([e["st_mv"] for e in elevated_leads], default=0.0)

        # STEMI criteria: ≥2 contiguous leads with significant elevation
        meets_stemi = len(elevated_leads) >= 2
        has_reciprocal = len(depressed_leads) >= 1

        # Severity
        if meets_stemi and has_reciprocal:
            severity = "CRITICAL"
            interpretation = (
                f"STEMI pattern in {terr_config['wall']} wall "
                f"({len(elevated_leads)} leads elevated, "
                f"{len(depressed_leads)} reciprocal). "
                f"Suspect {terr_config['artery']} occlusion."
            )
        elif meets_stemi:
            severity = "WARNING"
            interpretation = (
                f"ST elevation in {terr_config['wall']} leads "
                f"({len(elevated_leads)} leads). "
                f"No reciprocal changes — may be early STEMI, "
                f"pericarditis, or benign early repolarization."
            )
        elif len(elevated_leads) == 1:
            severity = "INFO"
            interpretation = (
                f"Isolated ST elevation in {elevated_leads[0]['lead']} "
                f"({elevated_leads[0]['st_mv']:.2f} mV). Monitor."
            )
        elif len(depressed_leads) >= 2:
            severity = "WARNING"
            interpretation = (
                f"ST depression in {', '.join(d['lead'] for d in depressed_leads)}. "
                f"Consider ischemia in {terr_config['wall']} territory or "
                f"reciprocal change from opposite territory STEMI."
            )
        else:
            severity = "NORMAL"
            interpretation = f"No significant ST changes in {terr_config['wall']} leads."

        territory_findings[terr_name] = {
            "severity": severity,
            "interpretation": interpretation,
            "elevated_leads": elevated_leads,
            "depressed_leads": depressed_leads,
            "max_elevation_mv": max_elev,
            "meets_stemi_criteria": meets_stemi,
            "has_reciprocal": has_reciprocal,
            "artery": terr_config["artery"],
            "wall": terr_config["wall"],
        }

    # Overall summary
    stemi_territories = [
        name for name, f in territory_findings.items()
        if f["meets_stemi_criteria"]
    ]
    any_stemi = len(stemi_territories) > 0

    if any_stemi:
        primary_territory = max(
            stemi_territories,
            key=lambda t: territory_findings[t]["max_elevation_mv"],
        )
        finding = territory_findings[primary_territory]
        summary = (
            f"STEMI detected — {finding['artery']} territory "
            f"({finding['wall']} wall). "
            f"Max ST elevation: {finding['max_elevation_mv']:.2f} mV."
        )
        urgency = "EMERGENCY" if finding["has_reciprocal"] else "URGENT"
    else:
        primary_territory = None
        # Check for significant depression anywhere
        depressed_territories = [
            name for name, f in territory_findings.items()
            if len(f["depressed_leads"]) >= 2
        ]
        if depressed_territories:
            summary = "ST depression pattern detected — consider ischemia."
            urgency = "URGENT"
        else:
            summary = "No significant ST-segment changes across all territories."
            urgency = "NORMAL"

    return {
        "lead_results": lead_results,
        "territories": territory_findings,
        "summary": summary,
        "urgency": urgency,
        "stemi_criteria_met": any_stemi,
        "affected_territory": primary_territory,
    }
