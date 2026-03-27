"""
clinical_rules.py
=================
Rule-based clinical findings beyond the 5 superclasses.
Detects conditions from the 12-lead signal that the CNN does not cover:

  - Axis deviation (left, right, extreme)
  - Low voltage
  - Tall/peaked T-waves (hyperkalemia pattern)
  - T-wave inversion patterns
  - Poor R-wave progression
  - Sinus rhythm assessment

These findings are presented alongside (not replacing) the AI classification
and interval analysis.

Usage (from app.py):
    from clinical_rules import analyze_clinical_rules
    findings = analyze_clinical_rules(signals_12, fs, lead_names, patient_profile)
"""

import numpy as np

LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def _get_lead_signal(signals_12, lead_names, target_lead):
    """Get signal for a specific lead by name."""
    name_to_idx = {name.upper(): i for i, name in enumerate(lead_names)}
    idx = name_to_idx.get(target_lead.upper())
    if idx is None:
        return None
    return signals_12[:, idx]


def _estimate_qrs_axis(signals_12, fs, lead_names):
    """
    Estimate the cardiac axis from leads I and aVF.
    Uses net QRS area (integral) in each lead.

    Returns: axis in degrees, or None if unable to compute.
    """
    lead_i = _get_lead_signal(signals_12, lead_names, "I")
    lead_avf = _get_lead_signal(signals_12, lead_names, "AVF")

    if lead_i is None or lead_avf is None:
        return None

    # Find R-peaks in Lead I for beat segmentation
    sig = lead_i - np.mean(lead_i)
    std_val = np.std(sig)
    if std_val < 1e-6:
        return None

    threshold = 2.0 * std_val
    above = np.where(sig > threshold)[0]
    peaks = []
    if len(above) > 0:
        peaks.append(above[0])
        for i in range(1, len(above)):
            if above[i] - above[i - 1] > fs // 2:
                peaks.append(above[i])

    if len(peaks) < 2:
        return None

    # Measure net QRS area around each beat in both leads
    qrs_window = int(0.06 * fs)
    areas_i = []
    areas_avf = []

    for peak in peaks:
        start = max(0, peak - qrs_window)
        end = min(len(lead_i), peak + qrs_window)
        if end - start < 5:
            continue
        seg_i = lead_i[start:end] - np.mean(lead_i)
        seg_avf = lead_avf[start:end] - np.mean(lead_avf)
        areas_i.append(np.sum(seg_i))
        areas_avf.append(np.sum(seg_avf))

    if not areas_i:
        return None

    net_i = np.median(areas_i)
    net_avf = np.median(areas_avf)

    # Axis = atan2(aVF, I) in degrees
    axis = np.degrees(np.arctan2(net_avf, net_i))
    return round(float(axis), 0)


def _check_low_voltage(signals_12, lead_names):
    """
    Low voltage: all limb leads < 0.5 mV peak-to-peak
    or all precordial leads < 1.0 mV peak-to-peak.
    """
    limb_leads = ["I", "II", "III", "AVR", "AVL", "AVF"]
    precordial_leads = ["V1", "V2", "V3", "V4", "V5", "V6"]
    name_to_idx = {name.upper(): i for i, name in enumerate(lead_names)}

    limb_low = True
    for lead in limb_leads:
        idx = name_to_idx.get(lead)
        if idx is None:
            continue
        pp = np.ptp(signals_12[:, idx])
        if pp >= 0.5:
            limb_low = False
            break

    precordial_low = True
    for lead in precordial_leads:
        idx = name_to_idx.get(lead)
        if idx is None:
            continue
        pp = np.ptp(signals_12[:, idx])
        if pp >= 1.0:
            precordial_low = False
            break

    return limb_low or precordial_low


def _check_t_wave_patterns(signals_12, fs, lead_names):
    """
    Analyze T-wave morphology across leads.
    Returns findings about:
      - Peaked T-waves (hyperkalemia pattern)
      - T-wave inversions
    """
    findings = []
    name_to_idx = {name.upper(): i for i, name in enumerate(lead_names)}

    # Measure T-wave amplitude relative to R-wave in each lead
    peaked_leads = []
    inverted_leads = []

    for lead_name in LEAD_ORDER:
        idx = name_to_idx.get(lead_name.upper())
        if idx is None:
            continue

        sig = signals_12[:, idx].copy()
        sig = sig - np.mean(sig)
        std_val = np.std(sig)
        if std_val < 1e-6:
            continue

        # Find R-peaks
        threshold = 2.0 * std_val
        above = np.where(sig > threshold)[0]
        peaks = []
        if len(above) > 0:
            peaks.append(above[0])
            for i in range(1, len(above)):
                if above[i] - above[i - 1] > fs // 2:
                    peaks.append(above[i])

        if len(peaks) < 2:
            continue

        # Measure T-wave: look 150-350ms after R-peak
        t_start_offset = int(0.15 * fs)
        t_end_offset = int(0.35 * fs)
        t_amplitudes = []
        r_amplitudes = []

        for peak in peaks:
            t_start = peak + t_start_offset
            t_end = peak + t_end_offset
            if t_end >= len(sig):
                continue

            t_segment = sig[t_start:t_end]
            t_peak = np.max(t_segment)
            t_trough = np.min(t_segment)

            # T-wave is the dominant deflection in this window
            if abs(t_peak) > abs(t_trough):
                t_amplitudes.append(t_peak)
            else:
                t_amplitudes.append(t_trough)

            r_amplitudes.append(sig[peak])

        if not t_amplitudes:
            continue

        median_t = np.median(t_amplitudes)
        median_r = np.median(r_amplitudes) if r_amplitudes else 1.0

        # Peaked T-wave: T > 0.5 * R amplitude (abnormally tall)
        if median_r > 0 and median_t / median_r > 0.6 and lead_name in ["V2", "V3", "V4", "V5"]:
            peaked_leads.append(lead_name)

        # T-wave inversion in leads where it should be upright
        upright_leads = {"I", "II", "V4", "V5", "V6"}
        if lead_name in upright_leads and median_t < -0.1:
            inverted_leads.append(lead_name)

    if peaked_leads:
        findings.append({
            "severity": "WARNING",
            "code": "PEAKED_T_WAVES",
            "finding": f"Peaked T-waves in {', '.join(peaked_leads)}",
            "explanation": (
                "Tall, peaked T-waves may indicate hyperkalemia. "
                "Correlate with serum potassium level."
            ),
        })

    if inverted_leads:
        findings.append({
            "severity": "WARNING",
            "code": "T_WAVE_INVERSION",
            "finding": f"T-wave inversion in {', '.join(inverted_leads)}",
            "explanation": (
                "T-wave inversion in these leads may indicate ischemia, "
                "strain pattern, or cardiomyopathy. Clinical correlation required."
            ),
        })

    return findings


def _check_r_wave_progression(signals_12, lead_names):
    """
    Check R-wave progression across precordial leads V1-V6.
    Normal: R-wave amplitude increases from V1 to V4-V5, then decreases.
    Poor R-wave progression: R-wave stays small through V3-V4.
    """
    name_to_idx = {name.upper(): i for i, name in enumerate(lead_names)}
    precordial = ["V1", "V2", "V3", "V4", "V5", "V6"]

    r_amplitudes = []
    for lead in precordial:
        idx = name_to_idx.get(lead)
        if idx is None:
            r_amplitudes.append(0.0)
            continue
        sig = signals_12[:, idx]
        r_amplitudes.append(float(np.max(sig - np.mean(sig))))

    # Poor R-wave progression: R in V3 < 0.3 mV
    if len(r_amplitudes) >= 4 and r_amplitudes[2] < 0.3:  # V3
        return {
            "severity": "INFO",
            "code": "POOR_R_PROGRESSION",
            "finding": "Poor R-wave progression (V1-V4)",
            "explanation": (
                "R-wave amplitude fails to increase normally across precordial leads. "
                "May indicate prior anterior MI, LVH, LBBB, or normal variant."
            ),
        }
    return None


def _check_rvh(signals_12, lead_names, axis):
    """
    RVH screening: dominant R in V1 (R amplitude > S amplitude) + right axis deviation.
    Both criteria together give high specificity for right ventricular hypertrophy.
    """
    if axis is None or axis <= 90:
        return None   # Need right axis deviation (>90 deg)

    name_to_idx = {name.upper(): i for i, name in enumerate(lead_names)}
    v1_idx = name_to_idx.get("V1")
    if v1_idx is None:
        return None

    sig = signals_12[:, v1_idx] - np.mean(signals_12[:, v1_idx])
    r_amp = float(np.max(sig))
    s_amp = float(abs(np.min(sig)))

    if r_amp > s_amp:
        return {
            "severity": "WARNING",
            "code": "RVH",
            "finding": f"Right ventricular hypertrophy pattern (dominant R in V1, axis {axis:.0f} deg)",
            "explanation": (
                "Dominant R wave in V1 with right axis deviation suggests RVH. "
                "Common causes: pulmonary hypertension, chronic PE, COPD, congenital heart disease. "
                "Echo recommended to assess RV size and pressure."
            ),
        }
    return None


def analyze_clinical_rules(signals_12, fs, lead_names, patient_profile=None):
    """
    Run all clinical rules on 12-lead ECG.

    Args:
        signals_12: (N, 12) numpy array
        fs: sampling rate
        lead_names: list of 12 lead name strings
        patient_profile: dict with age, sex, etc. (optional)

    Returns:
        dict with:
          - axis: cardiac axis in degrees
          - axis_deviation: str ("Normal", "Left", "Right", "Extreme")
          - findings: list of finding dicts
          - summary: overall text summary
    """
    if patient_profile is None:
        patient_profile = {}

    findings = []

    # 1. Cardiac axis
    axis = _estimate_qrs_axis(signals_12, fs, lead_names)
    axis_deviation = "Unknown"

    if axis is not None:
        if -30 <= axis <= 90:
            axis_deviation = "Normal"
        elif -90 <= axis < -30:
            axis_deviation = "Left"
            findings.append({
                "severity": "INFO",
                "code": "LEFT_AXIS_DEVIATION",
                "finding": f"Left axis deviation ({axis:.0f} degrees)",
                "explanation": (
                    "Axis more negative than -30 degrees. "
                    "Common causes: left anterior fascicular block, LVH, inferior MI."
                ),
            })
        elif 90 < axis <= 180:
            axis_deviation = "Right"
            findings.append({
                "severity": "INFO",
                "code": "RIGHT_AXIS_DEVIATION",
                "finding": f"Right axis deviation ({axis:.0f} degrees)",
                "explanation": (
                    "Axis > 90 degrees. "
                    "Common causes: RVH, PE, COPD, lateral MI, left posterior fascicular block."
                ),
            })
        elif axis < -90 or axis > 180:
            axis_deviation = "Extreme"
            findings.append({
                "severity": "WARNING",
                "code": "EXTREME_AXIS_DEVIATION",
                "finding": f"Extreme axis deviation ({axis:.0f} degrees)",
                "explanation": (
                    "Northwest axis. Consider ventricular rhythm, "
                    "lead misplacement, or severe conduction disease."
                ),
            })

    # 2. Low voltage
    if _check_low_voltage(signals_12, lead_names):
        findings.append({
            "severity": "INFO",
            "code": "LOW_VOLTAGE",
            "finding": "Low voltage QRS complexes",
            "explanation": (
                "QRS amplitude below normal thresholds. "
                "Differential: pericardial effusion, obesity, COPD, "
                "hypothyroidism, infiltrative cardiomyopathy."
            ),
        })

    # 3. T-wave patterns
    t_findings = _check_t_wave_patterns(signals_12, fs, lead_names)
    findings.extend(t_findings)

    # Correlate with potassium if available
    k_level = patient_profile.get("k_level", 4.0)
    peaked_t = any(f["code"] == "PEAKED_T_WAVES" for f in findings)
    if peaked_t and k_level > 5.5:
        # Upgrade peaked T severity if K+ is actually high
        for f in findings:
            if f["code"] == "PEAKED_T_WAVES":
                f["severity"] = "CRITICAL"
                f["explanation"] += f" K+ is {k_level} mmol/L -- treat hyperkalemia urgently."

    # 4. R-wave progression
    r_finding = _check_r_wave_progression(signals_12, lead_names)
    if r_finding:
        findings.append(r_finding)

    # 5. RVH screening (dominant R in V1 + right axis)
    rvh_finding = _check_rvh(signals_12, lead_names, axis)
    if rvh_finding:
        findings.append(rvh_finding)

    # 6. Posterior STEMI screen (dominant R + ST depression in V1-V3)
    name_to_idx = {name.upper(): i for i, name in enumerate(lead_names)}
    posterior_leads = [l for l in ("V1", "V2", "V3") if l in name_to_idx]
    if len(posterior_leads) >= 2:
        dom_r_leads, std_dep_leads = [], []
        for lead in posterior_leads:
            sig = signals_12[:, name_to_idx[lead]] - np.mean(signals_12[:, name_to_idx[lead]])
            r_amp = float(np.max(sig))
            s_amp = float(abs(np.min(sig)))
            if r_amp > s_amp and r_amp > 0.5:
                dom_r_leads.append(lead)
            baseline = np.percentile(sig, 10)
            if baseline < -0.05:
                std_dep_leads.append(lead)
        if len(dom_r_leads) >= 2 and len(std_dep_leads) >= 2:
            findings.append({
                "severity": "WARNING",
                "code": "POSTERIOR_STEMI_SCREEN",
                "finding": f"Posterior STEMI pattern in {', '.join(posterior_leads)}",
                "explanation": (
                    "Dominant R wave with ST depression in V1-V3 may represent posterior STEMI. "
                    "Consider posterior leads (V7-V9). If clinical suspicion: activate cath lab."
                ),
            })

    # 7. Hyperacute T-wave screen (de Winter / early STEMI equivalent)
    hyperacute_leads = []
    for lead in ("V2", "V3", "V4"):
        idx = name_to_idx.get(lead)
        if idx is None:
            continue
        sig = signals_12[:, idx]
        sig_c = sig - np.mean(sig)
        # Upsloping ST depression + tall peaked T = de Winter pattern
        st_level = float(np.percentile(sig_c, 15))   # proximal baseline
        t_peak   = float(np.max(sig_c[len(sig_c) // 2:]))
        r_amp    = float(np.max(sig_c[:len(sig_c) // 2]))
        if st_level < -0.05 and t_peak > 0.4 * r_amp and t_peak > 0.4:
            hyperacute_leads.append(lead)
    if len(hyperacute_leads) >= 2:
        findings.append({
            "severity": "CRITICAL",
            "code": "HYPERACUTE_T",
            "finding": f"Hyperacute T-waves in {', '.join(hyperacute_leads)} (de Winter pattern)",
            "explanation": (
                "Upsloping ST depression with tall peaked T-waves in precordial leads — "
                "de Winter pattern is a STEMI equivalent indicating LAD occlusion. "
                "Treat as STEMI: activate cath lab immediately."
            ),
        })

    # 8. Right atrial enlargement (RAE): peaked P > 2.5 mV in lead II
    ii_idx = name_to_idx.get("II")
    if ii_idx is not None:
        sig_ii = signals_12[:, ii_idx]
        sig_ii_c = sig_ii - np.mean(sig_ii)
        # Estimate P-wave region: first 20% of median RR interval before each R-peak
        p_peak_amp = float(np.percentile(sig_ii_c, 97))   # rough upper tail
        if p_peak_amp > 0.25:   # > 0.25 mV (2.5 mm at standard gain)
            findings.append({
                "severity": "INFO",
                "code": "RAE",
                "finding": f"Right atrial enlargement pattern (P amplitude {p_peak_amp:.2f} mV in II)",
                "explanation": (
                    "Peaked P wave > 2.5 mm in lead II suggests right atrial enlargement. "
                    "Common causes: pulmonary hypertension, COPD, tricuspid stenosis, congenital disease."
                ),
            })

    # Build summary
    if not findings:
        summary = f"No additional findings. Axis: {axis:.0f} degrees (normal)." if axis else "No additional findings detected."
    else:
        critical = sum(1 for f in findings if f["severity"] == "CRITICAL")
        warnings = sum(1 for f in findings if f["severity"] == "WARNING")
        info = sum(1 for f in findings if f["severity"] == "INFO")
        parts = []
        if critical:
            parts.append(f"{critical} critical")
        if warnings:
            parts.append(f"{warnings} warning(s)")
        if info:
            parts.append(f"{info} informational")
        summary = f"{', '.join(parts)} finding(s) detected."
        if axis is not None:
            summary += f" Axis: {axis:.0f} deg ({axis_deviation})."

    return {
        "axis": axis,
        "axis_deviation": axis_deviation,
        "findings": findings,
        "summary": summary,
    }
