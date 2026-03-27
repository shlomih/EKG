"""
interval_calculator.py
======================
Real clinical interval measurement using NeuroKit2.
Computes HR, PR, QRS, QTc from a raw ECG signal and applies
patient-context logic to flag or suppress findings.

Usage (standalone test):
    python interval_calculator.py

Usage (from app.py):
    from interval_calculator import calculate_intervals, apply_clinical_context
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=NeuroKitWarning if False else Warning)

# ─────────────────────────────────────────────────────────────
# Clinical Reference Ranges
# These are standard cardiology reference values.
# All intervals in milliseconds unless noted.
# ─────────────────────────────────────────────────────────────

REFERENCE = {
    "hr":  {"normal": (60, 100),   "low": 60,   "high": 100,  "unit": "bpm"},
    "pr":  {"normal": (120, 200),  "low": 120,  "high": 200,  "unit": "ms"},
    "qrs": {"normal": (70, 110),   "low": 70,   "high": 110,  "unit": "ms"},
    "qtc": {"normal": (350, 450),  "low": 350,  "high": 450,  "unit": "ms",
            "critical_high": 500,  # Risk of Torsades de Pointes
            "borderline_high": 470},
    "rr_cv": {"normal_max": 0.15,  "unit": "coefficient of variation"},
}

# QTc thresholds differ slightly by sex (AHA guidelines)
QTC_HIGH_MALE   = 450   # ms
QTC_HIGH_FEMALE = 460   # ms
QTC_CRITICAL    = 500   # ms — Torsades risk regardless of sex


# ─────────────────────────────────────────────────────────────
# Core Measurement Function
# ─────────────────────────────────────────────────────────────

def calculate_intervals(signal: np.ndarray, sampling_rate: int = 500) -> dict:
    """
    Compute clinical ECG intervals from a 1D Lead II signal.

    Args:
        signal:        1D numpy array of voltage values (mV)
        sampling_rate: Hz — PTB-XL default is 500Hz

    Returns:
        dict with keys: hr, pr, qrs, qtc, rr_intervals,
                        r_peaks, p_peaks, quality_score, raw_delineation
        All intervals in milliseconds. Returns error dict on failure.
    """
    try:
        import neurokit2 as nk
    except ImportError:
        return {"error": "neurokit2 not installed. Run: pip install neurokit2"}

    results = {
        "hr": None, "hr_variability": None,
        "pr": None, "qrs": None, "qtc": None,
        "rr_intervals": [],
        "r_peaks": [], "p_peaks": [],
        "quality_score": None,
        "warnings": [],
        "error": None,
    }

    min_samples = sampling_rate * 3  # NeuroKit2 needs at least ~3 seconds
    if len(signal) < min_samples:
        duration = len(signal) / sampling_rate
        results["error"] = (
            f"Signal too short ({duration:.1f}s, {len(signal)} samples). "
            f"Need at least 3s at {sampling_rate}Hz. "
            "For scanned images, ensure the full EKG strip is visible and well-lit."
        )
        return results

    def _safe_to_int_indices(arr):
        a = np.array(arr, dtype=float)
        a = a[np.isfinite(a)]
        return a.astype(int)

    try:
        # ── Step 1: Clean the signal ──────────────────────────
        cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)

        # ── Step 2: Signal quality check ──────────────────────
        quality = nk.ecg_quality(cleaned, sampling_rate=sampling_rate)
        results["quality_score"] = round(float(np.mean(quality)), 3)

        if results["quality_score"] < 0.3:
            results["warnings"].append(
                f"Low signal quality ({results['quality_score']:.2f}). "
                "Results may be unreliable — check lead placement."
            )

        # ── Step 3: R-peak detection ───────────────────────────
        _, r_info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
        r_peaks = r_info["ECG_R_Peaks"]

        if len(r_peaks) < 2:
            results["error"] = "Insufficient R-peaks detected (< 2). Signal too short or too noisy."
            return results

        results["r_peaks"] = r_peaks.tolist()

        # ── Step 4: Heart Rate ─────────────────────────────────
        rr_samples = np.diff(r_peaks)
        rr_ms = (rr_samples / sampling_rate) * 1000
        results["rr_intervals"] = rr_ms.tolist()

        hr_values = 60000 / rr_ms
        results["hr"] = round(float(np.mean(hr_values)), 1)

        # RR variability (coefficient of variation — proxy for rhythm regularity)
        results["hr_variability"] = round(float(np.std(rr_ms) / np.mean(rr_ms)), 3)

        # ── Step 5: Full waveform delineation ─────────────────
        # This gives us P, Q, R, S, T peak locations
        try:
            _, waves = nk.ecg_delineate(
                cleaned,
                r_info,
                sampling_rate=sampling_rate,
                method="dwt",   # Discrete Wavelet Transform — most reliable
            )
            results["raw_delineation"] = waves

            # ── PR Interval ────────────────────────────────────
            # PR = P onset to R peak
            p_onsets  = _safe_to_int_indices(waves.get("ECG_P_Onsets", []))
            results["p_peaks"] = p_onsets.tolist()

            if len(p_onsets) >= 2 and len(r_peaks) >= 2:
                pr_intervals = []
                for r in r_peaks:
                    # Find the nearest P onset before this R peak
                    candidates = p_onsets[p_onsets < r]
                    if len(candidates) > 0:
                        p = candidates[-1]
                        pr_ms = ((r - p) / sampling_rate) * 1000
                        if 60 < pr_ms < 400:  # sanity bounds
                            pr_intervals.append(pr_ms)
                if pr_intervals:
                    results["pr"] = round(float(np.median(pr_intervals)), 1)

            # ── QRS Duration ───────────────────────────────────
            # QRS = Q onset to S offset
            q_onsets   = _safe_to_int_indices(waves.get("ECG_Q_Peaks", []))
            s_offsets  = _safe_to_int_indices(waves.get("ECG_S_Peaks", []))

            if len(q_onsets) >= 2 and len(s_offsets) >= 2:
                qrs_durations = []
                for q in q_onsets:
                    candidates = s_offsets[s_offsets > q]
                    if len(candidates) > 0:
                        s = candidates[0]
                        qrs_ms = ((s - q) / sampling_rate) * 1000
                        if 40 < qrs_ms < 250:  # sanity bounds
                            qrs_durations.append(qrs_ms)
                if qrs_durations:
                    results["qrs"] = round(float(np.median(qrs_durations)), 1)

            # ── QTc (Bazett's Formula) ─────────────────────────
            # QTc = QT / sqrt(RR in seconds)
            # Using T-wave offsets for QT measurement
            t_offsets = _safe_to_int_indices(waves.get("ECG_T_Offsets", []))

            if len(t_offsets) >= 2:
                qtc_values = []
                for i, r in enumerate(r_peaks[:-1]):
                    # Find T offset after this R
                    t_candidates = t_offsets[t_offsets > r]
                    if len(t_candidates) > 0:
                        t_off = t_candidates[0]
                        qt_ms = ((t_off - r) / sampling_rate) * 1000
                        rr_s  = rr_ms[i] / 1000  # RR in seconds for Bazett

                        if 200 < qt_ms < 700 and rr_s > 0:  # sanity bounds
                            qtc = qt_ms / np.sqrt(rr_s)
                            if 250 < qtc < 700:  # post-correction sanity
                                qtc_values.append(qtc)

                if qtc_values:
                    results["qtc"] = round(float(np.median(qtc_values)), 1)

        except Exception as delineation_error:
            # Delineation can fail on noisy signals — HR is still valid
            results["warnings"].append(
                f"Waveform delineation incomplete: {str(delineation_error)[:80]}. "
                "HR is reliable; PR/QRS/QTc may be unavailable."
            )

    except Exception as e:
        results["error"] = f"Analysis failed: {str(e)}"

    return results


# ─────────────────────────────────────────────────────────────
# Multi-Lead Interval Analysis (Phase 1 Improvement)
# Extract and analyze intervals from all 12 leads separately
# ─────────────────────────────────────────────────────────────

def calculate_intervals_all_leads(signal_12: np.ndarray, lead_names: list, 
                                   sampling_rate: int = 500) -> dict:
    """
    Calculate clinical intervals (HR, PR, QRS, QTc) for each lead separately,
    then compute consensus and dispersion metrics.

    Args:
        signal_12:      (N, 12) numpy array of 12-lead signal
        lead_names:     List of 12 lead name strings (e.g., ["I", "II", "III", ...])
        sampling_rate:  Hz (default 500)

    Returns:
        dict with keys:
            - per_lead: { "II": {...}, "V1": {...}, } — intervals per lead
            - consensus: { "hr": float, "pr": float, ... } — median across leads
            - dispersion: { "qrs_std": float, ... } — std dev (repolarization variability)
            - quality_per_lead: { "II": 0.57, ... } — signal quality per lead
            - warnings: [] — list of anomalies detected
    """
    if signal_12.shape[1] != len(lead_names):
        return {"error": f"Signal shape {signal_12.shape[1]} doesn't match lead_names length {len(lead_names)}"}

    per_lead_results = {}
    warnings = []
    
    # Extract intervals from each lead
    for i, lead_name in enumerate(lead_names):
        try:
            lead_signal = signal_12[:, i]
            intervals = calculate_intervals(lead_signal, sampling_rate)
            per_lead_results[lead_name] = intervals
        except Exception as e:
            warnings.append(f"Lead {lead_name} analysis failed: {str(e)[:50]}")
            per_lead_results[lead_name] = {"error": str(e)[:80]}

    # Compute consensus metrics (median across leads)
    consensus = _compute_consensus_metrics(per_lead_results)
    
    # Compute dispersion (variability across leads) — marker of arrhythmia/repolarization heterogeneity
    dispersion = _compute_dispersion_metrics(per_lead_results)
    
    # Extract quality scores per lead
    quality_per_lead = {
        lead: results.get("quality_score", np.nan)
        for lead, results in per_lead_results.items()
    }
    
    # Flag leads with poor quality
    poor_quality = [lead for lead, q in quality_per_lead.items() if q is not None and q < 0.4]
    if poor_quality:
        warnings.append(f"Poor signal quality in leads: {', '.join(poor_quality)}")
    
    # Flag high dispersion (potential arrhythmia pattern)
    if dispersion.get("qrs_std", 0) > 20:
        warnings.append(f"High QRS dispersion ({dispersion['qrs_std']:.1f} ms) — possible aberrancy pattern")
    
    if dispersion.get("qtc_std", 0) > 40:
        warnings.append(f"High QTc dispersion ({dispersion['qtc_std']:.1f} ms) — repolarization heterogeneity")

    return {
        "per_lead": per_lead_results,
        "consensus": consensus,
        "dispersion": dispersion,
        "quality_per_lead": quality_per_lead,
        "warnings": warnings,
    }


def _compute_consensus_metrics(per_lead_results: dict) -> dict:
    """Extract median intervals across leads (consensus metric)."""
    consensus = {}
    
    for metric in ["hr", "pr", "qrs", "qtc", "hr_variability"]:
        values = [
            r.get(metric)
            for r in per_lead_results.values()
            if isinstance(r, dict) and r.get(metric) is not None and not np.isnan(r.get(metric, np.nan))
        ]
        if values:
            consensus[metric] = round(float(np.median(values)), 1)
        else:
            consensus[metric] = None
    
    return consensus


def _compute_dispersion_metrics(per_lead_results: dict) -> dict:
    """Compute inter-lead variability (repolarization heterogeneity marker)."""
    dispersion = {}
    
    for metric in ["qrs", "qtc", "pr"]:
        values = [
            r.get(metric)
            for r in per_lead_results.values()
            if isinstance(r, dict) and r.get(metric) is not None and not np.isnan(r.get(metric, np.nan))
        ]
        if len(values) >= 2:
            dispersion[f"{metric}_std"] = round(float(np.std(values)), 1)
            dispersion[f"{metric}_min"] = round(float(np.min(values)), 1)
            dispersion[f"{metric}_max"] = round(float(np.max(values)), 1)
        else:
            dispersion[f"{metric}_std"] = None

    return dispersion


def _get_age_adjusted_hr_lower_threshold(age: int, is_athlete: bool = False) -> int:
    """
    Return age- and fitness-appropriate lower heart rate threshold (bradycardia cutoff).
    
    Phase 1 Clinical Improvement: Age-specific thresholds replace hardcoded 60 bpm.
    
    Standard cardiology reference ranges:
    - Infants (0-3m): 100-160 bpm
    - Children (3m-1y): 90-150 bpm
    - Toddlers (1-3y): 80-130 bpm
    - Preschool (3-6y): 70-110 bpm
    - School age (6-12y): 60-100 bpm
    - Teens (12-18y): 55-95 bpm
    - Adults (18-65y): 60-100 bpm (or 50+ for athletes)
    - Elderly (>65y): 50+ acceptable
    
    Args:
        age: Patient age in years
        is_athlete: True if regular athletic training
    
    Returns:
        Heart rate lower bound (bpm) for clinical bradycardia threshold
    """
    if is_athlete:
        return 40  # Trained athletes can have resting HR in 40s
    elif age < 12:
        return 60  # Children tolerate lower HR better
    elif age > 65:
        return 50  # Elderly: 50 bpm acceptable baseline
    else:
        return 60  # Standard adult threshold


# ─────────────────────────────────────────────────────────────
# Clinical Flag Engine
# Applies patient context to produce final flags.
# This is the "logic inverter" layer from the PRD.
# ─────────────────────────────────────────────────────────────

def apply_clinical_context(intervals: dict, patient: dict) -> dict:
    """
    Takes raw interval measurements and a patient profile dict,
    returns a list of clinical flags with severity and explanation.

    Patient dict keys (all optional, safe defaults applied):
        age (int), sex (str: 'M'/'F'), has_pacemaker (bool),
        is_athlete (bool), is_pregnant (bool), k_level (float)

    Returns:
        {
          "flags": [ { "severity": "CRITICAL|WARNING|INFO|SUPPRESSED",
                       "code": str,
                       "finding": str,
                       "explanation": str } ],
          "urgency": "EMERGENCY|URGENT|ROUTINE|NORMAL",
          "suppressed": [ ... ]   # findings that were overridden by context
        }
    """
    flags = []
    suppressed = []

    # Safe defaults
    age          = patient.get("age", 50)
    sex          = patient.get("sex", "M").upper()
    pacemaker    = patient.get("has_pacemaker", False)
    athlete      = patient.get("is_athlete", False)
    pregnant     = patient.get("is_pregnant", False)
    k_level      = patient.get("k_level", 4.0)

    hr   = intervals.get("hr")
    pr   = intervals.get("pr")
    qrs  = intervals.get("qrs")
    qtc  = intervals.get("qtc")
    rr_cv = intervals.get("hr_variability")
    quality = intervals.get("quality_score", 1.0)

    # ── Age-aware thresholds (Phase 1 Improvement) ──────────────
    hr_lower_threshold = _get_age_adjusted_hr_lower_threshold(age, athlete)

    # ── Signal quality warning ─────────────────────────────────
    if quality is not None and quality < 0.5:
        flags.append({
            "severity": "WARNING",
            "code": "LOW_QUALITY",
            "finding": f"Signal quality low ({quality:.2f}/1.0)",
            "explanation": "Measurements may be inaccurate. Repeat acquisition or check lead contact."
        })

    # ── Heart Rate ─────────────────────────────────────────────
    if hr is not None:

        if hr < 40:
            if pacemaker:
                suppressed.append({
                    "code": "SEVERE_BRADYCARDIA",
                    "finding": f"Severe bradycardia ({hr:.0f} bpm)",
                    "suppressed_reason": "Pacemaker present — paced rhythm expected. Alert suppressed."
                })
            else:
                flags.append({
                    "severity": "CRITICAL",
                    "code": "SEVERE_BRADYCARDIA",
                    "finding": f"Severe bradycardia — {hr:.0f} bpm",
                    "explanation": "HR < 40 bpm. Risk of haemodynamic compromise. Urgent assessment required."
                })

        elif hr < hr_lower_threshold:
            if pacemaker:
                suppressed.append({
                    "code": "BRADYCARDIA",
                    "finding": f"Bradycardia ({hr:.0f} bpm)",
                    "suppressed_reason": "Pacemaker present. Suppressed."
                })
            elif athlete:
                suppressed.append({
                    "code": "BRADYCARDIA",
                    "finding": f"Bradycardia ({hr:.0f} bpm)",
                    "suppressed_reason": "Athlete status — resting bradycardia is physiologically normal in trained athletes."
                })
            else:
                flags.append({
                    "severity": "WARNING",
                    "code": "BRADYCARDIA",
                    "finding": f"Bradycardia — {hr:.0f} bpm",
                    "explanation": f"HR < {hr_lower_threshold} bpm (age {age} threshold). May be physiologic or indicate conduction disease."
                })

        elif hr > 150:
            flags.append({
                "severity": "CRITICAL",
                "code": "TACHYCARDIA_SEVERE",
                "finding": f"Severe tachycardia — {hr:.0f} bpm",
                "explanation": "HR > 150 bpm. Consider SVT, VT, or haemodynamic instability."
            })

        elif hr > 100:
            flags.append({
                "severity": "WARNING",
                "code": "TACHYCARDIA",
                "finding": f"Tachycardia — {hr:.0f} bpm",
                "explanation": "HR > 100 bpm. May indicate pain, fever, dehydration, or arrhythmia."
            })

    # ── PR Interval ────────────────────────────────────────────
    if pr is not None:

        if pr > 200:
            if pacemaker:
                suppressed.append({
                    "code": "FIRST_DEGREE_BLOCK",
                    "finding": f"Prolonged PR ({pr:.0f} ms)",
                    "suppressed_reason": "Pacemaker present — PR measurement unreliable in paced rhythm."
                })
            else:
                severity = "WARNING" if pr < 300 else "CRITICAL"
                flags.append({
                    "severity": severity,
                    "code": "FIRST_DEGREE_BLOCK",
                    "finding": f"Prolonged PR interval — {pr:.0f} ms",
                    "explanation": (
                        "PR > 200 ms indicates first-degree AV block. "
                        f"{'Marked prolongation (>300ms) — consider higher-degree block.' if pr >= 300 else 'Monitor for progression.'}"
                    )
                })

        elif pr < 120:
            if qrs is not None and qrs >= 110 and not pacemaker:
                # Short PR + borderline/wide QRS → specific WPW pattern
                flags.append({
                    "severity": "WARNING",
                    "code": "WPW_SCREEN",
                    "finding": f"WPW pattern — short PR ({pr:.0f} ms) + wide QRS ({qrs:.0f} ms)",
                    "explanation": (
                        "Short PR with wide or borderline-wide QRS strongly suggests "
                        "Wolff-Parkinson-White syndrome or other pre-excitation. "
                        "Delta wave may be visible in V1-V3. Risk of rapid conduction during AF. "
                        "Electrophysiology referral recommended."
                    ),
                })
            else:
                flags.append({
                    "severity": "WARNING",
                    "code": "SHORT_PR",
                    "finding": f"Short PR interval — {pr:.0f} ms",
                    "explanation": "PR < 120 ms. Consider pre-excitation (WPW syndrome) or AV nodal bypass tract."
                })

    # ── QRS Duration ───────────────────────────────────────────
    if qrs is not None:

        if qrs > 120:
            if pacemaker:
                suppressed.append({
                    "code": "WIDE_QRS",
                    "finding": f"Wide QRS ({qrs:.0f} ms)",
                    "suppressed_reason": "Pacemaker present — wide QRS is expected in paced rhythm."
                })
            else:
                flags.append({
                    "severity": "WARNING",
                    "code": "WIDE_QRS",
                    "finding": f"Wide QRS complex — {qrs:.0f} ms",
                    "explanation": (
                        "QRS > 120 ms. Possible bundle branch block (LBBB/RBBB) or "
                        "ventricular conduction delay. Compare to prior EKG if available."
                    )
                })
        elif qrs > 110:
            flags.append({
                "severity": "INFO",
                "code": "BORDERLINE_QRS",
                "finding": f"Borderline QRS width — {qrs:.0f} ms",
                "explanation": "QRS 110–120 ms — incomplete bundle branch block pattern. Monitor."
            })

    # ── QTc ────────────────────────────────────────────────────
    if qtc is not None:
        qtc_threshold = QTC_HIGH_FEMALE if sex == "F" else QTC_HIGH_MALE
        if pregnant:
            qtc_threshold = 460  # Pregnancy shifts threshold

        if qtc >= QTC_CRITICAL:
            flags.append({
                "severity": "CRITICAL",
                "code": "QTC_CRITICAL",
                "finding": f"Critically prolonged QTc — {qtc:.0f} ms",
                "explanation": (
                    f"QTc ≥ {QTC_CRITICAL} ms. HIGH RISK of Torsades de Pointes. "
                    "Urgently review QT-prolonging medications. "
                    f"{'Potassium is low — hypokalaemia worsens QTc risk.' if k_level < 3.5 else ''}"
                )
            })
        elif qtc > qtc_threshold:
            flags.append({
                "severity": "WARNING",
                "code": "QTC_PROLONGED",
                "finding": f"Prolonged QTc — {qtc:.0f} ms",
                "explanation": (
                    f"QTc > {qtc_threshold} ms ({'female' if sex == 'F' else 'male'} threshold). "
                    "Review QT-prolonging medications. "
                    f"{'Pregnancy noted — monitor closely.' if pregnant else ''}"
                    f"{'Potassium at {k_level} — consider replacement.' if k_level < 3.5 else ''}"
                )
            })
        elif qtc < 350:
            flags.append({
                "severity": "WARNING",
                "code": "QTC_SHORT",
                "finding": f"Short QTc — {qtc:.0f} ms",
                "explanation": "QTc < 350 ms. Short QT syndrome or hypercalcaemia. Check serum calcium."
            })

    # ── Electrolyte Context (K+) ───────────────────────────────
    if k_level < 3.0:
        flags.append({
            "severity": "WARNING",
            "code": "SEVERE_HYPOKALAEMIA",
            "finding": f"Severe hypokalaemia — K⁺ {k_level} mmol/L",
            "explanation": "K⁺ < 3.0 mmol/L markedly increases arrhythmia risk and prolongs QTc. Urgent replacement indicated."
        })
    elif k_level < 3.5:
        flags.append({
            "severity": "INFO",
            "code": "HYPOKALAEMIA",
            "finding": f"Mild hypokalaemia — K⁺ {k_level} mmol/L",
            "explanation": "K⁺ 3.0–3.5 mmol/L. Monitor QTc closely. Consider oral replacement."
        })
    elif k_level > 6.0:
        flags.append({
            "severity": "CRITICAL",
            "code": "HYPERKALAEMIA_SEVERE",
            "finding": f"Severe hyperkalaemia — K⁺ {k_level} mmol/L",
            "explanation": "K⁺ > 6.0 mmol/L. Risk of fatal arrhythmia. Peaked T-waves and wide QRS are EKG hallmarks."
        })
    elif k_level > 5.5:
        flags.append({
            "severity": "WARNING",
            "code": "HYPERKALAEMIA",
            "finding": f"Hyperkalaemia — K⁺ {k_level} mmol/L",
            "explanation": "K⁺ > 5.5 mmol/L. Look for peaked T-waves. Restrict potassium intake."
        })

    # ── RR Regularity (crude AF screen) ───────────────────────
    if rr_cv is not None and rr_cv > 0.15:
        if not pacemaker:
            flags.append({
                "severity": "WARNING",
                "code": "IRREGULAR_RHYTHM",
                "finding": f"Irregular rhythm detected (RR variability: {rr_cv:.2f})",
                "explanation": (
                    "High beat-to-beat variability may indicate atrial fibrillation, "
                    "frequent ectopics, or second-degree AV block. "
                    "Full 12-lead analysis recommended."
                )
            })

    # ── Urgency Score ──────────────────────────────────────────
    severities = [f["severity"] for f in flags]
    if "CRITICAL" in severities:
        urgency = "EMERGENCY"
    elif "WARNING" in severities:
        urgency = "URGENT"
    elif "INFO" in severities:
        urgency = "ROUTINE"
    else:
        urgency = "NORMAL"

    # If all findings were suppressed by context, downgrade urgency
    if not flags and suppressed:
        urgency = "ROUTINE"

    return {
        "flags": flags,
        "urgency": urgency,
        "suppressed": suppressed,
    }


# ─────────────────────────────────────────────────────────────
# Formatting helpers (for app.py display)
# ─────────────────────────────────────────────────────────────

def format_interval(value, unit="ms", low=None, high=None):
    """Returns (display_string, status) for a Streamlit metric delta."""
    if value is None:
        return "N/A", None
    label = f"{value:.0f} {unit}"
    if low and high:
        if value < low or value > high:
            return label, "abnormal"
        return label, "normal"
    return label, None


URGENCY_CONFIG = {
    "EMERGENCY": {"color": "#FF4444", "emoji": "🔴", "label": "EMERGENCY"},
    "URGENT":    {"color": "#FF8C00", "emoji": "🟠", "label": "URGENT"},
    "ROUTINE":   {"color": "#00C49F", "emoji": "🟢", "label": "ROUTINE"},
    "NORMAL":    {"color": "#00C49F", "emoji": "✅", "label": "NORMAL"},
}

SEVERITY_EMOJI = {
    "CRITICAL":   "🔴",
    "WARNING":    "🟠",
    "INFO":       "🔵",
    "SUPPRESSED": "⚪",
}


# ─────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  Interval Calculator — Self Test")
    print("═" * 60)

    # Try to load a real PTB-XL record
    from pathlib import Path
    import wfdb

    ptbxl_path = Path("./ekg_datasets/ptbxl")
    test_record = None

    if ptbxl_path.exists():
        dat_files = list(ptbxl_path.rglob("*.dat"))
        if dat_files:
            record_path = str(dat_files[0]).replace(".dat", "")
            record = wfdb.rdrecord(record_path)
            lead_ii_idx = record.sig_name.index("II") if "II" in record.sig_name else 1
            test_signal = record.p_signal[:, lead_ii_idx]
            fs = record.fs
            print(f"\n  Using real record: {dat_files[0].name}")
            print(f"  Duration: {len(test_signal)/fs:.1f}s  |  Rate: {fs}Hz")
        else:
            test_signal = None
    else:
        test_signal = None

    # Fall back to synthetic signal if no data
    if test_signal is None:
        print("\n  No PTB-XL data found — generating synthetic signal for test...")
        fs = 500
        t = np.linspace(0, 10, 10 * fs)
        # Synthetic ECG-like signal: sum of gaussians mimicking PQRST
        test_signal = np.zeros_like(t)
        for beat_t in np.arange(0.5, 10, 0.857):  # ~70 bpm
            test_signal += 0.1 * np.exp(-((t - beat_t + 0.1)**2) / (2 * 0.003**2))   # P
            test_signal += 1.0 * np.exp(-((t - beat_t)**2)        / (2 * 0.002**2))   # R
            test_signal -= 0.2 * np.exp(-((t - beat_t + 0.04)**2) / (2 * 0.003**2))  # S
            test_signal += 0.3 * np.exp(-((t - beat_t - 0.15)**2) / (2 * 0.02**2))   # T
        test_signal += np.random.normal(0, 0.02, len(t))

    # Run measurement
    print("\n  Running interval calculation...")
    intervals = calculate_intervals(test_signal, sampling_rate=fs)

    if intervals.get("error"):
        print(f"\n  ✗ Error: {intervals['error']}")
    else:
        print(f"\n  Results:")
        print(f"  Heart Rate    : {intervals['hr']} bpm")
        print(f"  PR Interval   : {intervals['pr']} ms")
        print(f"  QRS Duration  : {intervals['qrs']} ms")
        print(f"  QTc (Bazett)  : {intervals['qtc']} ms")
        print(f"  RR Variability: {intervals['hr_variability']}")
        print(f"  Quality Score : {intervals['quality_score']}")
        if intervals["warnings"]:
            print(f"\n  Warnings:")
            for w in intervals["warnings"]:
                print(f"    ⚠ {w}")

    # Test clinical context
    print("\n  Testing clinical context (pacemaker patient, low K+)...")
    patient = {
        "age": 72, "sex": "M",
        "has_pacemaker": True,
        "is_athlete": False,
        "is_pregnant": False,
        "k_level": 3.1,
    }
    context = apply_clinical_context(intervals, patient)
    print(f"\n  Urgency: {context['urgency']}")
    print(f"\n  Flags ({len(context['flags'])}):")
    for f in context["flags"]:
        print(f"    {SEVERITY_EMOJI.get(f['severity'], '?')} [{f['severity']}] {f['finding']}")
        print(f"      → {f['explanation'][:90]}")
    print(f"\n  Suppressed by context ({len(context['suppressed'])}):")
    for s in context["suppressed"]:
        print(f"    ⚪ {s['finding']} — {s['suppressed_reason'][:80]}")

    print("\n  ✅ interval_calculator.py ready\n")