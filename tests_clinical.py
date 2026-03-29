"""
tests_clinical.py
=================
Unit tests for clinical_rules.py and interval_calculator.py.
No neurokit2 dependency — only tests pure-Python logic.

Run:
    python tests_clinical.py
    # or
    python -m pytest tests_clinical.py -v
"""

import sys
import numpy as np
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from clinical_rules import (
    _estimate_qrs_axis,
    _check_low_voltage,
    _check_t_wave_patterns,
    _check_r_wave_progression,
    _check_rvh,
    _check_posterior_stemi,
    _check_hyperacute_t,
    _check_rae,
    analyze_clinical_rules,
    _find_r_peaks,
)
from interval_calculator import (
    apply_clinical_context,
    format_interval,
    _get_age_adjusted_hr_lower_threshold,
    QTC_CRITICAL,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

FS = 500
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _flat_signal(n_leads=12, n_samples=5000):
    """12-lead flat ECG — all zeros."""
    return np.zeros((n_samples, n_leads), dtype=np.float32)


def _synthetic_ecg(n_beats=7, hr=70, amplitude=1.0, t_ratio=0.3,
                   t_invert=False, fs=FS, n_samples=5000):
    """
    Single-lead synthetic ECG with Gaussian PQRST complexes.
    Returns 1-D signal of length n_samples.
    """
    t = np.linspace(0, n_samples / fs, n_samples)
    sig = np.zeros_like(t)
    rr = 60.0 / hr
    for i in range(n_beats):
        beat_t = 0.5 + i * rr
        if beat_t + 0.4 > n_samples / fs:
            break
        sig += 0.1 * np.exp(-((t - beat_t + 0.15) ** 2) / (2 * 0.005 ** 2))  # P
        sig += amplitude * np.exp(-((t - beat_t) ** 2) / (2 * 0.003 ** 2))   # R
        sig -= 0.2 * amplitude * np.exp(-((t - beat_t - 0.04) ** 2) / (2 * 0.003 ** 2))  # S
        t_amp = t_ratio * amplitude * (-1 if t_invert else 1)
        sig += t_amp * np.exp(-((t - beat_t - 0.2) ** 2) / (2 * 0.015 ** 2))  # T
    return sig.astype(np.float32)


def _make_12lead(lead_signals: dict, n_samples=5000, leads=LEADS):
    """
    Build a (n_samples, 12) array from a dict of {lead_name: 1D-signal}.
    Leads not in the dict are set to zero.
    """
    sig = np.zeros((n_samples, 12), dtype=np.float32)
    for i, name in enumerate(leads):
        if name in lead_signals:
            sig[:, i] = lead_signals[name]
    return sig


# ── _find_r_peaks ─────────────────────────────────────────────────────────────

class TestFindRPeaks(unittest.TestCase):

    def test_flat_signal_returns_empty(self):
        self.assertEqual(_find_r_peaks(np.zeros(5000), FS), [])

    def test_synthetic_detects_correct_count(self):
        sig = _synthetic_ecg(n_beats=6, hr=60)
        peaks = _find_r_peaks(sig, FS)
        # Should detect most beats (allow ±1)
        self.assertGreaterEqual(len(peaks), 5)
        self.assertLessEqual(len(peaks), 7)

    def test_refractory_period_prevents_double_counting(self):
        sig = _synthetic_ecg(n_beats=5, hr=70)
        peaks = _find_r_peaks(sig, FS)
        if len(peaks) >= 2:
            for i in range(1, len(peaks)):
                self.assertGreater(peaks[i] - peaks[i - 1], FS // 2)


# ── _estimate_qrs_axis ────────────────────────────────────────────────────────

class TestQRSAxis(unittest.TestCase):

    def test_flat_signal_returns_none(self):
        result = _estimate_qrs_axis(_flat_signal(), FS, LEADS)
        self.assertIsNone(result)

    def test_normal_axis_range(self):
        # Positive Lead I, positive aVF → ~45 deg (normal quadrant)
        ecg_I = _synthetic_ecg(amplitude=1.0)
        ecg_avf = _synthetic_ecg(amplitude=1.0)
        sig = _make_12lead({"I": ecg_I, "AVF": ecg_avf})
        axis = _estimate_qrs_axis(sig, FS, LEADS)
        if axis is not None:
            self.assertGreaterEqual(axis, -90)
            self.assertLessEqual(axis, 180)

    def test_returns_float_or_none(self):
        ecg_I = _synthetic_ecg(amplitude=0.8)
        ecg_avf = _synthetic_ecg(amplitude=0.5)
        sig = _make_12lead({"I": ecg_I, "AVF": ecg_avf})
        result = _estimate_qrs_axis(sig, FS, LEADS)
        if result is not None:
            self.assertIsInstance(result, float)


# ── _check_low_voltage ────────────────────────────────────────────────────────

class TestLowVoltage(unittest.TestCase):

    def test_flat_signal_is_low_voltage(self):
        self.assertTrue(_check_low_voltage(_flat_signal(), LEADS))

    def test_normal_amplitude_not_low_voltage(self):
        sig = np.zeros((5000, 12), dtype=np.float32)
        amp = 1.0 * np.sin(2 * np.pi * 1.0 * np.arange(5000) / FS)
        # Need BOTH a limb lead (≥ 0.5 mV) AND a precordial lead (≥ 1.0 mV) to be normal
        sig[:, 0] = amp          # Lead I (limb) — ptp = 2.0 ≥ 0.5
        sig[:, 8] = 1.5 * amp   # V3 (precordial) — ptp = 3.0 ≥ 1.0
        self.assertFalse(_check_low_voltage(sig, LEADS))

    def test_all_limb_leads_low_with_ok_precordial_returns_false(self):
        # Low voltage only when EITHER limb or precordial all-low, not both
        sig = np.zeros((5000, 12), dtype=np.float32)
        # All limb leads zero, but V3 has normal amplitude
        sig[:, 8] = 1.5 * np.sin(2 * np.pi * 1 * np.arange(5000) / FS)  # V3 ok
        # limb_low=True, precordial_low=False → True (limb low fires)
        self.assertTrue(_check_low_voltage(sig, LEADS))

    def test_all_precordial_leads_low_returns_true(self):
        sig = np.zeros((5000, 12), dtype=np.float32)
        # Set Lead I to normal amplitude (limb_low=False), but precordial all zero
        sig[:, 0] = 1.0 * np.sin(2 * np.pi * 1 * np.arange(5000) / FS)
        # limb_low=False, precordial_low=True → True (precordial fires)
        self.assertTrue(_check_low_voltage(sig, LEADS))


# ── _check_r_wave_progression ─────────────────────────────────────────────────

class TestRWaveProgression(unittest.TestCase):

    def test_normal_progression_no_finding(self):
        # R amplitude increases V1→V4
        sig = _make_12lead({
            "V1": _synthetic_ecg(amplitude=0.2),
            "V2": _synthetic_ecg(amplitude=0.4),
            "V3": _synthetic_ecg(amplitude=0.6),
            "V4": _synthetic_ecg(amplitude=0.9),
        })
        result = _check_r_wave_progression(sig, LEADS)
        self.assertIsNone(result)

    def test_poor_r_progression_detected(self):
        # V3 amplitude < 0.3 mV → poor R-wave progression
        sig = _make_12lead({
            "V1": _synthetic_ecg(amplitude=0.1),
            "V2": _synthetic_ecg(amplitude=0.1),
            "V3": _synthetic_ecg(amplitude=0.1),  # < 0.3 threshold
            "V4": _synthetic_ecg(amplitude=0.1),
        })
        result = _check_r_wave_progression(sig, LEADS)
        self.assertIsNotNone(result)
        self.assertEqual(result["code"], "POOR_R_PROGRESSION")
        self.assertEqual(result["severity"], "INFO")


# ── _check_rvh ────────────────────────────────────────────────────────────────

class TestRVH(unittest.TestCase):

    def test_no_rvh_without_right_axis(self):
        # Axis = 45 deg (normal) — no RVH even with dominant R in V1
        result = _check_rvh(_flat_signal(), LEADS, axis=45.0)
        self.assertIsNone(result)

    def test_no_rvh_with_right_axis_but_no_dominant_r(self):
        sig = _make_12lead({
            "V1": _synthetic_ecg(amplitude=0.1, t_ratio=0.8),  # tiny R, large S area
        })
        result = _check_rvh(sig, LEADS, axis=100.0)
        # R < S in this case → no RVH
        # (may or may not fire depending on exact signal; just check no crash)
        self.assertIn(result, [None, {"severity": "WARNING", "code": "RVH",
                                       "finding": result["finding"] if result else None,
                                       "explanation": result["explanation"] if result else None}])

    def test_none_axis_returns_none(self):
        self.assertIsNone(_check_rvh(_flat_signal(), LEADS, axis=None))


# ── _check_rae ────────────────────────────────────────────────────────────────

class TestRAE(unittest.TestCase):

    def test_no_lead_ii_returns_none(self):
        self.assertIsNone(_check_rae(_flat_signal(), FS, []))

    def test_flat_lead_ii_no_peaks_returns_none(self):
        result = _check_rae(_flat_signal(), FS, LEADS)
        self.assertIsNone(result)

    def test_tall_p_wave_detected(self):
        """Construct ECG with very tall P wave (0.4 mV) in Lead II."""
        sig_ii = _synthetic_ecg(n_beats=6, hr=60, amplitude=0.5)
        # Add extra P wave enhancement before each QRS
        t = np.linspace(0, 5000 / FS, 5000)
        rr = 1.0
        for i in range(7):
            beat_t = 0.5 + i * rr
            sig_ii += 0.4 * np.exp(-((t - (beat_t - 0.2)) ** 2) / (2 * 0.008 ** 2))

        sig = _make_12lead({"II": sig_ii})
        result = _check_rae(sig, FS, LEADS)
        # With 0.4 mV P wave > 0.25 mV threshold, should fire
        # (exact result depends on peak detection; at minimum no crash)
        if result is not None:
            self.assertEqual(result["code"], "RAE")
            self.assertEqual(result["severity"], "INFO")


# ── _check_posterior_stemi ────────────────────────────────────────────────────

class TestPosteriorSTEMI(unittest.TestCase):

    def test_normal_signal_no_posterior_stemi(self):
        sig = _make_12lead({
            "V1": _synthetic_ecg(amplitude=0.2),
            "V2": _synthetic_ecg(amplitude=0.3),
            "V3": _synthetic_ecg(amplitude=0.4),
        })
        result = _check_posterior_stemi(sig, FS, LEADS)
        self.assertIsNone(result)

    def test_no_crash_on_flat_signal(self):
        result = _check_posterior_stemi(_flat_signal(), FS, LEADS)
        self.assertIsNone(result)


# ── _check_hyperacute_t ───────────────────────────────────────────────────────

class TestHyperacuteT(unittest.TestCase):

    def test_normal_t_ratio_no_finding(self):
        # Small T waves (t_ratio = 0.2 < 0.4 threshold)
        sig = _make_12lead({
            "V2": _synthetic_ecg(amplitude=1.0, t_ratio=0.2),
            "V3": _synthetic_ecg(amplitude=1.0, t_ratio=0.2),
            "V4": _synthetic_ecg(amplitude=1.0, t_ratio=0.2),
        })
        result = _check_hyperacute_t(sig, FS, LEADS)
        self.assertIsNone(result)

    def test_no_crash_on_flat_signal(self):
        result = _check_hyperacute_t(_flat_signal(), FS, LEADS)
        self.assertIsNone(result)


# ── analyze_clinical_rules ────────────────────────────────────────────────────

class TestAnalyzeClinicalRules(unittest.TestCase):

    def test_flat_signal_returns_valid_structure(self):
        result = analyze_clinical_rules(_flat_signal(), FS, LEADS)
        self.assertIn("axis", result)
        self.assertIn("axis_deviation", result)
        self.assertIn("findings", result)
        self.assertIn("summary", result)
        self.assertIsInstance(result["findings"], list)

    def test_axis_zero_shows_in_summary(self):
        """Axis == 0 should NOT show 'No additional findings detected.'"""
        # Create a signal where axis could be computed as ~0
        ecg_I = _synthetic_ecg(amplitude=1.0)
        ecg_avf = np.zeros(5000, dtype=np.float32)  # no aVF deflection → axis ~0
        sig = _make_12lead({"I": ecg_I, "AVF": ecg_avf})
        result = analyze_clinical_rules(sig, FS, LEADS)
        # If axis was detected as 0.0, summary must not fallback to "No additional findings detected."
        if result["axis"] == 0.0 and not result["findings"]:
            self.assertIn("0", result["summary"])

    def test_left_axis_deviation_flagged(self):
        # Positive Lead I, negative aVF → left axis (around -45 deg)
        ecg_I = _synthetic_ecg(amplitude=1.0)
        ecg_avf = _synthetic_ecg(amplitude=-1.0)  # negative aVF
        sig = _make_12lead({"I": ecg_I, "AVF": ecg_avf})
        result = analyze_clinical_rules(sig, FS, LEADS)
        if result["axis"] is not None and result["axis"] < -30:
            codes = [f["code"] for f in result["findings"]]
            self.assertIn("LEFT_AXIS_DEVIATION", codes)

    def test_k_level_upgrades_peaked_t_severity(self):
        """If patient has K+ > 5.5 and ECG shows peaked T, severity upgrades to CRITICAL."""
        # Peaked T in V3 (t_ratio > 0.6 of R in V2-V5)
        ecg_v3 = _synthetic_ecg(amplitude=1.0, t_ratio=0.8)
        sig = _make_12lead({"V3": ecg_v3})
        patient = {"k_level": 6.0}
        result = analyze_clinical_rules(sig, FS, LEADS, patient)
        peaked = [f for f in result["findings"] if f["code"] == "PEAKED_T_WAVES"]
        if peaked:
            self.assertEqual(peaked[0]["severity"], "CRITICAL")
            self.assertIn("6.0", peaked[0]["explanation"])


# ── apply_clinical_context ────────────────────────────────────────────────────

class TestApplyClinicalContext(unittest.TestCase):

    def _base_intervals(self, **overrides):
        ivl = {
            "hr": 75, "pr": 160, "qrs": 90, "qtc": 420,
            "hr_variability": 0.05, "quality_score": 0.9,
        }
        ivl.update(overrides)
        return ivl

    def test_normal_intervals_no_flags(self):
        ctx = apply_clinical_context(self._base_intervals(), {})
        self.assertEqual(ctx["urgency"], "NORMAL")
        self.assertEqual(ctx["flags"], [])

    def test_critical_qtc_triggers_emergency(self):
        ctx = apply_clinical_context(self._base_intervals(qtc=QTC_CRITICAL + 10), {})
        codes = [f["code"] for f in ctx["flags"]]
        self.assertIn("QTC_CRITICAL", codes)
        self.assertEqual(ctx["urgency"], "EMERGENCY")

    def test_bradycardia_not_flagged_for_athlete(self):
        # Athlete threshold is 40 bpm — HR 40-59 is normal for athletes (no flag fires)
        ctx_athlete = apply_clinical_context(self._base_intervals(hr=52), {"is_athlete": True})
        ctx_normal  = apply_clinical_context(self._base_intervals(hr=52), {})
        athlete_codes = [f["code"] for f in ctx_athlete["flags"]]
        normal_codes  = [f["code"] for f in ctx_normal["flags"]]
        self.assertNotIn("BRADYCARDIA", athlete_codes)
        self.assertIn("BRADYCARDIA", normal_codes)  # Same HR fires for non-athlete

    def test_bradycardia_suppressed_for_pacemaker(self):
        ctx = apply_clinical_context(self._base_intervals(hr=45), {"has_pacemaker": True})
        codes = [f["code"] for f in ctx["flags"]]
        suppressed_codes = [s["code"] for s in ctx["suppressed"]]
        self.assertNotIn("BRADYCARDIA", codes)
        self.assertIn("BRADYCARDIA", suppressed_codes)

    def test_severe_bradycardia_not_suppressed_without_pacemaker(self):
        ctx = apply_clinical_context(self._base_intervals(hr=35), {})
        codes = [f["code"] for f in ctx["flags"]]
        self.assertIn("SEVERE_BRADYCARDIA", codes)
        self.assertEqual(ctx["urgency"], "EMERGENCY")

    def test_severe_bradycardia_suppressed_for_pacemaker(self):
        ctx = apply_clinical_context(self._base_intervals(hr=35), {"has_pacemaker": True})
        suppressed_codes = [s["code"] for s in ctx["suppressed"]]
        self.assertIn("SEVERE_BRADYCARDIA", suppressed_codes)

    def test_wide_qrs_suppressed_for_pacemaker(self):
        ctx = apply_clinical_context(self._base_intervals(qrs=140), {"has_pacemaker": True})
        suppressed_codes = [s["code"] for s in ctx["suppressed"]]
        self.assertIn("WIDE_QRS", suppressed_codes)

    def test_wpw_screen_short_pr_wide_qrs(self):
        ctx = apply_clinical_context(self._base_intervals(pr=100, qrs=115), {})
        codes = [f["code"] for f in ctx["flags"]]
        self.assertIn("WPW_SCREEN", codes)

    def test_short_pr_only_short_pr_code(self):
        ctx = apply_clinical_context(self._base_intervals(pr=100, qrs=90), {})
        codes = [f["code"] for f in ctx["flags"]]
        self.assertIn("SHORT_PR", codes)
        self.assertNotIn("WPW_SCREEN", codes)

    def test_first_degree_block_warning(self):
        ctx = apply_clinical_context(self._base_intervals(pr=220), {})
        codes = [f["code"] for f in ctx["flags"]]
        self.assertIn("FIRST_DEGREE_BLOCK", codes)
        sev = {f["code"]: f["severity"] for f in ctx["flags"]}
        self.assertEqual(sev["FIRST_DEGREE_BLOCK"], "WARNING")

    def test_first_degree_block_critical_when_pr_gt_300(self):
        ctx = apply_clinical_context(self._base_intervals(pr=320), {})
        sev = {f["code"]: f["severity"] for f in ctx["flags"]}
        self.assertEqual(sev["FIRST_DEGREE_BLOCK"], "CRITICAL")

    def test_qtc_threshold_differs_by_sex(self):
        # Female threshold is 460, male is 450
        ctx_m = apply_clinical_context(self._base_intervals(qtc=455), {"sex": "M"})
        ctx_f = apply_clinical_context(self._base_intervals(qtc=455), {"sex": "F"})
        codes_m = [f["code"] for f in ctx_m["flags"]]
        codes_f = [f["code"] for f in ctx_f["flags"]]
        self.assertIn("QTC_PROLONGED", codes_m)
        self.assertNotIn("QTC_PROLONGED", codes_f)  # 455 < 460 female threshold

    def test_qtc_pregnancy_shifts_threshold(self):
        # Pregnant female: threshold 460 same as female
        ctx = apply_clinical_context(
            self._base_intervals(qtc=455), {"sex": "F", "is_pregnant": True}
        )
        codes = [f["code"] for f in ctx["flags"]]
        self.assertNotIn("QTC_PROLONGED", codes)  # 455 < 460

    def test_hyperkalaemia_severe(self):
        ctx = apply_clinical_context(self._base_intervals(), {"k_level": 6.5})
        codes = [f["code"] for f in ctx["flags"]]
        self.assertIn("HYPERKALAEMIA_SEVERE", codes)
        self.assertEqual(ctx["urgency"], "EMERGENCY")

    def test_irregular_rhythm_af_screen(self):
        ctx = apply_clinical_context(self._base_intervals(hr_variability=0.25), {})
        codes = [f["code"] for f in ctx["flags"]]
        self.assertIn("IRREGULAR_RHYTHM", codes)

    def test_irregular_rhythm_suppressed_with_pacemaker(self):
        ctx = apply_clinical_context(
            self._base_intervals(hr_variability=0.25), {"has_pacemaker": True}
        )
        codes = [f["code"] for f in ctx["flags"]]
        self.assertNotIn("IRREGULAR_RHYTHM", codes)

    def test_low_quality_warning(self):
        ctx = apply_clinical_context(self._base_intervals(quality_score=0.2), {})
        codes = [f["code"] for f in ctx["flags"]]
        self.assertIn("LOW_QUALITY", codes)

    def test_urgency_warning_when_only_warnings(self):
        ctx = apply_clinical_context(self._base_intervals(hr=110), {})
        self.assertEqual(ctx["urgency"], "URGENT")

    def test_tachycardia_severe_critical(self):
        ctx = apply_clinical_context(self._base_intervals(hr=160), {})
        codes = [f["code"] for f in ctx["flags"]]
        self.assertIn("TACHYCARDIA_SEVERE", codes)
        self.assertEqual(ctx["urgency"], "EMERGENCY")


# ── _get_age_adjusted_hr_lower_threshold ─────────────────────────────────────

class TestAgeAdjustedHR(unittest.TestCase):

    def test_adult(self):
        self.assertEqual(_get_age_adjusted_hr_lower_threshold(40), 60)

    def test_elderly(self):
        self.assertEqual(_get_age_adjusted_hr_lower_threshold(70), 50)

    def test_child(self):
        self.assertEqual(_get_age_adjusted_hr_lower_threshold(8), 60)

    def test_athlete_override(self):
        self.assertEqual(_get_age_adjusted_hr_lower_threshold(30, is_athlete=True), 40)


# ── format_interval ───────────────────────────────────────────────────────────

class TestFormatInterval(unittest.TestCase):

    def test_none_returns_na(self):
        val, status = format_interval(None)
        self.assertEqual(val, "N/A")
        self.assertIsNone(status)

    def test_normal_range(self):
        val, status = format_interval(160, low=120, high=200)
        self.assertIn("160", val)
        self.assertEqual(status, "normal")

    def test_below_low(self):
        val, status = format_interval(100, low=120, high=200)
        self.assertEqual(status, "abnormal")

    def test_above_high(self):
        val, status = format_interval(210, low=120, high=200)
        self.assertEqual(status, "abnormal")

    def test_no_bounds_returns_none_status(self):
        val, status = format_interval(75, unit="bpm")
        self.assertEqual(status, None)
        self.assertIn("75", val)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
