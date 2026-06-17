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

import logging
import traceback
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=NeuroKitWarning if False else Warning)

logging.basicConfig(
    filename="ekg_app.log",
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    encoding="utf-8",
)

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


def _recover_missed_rpeaks(cleaned, r_peaks, sampling_rate):
    """Recover R-peaks that NeuroKit missed on low-amplitude scanned beats.

    On digitized paper ECGs a beat with a slightly attenuated R-wave is
    sometimes skipped, leaving one RR interval that is ~2-3x the surrounding
    rhythm (the "one RR ~= 2x its neighbours" signature). A single miss
    corrupts HR (averaged over a wrong long RR) and breaks QTc (Bazett divides
    by the wrong sqrt(RR), pushing the value out of physiological bounds).

    Strategy: for every gap noticeably longer than the *local* median RR,
    estimate how many beats are missing, place candidate peaks at evenly
    spaced positions inside the gap, snap each to the nearest local maximum in
    the cleaned signal, and accept it only if its amplitude is comparable to
    the surrounding real R-peaks. Purely additive — never deletes a detected
    peak — so it cannot make a clean trace worse.

    Returns a sorted numpy array of R-peak sample indices.
    """
    r_peaks = np.asarray(r_peaks, dtype=int)
    if len(r_peaks) < 3:
        return r_peaks

    rr = np.diff(r_peaks)
    med_rr = float(np.median(rr))
    if med_rr <= 0:
        return r_peaks

    # Reference R-amplitude from the beats we already trust.
    r_amps = cleaned[r_peaks]
    med_amp = float(np.median(r_amps))
    # Snap window: +/- 12% of a median RR around each expected position.
    snap = max(1, int(0.12 * med_rr))

    recovered = list(r_peaks)
    for i in range(len(r_peaks) - 1):
        left, right = r_peaks[i], r_peaks[i + 1]
        gap = right - left
        n_missing = int(round(gap / med_rr)) - 1
        if n_missing < 1 or gap < 1.5 * med_rr:
            continue
        step = gap / (n_missing + 1)
        for k in range(1, n_missing + 1):
            expected = int(left + step * k)
            lo = max(left + snap, expected - snap)
            hi = min(right - snap, expected + snap)
            if hi <= lo:
                continue
            seg = cleaned[lo:hi]
            if len(seg) == 0:
                continue
            cand = lo + int(np.argmax(seg))
            # Accept only a genuine R-like deflection (>= 40% of median R amp).
            if cleaned[cand] >= 0.4 * med_amp:
                recovered.append(cand)

    return np.array(sorted(set(int(p) for p in recovered)), dtype=int)


# Detector panel for consensus R-peak detection. These NeuroKit methods run
# reliably on short, noisy, scan-reconstructed signals. Excluded on purpose:
# 'gamboa', 'zong', 'manikandan' (observed to raise on these signals).
_RPEAK_METHODS = [
    "neurokit", "pantompkins1985", "hamilton2002",
    "elgendi2010", "kalidas2017", "rodrigues2021",
]


def _score_rpeak_train(train, cleaned, sampling_rate):
    """Physiological-plausibility score for a candidate R-peak train (0-1).

    Combines three cues that separate a real beat train from a regular train of
    noise spikes:
      - regularity      : low RR coefficient of variation
      - coverage        : peaks span the signal with no oversized gap
      - amplitude consistency : real R-peaks have similar heights in `cleaned`
    Returns None if the train is implausible (too few peaks, HR out of range).
    """
    train = np.asarray(train, dtype=int)
    if len(train) < 3:
        return None
    rr = np.diff(train).astype(float)
    med_rr = float(np.median(rr))
    if med_rr <= 0:
        return None
    hr = 60000.0 / (med_rr / sampling_rate * 1000.0)
    if not (30.0 <= hr <= 230.0):
        return None

    cv = float(np.std(rr) / max(np.mean(rr), 1e-9))
    regularity = 1.0 - min(cv, 1.0)

    # Coverage: fraction of the signal spanned, penalized if any RR gap is much
    # larger than the median (a sparse train that skips beats). The gap penalty
    # is what rejects a sparse-but-regular noise train.
    span = (train[-1] - train[0]) / max(len(cleaned) - 1, 1)
    max_gap_ratio = float(np.max(rr) / med_rr)
    gap_penalty = 1.0 if max_gap_ratio <= 1.8 else max(0.0, 1.0 - (max_gap_ratio - 1.8))
    coverage = min(span, 1.0) * gap_penalty

    amps = cleaned[train].astype(float)
    amp_med = float(np.median(np.abs(amps))) + 1e-9
    amp_cv = float(np.std(amps) / amp_med)
    amp_consistency = 1.0 - min(amp_cv, 1.0)

    return 0.45 * regularity + 0.35 * coverage + 0.20 * amp_consistency


def _filter_by_peak_width(peaks, signal, sampling_rate):
    """Post-filter: remove peaks whose half-max width is outside QRS physiology.

    True QRS R-peaks have a narrow, sharp spike (half-max width ~15–100 ms).
    Broad deflections like T-waves (typically 80–300 ms half-max) and noise
    spikes (<15 ms) fall outside this range and are rejected.

    This directly targets the HR84 failure mode: consensus picks a spurious-
    but-tall train of broad T/P-wave deflections at ~130 bpm. T-waves have
    roughly 2–5× the half-max width of a real QRS, so the width gate removes
    them cleanly without touching genuine R-peaks.

    Upper bound (100 ms) is generous enough to pass bundle-branch-block beats
    (~80–90 ms half-max even with QRS duration of 120–200 ms), because the
    sharp R spike is still narrow even inside a wide complex.

    Safety guard: if filtering would leave fewer than 2 peaks, the original
    set is returned unchanged.

    Parameters
    ----------
    peaks        : array of R-peak sample indices from the consensus train
    signal       : the cleaned 1-D ECG signal (from nk.ecg_clean)
    sampling_rate: Hz (typically 500)

    Returns
    -------
    Sorted numpy array of accepted peak sample indices.
    """
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 2:
        return peaks

    min_samp = max(1, int(0.015 * sampling_rate))   # 15 ms
    max_samp = int(0.100 * sampling_rate)             # 100 ms

    sig = signal.astype(float)
    # If R-peaks are predominantly negative (inverted lead), flip for measurement.
    if np.median(sig[peaks]) < 0:
        sig = -sig

    try:
        from scipy.signal import peak_widths as _peak_widths
        # Clamp peak indices to valid range to avoid IndexError at edges.
        valid_peaks = np.clip(peaks, 1, len(sig) - 2)
        widths, _, _, _ = _peak_widths(sig, valid_peaks, rel_height=0.5)
        # Width=0 means peak_widths couldn't measure (zero prominence) — keep by
        # default rather than rejecting a potentially valid QRS peak.  Only reject
        # peaks whose measured width is in the noise range (1-14 ms) or clearly
        # too wide for a QRS (> 100 ms).
        keep_mask = (widths == 0) | ((widths >= min_samp) & (widths <= max_samp))
    except Exception:
        # Fallback: manual half-max crossing search (no scipy dependency).
        n = len(sig)
        keep_mask = np.ones(len(peaks), dtype=bool)
        for i, r in enumerate(peaks):
            pv = sig[r]
            if pv <= 0:
                continue  # inverted or flat — can't measure; keep by default
            half = pv * 0.5
            left = r
            while left > 0 and sig[left] >= half:
                left -= 1
            right = r
            while right < n - 1 and sig[right] >= half:
                right += 1
            w = right - left
            if w < min_samp or w > max_samp:
                keep_mask[i] = False

    filtered = peaks[keep_mask]
    # Safety: never return fewer than 2 peaks.
    return filtered if len(filtered) >= 2 else peaks


def _consensus_rpeaks(cleaned, sampling_rate, nk):
    """Multi-detector consensus R-peak detection.

    No single detector is reliable on scan-reconstructed ECGs — each is best on
    some traces and badly wrong on others. This runs a panel of detectors,
    clusters the peaks they *agree* on (a real beat is found by most detectors;
    a noise spike by only one → voted out), then selects the most
    physiologically plausible consensus train across vote thresholds.

    Falls back to the default `nk.ecg_peaks` if fewer than two detectors
    succeed or no candidate train is plausible.

    Returns a sorted numpy array of R-peak sample indices.
    """
    # 1. Run the panel; record (peak_position, detector_index) for vote counting.
    labeled = []
    n_ok = 0
    for di, method in enumerate(_RPEAK_METHODS):
        try:
            _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, method=method)
            pk = np.asarray(info["ECG_R_Peaks"], dtype=float)
            pk = pk[np.isfinite(pk)].astype(int)
        except Exception:
            continue
        if len(pk) >= 2:
            n_ok += 1
            for p in pk:
                labeled.append((int(p), di))

    if n_ok < 2:
        _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
        return np.asarray(info["ECG_R_Peaks"], dtype=float)[
            np.isfinite(np.asarray(info["ECG_R_Peaks"], dtype=float))
        ].astype(int)

    # 2. Cluster peaks within a ~120 ms tolerance window; vote = distinct detectors.
    labeled.sort()
    win = int(0.12 * sampling_rate)
    clusters = [[labeled[0]]]
    for pos, di in labeled[1:]:
        if pos - clusters[-1][-1][0] <= win:
            clusters[-1].append((pos, di))
        else:
            clusters.append([(pos, di)])

    cl_pos = np.array([int(np.median([p for p, _ in c])) for c in clusters])
    cl_vote = np.array([len({di for _, di in c}) for c in clusters])
    order = np.argsort(cl_pos)
    cl_pos, cl_vote = cl_pos[order], cl_vote[order]

    # 3. Candidate trains at vote thresholds k = 2..n_ok; 4. pick best score.
    best_score, best_train = -np.inf, None
    for k in range(2, n_ok + 1):
        train = cl_pos[cl_vote >= k]
        score = _score_rpeak_train(train, cleaned, sampling_rate)
        if score is not None and score > best_score:
            best_score, best_train = score, train

    if best_train is None:
        _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
        return np.asarray(info["ECG_R_Peaks"], dtype=float)[
            np.isfinite(np.asarray(info["ECG_R_Peaks"], dtype=float))
        ].astype(int)

    # ── Peak-width gate ────────────────────────────────────────────────────
    # Remove broad deflections (T-waves, P-waves) that score well on the
    # regularity/coverage scorer but are physiologically too wide to be QRS.
    # Main target: HR84 spurious train of tall-but-broad T/P deflections.
    # See Research findings 2026-06-13 in SCAN_DEBUG_ANALYSIS.md (Approach 2).
    best_train = _filter_by_peak_width(best_train, cleaned, sampling_rate)

    return np.sort(best_train).astype(int)


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
        # nk.ecg_quality crashes with "cannot convert float NaN to integer"
        # when the signal has too few R-peaks to form valid RR intervals
        # (NeuroKit2's internal ecg_segment fails). Default to 0.5 so the
        # rest of the pipeline can still report intervals.
        try:
            quality = nk.ecg_quality(cleaned, sampling_rate=sampling_rate)
            results["quality_score"] = round(float(np.mean(quality)), 3)
        except (ValueError, TypeError, IndexError):
            results["quality_score"] = 0.5

        if results["quality_score"] < 0.3:
            results["warnings"].append(
                f"Low signal quality ({results['quality_score']:.2f}). "
                "Results may be unreliable — check lead placement."
            )

        # ── Step 3: R-peak detection (multi-detector consensus) ─
        # No single detector is reliable on scan-reconstructed ECGs, so we
        # combine a panel and keep the peaks they agree on. See
        # _consensus_rpeaks. Applied to all signals for one consistent path.
        r_peaks = _consensus_rpeaks(cleaned, sampling_rate, nk)
        r_info = {"ECG_R_Peaks": r_peaks}
        # NB: missed-beat recovery (_recover_missed_rpeaks) is intentionally NOT
        # run here — consensus voting already rescues beats a single detector
        # missed, and re-running recovery on the consensus train re-introduced
        # spurious short RRs (verified on fx8200/HR167). Kept available for the
        # single-detector fallback path only.

        if len(r_peaks) < 2:
            results["error"] = "Insufficient R-peaks detected (< 2). Signal too short or too noisy."
            return results

        results["r_peaks"] = r_peaks.tolist()

        # ── Step 4: Heart Rate ─────────────────────────────────
        rr_samples = np.diff(r_peaks)
        rr_ms = (rr_samples / sampling_rate) * 1000
        results["rr_intervals"] = rr_ms.tolist()

        valid_rr = rr_ms[rr_ms > 0]
        if len(valid_rr) == 0:
            results["error"] = "R-peaks detected at identical positions — signal too noisy or flat."
            return results

        # HR from the *median* RR, not the mean of instantaneous HR: a single
        # spurious short or long RR (a missed/extra beat the recovery pass
        # didn't catch) skews mean-instantaneous-HR badly, whereas the median
        # RR is robust to one outlier interval and is the standard estimator
        # for average rate in mildly irregular rhythm.
        results["hr"] = round(float(60000.0 / np.median(valid_rr)), 1)

        # RR variability (coefficient of variation — proxy for rhythm regularity)
        mean_rr = float(np.mean(valid_rr))
        rr_cv = float(np.std(valid_rr) / mean_rr) if mean_rr > 0 else None
        results["hr_variability"] = round(rr_cv, 3) if rr_cv is not None else None

        # Safety note: high RR variability means either a genuinely irregular
        # rhythm (e.g. AFib) OR — on scan-reconstructed signals — that the
        # consensus detector could not lock onto a clean beat train (spurious/
        # missed beats). Either way the rate and intervals are provisional and
        # should be verified against the trace. See SCAN_DEBUG_ANALYSIS.md task
        # "Scan-derived HR is provisional on noisy traces".
        if rr_cv is not None and rr_cv > 0.25:
            results["warnings"].append(
                f"Irregular R-R intervals (variability {rr_cv:.0%}). Rate/intervals are "
                "provisional — confirm against the trace; on scans this often means noisy "
                "beat detection rather than true arrhythmia."
            )

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

            # ── Edge-beat exclusion ────────────────────────────
            # The DWT delineator is unstable at the strip boundaries: the first
            # and last beat frequently come back with NaN P/T fiducials or with
            # onsets/offsets latched onto edge artifacts (grid lines, the 0.18
            # left-text mask). On scanned strips this inflates PR and wipes out
            # QTc. When we have enough beats (>=4), restrict per-beat fiducial
            # pairing to the *interior* beats, which have full waveform context
            # on both sides. This does not shorten the signal (so it is safe
            # against the A.5 3-second DWT-minimum precedent) — it only ignores
            # boundary beats' fiducials.
            if len(r_peaks) >= 4:
                beat_r_peaks = r_peaks[1:-1]
            else:
                beat_r_peaks = r_peaks

            # ── PR Interval ────────────────────────────────────
            # PR = P onset to R peak
            p_onsets  = _safe_to_int_indices(waves.get("ECG_P_Onsets", []))
            results["p_peaks"] = p_onsets.tolist()

            if len(p_onsets) >= 2 and len(r_peaks) >= 2:
                pr_intervals = []
                # Attempt 10B: Restrict P-onset search to a physiological window
                # (60–250ms before R). DWT on scanned strips places spurious P-onsets
                # at 300–500ms when residual grid-line artifacts create P-like
                # inflections well outside the true PR interval. The 250ms ceiling
                # still covers mild 1st-degree AVB (PR 200-250ms) while cleanly
                # rejecting artifact P-onsets that _suppress_grid_crossing_artifacts
                # does not fully eliminate. Safety: if no P found in the tight window,
                # fall back to the wide search (60-400ms) so we don't lose PR on
                # genuinely long-PR beats.
                p_win_min = int(0.060 * sampling_rate)   # 60 ms
                p_win_max = int(0.250 * sampling_rate)   # 250 ms tight window
                for r in beat_r_peaks:
                    # Tight search first: P onset 60–250ms before R
                    cands_tight = p_onsets[(p_onsets >= r - p_win_max) & (p_onsets < r - p_win_min)]
                    if len(cands_tight) > 0:
                        p = cands_tight[-1]  # closest in window
                    else:
                        # Fallback to wide search: any P onset 60–400ms before R
                        cands_wide = p_onsets[(p_onsets >= r - int(0.400 * sampling_rate)) & (p_onsets < r - p_win_min)]
                        if len(cands_wide) == 0:
                            continue
                        p = cands_wide[-1]
                    pr_ms = ((r - p) / sampling_rate) * 1000
                    if 60 < pr_ms < 400:  # sanity bounds
                        pr_intervals.append(pr_ms)
                if pr_intervals:
                    results["pr"] = round(float(np.median(pr_intervals)), 1)

            # ── QRS Duration ───────────────────────────────────
            # DWT outputs ECG_R_Onsets (start of QRS) and ECG_R_Offsets (end of QRS = J-point).
            # Pair per-beat via R peak — not "first-after", which crosses beats when a
            # boundary is missed and inflates QRS to hundreds of ms.
            r_onsets  = _safe_to_int_indices(waves.get("ECG_R_Onsets",  []))
            r_offsets = _safe_to_int_indices(waves.get("ECG_R_Offsets", []))

            if len(r_onsets) >= 2 and len(r_offsets) >= 2:
                qrs_durations = []
                # Beat window: QRS boundary must fall within a short pre/post-R window
                # (max ~120ms either side of R — wider than any physiological QRS half-width).
                win = int(0.12 * sampling_rate)
                for r in beat_r_peaks:
                    onset_candidates  = r_onsets[(r_onsets  >= r - win) & (r_onsets  <  r)]
                    offset_candidates = r_offsets[(r_offsets > r)       & (r_offsets <= r + win)]
                    if len(onset_candidates) == 0 or len(offset_candidates) == 0:
                        continue
                    ons = onset_candidates[-1]   # closest onset before R
                    off = offset_candidates[0]   # closest offset after R
                    qrs_ms = ((off - ons) / sampling_rate) * 1000
                    if 40 < qrs_ms < 200:  # physiological bound
                        qrs_durations.append(qrs_ms)
                if qrs_durations:
                    results["qrs"] = round(float(np.median(qrs_durations)), 1)

            # ── QTc (Bazett's Formula) ─────────────────────────
            # QTc = QT / sqrt(RR in seconds)
            # Per-beat T-offset pairing: find the T_Offset that falls between R[i] and
            # R[i]+0.95*RR. Window extended to 95% of RR to support tachycardia (HR>150 bpm)
            # where QT can reach 90%+ of the RR interval; 0.70 was too narrow and caused
            # missed T-offsets at fast rates. On noisy scans DWT often fires at the T *peak*;
            # bound QT ≥ 280ms to reject those (T-peak timing is ~240-280ms post-R, well below real QT).
            t_offsets = _safe_to_int_indices(waves.get("ECG_T_Offsets", []))

            if len(t_offsets) >= 2:
                qtc_values = []
                # Skip the first beat too when we have enough beats — its T-offset
                # is the most edge-affected; the last beat is already excluded by [:-1].
                skip_first = len(r_peaks) >= 4
                for i, r in enumerate(r_peaks[:-1]):
                    if skip_first and i == 0:
                        continue
                    rr_samples_i = r_peaks[i + 1] - r
                    max_t_offset = r + int(0.95 * rr_samples_i)
                    t_candidates = t_offsets[(t_offsets > r) & (t_offsets <= max_t_offset)]
                    if len(t_candidates) == 0:
                        continue
                    t_off = t_candidates[-1]   # last T_offset in the window — most likely the true T-end
                    qt_ms = ((t_off - r) / sampling_rate) * 1000
                    rr_s  = rr_ms[i] / 1000
                    if 280 < qt_ms < 600 and rr_s > 0:  # tighter floor rejects T-peak mis-detection
                        qtc = qt_ms / np.sqrt(rr_s)
                        if 300 < qtc < 600:
                            qtc_values.append(qtc)

                if qtc_values:
                    results["qtc"] = round(float(np.median(qtc_values)), 1)

            # ── QTc fallback: 'peak' method ────────────────────
            # DWT mislabels the J-point as T-offset for fast (HR>130 bpm) or
            # wide-QRS beats — all T-offsets land at 50-80ms from R (within the
            # QRS), are rejected by the 280ms floor, and QTc stays None.
            # The 'peak' method uses a simpler threshold approach that degrades
            # more gracefully under these conditions.  Only attempted when DWT
            # produced no valid QTc.
            if results.get("qtc") is None and len(r_peaks) >= 3:
                try:
                    _, waves_peak = nk.ecg_delineate(
                        cleaned, r_info, sampling_rate=sampling_rate, method="peak"
                    )
                    t_offs_peak = _safe_to_int_indices(
                        waves_peak.get("ECG_T_Offsets", [])
                    )
                    if len(t_offs_peak) >= 2:
                        qtc_peak_vals = []
                        skip_first_p = len(r_peaks) >= 4
                        for i_p, r_p in enumerate(r_peaks[:-1]):
                            if skip_first_p and i_p == 0:
                                continue
                            rr_samp_p = r_peaks[i_p + 1] - r_p
                            max_t_p = r_p + int(0.95 * rr_samp_p)
                            t_cands_p = t_offs_peak[
                                (t_offs_peak > r_p) & (t_offs_peak <= max_t_p)
                            ]
                            if len(t_cands_p) == 0:
                                continue
                            t_off_p = t_cands_p[-1]
                            qt_ms_p = ((t_off_p - r_p) / sampling_rate) * 1000
                            rr_s_p = rr_ms[i_p] / 1000
                            if 280 < qt_ms_p < 600 and rr_s_p > 0:
                                qtc_p = qt_ms_p / np.sqrt(rr_s_p)
                                if 300 < qtc_p < 600:
                                    qtc_peak_vals.append(qtc_p)
                        if qtc_peak_vals:
                            results["qtc"] = round(float(np.median(qtc_peak_vals)), 1)
                except Exception:
                    pass  # peak method also failed — QTc stays None

        except Exception as delineation_error:
            # Delineation can fail on noisy signals — HR is still valid
            results["warnings"].append(
                f"Waveform delineation incomplete: {str(delineation_error)[:80]}. "
                "HR is reliable; PR/QRS/QTc may be unavailable."
            )

    except Exception as e:
        logging.error("calculate_intervals failed\n%s", traceback.format_exc())
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
