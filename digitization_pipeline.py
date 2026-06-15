"""
digitization_pipeline.py
========================
Enhanced EKG paper strip digitization using OpenCV.

Improvements over v1:
  - Adaptive grid detection (handles red, blue, and green grids)
  - Morphological cleanup for better trace isolation
  - Multi-lead extraction (detects horizontal strips in the image)
  - Weighted median for smoother signal extraction
  - Baseline drift removal via high-pass filter
  - Quality score for the extracted signal

Usage:
    from digitization_pipeline import extract_signal_from_image, extract_multi_lead
    signal = extract_signal_from_image("scan.png")
    leads = extract_multi_lead("12_lead_scan.png")
"""

from fractions import Fraction

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

# Optional OCR for lead-label routing. The pipeline degrades to the
# heuristic R-peak-regularity scorer when this import or the underlying
# Tesseract binary isn't available, so the app still runs without it.
try:
    import pytesseract  # noqa: F401
    _PYTESSERACT_AVAILABLE = True
except ImportError:
    _PYTESSERACT_AVAILABLE = False

_TESSERACT_BINARY_OK = None  # lazily probed; see _ocr_available()

# Common Windows install locations to try before giving up. The
# UB-Mannheim installer writes to Program Files by default but doesn't
# add itself to PATH, so we have to look for it ourselves.
_TESSERACT_FALLBACK_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)

# Standard ECG paper calibration constants
# Small grid square = 1 mm × 1 mm
# At 25 mm/s  and 10 mm/mV: 1 small sq = 40 ms × 0.1 mV
DEFAULT_PAPER_SPEED = 25   # mm/s  (25 or 50 are the two clinical standards)
DEFAULT_MM_PER_MV   = 10   # mm/mV (5, 10, or 20 mm/mV)
TARGET_FS           = 500  # Hz — output sampling rate expected by the rest of the pipeline


def _detect_grid_spacing(grid_mask):
    """
    Measure the pixel spacing between grid lines in X and Y directions.

    Returns:
        (px_per_mm_x, px_per_mm_y): pixels per 1 mm of ECG paper.
        Either value is None if detection failed (caller should use fallback).
    """
    def _find_median_line_spacing(proj):
        """Find the dominant grid-line spacing in a 1-D projection."""
        if np.max(proj) < 5:
            return None   # no grid visible
        peaks, _ = find_peaks(proj, distance=3,
                               prominence=np.std(proj) * 0.4)
        if len(peaks) < 4:
            return None
        spacings = np.diff(peaks)
        # The small-square spacing is the most common short spacing.
        # Large squares (every 5th line) appear as 5× the small spacing — exclude them.
        median_sp = float(np.median(spacings))
        small = spacings[spacings < median_sp * 1.8]
        if len(small) < 2:
            small = spacings
        sp = float(np.median(small))
        # Sanity: a grid square should be between 3 and 60 pixels wide
        return sp if 3 < sp < 60 else None

    col_proj = np.sum(grid_mask, axis=0).astype(float)
    row_proj = np.sum(grid_mask, axis=1).astype(float)

    px_x = _find_median_line_spacing(col_proj)
    px_y = _find_median_line_spacing(row_proj)
    return px_x, px_y


def _detect_grid_mask(img_bgr):
    """
    Detect the EKG paper grid regardless of color (red, blue, green).
    Uses HSV color space with multiple range checks.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Red grid (wraps around H=0 in HSV)
    red_low1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255]))
    red_low2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_low1, red_low2)

    # Blue grid
    blue_mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([130, 255, 255]))

    # Green grid
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))

    # Combine all grid colors
    grid_mask = cv2.bitwise_or(red_mask, cv2.bitwise_or(blue_mask, green_mask))

    # Dilate slightly to catch grid edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid_mask = cv2.dilate(grid_mask, kernel, iterations=1)

    return grid_mask


def _extract_trace(gray, grid_mask):
    """
    Extract the signal trace from grayscale image after grid removal.
    Uses adaptive thresholding and morphological operations.
    """
    # Adaptive threshold for the trace (handles varying illumination)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=8,
    )

    # Remove grid from trace
    cleaned = cv2.bitwise_and(binary, binary, mask=cv2.bitwise_not(grid_mask))

    # Morphological close to connect broken trace segments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Remove small noise blobs
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

    return cleaned


def _select_dominant_cluster(col_pixels, prev_value, gap_tol=1):
    """Pick the dominant (largest) contiguous cluster of lit pixels in a column.

    ECGtizer-style fragmented extraction: when a column contains pixels from
    multiple sources (two nearby leads, a grid-line crossing the trace), splitting
    into contiguous runs and selecting the largest one preferentially follows
    the real trace ink over artifacts or adjacent-lead bleed.

    Gap tolerance of 1 pixel lets a single-pixel anti-aliasing hole stay within
    the same cluster without merging two genuinely separate traces.

    For size ties, the cluster whose centroid is closest to ``prev_value``
    (the preceding column's extracted row position) is chosen — enforcing
    trace continuity across columns.

    Parameters
    ----------
    col_pixels  : sorted 1-D int array of lit row indices (from np.where)
    prev_value  : float (may be NaN) — the previous column's signal[x-1]
    gap_tol     : int — max row-gap within one cluster (default 1)

    Returns
    -------
    float — median row index of the dominant cluster
    """
    diffs = np.diff(col_pixels)
    split_idx = np.where(diffs > gap_tol)[0] + 1
    clusters = np.split(col_pixels, split_idx)

    if len(clusters) == 1:
        return float(np.median(clusters[0]))

    sizes = np.array([len(c) for c in clusters])
    max_size = sizes.max()
    candidates = [c for c in clusters if len(c) == max_size]

    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        centroids = np.array([np.mean(c) for c in candidates])
        if not np.isnan(prev_value):
            ref = float(prev_value)
        else:
            ref = float(np.mean(col_pixels))
        chosen = candidates[int(np.argmin(np.abs(centroids - ref)))]

    return float(np.median(chosen))


def _trace_to_signal(trace_binary, use_weighted=True):
    """Convert binary trace image to 1D signal.

    When ``use_weighted=True`` (default), uses ECGtizer-style fragmented
    extraction: per column, split lit pixels into contiguous runs and pick
    the largest cluster (most ink). This preferentially follows the real trace
    over grid-line crossings or adjacent-lead bleed. Ties are broken by
    continuity with the previous column.

    When ``use_weighted=False``, falls back to simple median of all lit pixels.
    """
    rows, cols = trace_binary.shape
    signal = np.full(cols, np.nan)

    for x in range(cols):
        col_pixels = np.where(trace_binary[:, x] > 0)[0]
        if len(col_pixels) == 0:
            continue

        if use_weighted:
            prev = signal[x - 1] if x > 0 else np.nan
            signal[x] = _select_dominant_cluster(col_pixels, prev)
        else:
            signal[x] = np.median(col_pixels)

    # Interpolate NaN gaps
    valid = ~np.isnan(signal)
    if np.sum(valid) < cols * 0.3:
        # Too few valid points — fall back to simple extraction
        signal = np.full(cols, rows / 2.0)
        for x in range(cols):
            col_pixels = np.where(trace_binary[:, x] > 0)[0]
            if len(col_pixels) > 0:
                signal[x] = np.median(col_pixels)
    else:
        # Interpolate gaps
        x_valid = np.where(valid)[0]
        y_valid = signal[valid]
        signal = np.interp(np.arange(cols), x_valid, y_valid)

    # Invert Y (image 0 is top) and center
    signal = -signal  # Invert so up = positive voltage
    signal = signal - np.mean(signal)

    return signal


def _remove_baseline_drift(signal, fs=500, cutoff=0.5):
    """Remove baseline wander using a high-pass Butterworth filter."""
    if len(signal) < 20:
        return signal
    nyq = fs / 2
    if cutoff >= nyq:
        return signal - np.mean(signal)
    b, a = butter(2, cutoff / nyq, btype='high')
    try:
        return filtfilt(b, a, signal)
    except Exception:
        return signal - np.mean(signal)


def _lowpass_40hz(signal, fs):
    # Anything above 40 Hz in a photo-reconstructed trace is pixel-quantization
    # jitter, not physiology — DWT delineation misreads it as spurious features.
    if len(signal) < 20 or fs < 90:
        return signal
    nyq = fs / 2
    b, a = butter(4, 40.0 / nyq, btype='low')
    try:
        return filtfilt(b, a, signal)
    except Exception:
        return signal


def _suppress_grid_crossing_artifacts(signal_px, grid_mask, text_cols=0, min_density=0.35):
    """Linear-interpolate across vertical grid-line columns to remove step artifacts.

    When the ECG trace crosses a vertical grid line, the grid-removal mask
    (`cv2.bitwise_and(binary, ~grid_mask)`) erases pixels at the crossing
    columns. The resulting gap is NaN-interpolated by `_trace_to_signal`,
    leaving a smooth inflection that the DWT delineator (`ecg_delineate
    method="dwt"`) misidentifies as a P-wave onset, inflating PR by
    100-200 ms (fx8200: 254 ms measured vs 131 ms truth).

    Vertical grid lines span the full band height → per-column grid density
    (`grid_mask > 0).mean(axis=0)`) ≈ 0.8-1.0. Horizontal grid lines give
    density ≈ 0.01-0.04. QRS peaks are not in the grid mask (different color
    ink). Threshold 0.35 cleanly separates vertical grid columns from everything
    else.

    One sample per image column (1:1 mapping from _trace_to_signal). If that
    invariant breaks the shape guard returns unchanged.

    Parameters
    ----------
    signal_px  : 1-D float array — extracted pixel-space signal (len = n_cols)
    grid_mask  : 2-D uint8 array (band_height, n_cols), axis=0 is rows
    text_cols  : int — skip leftmost text region (already zeroed in trace)
    min_density: float — vertical-grid-line detection threshold (default 0.35)

    Returns
    -------
    1-D float array — same length as signal_px, grid crossings interpolated
    """
    # Safety guards — return input unchanged rather than crash
    if grid_mask is None or grid_mask.ndim != 2:
        return signal_px
    if grid_mask.shape[1] != len(signal_px) or len(signal_px) < 3:
        return signal_px

    out = signal_px.astype(float).copy()
    N = len(out)

    # Per-column grid density: axis=0 collapses rows → 1-D array length N
    density = (grid_mask > 0).mean(axis=0)

    # Mark columns that are dominated by a vertical grid line
    bad = density >= min_density

    # Never interpolate over the left text block or use it as an anchor
    bad[:text_cols] = False

    # Anchor indices: clean columns outside the text region
    good_idx = np.where(~bad & (np.arange(N) >= text_cols))[0]
    if good_idx.size < 2:
        return out  # too few clean columns to interpolate from

    # Repair indices: bad columns that are in the signal region
    repair_idx = np.where(bad & (np.arange(N) >= text_cols))[0]
    if repair_idx.size:
        # np.interp linearly interpolates from nearest good columns on each side;
        # clamps to edge value for any bad column outside the good-column range.
        out[repair_idx] = np.interp(repair_idx, good_idx, out[good_idx])

    return out


def _detect_lead_rows(trace_binary, min_row_height_ratio=0.05):
    """
    Find horizontal bands where ECG traces exist.
    Returns list of (y_start, y_end) tuples, one per detected lead row.

    Two-pass detection:
      1. Coarse threshold at 10% of peak finds overall content regions.
      2. For any band tall enough to plausibly contain multiple stacked leads
         (FX-8200 prints 3 cell rows + rhythm strip with no whitespace gap
         between them, just shallow valleys), find internal valleys via
         find_peaks on the negated row-sum and split at those positions.
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    rows = trace_binary.shape[0]
    row_sum = trace_binary.sum(axis=1).astype(float)
    smoothed = uniform_filter1d(row_sum, size=max(5, rows // 40))
    peak = smoothed.max()
    if peak == 0:
        return []

    # Pass 1 — coarse content regions
    in_band = smoothed > peak * 0.10
    coarse_bands = []
    start = None
    for i, v in enumerate(in_band):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if (i - start) >= rows * min_row_height_ratio:
                coarse_bands.append((start, i))
            start = None
    if start is not None and (rows - start) >= rows * min_row_height_ratio:
        coarse_bands.append((start, rows))

    # Pass 2 — split each coarse band on internal valleys.
    # Two conditions must both hold for a band to be split:
    #   (a) Tall enough to plausibly stack ≥2 cell rows (~10 mm each).
    #   (b) Dense enough that its mean row-sum is comparable to the global peak —
    #       stacked rows of traces accumulate pixels per row, so the merged block
    #       is bright; a single sparse rhythm strip stays dim and should NOT be
    #       split on its own intra-wave morphology dips.
    min_cell_px = max(80, int(rows * 0.05))
    # Re-smooth at finer resolution for valley detection. The pass-1 window
    # (~rows/40) is tuned for finding outer edges of content regions; it
    # over-flattens the ~10pp inter-row dips on tightly-stacked layouts like
    # FX-8200, where cell rows touch each other with no whitespace between them.
    smoothed_fine = uniform_filter1d(row_sum, size=max(5, rows // 150))
    refined = []
    for s, e in coarse_bands:
        height = e - s
        local = smoothed[s:e]
        local_mean = float(local.mean())
        local_peak_in_band = float(local.max())

        if (
            height < min_cell_px * 2.0
            or local_peak_in_band <= 0
            or local_mean < peak * 0.50
        ):
            refined.append((s, e))
            continue

        # Run valley detection on the finely-smoothed profile. Prominence is
        # measured against the GLOBAL peak (absolute density drop) rather than
        # the local peak so the threshold means the same thing in every band.
        local_fine = smoothed_fine[s:e]
        valleys, _ = find_peaks(
            -local_fine,
            prominence=peak * 0.08,
            distance=min_cell_px,
        )
        if len(valleys) == 0:
            refined.append((s, e))
            continue

        cuts = [0] + sorted(int(v) for v in valleys) + [height]
        for i in range(len(cuts) - 1):
            sub_s = s + cuts[i]
            sub_e = s + cuts[i + 1]
            if (sub_e - sub_s) >= min_cell_px * 0.6:
                refined.append((sub_s, sub_e))

    return refined


def _band_traceness(trace_binary, start, end):
    """
    Return a 0-1 score of how trace-like a horizontal band is.

    Real ECG bands have:
      - high column coverage (most columns have at least one trace pixel)
      - thin per-column trace (a few px tall, not a wide blob)
    Wrist-rest / paper-edge / wood-texture bands have low coverage or wide
    blobs. This is the cheapest discriminator that doesn't require physiology.
    """
    strip = trace_binary[start:end, :]
    rows_b, cols_b = strip.shape
    if rows_b == 0 or cols_b == 0:
        return 0.0
    coverage = float((strip.sum(axis=0) > 0).mean())
    # Per-column trace thickness (how many pixels deep is the trace).
    # Real ECG: 2-6 px out of ~150 px row height (≤5%). Background noise
    # blobs cover much more of each column.
    thicknesses = [
        int(np.sum(strip[:, x] > 0)) for x in range(0, cols_b, max(1, cols_b // 200))
    ]
    if not thicknesses:
        return 0.0
    median_thickness_ratio = float(np.median(thicknesses)) / rows_b
    # Score: coverage rewarded, thickness penalized
    thin_score = max(0.0, 1.0 - median_thickness_ratio / 0.20)
    return coverage * 0.5 + thin_score * 0.5


def _select_best_row(trace_binary, bands):
    """
    Select the best single lead row to analyze.

    Trace-likeness dominates: a high-coverage thin-trace band beats a
    big band of wood-texture noise (which has high column "coverage" but
    very thick per-column blobs). The tiebreak is moderate height (~150 px
    is a typical ECG cell row at common photo resolutions; much taller
    bands are usually noise/text regions, much shorter ones are
    separator strips between rows).
    """
    rows = trace_binary.shape[0]
    min_acceptable_height = max(60, rows // 30)
    target_height = max(100, rows // 12)   # rough rhythm-strip / lead-row height
    candidates = []
    for s, e in bands:
        height = e - s
        if height < min_acceptable_height:
            continue
        tr = _band_traceness(trace_binary, s, e)
        if tr < 0.30:
            continue   # band looks like noise, not ECG
        # Height fit: peak at target, falls off in both directions.
        # 1.0 at target_height, 0.0 at 4× target or 0.25× target.
        ratio = height / target_height
        if ratio < 1:
            height_fit = ratio
        else:
            height_fit = max(0.0, 2.0 - ratio) / 1.0  # 1.0 at ratio=1, 0 at ratio=2
        bottom_bonus = (s + e) / (2.0 * rows)
        score = tr * 0.55 + height_fit * 0.30 + bottom_bonus * 0.15
        candidates.append(((s, e), score))
    if not candidates:
        return bands[0]
    return max(candidates, key=lambda t: t[1])[0]


# ── A.7 OCR-based lead identification (PMcardio "Identify Lead → Orient Signal") ──

# Canonical lead names we recognize. Anything OCR returns gets normalized
# to one of these or rejected.
_CANONICAL_LEADS = {"I", "II", "III", "aVR", "aVL", "aVF",
                    "V1", "V2", "V3", "V4", "V5", "V6"}

# Common OCR confusions, applied after uppercasing (except the aV* prefix).
# Tesseract often reads 'I'/'1'/'l' interchangeably and confuses 'II' as 'Il' or 'l1'.
_OCR_NORMALIZE = {
    "L": "I", "1": "I", "0": "O",
}


def _ocr_available():
    """Probe whether the Tesseract binary is callable. Cached after first call.

    Tries PATH first, then falls back to known Windows install locations.
    The UB-Mannheim installer doesn't update PATH automatically, so without
    this fallback users would need to manually edit their environment.
    """
    global _TESSERACT_BINARY_OK
    if _TESSERACT_BINARY_OK is not None:
        return _TESSERACT_BINARY_OK
    if not _PYTESSERACT_AVAILABLE:
        _TESSERACT_BINARY_OK = False
        return False
    try:
        pytesseract.get_tesseract_version()
        _TESSERACT_BINARY_OK = True
        return True
    except Exception:
        pass
    import os as _os
    for path in _TESSERACT_FALLBACK_PATHS:
        if _os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            try:
                pytesseract.get_tesseract_version()
                _TESSERACT_BINARY_OK = True
                return True
            except Exception:
                continue
    _TESSERACT_BINARY_OK = False
    return False


def _normalize_lead_name(raw):
    """
    Best-effort cleanup of OCR output to a canonical lead name.

    Returns the canonical name (e.g. "II", "aVR", "V3") or None if the text
    doesn't plausibly look like a lead label.
    """
    if not raw:
        return None
    # Strip whitespace, dots, commas, dashes
    s = raw.strip().replace(".", "").replace(",", "").replace("-", "")
    if not s:
        return None
    # aV-family: case-insensitive prefix match
    low = s.lower()
    if low.startswith("av") and len(s) >= 3:
        suffix = s[2].upper()
        if suffix in {"R", "L", "F"}:
            return f"aV{suffix}"
        # Common confusions: 'avi' → aVL, 'av1' → aVL (1↔L), 'avr/avf' obvious
        if suffix in {"I", "1", "L"}:
            return "aVL"
    # V-leads: V1..V6
    upper = s.upper()
    if upper.startswith("V") and len(upper) >= 2 and upper[1] in "123456":
        return f"V{upper[1]}"
    # Limb leads I, II, III — only accept if the whole string reduces to I's
    # after applying common OCR substitutions (otherwise "gibberish" matches
    # because it contains an 'i').
    swapped = "".join(_OCR_NORMALIZE.get(ch, ch) for ch in upper)
    if swapped in {"I", "II", "III"}:
        return swapped
    return None


import re as _re

# Match a canonical lead name as a standalone token. Anchored on word
# boundaries so embedded-letter false positives (the 'i' in "i") don't
# leak through. Must NOT match inside dates ("18:12") or HR numbers.
_LEAD_TOKEN_RE = _re.compile(
    r"(?<![A-Za-z0-9])(aV[RLF]|V[1-6]|III|II|I)(?![A-Za-z0-9])",
    _re.IGNORECASE,
)


def _read_band_label(paper_bgr, band, label_col_frac=None):
    """
    OCR the band and return a canonical lead name (e.g. "II", "aVR", "V1")
    or None.

    Empirical finding on the 4 fixtures: FX-8200 thermal printer prints
    the lead label *inside* each row (mid/right of the trace), not in a
    leftmost label column. PSM 7 (single line) + character whitelist
    finds nothing because each band is a 100-300px strip with a label
    surrounded by ECG trace and grid noise — not a clean text line.

    Strategy that works: pass the full-width band gray crop to Tesseract
    in PSM 11 (sparse text — finds text in any orientation/position),
    *without* a character whitelist, and post-filter the output for
    canonical lead-name tokens via regex. The existing date/time/HR
    header noise on each band gets discarded by the regex anchors.

    `label_col_frac` is accepted for backwards compatibility but ignored
    — we now scan the entire band width.
    """
    if not _ocr_available():
        return None
    s, e = band
    h, w = paper_bgr.shape[:2]
    s = max(0, int(s)); e = min(h, int(e))
    if e - s < 20:
        return None
    crop = paper_bgr[s:e, :]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    try:
        raw = pytesseract.image_to_string(gray, config="--psm 11")
    except Exception:
        return None
    # Pull all canonical-lead-shaped tokens from the OCR output. Tesseract
    # often returns multiple lines including header noise (date, HR), so
    # we rely on the regex to discard non-leads.
    matches = _LEAD_TOKEN_RE.findall(raw)
    if not matches:
        return None
    normed = [n for n in (_normalize_lead_name(m) for m in matches) if n]
    if not normed:
        return None
    # Multi-letter labels (II, III, aVR/aVL/aVF, V1..V6) are highly specific
    # and only appear as true labels. Single "I" tokens come from PSM 11
    # reading any vertical grid-line mark as the letter "I", so we trust
    # them only when nothing more distinctive shows up.
    multi = [n for n in normed if len(n) >= 2]
    if multi:
        from collections import Counter
        return Counter(multi).most_common(1)[0][0]
    # Only single-letter matches — accept "I" only if it appears at least
    # twice (single occurrence is too easily one stray vertical mark).
    if normed.count("I") >= 2:
        return "I"
    return None


# Priority order when picking which band to extract as the rhythm strip.
# II is the most common rhythm strip on FX-8200 and most A4 hospital
# prints. V1 is a common alternative. After that we accept any labeled
# band, preferring the longest (which on a multi-row layout is usually
# the dedicated rhythm strip rather than a per-cell snippet).
_LEAD_PRIORITY = ("II", "V1", "I", "III", "aVR", "aVL", "aVF",
                  "V2", "V3", "V4", "V5", "V6")


def _pick_by_label(labeled, all_bands, trace_binary):
    """
    Pick a single band using OCR'd lead labels. Falls back to
    _select_best_row when no labels are available or none match.

    Args:
        labeled  : list of ((s, e), lead_name) — bands with confident OCR.
        all_bands: full list of (s, e) bands from _detect_lead_rows.
        trace_binary: trace mask, passed through to the heuristic fallback.

    Returns:
        (start, end) of the chosen band.
    """
    if labeled:
        by_name = {}
        for band, name in labeled:
            # If a lead is detected on multiple rows, prefer the longest.
            prev = by_name.get(name)
            if prev is None or (band[1] - band[0]) > (prev[1] - prev[0]):
                by_name[name] = band
        for name in _LEAD_PRIORITY:
            if name in by_name:
                return by_name[name]
        # Some lead got detected but it's not in our priority list — pick the
        # longest labeled band as a safe default.
        return max((b for b, _ in labeled), key=lambda b: b[1] - b[0])
    # No OCR results — fall back to the existing heuristic.
    return _select_best_row(trace_binary, all_bands)


def _compute_quality_score(trace_binary):
    """
    Estimate signal extraction quality (0-1).
    Based on: column coverage, trace continuity, column-to-column jitter.
    """
    rows, cols = trace_binary.shape

    # Column coverage: what fraction of columns have trace pixels
    has_trace = np.array([np.any(trace_binary[:, x] > 0) for x in range(cols)])
    coverage = float(np.mean(has_trace))

    # Continuity: what fraction of columns have a clean single-cluster trace
    clean_cols = 0
    for x in range(cols):
        pixels = np.where(trace_binary[:, x] > 0)[0]
        if len(pixels) > 0:
            spread = pixels[-1] - pixels[0]
            if spread < rows * 0.15:  # trace occupies < 15% of column height
                clean_cols += 1
    continuity = clean_cols / cols if cols > 0 else 0

    # Jitter: median column-to-column step as a fraction of signal range.
    # Clean ECG baseline is near-flat between events (ratio <1%); phone-photo
    # sharp-toothed noise moves by several % every column. Catches scans that
    # pass coverage + continuity but produce unusable waveforms.
    y_by_col = _trace_to_signal(trace_binary, use_weighted=True) if cols > 0 else np.array([])
    if len(y_by_col) > 1:
        y_range = float(np.ptp(y_by_col))
        if y_range > 1e-6:
            med_diff = float(np.median(np.abs(np.diff(y_by_col))))
            jitter_ratio = med_diff / y_range
            # ≤1% of range → full credit; ≥6% → zero credit; linear in between.
            jitter_score = max(0.0, min(1.0, 1.0 - (jitter_ratio - 0.01) / 0.05))
        else:
            jitter_score = 0.0  # flat signal — useless
    else:
        jitter_score = 0.0

    return round(min(1.0, coverage * 0.4 + continuity * 0.3 + jitter_score * 0.3), 3)


def _detect_paper_region(img_bgr):
    """
    Find the ECG paper region in a phone photo and crop the image to it.

    Removes hand/desk/wrist-rest background that confuses downstream band
    detection. Returns an axis-aligned crop (no perspective warp) so the
    paper's printed orientation is preserved for A.6 to resolve.

    Detection cue: a block is "paper" if it is BOTH bright (CLAHE-normalized
    so dim-light photos work) AND has dense grid pixels from the existing
    red/blue/green grid mask. The bounding box of the largest connected
    component of paper-blocks is the crop region.

    Returns:
        (paper_bgr, bbox_xywh) — bbox is (x, y, w, h) in original image
                                 coordinates, or None if detection failed
                                 (paper_bgr is then the original image).
    """
    grid_mask = _detect_grid_mask(img_bgr)
    if grid_mask.sum() == 0:
        return img_bgr, None

    h, w = img_bgr.shape[:2]
    min_side = min(h, w)

    # Two-cue paper detection: a block is "paper" if it is bright AND has
    # dense grid pixels. CLAHE is applied first so dim photos (FX-8200 ref
    # at ~150 median brightness) are normalized to the same dynamic range
    # as well-lit ones (~200 median) before the gate. This was the cheapest
    # intervention to make a fixed-percentile gate work across photos.
    gray_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray_raw)

    block = max(20, int(min_side * 0.02))
    small_h = h // block
    small_w = w // block
    if small_h < 4 or small_w < 4:
        return img_bgr, None

    gray_c = gray[:small_h * block, :small_w * block].astype(np.float32)
    grid_c = grid_mask[:small_h * block, :small_w * block]
    brightness = gray_c.reshape(small_h, block, small_w, block).mean(axis=(1, 3))
    density    = grid_c.reshape(small_h, block, small_w, block).sum(axis=(1, 3))
    if density.max() == 0:
        return img_bgr, None

    bright_ok = brightness > 255 * 0.55
    dense_ok  = density > density.max() * 0.40
    paper_blocks = (bright_ok & dense_ok).astype(np.uint8)

    # Close gaps from text/labels inside the paper region (~5% of paper width).
    # Skip the open step: small papers under dim light can't afford erosion.
    k = max(3, int(small_w * 0.05))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    paper_blocks = cv2.morphologyEx(paper_blocks, cv2.MORPH_CLOSE, kernel)

    # Find all paper-block components and take the bounding box of every
    # component that is at least ~20% of the largest one. The largest-only
    # approach loses fragments of the same paper (label/header/footer
    # sections can disconnect from the dense trace region by text rows
    # that fail the density gate), shrinking the crop to a sliver of the
    # actual paper. Including the smaller fragments restores the full
    # paper extent while still ignoring tiny background noise blobs.
    nl, lbls, stats, _ = cv2.connectedComponentsWithStats(paper_blocks, 8)
    if nl <= 1:
        return img_bgr, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest_area = int(areas.max())
    if biggest_area < small_h * small_w * 0.03:
        return img_bgr, None

    keep = np.where(areas >= biggest_area * 0.20)[0] + 1
    xs0, ys0, xs1, ys1 = [], [], [], []
    for k in keep:
        bx_k, by_k, bw_k, bh_k, _ = stats[k]
        xs0.append(bx_k); ys0.append(by_k)
        xs1.append(bx_k + bw_k); ys1.append(by_k + bh_k)
    bx, by = min(xs0), min(ys0)
    bx_end, by_end = max(xs1), max(ys1)

    # Axis-aligned crop, no warp — A.6's orientation logic handles rotation.
    x0 = bx * block
    y0 = by * block
    x1 = min(w, bx_end * block)
    y1 = min(h, by_end * block)
    if (x1 - x0) < 100 or (y1 - y0) < 50:
        return img_bgr, None

    paper = img_bgr[y0:y1, x0:x1]
    return paper, (x0, y0, x1 - x0, y1 - y0)


def _load_image_exif_aware(image_path):
    """
    Load image as OpenCV BGR array, honoring EXIF orientation tags.

    cv2.imread() ignores EXIF — phone photos taken in portrait often arrive
    with landscape pixel dims plus an "rotate 90° CW" tag. PIL's exif_transpose
    applies the rotation so downstream shape checks see the user-intended
    orientation.

    Falls back to cv2.imread for formats PIL can't open.
    """
    try:
        with Image.open(image_path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            rgb = np.array(pil_img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
        return img


def _resolve_orientation(paper_bgr):
    """
    Resolve which of the 4 cardinal rotations puts the ECG paper labels
    right-side-up. Tests all 4 rotations and picks the one where the
    extracted trace has the strongest upward-deflecting QRS-like pattern.

    On a correctly-oriented ECG paper, the trace runs left-to-right and
    R-peaks deflect upward (positive in the numpy array, after Y-inversion).
    On a 180° rotation, R-peaks point downward; on 90°/270° rotations,
    the "trace" is actually a vertical line and the column-extracted
    signal is mostly flat with sparse sharp jumps.

    Returns the paper rotated to canonical orientation (R-peaks up,
    trace flowing left-to-right).
    """
    # Step 1: force landscape — ECG paper is canonically wider than tall
    # (paper roll runs left-to-right, time on the X axis). For roughly
    # square crops we still want landscape; tie-break by trying both and
    # picking the higher-scored 180° pair.
    h, w = paper_bgr.shape[:2]
    if h > w:
        # Rotate 90° (either direction) to make landscape; we then resolve
        # which 90° was right via the 0°-vs-180° score below.
        landscape = cv2.rotate(paper_bgr, cv2.ROTATE_90_CLOCKWISE)
    else:
        landscape = paper_bgr

    # Step 2: resolve 0°/180° flip — pick the orientation that gives a higher
    # _orientation_score (R-peaks deflecting upward).
    flipped = cv2.rotate(landscape, cv2.ROTATE_180)
    score_a = _orientation_score(landscape)
    score_b = _orientation_score(flipped)

    # Step 3: also try the OTHER 90° direction (if input was portrait, both
    # rotations of the original portrait can produce landscape — only one
    # is correct). Score that pair too and keep the global max.
    if h > w:
        landscape_alt = cv2.rotate(paper_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        flipped_alt = cv2.rotate(landscape_alt, cv2.ROTATE_180)
        score_c = _orientation_score(landscape_alt)
        score_d = _orientation_score(flipped_alt)
    else:
        landscape_alt = flipped_alt = landscape  # unused
        score_c = score_d = -np.inf

    scores = [(score_a, landscape), (score_b, flipped),
              (score_c, landscape_alt), (score_d, flipped_alt)]
    return max(scores, key=lambda t: t[0])[1]


def _orientation_score(paper_bgr):
    """
    Score how "right-side-up" a paper crop is.

    For the orientation to be correct:
      - the bottom 30% of the image (likely rhythm strip area) should contain
        a roughly-horizontal trace
      - extracted column-wise median Y position should swing through positive
        values more often than negative (R-peak deflections are upward)
      - extracted signal should have sharp narrow peaks (high kurtosis)

    Returns a real-valued score; higher = more likely upright.
    """
    gray = cv2.cvtColor(paper_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Look at the bottom 30% — most ECG layouts put the rhythm strip there;
    # for stacked-row layouts any row works the same.
    band = gray[int(h * 0.5):, :]
    # Adaptive threshold to isolate dark trace pixels
    binary = cv2.adaptiveThreshold(
        band, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=8,
    )
    # Drop columns with too many or too few dark pixels (background or text)
    sig = np.full(w, np.nan, dtype=np.float32)
    rows_band = binary.shape[0]
    for x in range(w):
        ys = np.where(binary[:, x] > 0)[0]
        if 1 <= len(ys) <= rows_band * 0.3:
            sig[x] = float(np.median(ys))
    valid = ~np.isnan(sig)
    if valid.sum() < w * 0.2:
        return -np.inf
    sig[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], sig[valid])
    # Invert Y (image rows grow downward, ECG amplitude positive = up)
    sig = -sig
    sig = sig - np.median(sig)
    # Score: positive deflections (R-peaks point up in correct orientation)
    # use the 95th percentile minus the 5th percentile, weighted by sign of
    # the bigger one. Real ECG: 95th percentile is large positive (R-peaks),
    # 5th percentile is small negative (S/Q dips). Upside-down: opposite.
    high = float(np.percentile(sig, 95))
    low  = float(np.percentile(sig, 5))
    return high + low  # both positive when R-peaks dominate upward


def extract_signal_from_image(image_path,
                              paper_speed: float = DEFAULT_PAPER_SPEED,
                              mm_per_mv: float = DEFAULT_MM_PER_MV,
                              debug: bool = False):
    """
    Extract 1D ECG signal from a paper strip image, calibrated to physical units.

    Args:
        image_path  : path to the image file
        paper_speed : recording paper speed in mm/s (25 or 50)
        mm_per_mv   : voltage gain in mm/mV (5, 10, or 20)
        debug       : if True, show intermediate images

    Returns:
        dict with keys:
            signal   : numpy array, calibrated in mV, resampled to TARGET_FS Hz
            quality  : float 0-1
            actual_fs: float — estimated source sampling rate (px/s) before resampling
            px_per_mm: (x, y) grid calibration in pixels/mm, or (None, None) if failed
    """
    img_full = _load_image_exif_aware(image_path)

    # A.6 — resolve which of the 4 cardinal rotations puts the paper
    # right-side-up. Operates on the whole image, which is fine because
    # _resolve_orientation only samples the bottom 30% for its scoring.
    # Paper segmentation (A.5) is currently disabled as a forward step:
    # the bbox-based crop was dropping content the trace pipeline needs
    # (HR84 was getting only 2.4 s of signal, below the 3 s DWT minimum).
    # Background filtering happens later via _band_traceness instead.
    img = _resolve_orientation(img_full)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE: boost local contrast for phone-camera images before thresholding
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Detect grid
    grid_mask = _detect_grid_mask(img)

    # Measure grid spacing for calibration
    px_per_mm_x, px_per_mm_y = _detect_grid_spacing(grid_mask)

    # Extract trace (returns Y-pixel positions, inverted, centered at 0)
    trace = _extract_trace(gray, grid_mask)

    # Auto-select one lead row. Two strategies:
    #   1. Filter bands via _select_best_row (heuristic — fast).
    #   2. If we have multiple plausible bands, also try each via R-peak detection
    #      and pick the band whose extracted signal looks most like a regular
    #      ECG (period in the 30-200 bpm range, low RR variability).
    bands = _detect_lead_rows(trace)
    n_rows_found = len(bands)
    selected_row = None
    if len(bands) >= 2:
        # Pre-filter: drop bands that fail the basic noise gate, keep up to 5 best.
        plausible = []
        for s, e in bands:
            if (e - s) < max(60, gray.shape[0] // 30):
                continue
            tr = _band_traceness(trace, s, e)
            if tr < 0.30:
                continue
            plausible.append((s, e, tr))
        plausible.sort(key=lambda t: t[2], reverse=True)
        plausible = plausible[:5]

        # A.7.2 OCR-based lead routing — DISABLED after empirical evaluation.
        # The helpers (_read_band_label, _pick_by_label, _normalize_lead_name)
        # remain available for future iterations, but routing by label hurts
        # accuracy on the current fixtures: Tesseract finds multi-letter
        # labels (aVR, aVF, V4) on the FX-8200 *header* band as often as
        # on real ECG rows, and even when it picks a true label the chosen
        # band isn't necessarily the one with the cleanest R-peak train.
        # Re-enabling will require either (a) excluding header bands more
        # robustly, or (b) combining label evidence with R-peak regularity
        # rather than letting label priority win outright.
        # If only one survived, use it.
        if len(plausible) == 1:
            selected_row = (plausible[0][0], plausible[0][1])
        elif len(plausible) >= 2:
            # Score each by R-peak regularity. Pick the most physiological.
            best_score = -np.inf
            for s, e, tr in plausible:
                strip = trace[s:e, :].copy()
                tcols = int(strip.shape[1] * 0.18)
                if tcols > 0:
                    strip[:, :tcols] = 0
                sig = _trace_to_signal(strip, use_weighted=True)
                if len(sig) < 100:
                    continue
                # Cheap R-peak detection: find peaks above 0.5 std with min spacing
                from scipy.signal import find_peaks as _find_peaks
                z = (sig - sig.mean()) / (sig.std() + 1e-9)
                pks, _ = _find_peaks(z, height=0.8, distance=max(20, len(sig) // 50))
                if len(pks) < 3:
                    continue
                rr = np.diff(pks)
                rr_mean = float(np.mean(rr))
                rr_std = float(np.std(rr))
                rr_cv = rr_std / max(rr_mean, 1)  # lower is more regular
                # Approx HR if px_per_mm_x known
                if px_per_mm_x and rr_mean > 0:
                    hr_est = 60.0 * px_per_mm_x * paper_speed / rr_mean
                else:
                    hr_est = 60.0 * len(sig) / 10.0 / rr_mean if rr_mean > 0 else 0
                # Reject implausible HRs
                if not (30 <= hr_est <= 220):
                    continue
                score = tr * 0.4 + (1.0 - min(rr_cv, 1.0)) * 0.6
                if score > best_score:
                    best_score = score
                    selected_row = (s, e)
        if selected_row is None:
            # fall back to the heuristic if scoring rejected everything
            row_start, row_end = _select_best_row(trace, bands)
            selected_row = (row_start, row_end)

        pad = 8
        row_start = max(0, selected_row[0] - pad)
        row_end = min(gray.shape[0], selected_row[1] + pad)
        gray      = gray[row_start:row_end, :]
        trace     = trace[row_start:row_end, :]
        grid_mask = grid_mask[row_start:row_end, :]
        selected_row = (row_start, row_end)
    elif len(bands) == 1:
        selected_row = bands[0]

    # Crop preview strip BEFORE text-block masking so user sees the actual lead content.
    # Preview is always set: cropped band when detection found something, full image otherwise.
    if selected_row is not None:
        r0, r1 = selected_row
        preview_strip_rgb = cv2.cvtColor(img[r0:r1, :], cv2.COLOR_BGR2RGB)
    else:
        preview_strip_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mask left text block (FX-8200 and similar machines print measurements at left edge)
    text_cols = int(trace.shape[1] * 0.18)
    if text_cols > 0:
        trace[:, :text_cols] = 0

    signal_px = _trace_to_signal(trace, use_weighted=True)

    # Task C: suppress residual vertical-grid-line dips that DWT delineation
    # misidentifies as P-onsets, inflating PR by 100-200 ms.
    signal_px = _suppress_grid_crossing_artifacts(
        signal_px, grid_mask, text_cols=text_cols
    )

    if len(signal_px) == 0:
        raise ValueError("Signal extraction produced empty array — image may be blank or unreadable")

    # PMcardio-style auto-polarity flip. R-peaks should deflect *more* than
    # S-waves; if the negative excursion dominates the median by 1.4× the
    # positive one, the trace is inverted (rotated band, or aVR which is
    # naturally negative). Applied before calibration so the sign carries
    # through to mV and to NeuroKit2 downstream.
    if len(signal_px) > 1:
        med = float(np.median(signal_px))
        neg_excursion = abs(float(signal_px.min()) - med)
        pos_excursion = abs(float(signal_px.max()) - med)
        if neg_excursion > 1.4 * pos_excursion:
            signal_px = -signal_px

    # ── Amplitude calibration ─────────────────────────────────────────────
    # Each pixel in Y corresponds to: 1 / (px_per_mm_y * mm_per_mv) mV
    if px_per_mm_y is not None and px_per_mm_y > 0:
        signal_mv = signal_px / (px_per_mm_y * mm_per_mv)
    else:
        # Fallback: assume the full pixel height = ±2 mV (rough estimate)
        img_height = max(trace.shape[0], 1)  # guard against zero-height strip
        signal_mv = signal_px / (img_height / 4.0)

    # ── Time axis calibration ─────────────────────────────────────────────
    # Each pixel column = 1 / (px_per_mm_x * paper_speed) seconds
    # → actual_fs = px_per_mm_x * paper_speed  (samples per second)
    if px_per_mm_x is not None and px_per_mm_x > 0:
        actual_fs = px_per_mm_x * paper_speed
    else:
        # Fallback: assume the image width covers 10 seconds
        actual_fs = len(signal_mv) / 10.0

    if actual_fs <= 0:
        raise ValueError(
            f"Could not determine sampling rate (actual_fs={actual_fs}). "
            "Grid detection failed and fallback could not recover — image may be unreadable."
        )

    # De-jitter before resampling: one pixel of column-to-column noise at ~250 px/s
    # source rate lands as >80 Hz garbage after resample, which DWT delineation reads
    # as fake P/T offsets. Low-pass at 40 Hz in the source-rate domain kills it.
    signal_mv = _lowpass_40hz(signal_mv, actual_fs)

    # Polyphase resample (time-domain) — FFT resample rings on QRS edges.
    ratio = Fraction(TARGET_FS, max(1, int(round(actual_fs)))).limit_denominator(1000)
    signal_resampled = resample_poly(signal_mv, ratio.numerator, ratio.denominator)
    if len(signal_resampled) < 100:
        # Safety floor — linear-interp back up if the ratio collapsed the signal
        signal_resampled = np.interp(
            np.linspace(0, len(signal_mv) - 1, 100),
            np.arange(len(signal_mv)),
            signal_mv,
        )

    # Remove baseline drift (after resampling, fs is now TARGET_FS)
    signal_out = _remove_baseline_drift(signal_resampled, fs=TARGET_FS, cutoff=0.5)

    quality = _compute_quality_score(trace)

    if debug:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original")
        axes[0, 1].imshow(grid_mask, cmap="gray")
        axes[0, 1].set_title(f"Grid Mask (px/mm x={px_per_mm_x:.1f} y={px_per_mm_y:.1f})"
                              if px_per_mm_x else "Grid Mask (calibration failed)")
        axes[1, 0].imshow(trace, cmap="gray")
        axes[1, 0].set_title("Cleaned Trace")
        axes[1, 1].plot(np.arange(len(signal_out)) / TARGET_FS, signal_out,
                        color="#CC0000", linewidth=0.7)
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("mV")
        axes[1, 1].set_title(f"Calibrated Signal (quality: {quality:.2f})")
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {
        "signal":              signal_out,
        "quality":             quality,
        "actual_fs":           actual_fs,
        "px_per_mm":           (px_per_mm_x, px_per_mm_y),
        "n_rows_found":        n_rows_found,
        "selected_row":        selected_row,
        "image_landscape_rgb": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        "preview_strip_rgb":   preview_strip_rgb,
    }


def extract_multi_lead(image_path, n_leads=None):
    """
    Extract multiple leads from a multi-strip EKG image.
    Detects horizontal strips by finding large horizontal gaps
    in the trace, then extracts each strip separately.

    Args:
        image_path: path to the image
        n_leads: expected number of leads (auto-detect if None)

    Returns:
        list of dicts: [{"signal": array, "y_range": (top, bottom), "quality": float}]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grid_mask = _detect_grid_mask(img)
    trace = _extract_trace(gray, grid_mask)

    rows, cols = trace.shape

    # Find horizontal strip boundaries by looking at row-wise trace density
    row_density = np.sum(trace > 0, axis=1).astype(float)

    # Smooth the density to find clear strips
    kernel_size = max(5, rows // 50)
    row_density_smooth = np.convolve(row_density, np.ones(kernel_size) / kernel_size, mode="same")

    # Threshold: rows with trace vs empty rows
    thresh = np.mean(row_density_smooth) * 0.2
    has_trace = row_density_smooth > thresh

    # Find contiguous strip regions
    strips = []
    in_strip = False
    strip_start = 0

    for y in range(rows):
        if has_trace[y] and not in_strip:
            strip_start = y
            in_strip = True
        elif not has_trace[y] and in_strip:
            strip_height = y - strip_start
            if strip_height > rows * 0.05:  # at least 5% of image height
                strips.append((strip_start, y))
            in_strip = False

    if in_strip:
        strip_height = rows - strip_start
        if strip_height > rows * 0.05:
            strips.append((strip_start, rows))

    if not strips:
        # No clear strips found — treat whole image as one lead
        signal = _trace_to_signal(trace, use_weighted=True)
        signal = _remove_baseline_drift(signal, fs=500)
        quality = _compute_quality_score(trace)
        return [{"signal": signal, "y_range": (0, rows), "quality": quality}]

    # Extract signal from each strip
    results = []
    for y_top, y_bottom in strips:
        strip_trace = trace[y_top:y_bottom, :]
        signal = _trace_to_signal(strip_trace, use_weighted=True)
        signal = _remove_baseline_drift(signal, fs=500)
        quality = _compute_quality_score(strip_trace)
        results.append({
            "signal": signal,
            "y_range": (y_top, y_bottom),
            "quality": quality,
        })

    return results


if __name__ == "__main__":
    print("Digitization Pipeline v2 Ready.")
    print("  extract_signal_from_image(path) -> 1D signal")
    print("  extract_multi_lead(path) -> list of lead signals")
