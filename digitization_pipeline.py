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
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

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


def _trace_to_signal(trace_binary, use_weighted=True):
    """
    Convert binary trace image to 1D signal.
    Uses weighted median of black pixel positions per column
    for smoother extraction.
    """
    rows, cols = trace_binary.shape
    signal = np.full(cols, np.nan)

    for x in range(cols):
        col_pixels = np.where(trace_binary[:, x] > 0)[0]
        if len(col_pixels) == 0:
            continue

        if use_weighted and len(col_pixels) >= 3:
            # Weight by distance from center of mass
            center = np.mean(col_pixels)
            weights = np.exp(-0.5 * ((col_pixels - center) / (len(col_pixels) * 0.3 + 1)) ** 2)
            signal[x] = np.average(col_pixels, weights=weights)
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


def _detect_lead_rows(trace_binary, min_row_height_ratio=0.05):
    """
    Find horizontal bands where ECG traces exist.
    Returns list of (y_start, y_end) tuples, one per detected lead row.
    """
    from scipy.ndimage import uniform_filter1d
    rows = trace_binary.shape[0]
    row_sum = trace_binary.sum(axis=1).astype(float)
    smoothed = uniform_filter1d(row_sum, size=max(5, rows // 40))
    peak = smoothed.max()
    if peak == 0:
        return []
    in_band = smoothed > peak * 0.10
    bands = []
    start = None
    for i, v in enumerate(in_band):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if (i - start) >= rows * min_row_height_ratio:
                bands.append((start, i))
            start = None
    if start is not None and (rows - start) >= rows * min_row_height_ratio:
        bands.append((start, rows))
    return bands


def _select_best_row(trace_binary, bands):
    """
    Select the best single lead row to analyze.
    Scores by trace coverage and height (height bonus favors the wide rhythm strip).
    Returns (y_start, y_end) of the best band.
    """
    rows = trace_binary.shape[0]
    best, best_score = bands[0], -1.0
    for start, end in bands:
        strip = trace_binary[start:end, :]
        coverage = float((strip.sum(axis=0) > 0).mean())
        width_bonus = (end - start) / rows
        # Bottom-preference: the rhythm strip is always at the bottom of the page
        bottom_bonus = (start + end) / (2.0 * rows)  # 0 at top, 1 at bottom
        # Coverage dominates; width and position break ties between similar-coverage bands.
        score = coverage * 0.80 + width_bonus * 0.10 + bottom_bonus * 0.10
        if score > best_score:
            best_score = score
            best = (start, end)
    return best


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
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Auto-rotate portrait to landscape — ECG paper is always wider than tall
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

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

    # Auto-select one lead row — prevents mixing signals from multiple rows
    bands = _detect_lead_rows(trace)
    n_rows_found = len(bands)
    selected_row = None
    if len(bands) > 1:
        row_start, row_end = _select_best_row(trace, bands)
        pad = 8
        row_start = max(0, row_start - pad)
        row_end = min(gray.shape[0], row_end + pad)
        gray      = gray[row_start:row_end, :]
        trace     = trace[row_start:row_end, :]
        grid_mask = grid_mask[row_start:row_end, :]
        selected_row = (row_start, row_end)

    # Crop preview strip BEFORE text-block masking so user sees the actual lead content
    preview_strip_rgb = None
    if selected_row is not None:
        r0, r1 = selected_row
        preview_strip_rgb = cv2.cvtColor(img[r0:r1, :], cv2.COLOR_BGR2RGB)

    # Mask left text block (FX-8200 and similar machines print measurements at left edge)
    text_cols = int(trace.shape[1] * 0.18)
    if text_cols > 0:
        trace[:, :text_cols] = 0

    signal_px = _trace_to_signal(trace, use_weighted=True)

    if len(signal_px) == 0:
        raise ValueError("Signal extraction produced empty array — image may be blank or unreadable")

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
