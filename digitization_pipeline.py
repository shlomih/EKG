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
            # When pixels are very spread out (multiple traces / row-boundary noise),
            # weights underflow to all-zero — fall back to median rather than divide-by-zero.
            if weights.sum() > 1e-12:
                signal[x] = np.average(col_pixels, weights=weights)
            else:
                signal[x] = np.median(col_pixels)
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


def _select_best_row(trace_binary, bands):
    """
    Select the best single lead row to analyze.
    Strongly biased toward the bottom-most band with reasonable coverage —
    standard 12-lead paper layouts always print the rhythm strip at the bottom,
    and the rhythm strip is what gives clean HR/PR/QRS/QT measurements (the
    cell rows above are 4 short ~2.5s clips of different leads spliced
    side-by-side, which produce step jumps in the extracted 1D signal).
    Returns (y_start, y_end) of the best band.
    """
    rows = trace_binary.shape[0]
    best, best_score = bands[0], -1.0
    for start, end in bands:
        strip = trace_binary[start:end, :]
        coverage = float((strip.sum(axis=0) > 0).mean())
        # Bottom-preference: the rhythm strip is always at the bottom of the page.
        bottom_bonus = (start + end) / (2.0 * rows)  # 0 at top, 1 at bottom
        # Bottom-bias dominates so the rhythm strip wins even when cell rows have
        # comparable column-coverage. Coverage stays as a guard against picking a
        # near-empty band that happens to be at the bottom.
        score = coverage * 0.40 + bottom_bonus * 0.60
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

    # Largest connected blob in the down-sampled density map.
    nl, lbls, stats, _ = cv2.connectedComponentsWithStats(paper_blocks, 8)
    if nl <= 1:
        return img_bgr, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(areas.argmax()) + 1
    if stats[biggest, cv2.CC_STAT_AREA] < small_h * small_w * 0.03:
        return img_bgr, None

    # Use the connected-component bbox directly — axis-aligned, no warp.
    # A perspective-warped minAreaRect adds rotation skew that fights with
    # A.6's orientation logic; an axis-aligned crop is a clean handoff to
    # A.6 (which then handles 0/90/180/270° label-up resolution).
    bx, by, bw, bh, _ = stats[biggest]
    x0 = bx * block
    y0 = by * block
    x1 = min(w, (bx + bw) * block)
    y1 = min(h, (by + bh) * block)
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
    candidates = {
        0:   paper_bgr,
        90:  cv2.rotate(paper_bgr, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(paper_bgr, cv2.ROTATE_180),
        270: cv2.rotate(paper_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    # Score every rotation; do NOT pre-filter portrait shapes — the
    # cropped paper region can be roughly square (FX-8200 reference is
    # 629x703 ≈ square), in which case every rotation is technically
    # portrait or landscape and pre-filtering would drop the correct
    # candidate. The score itself penalizes wrong orientations: wrong
    # rotations produce vertical "trace" pixels whose median Y stays
    # near the band centre, giving low high-low scores.
    best_angle, best_score = 0, -np.inf
    for angle, candidate in candidates.items():
        score = _orientation_score(candidate)
        if score > best_score:
            best_score = score
            best_angle = angle
    return candidates[best_angle]


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

    # A.5 — segment the paper region from background (desk, wrist rest, hands).
    # If detection fails, fall back to using the whole image.
    paper_img, paper_bbox = _detect_paper_region(img_full)

    # A.6 — resolve which of the 4 cardinal rotations puts the paper
    # right-side-up (R-peaks pointing up, trace flowing left-to-right).
    # This subsumes the old shape-based "if portrait, rotate 90° CW" check.
    img = _resolve_orientation(paper_img)

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
    elif len(bands) == 1:
        # Single-band path — record the band so the user still sees what was analyzed
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
