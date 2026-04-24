"""Tests for digitization_pipeline.py — run with: python -m pytest tests/test_digitization.py -v

Uses synthetic binary-trace images so tests don't depend on real ECG photos.
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from digitization_pipeline import _detect_lead_rows, _select_best_row


def make_band_image(height: int, width: int, bands: list[tuple[int, int, float]]) -> np.ndarray:
    """Build a (H, W) uint8 binary image with horizontal trace bands.

    Each band is (y_start, y_end, coverage_fraction). Coverage is the fraction of
    columns in that band that receive trace pixels (distributed evenly).
    """
    img = np.zeros((height, width), dtype=np.uint8)
    for y0, y1, cov in bands:
        if cov <= 0:
            continue
        n_cols = max(1, int(width * cov))
        cols = np.linspace(0, width - 1, n_cols).astype(int)
        img[y0:y1, cols] = 255
    return img


# ── _detect_lead_rows ──────────────────────────────────────────────────────

def test_detect_rows_empty_image_returns_empty():
    img = np.zeros((500, 1000), dtype=np.uint8)
    assert _detect_lead_rows(img) == []


def test_detect_rows_single_band():
    img = make_band_image(500, 1000, [(200, 260, 1.0)])
    bands = _detect_lead_rows(img)
    assert len(bands) == 1
    s, e = bands[0]
    # Smoothing can shift boundaries slightly — allow ±10 px slack
    assert abs(s - 200) <= 15
    assert abs(e - 260) <= 15


def test_detect_rows_three_bands_with_gaps():
    # 4-row layout: 3 narrow upper rows + 1 wide rhythm strip at bottom
    img = make_band_image(600, 1000, [
        (50,  90,  0.9),
        (180, 220, 0.9),
        (310, 350, 0.9),
        (480, 560, 0.95),
    ])
    bands = _detect_lead_rows(img)
    assert len(bands) == 4, f"expected 4 bands, got {len(bands)}: {bands}"


def test_detect_rows_filters_tiny_bands():
    # A 2-pixel band should be below the min-height filter (5% of 500 = 25)
    img = make_band_image(500, 1000, [(100, 102, 1.0), (200, 260, 1.0)])
    bands = _detect_lead_rows(img)
    assert len(bands) == 1  # the tiny one is filtered out


# ── _select_best_row ──────────────────────────────────────────────────────

def test_select_best_prefers_highest_coverage():
    # Two bands same width, different coverage — higher-coverage band wins
    img = make_band_image(500, 1000, [(100, 160, 0.5), (300, 360, 1.0)])
    bands = [(100, 160), (300, 360)]
    best = _select_best_row(img, bands)
    assert best == (300, 360)


def test_select_best_rhythm_strip_beats_multicolumn():
    # Simulate FX-8200-style layout: narrow bands with 0.7 coverage (two leads per row,
    # gap in the middle) vs. wider rhythm strip with 0.95 coverage.
    img = make_band_image(600, 1000, [
        (50,  90,  0.70),   # 2-column row: I + aVR side by side with gap
        (180, 220, 0.70),   # II + aVL
        (310, 350, 0.70),   # III + aVF
        (450, 540, 0.95),   # rhythm strip (wider + full coverage)
    ])
    bands = _detect_lead_rows(img)
    best = _select_best_row(img, bands)
    # Rhythm strip starts ~450, well below y=400 — must NOT be one of the top three rows
    assert best[0] >= 400, f"selected {best} — should be the bottom rhythm strip"


def test_select_best_bottom_tiebreak():
    # Two bands with identical coverage and width — bottom one should win via bottom-preference
    img = make_band_image(500, 1000, [(50, 110, 0.90), (350, 410, 0.90)])
    bands = [(50, 110), (350, 410)]
    best = _select_best_row(img, bands)
    assert best == (350, 410)


def test_select_best_single_band_returns_it():
    img = make_band_image(500, 1000, [(200, 260, 0.95)])
    bands = [(200, 260)]
    best = _select_best_row(img, bands)
    assert best == (200, 260)
