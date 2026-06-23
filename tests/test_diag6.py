"""Diagnostic test — per-detector output for HR84 and HR106. Nightly 2026-06-23."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import neurokit2 as nk
import pytest

from digitization_pipeline import extract_signal_from_image
from interval_calculator import calculate_intervals

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_METHODS = [
    "neurokit",
    "pantompkins1985",
    "hamilton2002",
    "elgendi2010",
    "engzeemod2012",
    "kalidas2017",
    "rodrigues2021",
]

CASES = [
    ("HR84",  "tests/fixtures/varied_HR84_reference.jpg",  500,  84, 4),
    ("HR106", "tests/fixtures/varied_HR106_reference.jpg", 500, 106, 5),
]


def _get_signal(relpath):
    imgpath = os.path.join(_PROJ, relpath)
    res = extract_signal_from_image(imgpath, paper_speed=25, mm_per_mv=10)
    return res["signal"], res.get("actual_fs", 500)


def test_per_detector_analysis():
    """Report per-detector R-peak trains for HR84 and HR106."""
    for name, relpath, fs, truth_hr, tol in CASES:
        sig, actual_fs = _get_signal(relpath)
        cleaned = nk.ecg_clean(sig, sampling_rate=fs)
        print(f"\n{'='*60}")
        print(f"=== {name} | actual_fs={actual_fs:.0f} | sig_len={len(sig)} ===")
        print(f"    Truth HR={truth_hr}, tol={tol}")
        print(f"    Cleaned range: [{cleaned.min():.3f}, {cleaned.max():.3f}]")

        best_hr_close = 999
        best_method = None
        best_peaks = None

        for method in _METHODS:
            try:
                _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method=method)
                pks = np.asarray(info["ECG_R_Peaks"], dtype=float)
                pks = pks[np.isfinite(pks)].astype(int)
                if len(pks) < 2:
                    print(f"  {method:20s}: {len(pks)} peaks — too few")
                    continue
                rr = np.diff(pks) / fs * 1000
                hr = 60000 / np.median(rr)
                diff = abs(hr - truth_hr)
                ok = "PASS" if diff <= tol else "FAIL"
                print(f"  {method:20s}: {len(pks)} peaks | HR={hr:.1f} ({ok} diff={diff:.1f}) | peaks={list(pks[:6])}")
                if diff < best_hr_close:
                    best_hr_close = diff
                    best_method = method
                    best_peaks = pks
            except Exception as e:
                print(f"  {method:20s}: ERROR {e}")

        print(f"  Closest method: {best_method} (diff={best_hr_close:.1f} from truth)")

        # Show signal values at best-detector peaks
        if best_peaks is not None:
            print(f"  Best peak amplitudes: {[round(float(cleaned[p]), 3) for p in best_peaks[:8]]}")

        # Standard pipeline result
        intervals = calculate_intervals(sig, sampling_rate=fs)
        print(f"  Standard pipeline: r_peaks={intervals.get('r_peaks')}, HR={intervals.get('hr')}")
        print(f"  Warnings: {intervals.get('warnings', [])}")
    assert True
