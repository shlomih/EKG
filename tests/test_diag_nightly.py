"""Per-detector diagnostic for HR84 and HR106. Nightly 2026-06-23."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import neurokit2 as nk
from digitization_pipeline import extract_signal_from_image
from interval_calculator import calculate_intervals

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_METHODS = ["neurokit", "pantompkins1985", "hamilton2002",
            "elgendi2010", "engzeemod2012", "kalidas2017", "rodrigues2021"]

CASES = [
    ("HR84",  "tests/fixtures/varied_HR84_reference.jpg",   84, 4),
    ("HR106", "tests/fixtures/varied_HR106_reference.jpg", 106, 5),
]


def test_per_detector_nightly():
    """Per-detector R-peak trains for HR84 and HR106 — 2026-06-23."""
    for name, relpath, truth_hr, tol in CASES:
        imgpath = os.path.join(_PROJ, relpath)
        res = extract_signal_from_image(imgpath, paper_speed=25, mm_per_mv=10)
        sig = res["signal"]
        actual_fs = res.get("actual_fs", 500)
        n_rows = res.get("n_rows_found", "?")
        sel_row = res.get("selected_row", "?")
        cleaned = nk.ecg_clean(sig, sampling_rate=500)
        print(f"\n{'='*65}")
        print(f"=== {name} | actual_fs={actual_fs:.0f} Hz | sig_len={len(sig)} "
              f"| n_rows={n_rows} | sel={sel_row} ===")
        print(f"    sig range: [{sig.min():.3f}, {sig.max():.3f}]  "
              f"cleaned range: [{cleaned.min():.3f}, {cleaned.max():.3f}]")

        best_diff = 9999
        best_method = None

        for method in _METHODS:
            try:
                _, info = nk.ecg_peaks(cleaned, sampling_rate=500, method=method)
                pks = np.asarray(info["ECG_R_Peaks"], dtype=float)
                pks = pks[np.isfinite(pks)].astype(int)
                if len(pks) < 2:
                    print(f"  {method:22s}: {len(pks)} peaks — too few")
                    continue
                rr = np.diff(pks) / 500 * 1000
                hr = 60000 / float(np.median(rr))
                diff = abs(hr - truth_hr)
                ok = "PASS" if diff <= tol else "FAIL"
                print(f"  {method:22s}: n={len(pks):2d} HR={hr:5.1f} "
                      f"({ok} diff={diff:.1f}) peaks={list(pks[:5])}")
                if diff < best_diff:
                    best_diff = diff
                    best_method = method
            except Exception as e:
                print(f"  {method:22s}: ERROR {e}")

        print(f"  → Best: {best_method} (diff={best_diff:.1f} from truth={truth_hr})")

        intervals = calculate_intervals(sig, sampling_rate=500)
        print(f"  → Pipeline: r_peaks={intervals.get('r_peaks')}, "
              f"HR={intervals.get('hr')}, RRs={intervals.get('rr_intervals')}")
    assert True
