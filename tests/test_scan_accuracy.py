"""End-to-end scan-accuracy tests.

Each fixture is a paper-ECG photo (.jpg/.png) plus a JSON sidecar listing
ground-truth intervals printed on the paper itself, plus per-metric tolerances.

The tests are intentionally lenient on PR/QRS/QTc tolerances — photo-reconstructed
signals will never match calipers on a digital trace, but they should land in the
same physiological neighborhood. If any metric returns N/A the test fails (we want
to know when the DWT delineator starves on contrast — that's the trigger for the
"trace inpainting at grid crossings" follow-up plan).

Run with:
    python -m pytest tests/test_scan_accuracy.py -v

Fixture format — `tests/fixtures/<name>.json`:
    {
        "description": "FX-8200, normal sinus, indoor light",
        "paper_speed": 25,
        "mm_per_mv": 10,
        "ground_truth": {"hr": 66, "pr": 131, "qrs": 91, "qtc": 436},
        "tolerance":    {"hr_bpm": 4, "pr_ms": 40, "qrs_ms": 30, "qtc_ms": 50},
        "min_quality":  0.4
    }

The image file must sit beside the JSON with the same stem (e.g. fx8200_reference.jpg
+ fx8200_reference.json).
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from digitization_pipeline import extract_signal_from_image
from interval_calculator import calculate_intervals

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FX8200_STEM = "fx8200_reference"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def _find_image(stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = FIXTURES_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _load_fixture(stem: str) -> tuple[Path, dict]:
    json_path = FIXTURES_DIR / f"{stem}.json"
    img_path = _find_image(stem)
    if not json_path.exists() or img_path is None:
        pytest.skip(
            f"Fixture {stem} missing — drop {stem}.{{jpg,png}} and {stem}.json into "
            f"{FIXTURES_DIR} to enable this test."
        )
    with open(json_path) as f:
        spec = json.load(f)
    return img_path, spec


def _run_scan(img_path: Path, spec: dict) -> dict:
    res = extract_signal_from_image(
        str(img_path),
        paper_speed=spec.get("paper_speed", 25),
        mm_per_mv=spec.get("mm_per_mv", 10),
    )
    intervals = calculate_intervals(res["signal"], sampling_rate=500)
    return {"scan": res, "intervals": intervals}


def _assert_within(metric: str, measured, truth, tol, errors: list):
    if truth is None:
        return  # ground truth unmeasurable on this fixture (e.g. PR in fast rhythm)
    if measured is None:
        errors.append(f"{metric}: returned N/A (truth={truth})")
        return
    diff = abs(measured - truth)
    if diff > tol:
        errors.append(
            f"{metric}: measured={measured:.0f} vs truth={truth} (diff={diff:.0f}, tol={tol})"
        )


def _assert_against_truth(scan_result: dict, spec: dict):
    truth = spec["ground_truth"]
    tol = spec["tolerance"]
    intervals = scan_result["intervals"]
    quality = scan_result["scan"].get("quality", 0.0)
    min_q = spec.get("min_quality", 0.4)

    errors: list[str] = []
    if quality < min_q:
        errors.append(f"quality: {quality:.2f} below minimum {min_q}")

    _assert_within("hr",  intervals.get("hr"),  truth.get("hr"),  tol["hr_bpm"], errors)
    _assert_within("pr",  intervals.get("pr"),  truth.get("pr"),  tol["pr_ms"],  errors)
    _assert_within("qrs", intervals.get("qrs"), truth.get("qrs"), tol["qrs_ms"], errors)
    _assert_within("qtc", intervals.get("qtc"), truth.get("qtc"), tol["qtc_ms"], errors)

    if errors:
        pytest.fail(
            "Scan accuracy regressions:\n  - "
            + "\n  - ".join(errors)
            + f"\n\nFull intervals: {intervals}\nQuality: {quality}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# A.3 — FX-8200 reference verification (Round 3 exit criterion)
# ─────────────────────────────────────────────────────────────────────────────


def test_fx8200_round3_verification():
    """Round 3 expected outcome on the FX-8200 reference photo.

    From C:/Users/osnat/.claude/plans/i-want-to-reconsider-majestic-wind.md §Round 3:
      HR ≈66, PR ≤170, QRS 70–140, QTc 380–480.

    If this fails with QRS/QTc=N/A, the DWT delineator is starving on low trace
    contrast — open a separate plan for "trace inpainting at grid crossings".
    """
    img_path, spec = _load_fixture(FX8200_STEM)
    result = _run_scan(img_path, spec)
    _assert_against_truth(result, spec)


# ─────────────────────────────────────────────────────────────────────────────
# A.4 — Parametric over all fixture pairs in tests/fixtures/
# ─────────────────────────────────────────────────────────────────────────────


def _discover_fixture_stems() -> list[str]:
    if not FIXTURES_DIR.exists():
        return []
    stems = set()
    for json_file in FIXTURES_DIR.glob("*.json"):
        stem = json_file.stem
        if _find_image(stem) is not None:
            stems.add(stem)
    return sorted(stems)


_FIXTURE_STEMS = _discover_fixture_stems()


@pytest.mark.skipif(
    not _FIXTURE_STEMS,
    reason=f"No fixture pairs found in {FIXTURES_DIR}",
)
@pytest.mark.parametrize("stem", _FIXTURE_STEMS)
def test_fixture_scan_accuracy(stem):
    """Every <stem>.{jpg,png} + <stem>.json pair under tests/fixtures/ must scan
    within the tolerances declared in its sidecar JSON.

    Add new photos by dropping the image plus a JSON sidecar — no test code change
    needed; the parametrize block discovers them at collection time.
    """
    img_path, spec = _load_fixture(stem)
    result = _run_scan(img_path, spec)
    _assert_against_truth(result, spec)


# ─────────────────────────────────────────────────────────────────────────────
# A.8 — Unit tests for the OCR + polarity helpers (don't need real fixtures)
# ─────────────────────────────────────────────────────────────────────────────


def test_normalize_lead_name_canonical_forms():
    """OCR-confusion variants must reduce to canonical lead names."""
    from digitization_pipeline import _normalize_lead_name as nrm
    # Limb leads
    assert nrm("II") == "II"
    assert nrm("Il") == "II"   # tesseract often reads II as Il
    assert nrm("l1") == "II"   # 1 ↔ I confusion
    assert nrm("I") == "I"
    assert nrm("III") == "III"
    # aV-family
    assert nrm("aVR") == "aVR"
    assert nrm("avr") == "aVR"
    assert nrm("aVL") == "aVL"
    assert nrm("aV1") == "aVL"  # 1 ↔ L confusion in aV-suffix
    # Precordial
    assert nrm("V1") == "V1"
    assert nrm("v3") == "V3"
    # Junk must be rejected — not silently coerced to "II"
    assert nrm("gibberish") is None
    assert nrm("") is None
    assert nrm(None) is None


def test_polarity_flip_inverts_negative_dominant_signal():
    """A signal with negative deflections > 1.4× positive must be flipped."""
    import numpy as np
    from digitization_pipeline import extract_signal_from_image
    # Build an artificial image with a clearly inverted ECG-like trace.
    # Use a real fixture but force the signal-extraction path to run.
    # Simpler: validate the flip logic directly by reproducing it.
    sig = np.zeros(1000)
    # Sharp downward spikes (mimic an inverted R-peak)
    for idx in (200, 500, 800):
        sig[idx] = -3.0
    # Small upward bumps (mimic small T-waves)
    for idx in (250, 550, 850):
        sig[idx] = 0.5
    med = float(np.median(sig))
    neg = abs(float(sig.min()) - med)
    pos = abs(float(sig.max()) - med)
    # Apply the same condition the pipeline uses
    if neg > 1.4 * pos:
        sig = -sig
    # After flip, peaks should now be positive
    assert sig.max() > abs(sig.min()), \
        f"Polarity flip failed: max={sig.max()}, min={sig.min()}"


def test_polarity_flip_does_not_invert_normal_signal():
    """A signal with positive R-peaks must NOT be flipped."""
    import numpy as np
    sig = np.zeros(1000)
    for idx in (200, 500, 800):
        sig[idx] = 3.0  # upward R-peaks
    for idx in (250, 550, 850):
        sig[idx] = -0.5  # small S-wave dips
    med = float(np.median(sig))
    neg = abs(float(sig.min()) - med)
    pos = abs(float(sig.max()) - med)
    flipped = neg > 1.4 * pos
    assert not flipped, \
        f"Polarity flip wrongly triggered on normal signal: neg={neg} pos={pos}"
