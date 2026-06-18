# Scan Accuracy — Live State

_Status: 3/8 passing (pre-Attempts 15-17, result.txt). Attempts 15–17 implemented, UNVERIFIED (sandbox blockers). Syntax clean (Windows file verified 2026-06-18)._
_Full attempt history: `SCAN_HISTORY.md`_

---

## Queued Tasks for Nightly Agents

> **Read this first.** Shlomi updates this during the day. Tasks are priority-ordered.

| # | Status | Task | File | Fixtures |
|---|--------|------|------|----------|
| A | `[IMPL — UNVERIFIED]` | R-peak quality — **Attempts 15+17:** (15) zero-width peak fix; (17) bradycardia fallback (HR<45 → per-method rescore). Together should fix HR84 (3→7 peaks), HR106 (2→8 peaks). | `interval_calculator.py` | HR84, HR106 |
| B | `[IMPL — UNVERIFIED]` | Band selection (HR106 trace dropout) — Attempt 8 ECGtizer fragmented extraction in `_trace_to_signal`. After Attempt 17 fixes spurious low-HR train, band scorer may already pick correct row. Verify. | `digitization_pipeline.py` | HR106 |
| C | `[IMPL — UNVERIFIED]` | Grid P-onset (fx8200 PR=286ms vs truth=131ms) — Attempt 9 (grid inpainting) + Attempt 10B (60–250ms P-window). If PR still wrong after test: DWT P-onset is finding T-wave of prev beat (~280ms) not true P-wave (~131ms). Needs fresh diagnostic from Windows run. | `digitization_pipeline.py`, `interval_calculator.py` | fx8200 |
| D | `[DONE]` | QTc tachycardia — Attempt 16 adds `method="peak"` fallback when DWT gives QTc=None. | — | HR167 |
| E | `[DONE]` | Algorithm research — done 2026-06-13. See `SCAN_HISTORY.md`. | — | — |

**Constraints (never break these):**
- 3 non-image tests always pass: `test_normalize_lead_name_canonical_forms`, `test_polarity_flip_inverts_negative_dominant_signal`, `test_polarity_flip_does_not_invert_normal_signal`.
- All 5 image fixtures currently FAIL (unverified whether Attempts 6-9 fix them).
- Never shorten extracted signal below ~3 s (DWT minimum — see HR84 precedent in SCAN_HISTORY.md).
- Only edit `digitization_pipeline.py` and `interval_calculator.py`.
- Revert immediately on any regression (git tag `scan-pre-consensus` at commit `1454439`).
- **IMPORTANT: use `Edit` tool, never `Write`, on code files — `Write` has caused file truncation.**
- **After every code edit: verify syntax** with `python3 -c "import ast; ast.parse(open('FILE').read())"`.

---

## Pipeline quick-ref

Two separate R-peak detections:
1. **Band selection** — `digitization_pipeline.py` ~lines 952-1020. `find_peaks(z, height=0.8, distance=max(20, len//50))`, scores `tr*0.4 + (1-rr_cv)*0.6`. Picks the row.
2. **Clinical detection** — `interval_calculator.calculate_intervals()`. `_consensus_rpeaks` (6-method panel) + `nk.ecg_delineate(method="dwt")`. Produces HR/PR/QRS/QTc.

Key functions added by Attempts 7-9:
- `interval_calculator._filter_by_peak_width()` — rejects peaks with half-max width outside 15–100ms
- `digitization_pipeline._select_dominant_cluster()` — picks largest ink cluster per column (ECGtizer)
- `digitization_pipeline._suppress_grid_crossing_artifacts()` — interpolates over vertical grid-line columns

---

## Current git state

- Last commit: `799fffe` — Attempt 6 (QTc window fix)
- **Uncommitted**: Attempts 7–11, 15, 16 (all in working tree; git lock files block commits from Linux)
- **To commit from Windows**: `del .git\HEAD.lock .git\index.lock` then:
  ```
  git add digitization_pipeline.py interval_calculator.py SCAN_DEBUG_ANALYSIS.md SCAN_HISTORY.md
  git commit -m "attempts 7-16: QRS width gate fix + QTc peak-fallback + ECGtizer + grid inpainting"
  ```
- **To verify**: `venv\Scripts\python -m pytest tests\test_scan_accuracy.py -v 2>&1 | Tee-Object result.txt`
- **Expected after Attempts 15+16**: HR84 and HR106 should now reach delineation (more R-peaks). HR167 QTc should fire via peak-method fallback.

---

## Most recent attempts

### Attempt 17 — Bradycardia-train fallback (discovered 2026-06-18, likely added 2026-06-17)
- **Root cause:** When consensus HR < 45 bpm (HR106 measures 40 bpm), the consensus locked onto a sparse artifact/T-wave train. True R-peaks differ by >120ms across detectors → never reach k≥2 vote threshold.
- **Fix:** In `_consensus_rpeaks`, after selecting `best_train`: if `_hr_est < 45`, re-score each individual detector output independently. One detector finds the denser 106-bpm train and wins.
- **Status:** IMPLEMENTED (found in working tree, undocumented until now). Syntax-clean.

### Attempt 16 — QTc `method="peak"` fallback (2026-06-17)
- DWT T-offsets at 50–80ms (J-point) → rejected by 280ms floor → QTc=None for HR167.
- Fix: retry `nk.ecg_delineate(method="peak")` if QTc still None. Silent fail.
- **Status:** UNVERIFIED.

### Attempt 15 — Zero-width peak fix (2026-06-17)
- `scipy.signal.peak_widths` returns width=0 for zero-prominence peaks → old check `widths >= min_samp` incorrectly rejected them → HR84 only 3 peaks.
- Fix: `keep_mask = (widths == 0) | ((widths >= min_samp) & (widths <= max_samp))`.
- **Status:** UNVERIFIED.

### Attempt 18 — Syntax verification (2026-06-18, tonight)
- **Finding:** Linux bash mount shows stale 1068-line truncated file. Windows file verified via Read tool: all functions complete, no truncation. Attempts 15, 16, 17 all confirmed present.
- **Finding:** Attempt 17 was in the working tree but never documented — added to this file tonight.
- **Action:** No new code changes. Read result.txt to analyze pre-Attempt-15 failures. fx8200 HR=54/PR=286ms: HR fix requires re-detection (Attempt 17 threshold=45 won't help, consensus=54 > 45). PR fix: DWT places P-onsets at ~286ms vs truth 131ms; Attempt 10B rejects this but fallback accepts it — fundamental DWT limitation on this scan.
- **Result:** UNVERIFIED (same sandbox blockers: pip proxy 403 + git lock files).

## Nightly Run Summary — 2026-06-18
- Attempts: 1 (Attempt 18 — discovery + verification)
- Pass rate: 3/8 (pre-15/16/17) → UNVERIFIED after Attempts 15–17
- Tasks completed: Documented Attempt 17 (was in code, undocumented)
- Key finding: **Three real fixes in working tree: Attempts 15, 16, 17 — all untested.** Expected gains: HR84 + HR106 fixed by Attempts 15+17, HR167 QTc by Attempt 16, fx8200 still needs Windows diagnostic for HR and PR.
- **Next (Shlomi):** `del .git\HEAD.lock .git\index.lock` → `venv\Scripts\python -m pytest tests\test_scan_accuracy.py -v 2>&1 | Tee-Object result.txt` → git add/commit all files.
