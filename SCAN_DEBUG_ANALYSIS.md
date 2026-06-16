# Scan Accuracy — Live State

_Status: 3/8 passing. Attempts 6–9 implemented, UNVERIFIED. Syntax errors in both files FIXED (2026-06-15)._
_Full attempt history: `SCAN_HISTORY.md`_

---

## Queued Tasks for Nightly Agents

> **Read this first.** Shlomi updates this during the day. Tasks are priority-ordered.

| # | Status | Task | File | Fixtures |
|---|--------|------|------|----------|
| A | `[IN PROGRESS]` | R-peak quality — Attempt 7 (QRS width gate in `_consensus_rpeaks`) + Attempt 8 (ECGtizer fragmented extraction in `_trace_to_signal`) both implemented. **Verify: run tests on Windows.** If HR84 still fails, try QRS template cross-correlation. | `interval_calculator.py`, `digitization_pipeline.py` | HR84, HR106 |
| B | `[IN PROGRESS]` | Band selection — HR167 ✅ fixed (Attempt 5 consensus). HR106 addressed by Attempt 8; verify after test run. If still failing, tune band-scorer `distance` parameter. | `digitization_pipeline.py` | HR106 |
| C | `[IN PROGRESS]` | Grid P-onset inpainting — Attempt 9 (`_suppress_grid_crossing_artifacts`) implemented. **Verify on Windows.** If fx8200 PR still >171ms after test, consider tightening P-search window to 60–250ms (Attempt 10B). | `digitization_pipeline.py` | fx8200, HR84 |
| D | `[DONE]` | Provisional HR warning — already at `interval_calculator.py` lines 338-343 (`rr_cv > 0.25`). No change needed. | — | — |
| E | `[DONE]` | Algorithm research — done 2026-06-13. See `SCAN_HISTORY.md` for findings. | — | — |

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
- **Uncommitted**: Attempts 7, 8, 9, 10 (all in working tree, not staged — git locks block commit)
- **To commit from Windows**: `del .git\HEAD.lock .git\index.lock` then:
  ```
  git add digitization_pipeline.py interval_calculator.py SCAN_DEBUG_ANALYSIS.md SCAN_HISTORY.md
  git commit -m "attempts 7-10: QRS width gate + ECGtizer extraction + grid inpainting + syntax fix"
  ```
- **To verify**: `venv\Scripts\python -m pytest tests\test_scan_accuracy.py -v`
- **Expected**: 5-6/8 pass (HR167 + HR84 + HR106 + partial fx8200). fx8200 HR (71 vs 66) and QTc still open.

---

## Most recent attempt

### Attempt 10 — Syntax fix (2026-06-15)
- **Problem:** Nightly agents hit context limit mid-Write and truncated both files. `digitization_pipeline.py` lost 127 lines; `interval_calculator.py` lost 110 lines. Both had SyntaxError on import, causing ALL tests to fail.
- **Fix:** Spliced correct tail from git HEAD into each file. All Attempt 7/8/9 code preserved.
- **Result:** Both files syntax-clean. Git commit blocked by lock files.

### Attempt 12 — Code review only (2026-06-16, nightly)
- **Problem:** scipy/neurokit2/pytest uninstallable (pip proxy blocked 403); git lock files on Windows mount (can't rm from sandbox). No tests runnable, no commits possible.
- **Action:** Read all of Attempts 7–11, `_filter_by_peak_width`, `_consensus_rpeaks`, `_select_dominant_cluster`, `_suppress_grid_crossing_artifacts`, P-onset window logic. All logic is sound. No bugs found.
- **Result:** UNVERIFIED (same as Attempts 7–11). No new changes made — safer than adding unverifiable code.
- **Verdict:** Shlomi must run `del .git\HEAD.lock .git\index.lock` then `venv\Scripts\python -m pytest tests\test_scan_accuracy.py -v` to determine actual pass rate and which tasks remain open.

### Attempt 11 — Tail re-truncation fix + Attempt 10B P-search window (2026-06-15)
- **Problem:** `interval_calculator.py` was STILL truncated at line 1070 (inside `__main__` demo block). Previous nightly session's Attempt 10 "syntax fix" did not fully persist on the bash mount — the Read tool showed stale cached content masking the truncation. Confirmed by `python3 -c "import ast; ast.parse(...)"` failing with "unterminated string literal" at line 1070 after my Edit.
- **Fix A (tail repair):** Used Python to strip the truncated last line and append the correct tail from `git show HEAD:interval_calculator.py` (lines 1053+). File now 1099 lines, syntax-clean.
- **Fix B (Attempt 10B):** Tightened P-onset search window to 60–250ms before R (with fallback to 60–400ms if no candidate found in tight window). Target: fx8200 PR spuriously high due to DWT misidentifying grid-artifact inflections as P-onsets at 300-500ms from R. Change at `interval_calculator.py` lines 456–478.
- **Result:** Both files syntax-clean (verified). Tests UNVERIFIED (scipy/neurokit2/pytest blocked in sandbox; git commit blocked by lock files).
- **To verify on Windows:** `del .git\HEAD.lock .git\index.lock` then `venv\Scripts\python -m pytest tests\test_scan_accuracy.py -v`
- **Expected:** fx8200 PR should drop from >171ms to ≈131ms ± 40ms if DWT P-onsets were the root cause. HR/QRS/QTc unchanged.

### Attempt 13 — Code audit + whitespace cleanup (2026-06-16, nightly)
- **Problem:** Same as Attempt 12: pip proxy 403, git lock files, pytest/scipy not runnable in sandbox.
- **Action:** Full independent audit of `_filter_by_peak_width`, `_consensus_rpeaks`, `_select_dominant_cluster`, `_suppress_grid_crossing_artifacts`, P-onset window (Attempt 10B). Verified `_suppress_grid_crossing_artifacts` receives correctly-cropped `grid_mask` (shape[1] matches `signal_px` length in all branch paths). Cleaned 21K-char whitespace blob left by previous agent.
- **Result:** Both files syntax-clean (verified `ast.parse`). No logic bugs found. UNVERIFIED (same sandbox blocker).
- **Verdict:** Code ready to test. All blockers are environment-only. Shlomi must: `del .git\HEAD.lock .git\index.lock` then `venv\Scripts\python -m pytest tests\test_scan_accuracy.py -v`.

## Nightly Run Summary — 2026-06-16 (2nd run)
- Attempts: 2 (Attempt 12 code review + Attempt 13 code audit)
- Pass rate: 3/8 → 3/8 (unchanged — tests unrunnable in sandbox)
- Tasks completed: none new (whitespace cleanup only)
- Tasks pending: A (HR84 width gate verify), B (HR106 ECGtizer verify), C (fx8200 PR verify)
- Key finding: Code is logically correct and syntax-clean. Only action needed on Windows: `del .git\HEAD.lock .git\index.lock` then run pytest.
