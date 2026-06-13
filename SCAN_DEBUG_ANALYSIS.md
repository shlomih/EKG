# Scan Accuracy — Root-Cause Analysis & Plan of Action

_Investigation date: 2026-06-13. Status: 3/8 fixture tests passing, 5/8 failing._

Failing fixtures: `fx8200_round3_verification`, `fx8200_reference`, `varied_HR106`, `varied_HR167`, `varied_HR84`.

---

## Queued Tasks for Nightly Agents

> **Agents: read this section first.** Shlomi may have updated this file during the day — always read it fresh at the start of each run. Tasks are listed in priority order. Mark a task `[IN PROGRESS]`, `[DONE]`, or `[BLOCKED: reason]` as you work.

| # | Status | Task | Target file | Fixtures affected |
|---|--------|------|-------------|-------------------|
| A | `[PENDING]` | **Trace-content quality (not post-hoc filtering)** — fix *which beats/leads land in the extracted trace*. NOTE: 1D denoising (medfilt/Savitzky-Golay) was tried and did NOT help — the spurious peaks are broad deflections (prominent T/P-waves or band artifacts), not high-freq spikes. Work the extraction/band step instead. | `digitization_pipeline.py` → `extract_signal_from_image()` / `_trace_to_signal` | HR84, HR106, fx8200 |
| B | `[PENDING]` | **Band selection for high HR** — scan at multiple `distance` values, pick most self-consistent RR train; or run quick `nk.ecg_peaks` on downsampled candidate band | `digitization_pipeline.py` → band selection scorer (~lines 897-929) | HR167 (every-3rd-beat), HR106 (only 3 beats extracted → delineation starved) |
| C | `[PENDING]` | **Grid-crossing inpainting for P-onsets** — suppress false P-onsets at grid-line crossings to fix PR overestimation | `digitization_pipeline.py` → grid artifact handling | fx8200, HR84 (PR overestimation) |
| D | `[PENDING]` | **Scan-derived HR is provisional on noisy traces** — consensus R-peak detection (now in `interval_calculator._consensus_rpeaks`) regressed HR84 (90→131): on a noisy trace it locks onto a denser, spurious-but-tall beat train. A provisional-rate warning is emitted when RR variability > 25% (interim safety). Real fix = cleaner trace content (items A/B). Until then, **scan HR on irregular/noisy traces must be treated as provisional, not clinical.** | `interval_calculator.py` (warning) + `digitization_pipeline.py` (real fix) | HR84 (regression), any noisy scan |

**Constraints agents must never break:**
- Currently passing (verified 2026-06-13): the 3 non-fixture tests — `test_normalize_lead_name_canonical_forms`, `test_polarity_flip_inverts_negative_dominant_signal`, `test_polarity_flip_does_not_invert_normal_signal`. **All 5 image fixtures currently FAIL.** Never let a green test go red.
- R-peak detection now uses **multi-detector consensus** (`_consensus_rpeaks`, all signals) — do not revert to single-detector without checking HR167/HR106 don't regress.
- Never shorten the extracted signal below ~3 s (DWT needs it — see HR84 precedent in this file)
- Only edit `digitization_pipeline.py` and `interval_calculator.py` unless Shlomi says otherwise
- Always run the full fixture suite after each change, not just the target fixture
- Safety net: `git tag scan-pre-consensus` (commit `1454439`) is the pre-consensus baseline for comparison/revert

---

## How the pipeline is wired today

The pipeline already does a row-first / peaks-second flow, and there are actually **two
separate R-peak detections**:

1. **Cheap peak scan for band selection** — inside `extract_signal_from_image()`
   (`digitization_pipeline.py`, ~lines 897-929). For each plausible band it runs
   `scipy.signal.find_peaks(z, height=0.8, distance=max(20, len(sig)//50))`, estimates HR,
   gates on `30 <= hr_est <= 220`, and scores `tr*0.4 + (1-rr_cv)*0.6`. This only *chooses
   the row*.
2. **Real clinical detection** — `interval_calculator.calculate_intervals()` runs
   NeuroKit's `nk.ecg_peaks` + `nk.ecg_delineate(method="dwt")` on the extracted 1D signal
   to produce HR/PR/QRS/QTc.

The failures live in different places depending on the fixture.

## Per-fixture root cause

### `varied_HR167` — wrong row picked (band-selection failure)
- Measured HR ≈ 67 vs truth 167 — almost exactly **2.5× too slow**; RR = `[2016, 1102, 514]` ms (wildly irregular for a regular fast rhythm).
- At 167 bpm beats are only ~9 px apart at typical resolution. The cheap scorer's
  `distance=max(20, len(sig)//50)` floor **merges adjacent beats**, so the band's *estimated*
  HR looks ~67 (passes the 30-220 gate) while it is really a 167 bpm trace under-sampled by
  the distance floor. The selected band may also straddle two leads' traces.
- **This is the one genuinely wrong-direction clinical error** and originates in the band
  selection step, not in `calculate_intervals`.

### `fx8200_reference` / `fx8200_round3_verification` / `varied_HR106` — right-ish row, delineation starves on edges
- HR closer (HR106: 125 vs 106; fx8200: 49 vs 66), but **QTc is always `None`** and PR/QRS
  are off or missing.
- `calculate_intervals` iterates over **all** R-peaks including first/last. The DWT
  delineator returns `nan` for the edge beat's `T_Peak/T_Onset/T_Offset` (visible in
  raw_delineation dumps). For HR106 the warning is literally *"cannot convert float NaN to
  integer"* — a whole beat's delineation came back NaN because its QRS/T complex sits at the
  edge of the cropped strip (band crop + 0.18 left-text mask).
- DWT wavelet coefficients are unstable near a strip edge or at a grid-line discontinuity in
  the gap-filled trace → NaN → QTc dropped.
- PR overestimation (fx8200: 254 vs 131; HR84: 185 vs 140) suggests `ECG_P_Onsets` fires too
  early, latching onto a residual grid-line artifact before the true P-wave.

### `varied_HR84` — closest miss
- HR 88 vs 84 (off by exactly 1, just over tolerance); same QTc-NaN issue; QRS over (149 vs 93).
- Underlying extraction is essentially correct, just noisy at onset/offset boundaries.

## Plan of action (the user's "remove rows → peaks → recheck rows" framing, split into separable problems)

1. **Edge-beat exclusion in `interval_calculator.py`** _(smallest, lowest risk, fixes the most fixtures)_
   - In the per-beat PR/QRS/QTc loops, skip the first and last R-peak's contribution when
     there are ≥4 beats. Interior beats have full waveform context on both sides and are far
     less likely to hit NaN. Should clear the QTc=N/A pattern on 4/5 fixtures **without
     touching image processing**.
2. **Band-selection robustness for high HR (`varied_HR167`)**
   - Don't gate on a single fixed `distance`: either run the cheap detector at 2-3 `distance`
     values and pick the most self-consistent RR train, or replace the cheap `find_peaks`
     scorer with a quick `nk.ecg_peaks` on a downsampled candidate band (more robust at high HR).
3. **Grid-crossing trace gaps → false P-onsets / wide QRS** _(defer)_
   - The bigger "trace inpainting at grid crossings" item already flagged in the
     `fx8200_round3_verification` docstring. Likely unnecessary if 1+2 get us to 6-7/8.

### Precedent / constraint
A.5 paper-segmentation crop was **disabled** because the bbox crop truncated the signal below
DWT's 3 s minimum (HR84 got only 2.4 s). Any reordering of row-selection vs. signal-length must
preserve enough signal duration for delineation. Edge-beat exclusion (item 1) does not shorten
the signal — it only ignores boundary beats' fiducials — so it is safe against this precedent.

### Order of attack
1. Edge-beat exclusion (item 1) → **test before proceeding**.
2. Only if tests improve / hold: band-selection fix (item 2).
3. Decide whether grid-inpainting (item 3) is still needed.

---

## Fix log

### Attempt 1 — Edge-beat exclusion in `interval_calculator.py` (2026-06-13)
- **Change:** when `len(r_peaks) >= 4`, restrict PR/QRS per-beat loops to interior beats
  (`r_peaks[1:-1]`) and skip the first beat in the QTc loop (last already excluded by `[:-1]`).
- **Result:** 3/8 pass, 5/8 fail — **no change in pass rate, no regression.**
- **What the test data revealed (this reframes the root cause):**
  - `varied_HR84`: `rr=[664,692,1308,462]ms`, `r_peaks=[467,799,1145,1799,2030]`. The `1308`
    is ~2× its neighbours → **NeuroKit missed one R-peak** (expected ~sample 1472). That single
    miss pushes HR 88 vs 84 and breaks QTc (Bazett over √1.308s → 297, under the 300 floor →
    rejected; all other QTc candidates pair to wrong beats because RR is inconsistent). QTc=N/A
    is therefore **not** an edge effect.
  - P-onsets `[680,1021,1733,1968]` pair correctly by position but sit ~240ms before R on two of
    three interior beats (one correct at 132ms) → PR median 238 vs truth 140. Genuine
    **false/early P-onset** from low trace contrast at grid crossings, not an edge effect.
- **Verdict:** Edge exclusion is correct and harmless (kept in), but the **dominant failure mode
  is missed/extra R-peaks** (same "one RR ≈ 2× others" signature in HR84, HR106, HR167), plus
  systematically early P-onsets. Re-prioritise: fix R-peak detection robustness first.

### Revised order of attack
1. **R-peak detection robustness (missed-beat recovery)** — #1 cause across 3 fixtures; upstream
   of both band selection and edge exclusion. Approach: physiological min-distance / RR
   post-processing to split RRs that are ≈2× the local median.
2. Band-selection robustness for high HR (`varied_HR167`).
3. P-onset / grid-crossing inpainting (PR overestimation).

### Attempt 2 — Missed-beat recovery (`_recover_missed_rpeaks`) (2026-06-13)
- **Change:** after `nk.ecg_peaks`, scan for RR gaps > 1.5× the local median RR, estimate how
  many beats are missing, place candidates at evenly-spaced positions, snap each to the nearest
  local max in the cleaned signal, accept if amplitude ≥ 0.4× median R-amp. Purely additive.
- **Result:** 3/8 pass, 5/8 fail — no test flipped, but **worked as designed where it fired**:
  - `varied_HR84`: recovered the real beat at sample 1452; RR `[664,692,1308,462]` →
    `[664,692,614,694,462]`; **hr_variability 0.405 → 0.138**; delineation now has 6 beats with
    real fiducials. But mean-of-instantaneous-HR rose 88 → 98: the old 1308ms gap (46bpm) had been
    *accidentally counterbalancing* the spurious 462ms final RR (130bpm); fixing the gap let the
    short-RR outlier dominate the mean. → motivated Attempt 3.
  - `varied_HR167`: fired once but cannot help — see below.

### Attempt 3 — Robust median HR (2026-06-13)
- **Change:** `hr = 60000 / median(RR)` instead of `mean(60000/RR)`. Outlier-resistant, clinically
  standard for mildly irregular rhythm.
- **Result:** 3/8 pass, 5/8 fail — no flip, but improved the close ones:
  - `varied_HR84`: 98 → **90** (truth 84, tol 4 → diff 6).
  - `varied_HR106`: 125 → **115** (truth 106, tol 5 → diff 9).
  - `varied_HR167`: 73 → 60 (worse, but fundamentally broken upstream).
  - `fx8200`: 49 → 50 (≈unchanged).

## KEY FINDING — remaining HR failures are UPSTREAM (extraction / calibration), not interval_calc
The signal handed to NeuroKit already has the wrong beat count; no amount of interval-side logic
fixes that:
- **`varied_HR167`** (60 vs 167): RR ≈ `[1076,940,1102,514]`ms ≈ 56-64 bpm. Detector catches
  ~**every 3rd beat** (1076/3 ≈ 359ms = 167bpm). Recovery's median-RR reference is itself the
  wrong 3× value, so it can't split correctly. → **band-selection / signal-resolution** (the trace
  for a 167bpm strip has beats too close to resolve at current extraction resolution).
- **`varied_HR106`** (115 vs 106): only 3 R-peaks extracted in a 3s+ strip + one spurious short RR
  (376ms) + delineation crash → **too few beats extracted** (extraction quality).
- **`fx8200`** (50 vs 66): RR uniformly ~1.32× too long, hr_variability=0.09 (looks like a *clean*
  slow rhythm). No single gap is 1.5× median so recovery never fires. A uniform 1.32× time stretch
  is the signature of a **time-axis calibration error** — `actual_fs = px_per_mm_x * paper_speed`
  drives the resample ratio; if `px_per_mm_x` is over-estimated, the resampled signal is stretched
  and every RR reads ~1.32× long. **Cheapest high-leverage suspect for fx8200.**
- Bias direction is **mixed** across fixtures (fx8200 slow, HR84/HR106 slightly fast) → not one
  global constant; it's per-image extraction/calibration quality.
- **PR over-detection** (fx8200 258 vs 131; HR84 187 vs 140): P-onset latches ~240ms pre-R on some
  beats (grid-crossing / low-contrast). Independent of HR. → grid-inpainting (item 3).

### Verdict on interval_calculator.py work
Attempts 1-3 are all **principled, no regressions, kept in** — they made the metrics greener
(HR84 closer, HR106 closer, rhythm regularity much better) but the thresholds need the **upstream**
signal to be right. Interval-side robustness is now ~exhausted.

### Next highest-leverage step
Move to `digitization_pipeline.py`:
- **(a) Verify time calibration** (`px_per_mm_x` → `actual_fs` → resample ratio). Quick to check;
  could fix fx8200's uniform 1.32× bias outright. **Do this first — cheapest.**
- **(b) Band-selection / extraction resolution** for high HR (HR167 every-3rd-beat, HR106 too-few).
- **(c) Grid-crossing inpainting** for PR/P-onset (item 3).

### Attempt 4 (investigation only) — calibration & detector probing (2026-06-13)
Instrumented `extract_signal_from_image` on the 4 failing fixtures. **No code change.**

**(a) Calibration RULED OUT.** Detected grid spacing: fx8200 px_x=33.5, HR84 px_x=34, HR106
px_x=33/px_y=32, HR167 px_x=32/px_y=32.5. Where both axes detected, **px_x ≈ px_y** (grid
squares are physically 1mm×1mm), cross-validating ~32-33 px/mm as correct → `actual_fs ≈ 800-850Hz`
is right, time axis is **not** stretched. Also the HR-error direction is **mixed** (fx8200 slow,
HR84/HR106 fast), inconsistent with a single calibration constant. **Calibration is not the bug.**

**(b) The beats are PRESENT in the extracted signal but detection is unreliable.**
Raw `find_peaks(prominence=0.4-0.6, distance=0.2s)` on the extracted signal:
- `varied_HR167`: **14 peaks → implied HR ≈ 168 ≈ truth 167** (by count). The 167bpm beats are
  cleanly in the trace; NeuroKit default `ecg_peaks` only catches every ~3rd → HR 60. But the
  find_peaks train is irregularly spaced (median-RR HR → 194, CV 0.23), so a naive swap overshoots.
- `fx8200`/`HR106`: find_peaks **overshoots** (HR 210/246 — noise-driven spurious peaks), while
  NeuroKit **undershoots** (50/115). Truth sits in between → the trace is genuinely noisy.

**(c) NeuroKit method sweep — no universal winner** (median-RR HR):
| fixture | truth | neurokit | pantompkins | hamilton | elgendi | kalidas | rodrigues |
|---------|-------|----------|-------------|----------|---------|---------|-----------|
| fx8200  | 66    | 50       | 122         | 149      | **76**  | 137     | 153       |
| HR167   | 167   | 54       | **170**     | 181      | 73      | **161** | 199       |
| HR106   | 106   | **115**  | 124         | 135      | 91      | 137     | 153       |
| HR84    | 84    | **88**   | 117         | 130      | 96      | 138     | 111       |

pantompkins/kalidas fix HR167 (within tol 8) but wreck fx8200; neurokit is best for the other
three. **The optimal detector differs per fixture because the traces have different noise
characteristics** → detector tuning can't win globally.

### CONCLUSION — remaining work is upstream trace/signal quality
Calibration is correct; interval-side robustness is exhausted (3 fixes landed, all kept).
The residual HR/PR/QRS errors trace to **noisy/distorted extracted 1D signals** (spurious peaks
on some, attenuated beats on others) and **early P-onsets at grid crossings**. The real levers are:
1. **Cleaner trace extraction & signal conditioning** — denoise the extracted 1D signal so a single
   detector finds the right beats (e.g., stronger smoothing tuned to expected beat width, spurious-
   peak suppression). Highest leverage; affects all four fixtures.
2. **Band selection** — confirm the chosen row is the cleanest rhythm strip (esp. fx8200, HR106).
3. **Grid-crossing inpainting** — for PR/P-onset over-detection.
These are `digitization_pipeline.py` changes and are a meaningful next chunk — checkpoint with
Shlomi before starting, since they carry regression risk to the real-ECG path in `app.py`.

---

## DECISION (2026-06-13) — Multi-detector consensus R-peak detection, for ALL signals
Pursuing the user's idea: combine several R-peak detectors (consensus/ensemble) instead of one.
Validated below. **Applied to all signals** (one code path), knowingly changing `app.py`'s
already-working digital-ECG path. Safety net: `git tag scan-pre-consensus` at commit `1454439`
(3/8 fixtures + the 3 interval-side fixes) for comparison/revert. Plan file:
`C:\Users\osnat\.claude\plans\flickering-twirling-pine.md`.

**Why consensus (validated this session):** single detectors have no universal winner; each wins
on some fixtures and fails on others (table above). Cluster-voting across 6 NeuroKit methods
(neurokit, pantompkins1985, hamilton2002, elgendi2010, kalidas2017, rodrigues2021) produced a
correct-range candidate for HR167 (vote≥3 → 163), HR106 (vote≥5 → 109), HR84 (vote≥5 → 84) —
which single detectors could not do consistently. Open problem: **automatic selection** of the
right consensus train (noisy traces need stricter agreement; clean-but-fast HR167 needs looser);
"lowest RR-CV alone" mis-picks HR84 → 131, so selection combines regularity + coverage +
amplitude consistency.

### Attempt 5 — Consensus detector `_consensus_rpeaks` in `interval_calculator.py` (2026-06-13)
- **Change:** Step-3 detection now runs the 6-method panel, clusters peaks by agreement (~120ms
  window, vote = distinct detectors), generates candidate trains at vote thresholds k=2..N, and
  selects the highest-scoring train (`_score_rpeak_train`: regularity + coverage + amplitude
  consistency, HR-plausibility gated). Removed the `_recover_missed_rpeaks` post-step (consensus
  subsumes it; re-running it re-introduced spurious short RRs on fx8200/HR167).
- **Result:** 3/8 pass, 5/8 fail. **No previously-passing test broke.** Big HR-accuracy gains on
  the catastrophic cases; one regression (HR84) and remaining non-HR blockers:
  | fixture | HR before (tag) | HR now | truth | tol | status |
  |---------|-----------------|--------|-------|-----|--------|
  | HR167   | 60 (every-3rd)  | **173** | 167 | 8 | HR ✅ — fails only on **QTc=N/A** |
  | HR106   | 115             | **109** | 106 | 5 | HR ✅ — fails on PR/QRS/QTc N/A (only 3 R-peaks → delineation starved) |
  | fx8200  | 50              | 71      | 66  | 4 | HR closer, still off by 5; PR still over |
  | HR84    | 90              | 116-131 | 84  | 4 | **HR regressed** — selector prefers a dense spurious-peak train |
- **Why HR84 regressed / can't be tuned away:** at no vote threshold does the clean ~84bpm beat
  train exist — high thresholds drop real beats (coverage collapses), low thresholds admit
  spurious-but-*tall*, regularly-spaced deflections. Amplitude/prominence cues don't separate them
  (the spurious peaks are tall). → genuinely needs a **cleaner extracted signal**, not selector
  tuning. Adding single-detector trains as candidates was tried and rejected (made fx8200→76 and
  HR106→91 worse).
- **Verdict:** Consensus is a clear **clinical-safety win** (HR167 60→173 is far safer than the
  reverse) and broke nothing green. But to flip tests green and undo the HR84 regression we need:
  (i) **QTc T-offset window** fix for tachycardia (QT can exceed 0.7×RR → HR167 green), and
  (ii) **extracted-signal denoising** (HR84/HR106/fx8200). Proceeding per plan Step 3.

### Attempt 5b — things tried and rejected to close the gap in `interval_calculator` (2026-06-13)
- **HR167 QTc window relax:** NOT the cause. HR167's consensus train still has spurious close
  peaks (RR 91-128 samples); T-offsets pair to give implausible ~100ms QT that the 280ms floor
  correctly rejects. Widening the window won't help while the R-train is imperfect.
- **Signal denoising (medfilt 5/9, Savitzky-Golay, combos) before detection:** no change
  (HR84 stays 131, etc.). The spurious peaks are **broad deflections** (likely prominent T/P-waves
  or band artifacts), not high-frequency spikes — `nk.ecg_clean`'s bandpass already passes them.
- **"Trust neurokit if healthy" guard:** rejected. HR84's neurokit train is itself unhealthy
  (cv 0.41, 1.93× gap from the missed beat) so the guard still picks consensus 131; and it
  reverts HR106 109→115 (loses a near-pass). neurokit "health" doesn't track correctness here.
- **CONCLUSION:** interval-side tuning is exhausted. HR84/fx8200/HR106 need better **extracted
  signal content** (which beats/leads land in the trace) — the band-selection / trace-quality work
  (queued items A/B/C above), not post-hoc filtering. Consensus stays as a validated net
  clinical-safety win; the HR84 HR-number regression sits inside an already-failing test.
