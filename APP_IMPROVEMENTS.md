# App Improvements — Prioritized

Each item is scoped to be a single session. Top section is "in-flight or blocking trust"; middle is "clinical polish"; bottom is "nice-to-have / polish". Items marked (plan) have existing design in `C:\Users\osnat\.claude\plans\`.

---

## P0 — Trust & correctness (blocking real use)

1. **Verify scan-accuracy Round 3 on the FX-8200 photo** (plan)
   Re-scan the reference photo. Expected: HR ≈66, QRS 70–140 ms, QTc 380–480 ms, PR ≤170 ms.
   If QRS/QTc still return N/A, the DWT delineator is starving on low trace contrast, not jitter — that moves us to "trace inpainting at grid crossings" work, not another measurement-code change.

2. **V3.2c operating-point regression** (-0.080 MacroF1 vs V3.2b, same MacroAUROC)
   After the `eval_v3_auroc.py` fix, combined MacroF1 is 0.521 vs 0.601. Options: (a) keep V3.2b in production, mark V3.2c as a generalization-but-not-operational-wins checkpoint; (b) retry temperature scaling with different val-fold weighting; (c) accept V3.2c and ship. Decision is clinical, not technical — which classes matter most for the target user.

3. **"Analyzed row" preview thumbnail** (deferred from scan plan, #3)
   Show the user the cropped lead-row band that was actually analyzed. Closes the loop on the current "Auto-selected 1 of N lead rows detected" caption — lets them spot when we picked the wrong band and retake.

4. **Auto-rotate portrait photos** (deferred, #5)
   Detect `width < height` on upload and rotate 90° before processing. Currently portrait photos silently give garbage because the rhythm strip gets sliced vertically.

---

## P1 — Clinical confidence & UX

5. **Block-on-low-quality tightening**
   Current gate is `quality < 0.4`. Re-calibrate once Round 3 jitter term is validated — clean scans should be ≥0.7, obvious junk <0.3. If there's a middle zone (0.4–0.6) where results are unreliable, add a "⚠ reduced confidence" badge rather than silent pass.

6. **Show per-beat variability, not just point estimates**
   Currently shows `HR: 66`. Better: `HR: 66 (range 62–71, n=11 beats)`. Same for PR/QRS/QTc. Physicians trust a range far more than a single number; it also exposes when beats are irregular (AFIB signal).

7. **Mobile camera flip reliability** (plan addressed, verify in field)
   `capture="environment"` was added via JS injection. Needs real-device testing — iOS Safari behaves differently from Android Chrome. If it still picks front cam on iOS, fall back to explicit "Take photo / Upload photo" buttons.

8. **Upload-from-gallery hint** (deferred, #8)
   Some users have pre-taken ECG photos. Currently the camera-first UX confuses them. Add a visible "or upload existing photo" link next to the camera widget.

9. **Finding-level confidence thresholds**
   App surfaces classes above their per-class threshold. Add a secondary threshold (e.g. threshold + 0.1) that bumps findings into a "high-confidence" section, so physicians see "DEFINITE: STACH, AFL" separately from "POSSIBLE: LAE, STc".

---

## P2 — Feature parity with competitors

(See memory `project_competitive_analysis.md` — PMcardio, ECG Buddy, Cardiomatics, Anumana.)

10. **12-lead extraction, not just rhythm strip**
    Current pipeline analyzes one selected lead row. PMcardio/ECG Buddy extract all 12 leads and run per-lead analysis. This is where axis deviation (LAD/RAD) and lead-specific findings (IMI = II/III/aVF) actually come from. Biggest clinical-value gap.

11. **PDF report polish**
    Current PDF exists (Sprint 2). Compare to PMcardio's report: patient banner, "what to do next" action lines, signed-off footer, QR code for referral. Prioritize what's missing by asking a cardiologist to rank, not by guessing.

12. **Demographic-gated interpretation**
    App already takes age/sex. Use them more aggressively — e.g. LVH voltage criteria differ by age; peds vs adult QTc formulas differ; NORM prevalence priors vary. Currently demographics feed the model via `aux`, but clinical thresholds are still one-size-fits-all.

---

## P3 — Polish / infra

13. **Unit-test the scan pipeline end-to-end**
    Current tests cover `_detect_lead_rows` / `_select_best_row` / interval math. Add a tiny fixture suite: 3–5 synthetic ECG images with known HR/PR/QRS, confirm the full pipeline produces values within tolerance. Without this, future refactors of the pipeline will silently regress.

14. **Streamlit reload / session-state audit**
    Signal gets stashed in `st.session_state`. Verify that navigating between pages or re-uploading cleanly drops the prior signal — stale signal bugs are hard to spot and produce confusing UIs.

15. **Streamlit dev-loop speed**
    Model loads take ~5–10 s on cold start. Confirm `@st.cache_resource` is correctly memoizing. If iterating on app code frequently gets slow, add a `--no-model` dev flag that skips model load and stubs predictions.

16. **Beta passcode → proper auth**
    Current passcode gate is a stopgap. Before broader beta, swap to Streamlit's built-in auth OR a single-link magic-login so invitees don't share a password.
