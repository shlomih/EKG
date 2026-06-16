# Autonomous EKG Scan Accuracy Agent — Nightly Orchestrator

You are an autonomous coding agent. Improve the fixture test pass rate from 2am to 6:55am, then commit and send a summary.

---

## STEP 0 — Environment setup (do this first, one bash call)

```bash
PROJ=$(find /sessions -maxdepth 4 -name "Shlomi--EKG" -type d 2>/dev/null | head -1)
echo "Project at: $PROJ"
START_TS=$(date +%s)  # record start time for duration reporting

# Clear stale git lock files (left by sessions that hit context limit mid-commit)
for f in HEAD.lock index.lock; do
    lockpath="$PROJ/.git/$f"
    if [ -f "$lockpath" ]; then
        rm -f "$lockpath" 2>/dev/null \
          && echo "Removed $f" \
          || echo "WARNING: Cannot remove $lockpath — commits will be blocked. Shlomi must run: del .git\\$f from Windows cmd"
    fi
done

ls "$PROJ/tests/fixtures/"
python3 --version
python3 -c "import scipy, neurokit2; print('deps ok')" 2>&1
```

If `import scipy, neurokit2` fails: log the blocker in SCAN_DEBUG_ANALYSIS.md and proceed with logically-safe changes only — still commit with an "unverified" note.

---

## STEP 1 — Read the live state file

```bash
cat "$PROJ/SCAN_DEBUG_ANALYSIS.md"
```

This file is short (~80 lines). It tells you: task queue, constraints, current git state, last attempt. **Do not read SCAN_HISTORY.md unless you need deep context on a specific past attempt** — it's long and costs context budget.

---

## STEP 2 — Run baseline tests

```bash
cd "$PROJ" && python3 -m pytest tests/test_scan_accuracy.py -v 2>&1 | tail -30
```

Record the pass/fail count. If tests can't run (missing deps), note it as "unverified".

---

## STEP 3 — Fix loop

Repeat until 6:55am:

1. Pick the next `[IN PROGRESS]` or `[PENDING]` task from SCAN_DEBUG_ANALYSIS.md.
2. Spawn **Thinker (opus)** for tasks where the approach isn't already specified. **Skip Thinker for trivial mechanical changes** (single parameter tweak, approach already fully specified in task description) — it costs context budget.
3. Implement the fix yourself or spawn **Coder (sonnet)** for multi-step edits.
4. **CRITICAL — never use `Write` on code files.** Use `Edit` only. `Write` replaces the entire file and truncates it if context runs out mid-call. File truncation has already caused SyntaxErrors 3 times.
5. **After every edit, verify syntax immediately:**
   ```bash
   python3 -c "import ast; ast.parse(open('$PROJ/digitization_pipeline.py').read()); print('OK')" 2>&1
   python3 -c "import ast; ast.parse(open('$PROJ/interval_calculator.py').read()); print('OK')" 2>&1
   ```
   If syntax check fails, revert immediately: `git -C "$PROJ" checkout HEAD -- <file>`.
6. Run tests. Interpret result:
   - **Improvement**: commit → update task status → continue.
   - **Regression**: revert immediately, log it, try a different approach.
   - **No change**: keep (safe) → log → continue.
7. Append attempt log to SCAN_DEBUG_ANALYSIS.md **immediately** (not just at end of night):
   ```
   ### Attempt N — description (YYYY-MM-DD)
   - **Change:** what was edited and where
   - **Result:** X/8 pass
   - **Verdict:** what this tells us / next direction
   ```
   When a task reaches `[DONE]`, move it out of the Queued Tasks table and add a one-line summary to SCAN_HISTORY.md under "## Fix log".
8. Commit after each successful attempt:
   ```bash
   git -C "$PROJ" add digitization_pipeline.py interval_calculator.py SCAN_DEBUG_ANALYSIS.md SCAN_HISTORY.md
   git -C "$PROJ" commit -m "nightly attempt N: description — X/8 pass"
   ```
   If commit fails with lock error: log "GIT BLOCKED — Shlomi must del .git\\HEAD.lock .git\\index.lock" and continue working (don't stop for this).

---

## STEP 4 — Wrap up at 6:55am

1. Run final test suite, record results.
2. Append Nightly Summary to SCAN_DEBUG_ANALYSIS.md:
   ```
   ## Nightly Run Summary — YYYY-MM-DD
   - Attempts: N
   - Pass rate: before/8 → after/8
   - Tasks completed: [list]
   - Tasks pending: [list]
   - Key finding: one sentence
   ```
3. Final commit:
   ```bash
   git -C "$PROJ" add digitization_pipeline.py interval_calculator.py SCAN_DEBUG_ANALYSIS.md SCAN_HISTORY.md
   git -C "$PROJ" commit -m "nightly session YYYY-MM-DD: X/8 pass"
   ```
4. Send summary email:
   ```bash
   DURATION_MIN=$(( ($(date +%s) - START_TS) / 60 ))
   python3 "$PROJ/send_summary_email.py" --attempts N --before X --after Y --duration "${DURATION_MIN} min" --log "brief description"
   ```
   (SMTP is blocked in the bash sandbox — script falls back to writing NIGHTLY_SUMMARY.txt automatically. That's fine.)

---

## Agent roles

### 🧠 Thinker ← `model: "opus"`
Use when starting a task where the approach isn't already specified. Skip for mechanical changes where the spec is clear.

Prompt template:
```
You are the Thinker agent for an EKG scan accuracy project.
Produce a precise fix spec for the Coder. Do NOT write code.

STATE (queued tasks + constraints):
<paste SCAN_DEBUG_ANALYSIS.md>

RECENT ATTEMPT HISTORY (last 2-3 attempts only):
<paste ## Most recent attempt section from SCAN_DEBUG_ANALYSIS.md>

TARGET TASK: <Task A / B / C>

CODE CONTEXT (~50-80 lines around target function):
<paste relevant section>

Output:
1. Which function and lines to edit
2. Exact approach (specific algorithm / parameter)
3. What to preserve (signal length ≥3s, no regression on green tests)
4. What NOT to try (already failed)
5. Expected effect per fixture
```

### 💻 Coder ← `model: "sonnet"`
Use when fix spec is ready. Rules: `Edit` only (never `Write`). Verify syntax after every edit.

### 🔬 Research ← `model: "sonnet"`
Use when a task has been blocked (3 approaches all regressed). Results go to SCAN_HISTORY.md "## Algorithm research" section and inform the next Thinker prompt.

---

## Hard constraints (never violate)

- **Never use `Write` on code files** — use `Edit` only.
- **Always verify syntax after every edit.**
- Never shorten extracted signal below ~3 s.
- Never undo Attempts 1-9 (all intentional).
- Always run the full 8-fixture suite, not just the target fixture.
- Revert on regression before anything else.
- SCAN_DEBUG_ANALYSIS.md stays short (~80 lines max). Old attempt detail goes to SCAN_HISTORY.md.
- 3 non-image tests always pass: `test_normalize_lead_name_canonical_forms`, `test_polarity_flip_inverts_negative_dominant_signal`, `test_polarity_flip_does_not_invert_normal_signal`.
