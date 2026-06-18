# Autonomous EKG Scan Accuracy Agent — Nightly Orchestrator

You are an autonomous coding agent. Improve the fixture test pass rate from 2am to 6:55am, then commit and send a summary.

---

## STEP 0 — Environment setup (one bash call, keep it short)

```bash
PROJ=$(find /sessions -maxdepth 4 -name "Shlomi--EKG" -type d 2>/dev/null | head -1)
echo "Project at: $PROJ"
START_TS=$(date +%s)

# Install deps from pre-downloaded wheels (pip outbound is blocked 403).
# Wheels are in linux_wheels/ — no platform flags needed at install time.
# --no-deps skips transitive resolution; all required packages are explicit.
# If a wheel is missing, add it to the pip download command below and commit.
#
# To refresh linux_wheels/ on Windows (run from EKG folder):
#   pip download --no-deps ^
#     "scipy==1.14.1" "neurokit2==0.2.13" ^
#     "scikit-learn==1.5.2" "joblib==1.4.2" "threadpoolctl==3.5.0" ^
#     "PyWavelets>=1.4.0" "exceptiongroup>=1.0.0rc8" ^
#     "pytest==8.3.4" "pluggy==1.5.0" "iniconfig==2.0.0" "packaging==24.2" ^
#     --platform manylinux_2_17_x86_64 --python-version 310 ^
#     --implementation cp --abi cp310 --only-binary=:all: -d linux_wheels
#
python3 -m pip install --break-system-packages --quiet --no-index --no-deps \
  --find-links "$PROJ/linux_wheels" \
  scipy neurokit2 scikit-learn joblib threadpoolctl pytest \
  PyWavelets pluggy iniconfig packaging exceptiongroup 2>&1 | tail -3

python3 -c "import scipy, neurokit2, pytest; print('deps OK')" 2>&1
python3 --version
ls "$PROJ/tests/fixtures/" | head -10
```

If `deps OK` does not print, something is missing from linux_wheels/. Check which package failed and add its wheel. Do not ask Shlomi to do anything — just log the blocker and work on logically-safe code changes only.

**IMPORTANT — do NOT do any of these:**
- Ask Shlomi to delete git lock files (the agent handles this automatically below)
- Ask Shlomi to run tests manually (always run in the sandbox)
- Ask Shlomi to run git commands (always commit from sandbox)

---

## STEP 1 — Read the live state file (short, no other files)

```bash
cat "$PROJ/SCAN_DEBUG_ANALYSIS.md"
```

**Do NOT read** `SCAN_HISTORY.md`, `interval_calculator.py`, or `digitization_pipeline.py` upfront. Only read specific sections of code files when a fix requires it. Reading full files wastes context budget and causes the session to end early.

---

## STEP 2 — Run baseline tests from the sandbox

```bash
cd "$PROJ" && python3 -m pytest tests/test_scan_accuracy.py -v 2>&1 | tee result_linux.txt | tail -40
```

Record pass/fail count. This is the baseline for tonight.

---

## STEP 3 — Fix loop

Repeat until 6:55am:

1. Pick the next `[IN PROGRESS]` or `[PENDING]` task from SCAN_DEBUG_ANALYSIS.md.
2. Read only the specific function(s) you need to edit (use `grep -n "def _function_name" FILE` to find line numbers, then `sed -n 'N,Mp' FILE` to read just those lines). **Do not read the whole file.**
3. Spawn **Thinker (opus)** ONLY when the approach is completely unknown. Skip Thinker for: (a) mechanical parameter changes, (b) approach already specified in SCAN_DEBUG_ANALYSIS.md, (c) simple threshold or window adjustments.
4. Implement the fix using **`Edit` only — never `Write`**. `Write` replaces the entire file and truncates it if context runs out mid-call.
5. **After every edit, verify syntax:**
   ```bash
   python3 -c "import ast; ast.parse(open('$PROJ/interval_calculator.py').read()); print('OK')" 2>&1
   python3 -c "import ast; ast.parse(open('$PROJ/digitization_pipeline.py').read()); print('OK')" 2>&1
   ```
   **KNOWN ISSUE:** The Linux bash mount sometimes shows a stale cached version of the Windows files, making it appear shorter than it is. If syntax check fails but the error is in the last few lines and looks like a truncation artefact, use the Read tool to verify the actual Windows file state — do not revert based on bash alone.
6. Run tests:
   ```bash
   cd "$PROJ" && python3 -m pytest tests/test_scan_accuracy.py -v 2>&1 | tee result_linux.txt | tail -40
   ```
7. Interpret result:
   - **Improvement**: commit (see Step 3.8) → update task status → continue.
   - **Regression**: revert immediately: `git -C "$PROJ" checkout HEAD -- <file>`, log it, try different approach.
   - **No change**: keep (safe) → log → continue.
8. **Commit after each improvement:**
   ```bash
   # Clear any stale lock files silently (safe even if they don't exist)
   rm -f "$PROJ/.git/HEAD.lock" "$PROJ/.git/index.lock" 2>/dev/null
   git -C "$PROJ" add digitization_pipeline.py interval_calculator.py \
       SCAN_DEBUG_ANALYSIS.md SCAN_HISTORY.md linux_wheels/ NIGHTLY_AGENT_PROMPT.md
   git -C "$PROJ" commit -m "nightly attempt N: description — X/8 pass"
   ```
   If commit still fails after removing locks, log the error message and keep working — don't ask Shlomi for anything.
9. Append attempt log to SCAN_DEBUG_ANALYSIS.md immediately (not just at end):
   ```
   ### Attempt N — description (YYYY-MM-DD)
   - **Change:** what was edited and where
   - **Result:** X/8 pass
   - **Verdict:** what this tells us / next direction
   ```
   When a task reaches `[DONE]`, move it out of the table and add a one-line summary to SCAN_HISTORY.md under "## Fix log".

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
3. Final commit (same as Step 3.8).
4. Send summary:
   ```bash
   DURATION_MIN=$(( ($(date +%s) - START_TS) / 60 ))
   python3 "$PROJ/send_summary_email.py" --attempts N --before X --after Y \
     --duration "${DURATION_MIN} min" --log "brief description"
   ```
   SMTP is blocked — script falls back to writing NIGHTLY_SUMMARY.txt automatically. That's fine.

---

## Agent roles (use sparingly — each spawn costs context budget)

### 🧠 Thinker ← `model: "opus"`
Use ONLY when the approach is genuinely unknown after reading the task description. Do NOT use for tasks where the approach is already specified.

Prompt template:
```
You are the Thinker agent for an EKG scan accuracy project.
Produce a precise fix spec for the Coder. Do NOT write code.

STATE (queued tasks + constraints):
<paste SCAN_DEBUG_ANALYSIS.md>

TARGET TASK: <Task A / B / C>

CODE CONTEXT (~50-80 lines around target function only):
<paste specific function, not the whole file>

Output:
1. Which function and lines to edit
2. Exact approach (specific algorithm / parameter)
3. What to preserve (signal length ≥3s, no regression on green tests)
4. What NOT to try (already failed)
5. Expected effect per fixture
```

### 💻 Coder ← `model: "sonnet"`
Use when fix spec is ready and the edit spans multiple functions or files. For single-function edits, do it yourself — spawning Coder just to edit 10 lines wastes context.

---

## Hard constraints (never violate)

- **Never use `Write` on code files** — use `Edit` only.
- **Always verify syntax after every edit** (accounting for bash mount lag — use Read tool if bash result looks wrong).
- Never shorten extracted signal below ~3 s.
- Never undo Attempts 1-9 (all intentional).
- Always run the full 8-fixture suite, not just the target fixture.
- Revert on regression before anything else.
- SCAN_DEBUG_ANALYSIS.md stays short (~80 lines max). Old attempt detail goes to SCAN_HISTORY.md.
- 3 non-image tests always pass: `test_normalize_lead_name_canonical_forms`, `test_polarity_flip_inverts_negative_dominant_signal`, `test_polarity_flip_does_not_invert_normal_signal`.
- **Never ask Shlomi to run tests, delete files, or run git commands.** The agent does all of this autonomously.
