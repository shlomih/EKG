# Copilot Tasks — Mobile-Web Pivot (Week 1-2)

**Context:** We pivoted on 2026-04-20 to mobile-web-first validation ([plan](../../.claude/plans/i-want-to-reconsider-majestic-wind.md)). These are small, scoped tasks Copilot can do without architectural judgement. Each task says **what**, **where** (file + approximate lines), and includes a ready-to-paste **Copilot prompt**.

**Browser target:** Chrome (Android + desktop) first. Safari/iOS is deferred unless a task explicitly calls it out. This is a deliberate scope cut to save time — revisit after first clinician test.

**Rule:** Copilot does UI, config, and wiring of existing functions. Copilot does **not** touch `digitization_pipeline.py`, model code, or any signal processing. If a task feels algorithmic, stop and flag it.

---

## Week 1 — Mobile CSS + Deploy

### Task 1 — Mobile-responsive CSS in app.py
**File:** [app.py](app.py) — add to the existing `st.markdown("<style>...)` block around lines 149-162.
**Why:** Current CSS is desktop-sized; buttons are hard to tap on a phone, sidebar steals screen width.

**Copilot prompt:**
> In app.py, extend the existing custom CSS `<style>` block (around line 149) with a `@media (max-width: 600px)` block that:
> 1. Sets `button`, `.stButton > button`, and `[data-testid="stCameraInput"] button` to `min-height: 44px; font-size: 16px; width: 100%;`.
> 2. Sets `[data-testid="stSidebar"]` to `display: none;` (hide sidebar on mobile; we'll add a mobile menu later).
> 3. Sets `[data-testid="column"]` to `flex: 100% !important; width: 100% !important;` so columns stack vertically.
> 4. Reduces default `padding` on `.main .block-container` to `1rem`.
> Keep the existing desktop rules untouched.

**Verify:** open DevTools → toggle device emulation → Pixel 7 → buttons should be tappable, single column, no sidebar.

---

### Task 2 — Research-use disclaimer banner
**File:** [app.py](app.py) — top of the main UI, right after the header/title block.
**Why:** Gate: no clinical diagnosis claim. Required before sharing any URL with a clinician.

**Copilot prompt:**
> In app.py, just after the existing title/header, add a yellow warning banner using `st.warning()` with this exact text: "Research use only. Not a medical device. Do not use for diagnosis. Do not upload real patient data without consent." Place it inside an `if` block keyed to a Streamlit session flag so it can be dismissed for the session (use `st.session_state.get('disclaimer_ack', False)` and a "Got it" button that sets it to True).

**Verify:** disclaimer shows on first load, dismisses on click, stays dismissed until browser refresh.

---

### Task 3 — Scan-as-primary-CTA on mobile landing
**File:** [app.py](app.py) — the screen/path the user lands on.
**Why:** On mobile, the main action is "scan a paper ECG." Don't bury it behind upload/select menus.

**Copilot prompt:**
> In app.py's main UI, add a section above the existing upload/camera widgets with a large primary button labelled "📸 Scan Paper ECG" (use `st.button(..., type='primary', use_container_width=True)`). When clicked, set `st.session_state['input_mode'] = 'camera'` and `st.rerun()`. Then gate the existing `st.camera_input()` widget behind `if st.session_state.get('input_mode') == 'camera'`. Keep the file-upload option available but move it to an `st.expander("Upload file instead")` block below.

**Verify:** on mobile viewport, camera button dominates the fold; upload is collapsible.

---

### Task 4 — Dockerfile for cloud deploy
**File:** new file `Dockerfile` at repo root.
**Why:** Streamlit Community Cloud is easiest, but if we end up on Cloud Run / Render we need this ready.

**Copilot prompt:**
> Create a Dockerfile at the repo root that:
> 1. Uses `python:3.11-slim` (not 3.14 — some deps don't have wheels).
> 2. Installs system libs needed by OpenCV: `apt-get install -y libgl1-mesa-glx libglib2.0-0`.
> 3. Copies `requirements.txt`, runs `pip install --no-cache-dir -r requirements.txt`.
> 4. Copies the rest of the project.
> 5. Exposes port 8080 (Cloud Run convention).
> 6. Runs `streamlit run app.py --server.port=8080 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false`.
> Don't copy `models/*.pt` files via `COPY . .` — add a `.dockerignore` that excludes `EKGMobile/`, `models/*.npz`, `*.ipynb`, `.git/`, `__pycache__/`.

**Verify:** `docker build -t ekg-web . && docker run -p 8080:8080 ekg-web` → open http://localhost:8080 → app loads.

---

### Task 5 — requirements.txt audit
**File:** [requirements.txt](requirements.txt) (if exists; else create).
**Why:** Deploy target needs a clean, pinned list. Exclude mobile/training-only deps.

**Copilot prompt:**
> Inspect the top-level `import` statements in app.py, digitization_pipeline.py, clinical_rules.py, database_setup.py. Produce a requirements.txt containing only the runtime deps needed by the Streamlit web app (streamlit, torch, torchvision, numpy, scipy, opencv-python-headless — NOT opencv-python, pandas, scikit-learn, wfdb, Pillow, matplotlib). Pin each to a known-working version. Do NOT include training-only deps (h5py, datasets, transformers, wandb, jupyter, onnxruntime). Put one package per line.

**Verify:** `pip install -r requirements.txt` in a fresh venv → `streamlit run app.py` starts without errors.

---

### Task 6 — .streamlit/config.toml for mobile-friendly defaults
**File:** new file `.streamlit/config.toml`.
**Why:** Hide hamburger, tighten theme, disable uploader 200MB default (too big for phone).

**Copilot prompt:**
> Create `.streamlit/config.toml` with:
> ```toml
> [server]
> maxUploadSize = 10
> enableCORS = false
> enableXsrfProtection = true
>
> [browser]
> gatherUsageStats = false
>
> [theme]
> base = "light"
> primaryColor = "#d32f2f"
> font = "sans serif"
>
> [client]
> toolbarMode = "minimal"
> showErrorDetails = false
> ```

**Verify:** `streamlit run app.py` → hamburger menu is minimal, upload limit is 10MB.

---

## Week 2 — Minimal auth + clinician polish

### Task 7 — Shared-passcode gate via st.secrets
**File:** [app.py](app.py) — very top of the main flow (before any PHI widget renders).
**Why:** Need to gate access to the beta URL. Full auth is overkill; a shared passcode in `st.secrets` is enough for a closed beta.

**Copilot prompt:**
> In app.py, add a passcode gate that runs before any other UI. Read the expected passcode from `st.secrets.get('beta_passcode', None)`. If set, render a password input (`st.text_input(type='password')`). Only proceed to the main UI when the entered value matches. Persist acceptance in `st.session_state['auth_ok'] = True`. If `st.secrets` has no `beta_passcode` key, log a warning and skip the gate (so local dev is unblocked). Do not log or display the passcode value.

**Also:** add a `.streamlit/secrets.toml.example` file at repo root documenting the `beta_passcode = "..."` key. Add `.streamlit/secrets.toml` to `.gitignore`.

**Verify:** local run without secrets → gate skipped; with `beta_passcode = "test"` in secrets.toml → must enter "test" to proceed.

---

### Task 8 — Urgency-grouped result display
**File:** [app.py](app.py) — the results rendering section (where the 26 classes are currently listed).
**Why:** A flat list of 26 conditions is a debug dump, not a clinician summary. Group by urgency.

**Prereq:** `clinical_rules.py` already classifies urgency (red/yellow/green). Copilot can read the existing urgency map and re-render; do NOT invent new clinical logic.

**Copilot prompt:**
> In app.py's result-display section, replace the flat class-probability list with three expanders (in order): "🔴 Urgent findings", "🟡 Notable findings", "🟢 Other findings". Use the urgency field from `analyze_clinical_rules()` output to bucket each detected condition. Show probability as percentage. Within each bucket, sort by probability descending. Only show conditions whose probability exceeds the class threshold (from `thresholds_v3.json`). Add a toggle below labelled "Show low-confidence findings" that, when enabled, reveals conditions below threshold in a fourth collapsible "Below threshold" section.

**Verify:** run a demo ECG → Urgent/Notable/Other expanders appear, top finding is sensible, no condition shown twice.

---

### Task 9 — Loading spinner + clearer error messages
**File:** [app.py](app.py) — around the digitization + inference call sites (~lines 524-545 and wherever inference runs).
**Why:** On a phone, a 3-second silent pause after scanning feels broken. Users retry, upload corrupts.

**Copilot prompt:**
> Wrap the `extract_signal_from_image()` call in `with st.spinner("Reading paper ECG..."):` and the model-inference call in `with st.spinner("Analyzing..."):`. Wrap both in try/except. On `ValueError`, show `st.error("Could not read the ECG paper. Try: better lighting, keep paper flat, fill the frame.")`. On any other exception, show `st.error("Something went wrong. Please try again.")` and log the full traceback via `st.exception()` only if `st.secrets.get('debug_mode', False)` is True. Do not leak tracebacks to clinicians.

**Verify:** upload a blank white image → friendly error, no traceback.

---

### Task 10 — Health-check page
**File:** new file `pages/99_Health.py` (Streamlit auto-routes `pages/` to sub-pages).
**Why:** Deploy pipelines need a cheap URL to probe.

**Copilot prompt:**
> Create `pages/99_Health.py` that renders: current datetime (UTC), Python version, torch version, whether `models/ecg_multilabel_v3_best.pt` exists and its SHA-256 (first 16 chars), and whether `thresholds_v3.json` loads as valid JSON. Use `st.json()` for the output. No auth gate on this page.

**Verify:** open `/Health` on the deployed URL → status shows.

---

## What Copilot MUST NOT do

These are off-limits — flag to Shlomi / Claude:

- **Anything inside `digitization_pipeline.py`.** It is the product moat. No "improvements," no refactors, no type hints.
- **Anything inside `cnn_classifier.py`, `multilabel_v3.py`, `temperature_scaling.py`.** Training pipeline.
- **Anything inside `EKGMobile/src/digitization/`.** Native path is parked; do not port algorithms here.
- **Deleting files.** See [DELETION_CANDIDATES.md](DELETION_CANDIDATES.md). Phase 1 is done; Phase 2 requires human review.
- **Committing `.streamlit/secrets.toml`.** Secrets go in `.gitignore`, templates go in `secrets.toml.example`.
- **Real clinical logic.** Urgency, thresholds, class definitions come from existing modules — do not invent.

---

## Tasks NOT suitable for Copilot (humans only)

- Deploying to Streamlit Community Cloud / Cloud Run (needs creds + account decisions)
- Running the deployed URL on a real phone to scan a real paper strip
- Recruiting and sitting with the first cardiologist tester
- The Phase-2 decision (continue web vs. build native thin-client on FastAPI)

---

## Order

Do tasks 1-6 before deploy. Do 7-10 before sharing URL with anyone outside the room.
Ask before starting 7 — the secrets setup is the one thing where "Copilot goes ahead and commits a file" could leak something.
