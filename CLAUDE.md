# EKG Intelligence Platform — Claude Context

## Project Overview

Streamlit-based 12-lead ECG analysis POC (`app.py`), targeting a native mobile app later.
Training happens on **Google Colab** (GPU/TPU). Local CPU is used as fallback only.
Python 3.14 at `C:\Users\osnat\AppData\Local\Programs\Python\Python314\python.exe`.

---

## Current Phase: V3 Multilabel (26 classes)

The active model is a multilabel CNN trained on PTB-XL with 26 SNOMED/SCP condition classes.

| File | Description |
|------|-------------|
| `multilabel_v3_colab.ipynb` | Main Colab notebook — primary training interface |
| `multilabel_v3.py` | Local training script (CPU fallback) |
| `models/ecg_multilabel_v3_best.pt` | Best Colab checkpoint — AUROC=0.9681 (26 classes, Apr 4) |
| `models/ecg_multilabel_v3.pt` | Local CPU training latest epoch |

**Best result so far:** AUROC=0.9681 from Colab (after fixing critical .mat extension bug that was zeroing all Challenge data).

---

## Architecture Decisions

- **CNN backbone:** ECGNetJoint (1D CNN with SE attention)
- **Loss:** ASL (Asymmetric Loss) or focal loss for class imbalance
- **26 classes:** Expanded from 12 → 14 → 26 via PTB-XL + Chapman + Challenge datasets
- **ECG-FM verdict:** Frozen backbone does NOT beat CNN (AUROC 0.927 vs 0.972). Full fine-tuning helps (HYP F1 0.478) but CNN still wins overall. Stay on CNN.

---

## Colab MCP Setup (BLOCKED — needs fix)

Goal: Claude controls Colab notebooks directly via `colab-mcp` MCP server.

**Status: Not working.** The `open_colab_browser_connection` tool never appears in Claude Code.

**Root cause identified:** `~/.claude.json` has `"hasTrustDialogAccepted": false` under the project config. VS Code extension blocks untrusted MCP servers.

**Fix:**
1. In VS Code, find and accept the trust dialog for `colab-proxy` MCP server
   — OR manually set `"hasTrustDialogAccepted": true` in `~/.claude.json` under `projects["C:/Users/osnat/Documents/Shlomi/EKG"]`
2. Fully restart VS Code (File → Exit, not Reload Window)
3. Verify by checking `C:\Users\osnat\.claude\wrapper_debug.log` for a new timestamp
4. Open `colab.new` in browser BEFORE calling `open_colab_browser_connection`

**Config location:** `~/.claude.json` → `projects[...].mcpServers` (NOT `~/.claude/mcp.json` — that's CLI only)

**Wrapper script:** `C:\Users\osnat\.claude\colab_mcp_wrapper.py` — delays `tools/list` by 4s so colab-mcp registers its tool before Claude disconnects. Works correctly when tested manually.

---

## Top 3 Open Problems

1. **colab-mcp trust dialog** — MCP server is configured but blocked by `hasTrustDialogAccepted: false`. Accept the trust dialog in VS Code to unblock.

2. **V3 training next steps** — After CPU training finishes (was at epoch ~42/60, best AUROC=0.971 at epoch 31), run threshold tuning. Compare CPU vs Colab checkpoints.

3. **26-class expansion quality** — Need to verify per-class AUROC on the Challenge classes (new additions). The .mat bug fix was recent — rerun full evaluation on test set.

---

## Key Previous Model Versions

| Model | Classes | AUROC | Notes |
|-------|---------|-------|-------|
| v10 CNN (ECGNetJoint) | 5 superclass | HYP F1=0.442 | PTB-XL only |
| v9+v10 Ensemble | 5 superclass | HYP F1=0.456 | Per-class thresholds |
| ECG-FM Stage 2 (Colab T4) | 5 superclass | HYP F1=0.478 | Full fine-tune |
| **V3 multilabel (current)** | **26** | **0.9681** | **Production target** |

---

## What's Working

- Colab training pipeline (multilabel_v3_colab.ipynb) — runs end-to-end
- .mat file loading bug is fixed (was zeroing all Challenge records)
- Local CPU training runs (slow but functional)
- Streamlit app (app.py) loads models and classifies

## What's Not Working

- colab-mcp → `open_colab_browser_connection` not available (trust dialog issue)
- No automated threshold tuning yet for 26-class model

---

## Next Steps (in order)

1. Fix colab-mcp trust issue → verify tool appears → call `open_colab_browser_connection`
2. Run threshold tuning on `ecg_multilabel_v3_best.pt`
3. Full per-class evaluation on 26-class test set (especially Challenge classes)
4. Update app.py to use v3 model (currently uses older 12-class model)
