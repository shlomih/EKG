# Files Recommended for Deletion

---

## ✅ DELETION SUMMARY (Completed — April 20, 2026)

### Phase 1 Deleted (Complete)
- ✅ **7 installation logs** (`install_log*.txt` files)
- ✅ **3 deprecated classifiers** (`ensemble_classifier.py`, `hybrid_classifier.py`, `poc_classifier.py`)
- ✅ **3 temp/patch files** (`cell1_updated.py`, `patch_email.py`, `patch_notebook.py`)
- ✅ **6 deprecated model files** (`colab_ecg_multilabel_v2_best.pt`, `ecg_multilabel_v3.pt`, `ecgfm_embeddings.npz`, `ecgfm_ml_embeddings.npz`, `thresholds_tuning_log.txt`, `a-large-scale-*.zip`)
- ✅ **3 stale documentation** (`MEMORY_GUIDE.md`, `NOTEBOOK_REVIEW.md`, `CODE15_STORAGE_PLAN.md`)
- ✅ **Archive file** (`EKG.zip`)
- ✅ **Empty directories** (`validation_results/`, `kaggle_upload/`, `ekg_strips/`)
- ✅ **ECG-FM model** (`models/ecgfm/` directory with pretrained checkpoint)

**Storage freed:** ~3-4 GB  
**Current models directory:** Only `ecg_multilabel_v3_best.pt`, `thresholds_v3.json`, `wfdb-4.3.1-py3-none-any.whl` remain ✓

---

## Category 1: Installation Logs (✅ DELETED)
Low-value logs from environment setup. Safely removed.

```
install_log.txt
install_log_chapman.txt
install_log_chapman_index.txt
install_log_ecgfm_precompute.txt
install_log_ecgfm_train.txt
install_log_local.txt
install_log_merged_train.txt
```

**Count:** 7 files | **Status:** ✅ Deleted  
**Reasoning:** Build/installation logs; not needed after environment is set up.

---

## Category 2: Deprecated Model Files (✅ DELETED)
Old model versions replaced by v3_best. Keeping only:
- `models/ecg_multilabel_v3_best.pt` (current best model)
- `models/thresholds_v3.json` (current thresholds)

```
models/colab_ecg_multilabel_v2_best.pt       [DELETED]
models/ecg_multilabel_v3.pt                  [DELETED]
models/ecgfm_embeddings.npz                  [DELETED]
models/ecgfm_ml_embeddings.npz               [DELETED]
models/thresholds_tuning_log.txt             [DELETED]
models/a-large-scale-12-lead-*.zip           [DELETED]
```

**Count:** 6 files | **Status:** ✅ Deleted  
**Reasoning:** Per CLAUDE.md, v3.2b uses `ecg_multilabel_v3_best.pt`. V2, ECGFM, and old logs are superseded.  
**Storage Freed:** ~2-3 GB (model files are large)

---

## Category 3: Deprecated Classifier Scripts — ✅ PARTIAL DELETED

**KEPT (Core Dependencies):**
- `cnn_classifier.py` — defines `ECGNetJoint` architecture; imported by `multilabel_v3.py`, `temperature_scaling.py`, `tune_thresholds.py`, `eval_v3_auroc.py`, `export_onnx.py`. **Actively used by the entire current pipeline.**
- `multilabel_classifier.py` — provides `load_demographics`, `preload_signals`, `MULTILABEL_CODES`; imported by `multilabel_v3.py`, `temperature_scaling.py`, `tune_thresholds.py`, `eval_v3_auroc.py`. **Actively used.**

**DELETED (app.py fallback chain, never reached since v3 always loads):**
```
ensemble_classifier.py          [DELETED]
hybrid_classifier.py            [DELETED]
poc_classifier.py               [DELETED]
```

**Count:** 3 files deleted  
**Status:** ✅ Deleted  
**Reasoning:** `app.py` loads v3 first; the fallback chain (ensemble → cnn → poc → hybrid) is never reached in normal operation. Core `cnn_classifier.py` and `multilabel_classifier.py` remain as dependencies of the training pipeline.

---
✅ DELETED)
One-off patches and temporary fixes.

```
cell1_updated.py        [DELETED]
patch_email.py          [DELETED]
patch_notebook.py       [DELETED]
```

**Count:** 3 files | **Status:** ✅ Deleted  
**Reasoning:** Ad-hoc fixes from earlier development phases; no longer needed
**Reasoning:** Appear to be ad-hoc fixes from earlier development phases.

---
⚠️ PARTIAL)
```
EKG.zip                                  [DELETED]
models/wfdb-4.3.1-py3-none-any.whl       [KEPT]
```

**Status:** Partial — Archive deleted, wheel kept  
**Reasoning:**
- `EKG.zip` (DELETED): Old backup archive; no longer needed
- `wfdb-4.3.1-py3-none-any.whl` (KEPT): May be referenced in deployment; kept for safet
- `wfdb-4.3.1-py3-none-any.whl`: Should be installed via pip; not needed locally

---

## Category 5b: ECG-FM Model (✅ DELETED)

```
models/ecgfm/mimic_iv_ecg_physionet_pretrained.pt   [DELETED]
models/ecgfm/.cache/huggingface/                     [DELETED]
```

**Status:** ✅ Deleted  
**Reasoning:** Per CLAUDE.md: "ECG-FM verdict: Frozen backbone AUROC=0.927 vs CNN AUROC=0.972 — stay on CNN." The pretrained ECG-FM checkpoint is no longer used.  
**Note:** `ecgfm_embeddings.npz` and `ecgfm_ml_embeddings.npz` already deleted in Category 2.

---
✅ DELETED)

```
validation_results/             [DELETED] (was completely empty)
kaggle_upload/                  [DELETED] (was empty notebook/ folder)
ekg_strips/                     [DELETED] (had 4 old demo PNG files)
```

**Count:** 3 directories | **Status:** ✅ Deleted
**Count:** 3 directories  
**Reasoning:** Not actively used in current workflow.

---
✅ PARTIAL DELETED)

| File | Status | Notes |
|------|--------|-------|
| `SCAN_STRIPS_PLAN.md` | **KEEP** | Sprint 3 plan — Step 2-5 not yet done |
| `APP_DEVELOPMENT_PLAN.md` | **KEEP** | Sprint 2 plan — Tasks 4,5,9 not yet done |
| `HIPAA_COMPLIANCE_CHECKLIST.md` | **KEEP** | Active compliance reference for mobile app |
| `MEMORY_GUIDE.md` | **DELETED** ✅ | One-time setup guide, obsolete |
| `NOTEBOOK_REVIEW.md` | **DELETED** ✅ | One-time review snapshot |
| `TEST_PLAN.md` | **KEPT** | May be referenced later; no tests yet |
| `CODE15_STORAGE_PLAN.md` | **DELETED** ✅ | Problem solved — now copies to SSD directly |

**Deleted
**Safe to delete:** 3 files (`MEMORY_GUIDE.md`, `NOTEBOOK_REVIEW.md`, `CODE15_STORAGE_PLAN.md`)

---

## Category 8: Verify Before Deleting (✅ RESOLVED — ALL KEEP)

Verified 2026-04-20 via grep. All four files are imported by `app.py` and cannot be deleted.

| File | Used by | Status |
|------|---------|--------|
| `digitization_pipeline.py` | `app.py:525` (`extract_signal_from_image`) | **KEEP — product moat** (load-bearing after mobile-web pivot; see `project_direction_pivot.md`) |
| `database_setup.py` | `app.py:23` | **KEEP** |
| `clinical_rules.py` | `app.py:140` (`analyze_clinical_rules`) | **KEEP** (also port source for future mobile `ClinicalRules.ts`) |
| `ekg_platform.db` | `database_setup.py:21` (`DB_PATH`); referenced by EKGMobile tests | **KEEP** |

**Count:** 4 files | **Status:** ✅ All KEEP — no deletions pending in this category.

---

## Summary

| Category | Count | Status | Impact |
|----------|-------|--------|--------|
| Installation Logs | 7 | ✅ DELETED | Cleaned up |
| Deprecated Models | 6 | ✅ DELETED | ~2-3 GB freed |
| ECG-FM model/cache | 1+ | ✅ DELETED | ~250-800 MB freed |
| Deprecated Classifiers | 3 | ✅ DELETED | app.py fallback chain |
| **cnn_classifier.py** | — | 🚫 **KEPT** | Active model architecture |
| **multilabel_classifier.py** | — | 🚫 **KEPT** | Active training utility |
| Temp/Patch Files | 3 | ✅ DELETED | One-off fixes |
| Archives/Wheels | 2 | ⚠️ PARTIAL | EKG.zip deleted; wheel kept |
| Empty Directories | 3 | ✅ DELETED | Unused folders |
| Stale Documentation | 3 | ✅ DELETED | MEMORY_GUIDE, NOTEBOOK_REVIEW, CODE15_STORAGE_PLAN |
| Active Documentation | 3 | 🚫 **KEPT** | SCAN_STRIPS_PLAN, APP_DEVELOPMENT_PLAN, HIPAA_COMPLIANCE |
| **Still Active** | 4 | ⚠️ KEPT | `clinical_rules.py`, `database_setup.py`, `digitization_pipeline.py`, `ekg_platform.db` |
| **TOTAL DELETED** | **~26 items** | ✅ | **~3-4 GB freed** |

---

## Deletion Order (Completed)

**Phase 1 (✅ COMPLETED — April 20, 2026):**
- ✅ All 7 installation logs
- ✅ 3 deprecated classifiers: `ensemble_classifier.py`, `hybrid_classifier.py`, `poc_classifier.py`
- ✅ All 3 temp/patch files
- ✅ All 6 deprecated model files from Category 2
- ✅ `models/ecgfm/` directory (ECG-FM pretrained checkpoint)
- ✅ 3 empty directories
- ✅ 3 stale docs: `MEMORY_GUIDE.md`, `NOTEBOOK_REVIEW.md`, `CODE15_STORAGE_PLAN.md`
- ✅ `EKG.zip` archive

**Freed Storage:** ~3-4 GB ✓

---

## Files Still Active (Not Deleted)

**Category 8 — Verified active on 2026-04-20, all KEEP:**
```
digitization_pipeline.py    → app.py:525 — product moat (mobile-web pivot)
database_setup.py           → app.py:23
clinical_rules.py           → app.py:140
ekg_platform.db             → database_setup.py:21 (live DB file)
```

**Models kept:**
- `models/wfdb-4.3.1-py3-none-any.whl` — May be referenced in deployment

**Documentation kept:**
- `SCAN_STRIPS_PLAN.md` — Sprint 3 plan active
- `APP_DEVELOPMENT_PLAN.md` — Sprint 2 plan active
- `HIPAA_COMPLIANCE_CHECKLIST.md` — Active compliance reference

---

## Final Status

✅ **Phase 1 cleanup complete:** ~26 files/directories deleted, ~3-4 GB freed  
🚫 **Do not delete:** `cnn_classifier.py`, `multilabel_classifier.py`, `SCAN_STRIPS_PLAN.md`, `APP_DEVELOPMENT_PLAN.md`, `HIPAA_COMPLIANCE_CHECKLIST.md`, `clinical_rules.py` (yet)
