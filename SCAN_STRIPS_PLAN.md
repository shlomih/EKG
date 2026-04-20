# EKG Mobile — Sprint 3: Scan Strips Plan

**Goal:** User can point phone at a paper ECG strip, capture it, and get an analysis result.  
**Prerequisite:** Sprint 2 tasks (inference pipeline, demo mode) should exist or be done in parallel.  
**Status:** Step 1 (ONNX export) is complete.

---

## Step 1 — Export ONNX Model ✅ DONE
**Effort:** ~30 min | **File:** `scripts/export_onnx.py`

`export_onnx.py` is fixed (onnxscript error handled with `dynamo=False` fallback).

Run to generate the model file:
```
python scripts/export_onnx.py --verify
```
Produces:
- `EKGMobile/assets/models/ecg_v3.onnx`
- `EKGMobile/assets/models/model_manifest.json` (SHA-256 for HIPAA 1.4.2)
- `EKGMobile/assets/models/thresholds_v3.json`

**Without this file, no on-device inference is possible — do this first.**

---

## Step 2 — Demo Mode Validation
**Effort:** ~1 day | **Copilot: PARTIAL** | **Files:** `src/ml/DemoData.ts`, `app/(main)/scan.tsx`

Validate the full inference pipeline before camera work, using synthetic ECG data.

### 2a. Create `src/ml/DemoData.ts`
Generate a synthetic 12-lead normal sinus rhythm (60 bpm, clean signal):
```typescript
export function generateNormalECG(): Float32Array {
  // Returns Float32Array[12 * 5000] — 12 leads × 5000 samples @ 500 Hz
  // Lead I: 60 bpm sinus with realistic PQRST morphology
  // Other leads: derived from Lead I + physiological relationships
}
```
**Copilot: NO** — requires ECG morphology knowledge.

### 2b. Wire Demo Button in scan.tsx
**Copilot: YES** — add "Use Demo ECG" button → calls `generateNormalECG()` → runs inference.

Copilot prompt: *"In scan.tsx, add a 'Use Demo ECG' button that calls DemoData.generateNormalECG() and passes the Float32Array[12*5000] result to the analysis pipeline. Follow the existing button pattern."*

### Verification
- Tap Demo → inference runs → shows 26-class results with NORM as top condition
- No crashes, <2 sec on device

---

## Step 3 — Camera Capture
**Effort:** ~0.5 day | **Copilot: YES** | **File:** `src/digitization/CameraCapture.tsx`

Capture a photo of a paper ECG strip with alignment guides.

### Implementation
```typescript
// CameraCapture.tsx — overlay shows:
//   - Corner markers for A4/Letter ECG paper alignment
//   - "Keep paper flat and fully visible" hint
//   - Capture button → returns temp URI
```
Uses `expo-camera` (already in package.json).

**Copilot prompt:** *"Create src/digitization/CameraCapture.tsx. Use expo-camera with a Camera component. Overlay a white rectangle guide showing where to place the ECG paper. On capture, save to a temp file via TempFileManager and return the URI. Follow the pattern from any existing screen."*

After capture: `TempFileManager.secureDelete()` the temp file when done (HIPAA 1.1.5).

### Verification
- Camera opens, overlay shows
- Capture saves a file, temp file deleted after use

---

## Step 4 — ECG Digitization (Signal Extraction from Image)

Choose **Option A** or **Option B**. Option B is simpler and ships faster.

---

### Option A — Custom TypeScript Digitizer
**Effort:** ~4-5 days | **Copilot: NO** | **File:** `src/digitization/SignalExtractor.ts`

Full image processing pipeline entirely in TypeScript:

1. **Preprocess** — use `expo-image-manipulator` to crop, grayscale, resize to fixed width
2. **Grid detection** — find horizontal/vertical grid lines via row/col intensity sums
3. **Baseline extraction** — median horizontal line per lead window
4. **Signal tracing** — for each column, find darkest pixel above baseline = signal sample
5. **Calibration** — use 1 mV calibration pulse (if visible) to scale to mV, else use 10mm/mV standard
6. **Lead assignment** — fixed layout (I, II, III, aVR, aVL, aVF, V1-V6 + rhythm)
7. **Resample** — linear interpolation to exactly 5000 samples

**Accuracy:** Works well for clean printed ECGs on white paper, poor for photos at angle/shadow.  
**Copilot: NO** — requires ECG paper layout knowledge and signal processing.

---

### Option B — Guided Capture + Scale-Based Extraction ⭐ RECOMMENDED
**Effort:** ~2 days | **Copilot: PARTIAL** | **File:** `src/digitization/SignalExtractor.ts`

Simpler approach: require the user to align the ECG paper within a fixed frame overlay, then use the known paper geometry to extract signals.

**How it works:**
- Camera overlay shows a fixed frame matching standard ECG paper proportions (A4 or 5" thermal paper)
- User aligns the paper exactly within the frame
- Since frame dimensions are known and fixed, pixel → mm mapping is deterministic
- No need to detect the grid — use the overlay geometry directly
- Signal extraction uses fixed lead row heights (standard: 40mm per lead block, 12 leads in 3×4 or 2×6 layout)

**What Copilot can build (YES):**
- Camera overlay UI with alignment markers and paper-size selector (A4/Letter/Thermal)
- Lead region slicing code (crop image into fixed lead rectangles)
- Basic pixel-intensity-to-voltage conversion given known mm/pixel scale

**What needs Claude (NO):**
- Signal tracing from pixel intensity (darkest pixel = signal column)
- Baseline correction (isoelectric line from TP segment)
- Calibration pulse detection (or fixed 10mm/mV assumption)

**Copilot prompt (UI part):** *"In CameraCapture.tsx, add a paper-size selector (A4 / US Letter / Thermal 5in). Overlay a rectangle that represents the selected paper size at correct aspect ratio. Show corner alignment arrows. When captured, pass the paper size along with the image URI to SignalExtractor."*

**Accuracy:** Better than Option A because the frame removes perspective distortion. Still sensitive to lighting.

---

### Option A vs B Summary

| | Option A | Option B |
|--|----------|----------|
| User effort | Just point camera | Must align paper precisely |
| Dev effort | 4-5 days, Claude only | 2 days, Copilot helps UI |
| Accuracy | Medium (needs grid detection) | Medium-high (fixed geometry) |
| Robustness | Fragile to angle/lighting | More robust (user aligns) |
| Dependencies | None | None |
| **Recommendation** | Later (v2) | **Ship first (v1)** |

**Can Copilot build Option B?**
- UI parts (camera overlay, paper selector, lead region slicing): **YES**
- Signal tracing, baseline correction, calibration: **NO (Claude only)**

---

## Step 5 — Full Scan Flow Wiring
**Effort:** ~1 day | **Copilot: PARTIAL** | **File:** `app/(main)/scan.tsx`

Wire the complete pipeline: camera → digitize → infer → store → display.

```
scan.tsx:
  1. CameraCapture → imageUri + paperSize
  2. SignalExtractor.extract(imageUri, paperSize) → Float32Array[12*5000]
  3. TempFileManager.secureDelete(imageUri)   ← HIPAA: delete immediately after extraction
  4. Inference.run(signal, patientContext) → DetectedCondition[]
  5. RecordRepository.saveRecord(patientId, signal, 'camera')
  6. AnalysisRepository.saveAnalysis(recordId, results)
  7. router.push('/analyze?analysisId=' + analysisId)
```

**Copilot: YES** for navigation/repo wiring.  
**Copilot: NO** for orchestrating the Inference + clinical analysis pipeline (Task 10 from Sprint 2 plan).

---

## Execution Order

| Step | Effort | Who | Blocks |
|------|--------|-----|--------|
| 1. ONNX export ✅ | Done | — | Step 2+ |
| 2a. DemoData.ts | 0.5 day | Claude | Step 2b |
| 2b. Demo button UI | 0.5 day | Copilot | Validation |
| 3. CameraCapture.tsx | 0.5 day | Copilot | Step 4 |
| 4. SignalExtractor.ts (Option B) | 2 days | Claude + Copilot | Step 5 |
| 5. scan.tsx wiring | 1 day | Copilot + Claude | End-to-end |

**Total: ~4-5 days to end-to-end scan → analyze flow.**

---

## Sprint 2 Tasks Still Needed Before Step 5

From the Sprint 2 plan (hazy-whistling-micali.md), these must exist for Step 5 to work:

| Task | Status | Needed for |
|------|--------|------------|
| Task 4: Preprocessor.ts | TODO | Step 5 (inference input) |
| Task 5: Inference.ts | TODO | Step 5 (model output) |
| Task 9: IntervalCalculator.ts | TODO | Step 5 (clinical analysis) |
| Task 1: Wire patient DB | TODO | Step 5 (save records) |

Tasks 4, 5, 9 are Claude-only. Task 1 is Copilot-ready.

---

## Verification After Each Step

- **After Step 2:** Tap Demo → see NORM result with probability >80%
- **After Step 3:** Camera opens → can take photo → file saved + deleted
- **After Step 4:** `SignalExtractor.extract(testImage)` → Float32Array[60000] with realistic amplitude
- **After Step 5:** Full flow: point phone at ECG strip → see analysis results screen
