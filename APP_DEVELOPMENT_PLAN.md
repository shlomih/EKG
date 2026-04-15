# EKG Intelligence Platform — Mobile App Development Plan

**Status:** Active development  
**Last updated:** 2026-04-13  
**Target:** HIPAA-compliant native mobile app (iOS + Android)  
**Current phase:** Phase 1 — Security Foundation  

---

## 1. Executive Summary

Transition the Streamlit POC (`app.py`) to a production HIPAA-compliant mobile app. The app performs **on-device** 26-class ECG classification using the trained ECGNetJoint model (AUROC=0.9682), with encrypted storage, biometric authentication, and a complete audit trail.

**Scope:** Tier 1 (MVP) only — fully offline, single user, no cloud backend. Tier 2 (multi-user, clinic sales) is deferred until there are paying users.

**Key principle:** HIPAA compliance is baked in from line 1, not bolted on later.

---

## 2. Tech Stack

| Layer | Technology | HIPAA Rationale |
|-------|-----------|-----------------|
| Framework | React Native + Expo SDK 52+ | Single codebase, managed native modules for biometrics/crypto |
| Language | TypeScript (strict mode) | Type safety prevents accidental PHI leaks at compile time |
| Inference | ONNX Runtime Mobile | On-device only — ECG data never leaves the phone (HIPAA 5.3.1) |
| Database | op-sqlite + SQLCipher | AES-256 encryption at rest (HIPAA 1.1.1) |
| Key Storage | react-native-keychain | Hardware-backed keystore (HIPAA 1.1.2) |
| Auth | expo-local-authentication | Biometric gate (HIPAA 1.2.1) |
| ECG Rendering | @shopify/react-native-skia | GPU-accelerated, no external data transfer |
| State | Zustand | In-memory only, never persisted unencrypted |
| Navigation | Expo Router | File-based routing, auth guard at root |
| PDF | react-native-html-to-pdf | On-device generation, no server |
| Camera | expo-camera | Paper ECG capture, temp files securely deleted (HIPAA 1.1.5) |
| i18n | i18next + react-i18next | Port existing EN/ES/FR translations |

### Why React Native (not Flutter or native)?

- **Solo developer:** Single codebase = half the work vs Kotlin + Swift
- **Expo ecosystem:** Managed native modules for every HIPAA requirement
- **ONNX Runtime:** Official React Native bindings (Microsoft-maintained)
- **SQLCipher:** Mature RN bindings via op-sqlite
- **Skia:** Shopify's production-tested rendering engine

---

## 3. Architecture

### 3.1 Security Architecture

```
┌─────────────────────────────────────────────────┐
│                 App Process                       │
│                                                   │
│  ┌──────────────┐    ┌───────────────────────┐   │
│  │ AuthProvider  │───>│ SessionManager        │   │
│  │ (biometric)   │    │ (5-min auto-lock)     │   │
│  └──────┬───────┘    └───────────────────────┘   │
│         │                                         │
│         ▼                                         │
│  ┌──────────────┐    ┌───────────────────────┐   │
│  │ KeyManager   │───>│ SQLCipher Database     │   │
│  │ (Keychain)   │    │ (AES-256 encrypted)    │   │
│  └──────────────┘    └───────────┬───────────┘   │
│                                   │               │
│  ┌──────────────┐    ┌───────────▼───────────┐   │
│  │ PHISanitizer │    │ AuditLogger           │   │
│  │ (log filter)  │    │ (hash-chained log)    │   │
│  └──────────────┘    └───────────────────────┘   │
│                                                   │
│  ┌──────────────┐    ┌───────────────────────┐   │
│  │ ScreenProtect│    │ TempFileManager       │   │
│  │ (FLAG_SECURE) │    │ (secure delete)       │   │
│  └──────────────┘    └───────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### 3.2 Data Flow

```
[Camera/File Input]
       │
       ▼
[In-Memory Buffer] ──────> [Secure Delete temp files]
       │
       ▼
[Signal Preprocessing]     (resample to 5000, normalize /5.0)
       │
       ├──> [VoltageFeatures.ts] ──> 14-dim voltage vector
       ├──> [RRFeatures.ts]     ──> 4-dim RR-interval vector
       │
       ▼
[ONNX Inference]           (signal + 18-dim aux → 26 logits)
       │
       ▼
[Temperature Scaling + Thresholds] ──> Detected conditions
       │
       ▼
[Clinical Analysis]        (intervals, rules, ST territory)
       │
       ├──> [Display Results]   (FLAG_SECURE active)
       ├──> [Store Encrypted]   (SQLCipher → patients/records/analyses)
       ├──> [Audit Log Entry]   (event type + resource ID, never PHI)
       └──> [PDF Export]        (temp file → share → secure delete)
```

### 3.3 Database Schema

```sql
-- Encrypted via SQLCipher (AES-256)
-- Key stored in Android Keystore / iOS Secure Enclave

CREATE TABLE patients (
    patient_id TEXT PRIMARY KEY,           -- UUID v4
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    id_number TEXT,
    age INTEGER,
    sex TEXT DEFAULT 'M' CHECK(sex IN ('M','F')),
    has_pacemaker INTEGER DEFAULT 0,
    is_athlete INTEGER DEFAULT 0,
    is_pregnant INTEGER DEFAULT 0,
    k_level REAL DEFAULT 4.0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE ekg_records (
    ekg_id TEXT PRIMARY KEY,               -- UUID v4
    patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    signal_data BLOB NOT NULL,             -- compressed float32 array (12 × 5000)
    sampling_rate INTEGER DEFAULT 500,
    lead_count INTEGER DEFAULT 12,
    acquisition_source TEXT,               -- 'camera', 'file_upload', 'bluetooth'
    checksum TEXT NOT NULL,                -- SHA-256 of signal_data (HIPAA 1.4.1)
    captured_at TEXT DEFAULT (datetime('now')),
    notes TEXT
);

CREATE TABLE analysis_results (
    analysis_id TEXT PRIMARY KEY,          -- UUID v4
    ekg_id TEXT NOT NULL REFERENCES ekg_records(ekg_id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,           -- 'v3_26class'
    model_hash TEXT NOT NULL,              -- SHA-256 of ONNX model used
    primary_condition TEXT NOT NULL,
    conditions_json TEXT NOT NULL,         -- JSON array of detected conditions
    scores_json TEXT NOT NULL,             -- JSON object: {code: probability}
    intervals_json TEXT,                   -- JSON: {hr, pr, qrs, qtc, ...}
    clinical_rules_json TEXT,             -- JSON: clinical rules findings
    st_territory_json TEXT,               -- JSON: ST territory analysis
    disclaimer_shown INTEGER DEFAULT 1,    -- HIPAA 5.3.5: track disclaimer display
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,               -- ISO 8601 UTC
    event_type TEXT NOT NULL,              -- AUTH_SUCCESS, AUTH_FAIL, PHI_VIEW, PHI_CREATE, etc.
    resource_type TEXT,                    -- 'patient', 'ekg_record', 'analysis', 'report'
    resource_id TEXT,                      -- UUID of accessed resource (NOT the PHI itself)
    action TEXT NOT NULL,                  -- 'create_patient', 'view_analysis', 'export_pdf', etc.
    details TEXT,                          -- Context (never contains PHI)
    integrity_hash TEXT NOT NULL           -- SHA-256(prev_hash + this_record)
);
-- Note: No UPDATE or DELETE triggers on audit_log — append-only by design

CREATE TABLE app_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
);
-- Stores: inactivity_timeout, language, theme, data_retention_days
```

---

## 4. Implementation Phases

### Phase 1: Security Foundation

**Goal:** HIPAA-compliant security shell before any feature code.

| # | Task | HIPAA Ref | File |
|---|------|-----------|------|
| 1.1 | Expo project + `allowBackup=false` | 2.1.4 | `app.json` |
| 1.2 | AES-256 key generation + hardware keystore | 1.1.2 | `src/security/KeyManager.ts` |
| 1.3 | SQLCipher database with schema | 1.1.1 | `src/db/Database.ts`, `src/db/schema.ts` |
| 1.4 | Biometric authentication gate | 1.2.1 | `src/security/AuthProvider.tsx` |
| 1.5 | Inactivity auto-lock (5 min default) | 1.2.2 | `src/security/SessionManager.ts` |
| 1.6 | Screen capture prevention | 1.2.3 | `src/security/ScreenProtection.ts` |
| 1.7 | Audit logger with hash chain | 1.3.1, 1.3.2 | `src/audit/AuditLogger.ts` |
| 1.8 | PHI sanitizer for logs/errors | 1.1.3 | `src/security/PHISanitizer.ts` |
| 1.9 | Root/jailbreak detection | 2.1.1 | `src/security/DeviceIntegrity.ts` |

**Exit criteria:**
- [ ] Database encrypted (hex dump = gibberish)
- [ ] App requires biometric to open
- [ ] Auto-locks after 5 min
- [ ] Screenshots blocked (Android)
- [ ] Audit log records auth events

### Phase 2: Model Export + On-Device Inference

**Goal:** ONNX model running on mobile with verified parity to PyTorch.

| # | Task | HIPAA Ref | File |
|---|------|-----------|------|
| 2.1 | PyTorch → ONNX export script | — | `scripts/export_onnx.py` |
| 2.2 | ONNX verification (PyTorch parity) | — | `scripts/verify_onnx.py` |
| 2.3 | Port voltage feature extraction | — | `src/ml/VoltageFeatures.ts` |
| 2.4 | Port RR-interval feature extraction | — | `src/ml/RRFeatures.ts` |
| 2.5 | Model loader + SHA-256 integrity | 1.4.2 | `src/ml/ModelManager.ts` |
| 2.6 | Inference pipeline (preprocess → ONNX → thresholds) | 5.3.1 | `src/ml/Inference.ts` |
| 2.7 | Condition metadata (26 codes, urgency, guidance) | — | `src/ml/ConditionMetadata.ts` |

**Source files to port:**
- `cnn_classifier.py:169-292` → `VoltageFeatures.ts` + `RRFeatures.ts`
- `multilabel_v3.py:636-731` → `ConditionMetadata.ts`
- `multilabel_v3.py:760-831` → `Inference.ts`

**Exit criteria:**
- [ ] ONNX output matches PyTorch within 1e-4 on 100 test ECGs
- [ ] TypeScript feature extraction matches Python within 0.01 per feature
- [ ] Model integrity check passes/fails correctly
- [ ] Zero network calls during inference

### Phase 3: Patient Management + Database UI

**Goal:** Full CRUD with audit trail, matching current `database_setup.py` schema.

| # | Task | HIPAA Ref | File |
|---|------|-----------|------|
| 3.1 | TypeScript types (Patient, ECGRecord, Analysis) | — | `src/types/*` |
| 3.2 | PatientRepository (CRUD + audit) | 1.3.1 | `src/db/PatientRepository.ts` |
| 3.3 | RecordRepository (CRUD + checksum) | 1.4.1 | `src/db/RecordRepository.ts` |
| 3.4 | AnalysisRepository (CRUD + audit) | 1.3.1 | `src/db/AnalysisRepository.ts` |
| 3.5 | Patient list screen | — | `app/(main)/patients.tsx` |
| 3.6 | Patient detail + history screen | — | `app/(main)/patient/[id].tsx` |
| 3.7 | "Delete All My Data" action | 5.1.2 | Settings screen |
| 3.8 | Data export (PDF + signal) | 5.1.3 | Export flow |

**Exit criteria:**
- [ ] All CRUD creates audit entries
- [ ] Cascade delete works (patient → records → analyses)
- [ ] Deletion creates audit entry (audit log preserved)
- [ ] Data export works via share sheet

### Phase 4: ECG Visualization + Clinical Analysis

**Goal:** Clinical-quality 12-lead display and interval measurement on mobile.

| # | Task | Source | File |
|---|------|--------|------|
| 4.1 | ECG background grid (1mm/5mm) | `app.py:263-343` | `src/ecg/ECGGrid.tsx` |
| 4.2 | Single lead strip renderer | `app.py:263-343` | `src/ecg/ECGLeadStrip.tsx` |
| 4.3 | 12-lead view (4×3 + rhythm) | `app.py:263-343` | `src/ecg/ECG12LeadView.tsx` |
| 4.4 | Pinch-to-zoom + pan gestures | — | Gesture handlers |
| 4.5 | R-peak detection | `cnn_classifier.py` | `src/analysis/IntervalCalculator.ts` |
| 4.6 | HR/PR/QRS/QTc calculation | `interval_calculator.py` | `src/analysis/IntervalCalculator.ts` |
| 4.7 | Clinical rules (axis, voltage, T-wave) | `clinical_rules.py` | `src/analysis/ClinicalRules.ts` |
| 4.8 | ST territory mapping | `st_territory.py` | `src/analysis/STTerritory.ts` |
| 4.9 | Clinical disclaimer on results | HIPAA 5.3.5 | All result screens |

### Phase 5: Camera + Digitization + Reports

| # | Task | HIPAA Ref | File |
|---|------|-----------|------|
| 5.1 | Camera capture with guides | 5.3.7 | `src/digitization/CameraCapture.tsx` |
| 5.2 | Signal extraction from image | — | `src/digitization/SignalExtractor.ts` |
| 5.3 | Secure temp file handling | 1.1.5 | `src/utils/TempFileManager.ts` |
| 5.4 | PDF report HTML template | — | `src/reports/ReportTemplate.ts` |
| 5.5 | PDF generation orchestrator | — | `src/reports/ReportGenerator.ts` |
| 5.6 | Share intent (iOS/Android) | — | Share flow |

### Phase 6: i18n + Testing + Polish

| # | Task | File |
|---|------|------|
| 6.1 | Port EN/ES/FR strings | `src/i18n/*` |
| 6.2 | HIPAA compliance test suite | `__tests__/security/*` |
| 6.3 | ML inference test suite | `__tests__/ml/*` |
| 6.4 | Clinical analysis test suite | `__tests__/analysis/*` |
| 6.5 | Integration tests (full workflow) | `__tests__/integration/*` |
| 6.6 | Performance benchmarks | `__tests__/performance/*` |
| 6.7 | Accessibility review | All screens |

---

## 5. ONNX Export Strategy

### 5.1 Export Script

```python
# scripts/export_onnx.py
# Exports ECGNetJoint from PyTorch .pt to ONNX format for mobile inference
#
# Input:  models/ecg_multilabel_v3_best.pt (PyTorch checkpoint)
# Output: EKGMobile/assets/models/ecg_v3.onnx
#         EKGMobile/assets/models/model_manifest.json (SHA-256 hash)
```

**Model inputs:**
- `signal`: float32 tensor (1, 12, 5000) — 12-lead ECG, 10 seconds at 500 Hz, normalized by /5.0
- `aux`: float32 tensor (1, 18) — voltage + demographic + RR features

**Model output:**
- `logits`: float32 tensor (1, 26) — raw logits, sigmoid applied post-inference

**Post-processing (in TypeScript, not in ONNX):**
1. Apply temperature scaling: `logits / T`
2. Apply sigmoid: `1 / (1 + exp(-logits))`
3. Compare each class probability against per-class threshold from `thresholds_v3.json`
4. Sort detected conditions by urgency (descending) then confidence (descending)

### 5.2 Feature Extraction Parity

The 18-dim auxiliary feature vector is computed **on-device in TypeScript**, not by the model. Must match Python exactly:

| Index | Feature | Python Source | Normalization |
|-------|---------|--------------|---------------|
| 0 | S(V1) depth | `cnn_classifier.py:248-250` | /3.0, clamp [0,2] |
| 1 | max(R_V5, R_V6) | `cnn_classifier.py:251-252` | /3.0, clamp [0,2] |
| 2 | Sokolow-Lyon | computed | /5.0, clamp [0,2] |
| 3 | Sokolow met | >3.5 mV → 1.0 | binary |
| 4 | R(aVL) | `cnn_classifier.py:253` | /2.0, clamp [0,2] |
| 5 | S(V3) | `cnn_classifier.py:254` | /2.0, clamp [0,2] |
| 6 | Cornell value | computed | /4.0, clamp [0,2] |
| 7 | R(V1) | `cnn_classifier.py:255` | /2.0, clamp [0,2] |
| 8 | Sex (F=1, M=0) | input | binary |
| 9 | Age normalized | input | /80.0, clamp [0,1] |
| 10 | Frontal QRS axis | `cnn_classifier.py:261-264` | /180.0, clamp [-1,1] |
| 11 | T-wave strain | `_t_wave_strain_score()` | [0,1] |
| 12 | QRS duration norm | `_qrs_duration_norm()` | /200ms, clamp [0,2] |
| 13 | Cornell VDP | computed | /2440, clamp [0,2] |
| 14 | Mean RR | `extract_rr_features()` | clamp [0.3, 2.0] seconds |
| 15 | SDNN norm | `extract_rr_features()` | /0.2s, clamp [0,1] |
| 16 | RMSSD norm | `extract_rr_features()` | /0.2s, clamp [0,1] |
| 17 | Irregularity | `extract_rr_features()` | clamp [0, 0.5] |

---

## 6. Security Implementation Details

### 6.1 Encryption Key Lifecycle

```
App First Launch:
  1. Generate 256-bit random key
  2. Store in Keychain with accessControl: BIOMETRY_ANY_OR_DEVICE_PASSCODE
  3. Key never leaves hardware security module

App Open:
  1. Biometric prompt → success → retrieve key from Keychain
  2. Open SQLCipher with key
  3. Start inactivity timer

Inactivity / Background:
  1. Close DB connection (key wiped from memory)
  2. Clear sensitive UI state
  3. Show blur overlay / lock screen
  4. Require re-authentication to resume
```

### 6.2 Audit Log Hash Chain

Each audit entry's integrity hash includes the previous entry's hash, creating a tamper-evident chain:

```
Entry 1: hash = SHA256("GENESIS" + entry1_data)
Entry 2: hash = SHA256(entry1_hash + entry2_data)
Entry 3: hash = SHA256(entry2_hash + entry3_data)
...
```

Tampering with any entry breaks the chain from that point forward. Verification scans the entire chain on-demand (Settings > Audit Log > Verify Integrity).

### 6.3 PHI Sanitization Rules

The `PHISanitizer` intercepts all logging/error reporting and strips:
- Patient names (first_name, last_name)
- Patient IDs (patient_id, id_number)
- Ages, dates of birth
- ECG signal values (any Float32Array or number[] > 100 elements)
- Analysis results (condition names in clinical context)
- File paths containing patient IDs

Allowed in logs:
- Screen names, navigation events
- Feature usage counters
- Error types and stack traces (with PHI stripped)
- App version, device model, OS version
- Performance metrics (inference time, render time)

---

## 7. npm Dependencies

### Security & HIPAA
```
react-native-keychain          # Biometric + Keychain/Keystore
expo-local-authentication      # Biometric prompts
expo-screen-capture            # Prevent screenshots
op-sqlite                      # SQLite + SQLCipher encryption
expo-crypto                    # SHA-256 hashing
```

### ML & Inference
```
onnxruntime-react-native       # On-device ONNX inference
```

### UI & Visualization
```
@shopify/react-native-skia     # ECG rendering
react-native-gesture-handler   # Pinch/pan gestures
react-native-reanimated        # Animations
expo-router                    # File-based navigation
```

### Data & Storage
```
zustand                        # State management
react-native-html-to-pdf       # PDF generation
expo-sharing                   # Share sheet
expo-file-system               # File operations
```

### Camera & Input
```
expo-camera                    # Paper ECG capture
```

### i18n
```
i18next                        # Internationalization framework
react-i18next                  # React bindings
```

### Utility
```
date-fns                       # Date formatting
uuid                           # UUID generation
```

---

## 8. App Configuration

### app.json (Expo)

```json
{
  "expo": {
    "name": "EKG Intelligence",
    "slug": "ekg-intelligence",
    "version": "1.0.0",
    "sdkVersion": "52.0.0",
    "platforms": ["ios", "android"],
    "android": {
      "allowBackup": false,
      "permissions": ["CAMERA"],
      "blockedPermissions": [
        "READ_CONTACTS", "ACCESS_FINE_LOCATION",
        "ACCESS_COARSE_LOCATION", "RECORD_AUDIO"
      ]
    },
    "ios": {
      "infoPlist": {
        "NSCameraUsageDescription": "Camera is used to capture paper ECG strips for analysis",
        "NSFaceIDUsageDescription": "Face ID is used to protect your health data"
      }
    },
    "plugins": [
      "expo-local-authentication",
      "expo-camera",
      "expo-screen-capture"
    ]
  }
}
```

---

## 9. Regulatory Positioning

### FDA Status
- **Deferred indefinitely.** App positioned as educational/informational tool.
- Every result screen shows: *"For educational purposes only. This is not a medical diagnosis. Not FDA-cleared. Always consult a qualified healthcare professional."*
- App store listing avoids medical device language.
- See `HIPAA_COMPLIANCE_CHECKLIST.md` Section 7 for details.

### HIPAA Compliance
- Full Tier 1 compliance (offline, single-user).
- See `HIPAA_COMPLIANCE_CHECKLIST.md` for item-by-item tracking.
- See `TEST_PLAN.md` for automated compliance verification.

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ONNX export breaks dual-input model | Medium | High | Verify early in Phase 2 with opset 13 |
| Feature extraction divergence (TS vs Python) | Medium | High | Test with 100+ real ECGs, use Float32Array |
| NeuroKit2 port loses accuracy | High | Medium | Accept simplified intervals for Tier 1; ML model carries diagnosis |
| SQLCipher performance on large audit logs | Low | Medium | Index on timestamp, configurable retention |
| Skia ECG rendering performance | Low | Medium | Profile early; fallback to react-native-svg |
| Expo managed workflow missing native module | Low | High | Eject to bare workflow only if absolutely needed |

---

## 11. Directory Reference

| Current POC File | Mobile Equivalent | Port Complexity |
|-----------------|-------------------|-----------------|
| `app.py` | `app/(main)/*.tsx` | High — full UI rewrite |
| `cnn_classifier.py` (features) | `src/ml/VoltageFeatures.ts`, `src/ml/RRFeatures.ts` | Medium — numerical port |
| `cnn_classifier.py` (model) | ONNX export → `assets/models/ecg_v3.onnx` | Low — automated export |
| `multilabel_v3.py` (inference) | `src/ml/Inference.ts` | Medium — pipeline port |
| `multilabel_v3.py` (metadata) | `src/ml/ConditionMetadata.ts` | Low — copy/adapt |
| `database_setup.py` | `src/db/Database.ts`, `src/db/schema.ts` | Low — schema copy + SQLCipher |
| `interval_calculator.py` | `src/analysis/IntervalCalculator.ts` | High — NeuroKit2 has no JS equivalent |
| `clinical_rules.py` | `src/analysis/ClinicalRules.ts` | Medium — pure logic port |
| `st_territory.py` | `src/analysis/STTerritory.ts` | Low — pure logic port |
| `report_generator.py` | `src/reports/ReportGenerator.ts` | Medium — HTML template approach |
| `digitization_pipeline.py` | `src/digitization/SignalExtractor.ts` | High — OpenCV → native/JS port |
| `translations/*.py` | `src/i18n/*.ts` | Low — string copy |
