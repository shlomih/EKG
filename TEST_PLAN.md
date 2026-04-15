# EKG Mobile App — Test Plan

**Status:** Active  
**Last updated:** 2026-04-13  
**Applies to:** EKGMobile React Native app (Tier 1 MVP)

---

## 1. Overview

This test plan covers functional tests, HIPAA compliance verification, ML inference validation, clinical accuracy checks, and performance benchmarks. Every HIPAA checklist item from `HIPAA_COMPLIANCE_CHECKLIST.md` that applies to Tier 1 has a corresponding automated or manual test.

**Test framework:** Jest + React Native Testing Library  
**E2E framework:** Detox (iOS/Android)  
**Coverage target:** 90%+ for `src/security/`, `src/audit/`, `src/db/`, `src/ml/`

---

## 2. HIPAA Compliance Test Suite

### 2.1 Encryption at Rest (HIPAA 1.1.1, 1.1.2)

**File:** `__tests__/security/encryption.test.ts`

| # | Test Case | HIPAA Ref | Type | Pass Criteria |
|---|-----------|-----------|------|---------------|
| E-1 | Database file is not readable as plaintext SQLite | 1.1.1 | Unit | First 16 bytes of .db file != "SQLite format 3\0" |
| E-2 | Database key stored in hardware-backed keystore | 1.1.2 | Unit | `Keychain.getSecurityLevel()` returns `SECURITY_LEVEL_SECURE_HARDWARE` or `SECURITY_LEVEL_SECURE_SOFTWARE` (min) |
| E-3 | Database cannot be opened without correct key | 1.1.1 | Unit | `db.open(wrongKey)` throws error |
| E-4 | No PHI in AsyncStorage | 1.1.1 | Integration | After full workflow, `AsyncStorage.getAllKeys()` contains no patient-related keys |
| E-5 | No PHI in app's shared preferences (Android) | 1.1.1 | E2E | Inspect XML shared_prefs files — no patient names, IDs, or ECG data |
| E-6 | Encryption key is wiped on app lock | 1.1.2 | Unit | After `SessionManager.lock()`, key reference is null and DB connection is closed |

### 2.2 Encryption in Transit (HIPAA 1.1.4)

| # | Test Case | HIPAA Ref | Type | Pass Criteria |
|---|-----------|-----------|------|---------------|
| T-1 | No HTTP (plaintext) network calls in entire app | 1.1.4 | Static | Grep source code for `http://` — zero matches (only `https://`) |
| T-2 | PDF export share uses secure channel | 1.1.4 | Manual | Share sheet offers only encrypted channels (email, messaging) |
| T-3 | App makes zero network calls during inference | 5.3.1 | E2E | Network monitor shows no requests during ECG analysis |

### 2.3 No PHI in Logs (HIPAA 1.1.3)

**File:** `__tests__/security/phiLeakage.test.ts`

| # | Test Case | HIPAA Ref | Type | Pass Criteria |
|---|-----------|-----------|------|---------------|
| P-1 | console.log intercepted and sanitized | 1.1.3 | Unit | After creating patient "John Smith" and viewing record, intercepted logs contain zero instances of "John" or "Smith" |
| P-2 | console.error sanitized | 1.1.3 | Unit | Trigger error during analysis — stack trace contains no patient data |
| P-3 | Crash report contains no PHI | 1.1.3 | Unit | Simulate crash during patient view — crash payload has no names, IDs, signal data |
| P-4 | No PHI in push notification content | 5.1.5 | Unit | Notification payloads contain only generic messages ("Analysis complete") |
| P-5 | Log output during full workflow has zero PHI | 1.1.3 | Integration | Run full create-analyze-export workflow, capture all log output, assert no patient identifiers |

### 2.4 Temporary File Security (HIPAA 1.1.5)

**File:** `__tests__/security/tempFiles.test.ts`

| # | Test Case | HIPAA Ref | Type | Pass Criteria |
|---|-----------|-----------|------|---------------|
| TF-1 | Temp files deleted after PDF generation | 1.1.5 | Integration | After `ReportGenerator.generate()`, all files in temp directory are deleted |
| TF-2 | Temp files deleted after camera capture processing | 1.1.5 | Integration | After signal extraction, captured image file is deleted |
| TF-3 | Temp files overwritten before deletion | 1.1.5 | Unit | `TempFileManager.secureDelete()` writes zeros to file before `unlink()` |
| TF-4 | No ECG images in Gallery/Photos | 5.3.6 | E2E | After camera capture + analysis, DCIM/Pictures directories have no new files |
| TF-5 | No files in Downloads directory | 5.3.6 | E2E | After PDF export via share, Downloads has no new EKG files |

### 2.5 Access Controls (HIPAA 1.2.1, 1.2.2, 1.2.3)

**File:** `__tests__/security/auth.test.ts`

| # | Test Case | HIPAA Ref | Type | Pass Criteria |
|---|-----------|-----------|------|---------------|
| A-1 | App requires biometric auth before showing any PHI | 1.2.1 | E2E | On cold start, patient list is inaccessible until biometric succeeds |
| A-2 | Failed biometric blocks all data access | 1.2.1 | Unit | `AuthProvider.isAuthenticated` = false after biometric failure |
| A-3 | PIN fallback available when biometrics unavailable | 1.2.1 | Unit | On device without biometrics, PIN entry screen is presented |
| A-4 | Auto-lock after 5 min inactivity | 1.2.2 | Unit | After `SessionManager` timer expires, `isAuthenticated` becomes false |
| A-5 | Auto-lock timer resets on user interaction | 1.2.2 | Unit | Touch event resets the timer to full 5 minutes |
| A-6 | Sensitive data cleared from memory on lock | 1.2.2 | Unit | After lock, Zustand stores have no patient data loaded |
| A-7 | FLAG_SECURE set on all PHI screens (Android) | 1.2.3 | E2E | Attempt screenshot on patient detail screen — result is blank |
| A-8 | Blur overlay on app backgrounding (iOS) | 1.2.3 | E2E | App switch shows blurred/hidden content, not actual PHI |
| A-9 | Device without passcode shows security warning | 1.2.1 | E2E | App shows warning and refuses to store data if no device lock set |

### 2.6 Audit Controls (HIPAA 1.3.1, 1.3.2, 1.3.3)

**File:** `__tests__/security/auditLog.test.ts`

| # | Test Case | HIPAA Ref | Type | Pass Criteria |
|---|-----------|-----------|------|---------------|
| AU-1 | Auth success logged | 1.3.1 | Unit | After biometric success, audit_log has AUTH_SUCCESS entry |
| AU-2 | Auth failure logged | 1.3.1 | Unit | After biometric failure, audit_log has AUTH_FAIL entry |
| AU-3 | Patient creation logged | 1.3.1 | Unit | After `savePatient()`, audit_log has PHI_CREATE entry with resource_id = patient_id |
| AU-4 | Patient view logged | 1.3.1 | Unit | After `getPatient()`, audit_log has PHI_VIEW entry |
| AU-5 | Analysis creation logged | 1.3.1 | Unit | After `saveAnalysis()`, audit_log has PHI_CREATE entry |
| AU-6 | PDF export logged | 1.3.1 | Unit | After `exportPDF()`, audit_log has PHI_EXPORT entry |
| AU-7 | Patient deletion logged | 1.3.1 | Unit | After `deletePatient()`, audit_log has PHI_DELETE entry |
| AU-8 | Audit entries contain NO PHI content | 1.3.1 | Unit | Scan all audit_log rows — no patient names, IDs, or signal data in any field |
| AU-9 | Hash chain is valid after 50 operations | 1.3.2 | Unit | After 50 mixed operations, `IntegrityVerifier.verify()` returns true |
| AU-10 | Tampered entry detected | 1.3.2 | Unit | Manually UPDATE one audit_log row → `IntegrityVerifier.verify()` returns false with index of tampered entry |
| AU-11 | Deleted entry detected | 1.3.2 | Unit | DELETE one audit_log row → `IntegrityVerifier.verify()` returns false |
| AU-12 | Audit log is append-only | 1.3.2 | Unit | Direct SQL `UPDATE audit_log SET ...` blocked by trigger or check |
| AU-13 | User can view their audit log | 1.3.3 | E2E | Settings > Audit Log screen displays entries with timestamps and event types |
| AU-14 | "Delete All Data" preserves audit log | 5.1.2 | Unit | After `deleteAllData()`, audit_log still contains all entries + new PHI_DELETE entry |

### 2.7 Data Integrity (HIPAA 1.4.1, 1.4.2)

**File:** `__tests__/security/integrity.test.ts`

| # | Test Case | HIPAA Ref | Type | Pass Criteria |
|---|-----------|-----------|------|---------------|
| I-1 | ECG record checksum verified on load | 1.4.1 | Unit | `RecordRepository.get()` verifies SHA-256 matches stored checksum |
| I-2 | Corrupted ECG detected | 1.4.1 | Unit | Manually corrupt signal_data blob → load throws integrity error |
| I-3 | ONNX model SHA-256 verified on load | 1.4.2 | Unit | `ModelManager.load()` checks hash against manifest |
| I-4 | Tampered model rejected | 1.4.2 | Unit | Provide wrong expected hash → `ModelManager.load()` throws |
| I-5 | Thresholds file integrity checked | 1.4.2 | Unit | `thresholds_v3.json` hash verified before use |

### 2.8 Physical Safeguards (HIPAA 2.1)

| # | Test Case | HIPAA Ref | Type | Pass Criteria |
|---|-----------|-----------|------|---------------|
| PH-1 | `allowBackup="false"` in Android manifest | 2.1.4 | Static | Parse `AndroidManifest.xml`, verify attribute present |
| PH-2 | iOS backup exclusion configured | 2.1.4 | Static | NSURLIsExcludedFromBackupKey set on data directory |
| PH-3 | No writes to external/SD storage | 2.1.3 | E2E | After full workflow, no files on external storage with app identifier |
| PH-4 | App data wiped on factory reset | 2.1.1 | Manual | Standard OS behavior — document in test report |

### 2.9 App-Specific Data Handling (HIPAA 5.1)

**File:** `__tests__/security/dataHandling.test.ts`

| # | Test Case | HIPAA Ref | Type | Pass Criteria |
|---|-----------|-----------|------|---------------|
| D-1 | "Delete All My Data" removes all patients | 5.1.2 | Unit | After delete, `SELECT COUNT(*) FROM patients` = 0 |
| D-2 | "Delete All My Data" removes all records | 5.1.2 | Unit | After delete, `SELECT COUNT(*) FROM ekg_records` = 0 |
| D-3 | "Delete All My Data" removes all analyses | 5.1.2 | Unit | After delete, `SELECT COUNT(*) FROM analysis_results` = 0 |
| D-4 | "Delete All My Data" preserves audit log | 5.1.2 | Unit | After delete, audit_log rows still exist + new PHI_DELETE entry |
| D-5 | Data export generates valid PDF | 5.1.3 | Integration | Exported PDF is readable, contains patient info + ECG image + results |
| D-6 | Data export generates signal file | 5.1.3 | Integration | Exported signal file is valid, contains correct lead count and sample count |
| D-7 | App requests only camera permission | 5.3.7 | Static | `app.json` permissions list contains only CAMERA |
| D-8 | Blocked permissions verified | 5.3.7 | Static | `app.json` blockedPermissions includes contacts, location, microphone |
| D-9 | Clinical disclaimer shown on every result | 5.3.5 | E2E | After analysis, disclaimer text is visible on screen |
| D-10 | Disclaimer includes "Not FDA-cleared" | 5.3.5 | E2E | Disclaimer contains required legal text |
| D-11 | No ad SDKs in bundle | 5.2.2 | Static | Scan `package.json` and `node_modules` for ad SDK identifiers |

---

## 3. ML Inference Test Suite

**File:** `__tests__/ml/`

### 3.1 Feature Extraction Parity

| # | Test Case | File | Pass Criteria |
|---|-----------|------|---------------|
| ML-1 | Voltage features match Python for synthetic normal ECG | `voltageFeatures.test.ts` | Max abs diff < 0.01 per feature |
| ML-2 | Voltage features match Python for LVH ECG (high Sokolow) | `voltageFeatures.test.ts` | Max abs diff < 0.01 per feature |
| ML-3 | Voltage features match Python for AFIB ECG (irregular RR) | `voltageFeatures.test.ts` | Max abs diff < 0.01 per feature |
| ML-4 | RR features match Python for regular sinus rhythm | `rrFeatures.test.ts` | Max abs diff < 0.01 per feature |
| ML-5 | RR features match Python for AFIB (high irregularity) | `rrFeatures.test.ts` | Max abs diff < 0.01 per feature |
| ML-6 | RR features return fallback for flat/zero signal | `rrFeatures.test.ts` | Returns [0.80, 0.03, 0.03, 0.04] |
| ML-7 | RR features return fallback for < 3 peaks | `rrFeatures.test.ts` | Returns fallback values |
| ML-8 | Feature extraction handles 250 Hz input (resampled) | `voltageFeatures.test.ts` | Signal resampled to 5000 before extraction |
| ML-9 | Feature extraction handles all-negative leads | `voltageFeatures.test.ts` | No NaN or Inf in output |
| ML-10 | Age normalization: 0-year-old → 0.0, 80 → 1.0 | `voltageFeatures.test.ts` | Exact values |

### 3.2 ONNX Inference

| # | Test Case | File | Pass Criteria |
|---|-----------|------|---------------|
| ML-11 | ONNX output matches PyTorch for reference ECG #1 | `inference.test.ts` | Max logit diff < 1e-4 |
| ML-12 | ONNX output matches PyTorch for reference ECG #2 (AFIB) | `inference.test.ts` | Max logit diff < 1e-4 |
| ML-13 | ONNX output matches PyTorch for reference ECG #3 (Brady) | `inference.test.ts` | Max logit diff < 1e-4 |
| ML-14 | Temperature scaling applied correctly (global T) | `inference.test.ts` | logits / T matches expected |
| ML-15 | Temperature scaling applied correctly (per-class T) | `inference.test.ts` | logits / T_array matches expected |
| ML-16 | Per-class thresholds applied correctly | `inference.test.ts` | Detected conditions match expected list |
| ML-17 | Conditions sorted by urgency desc, then confidence desc | `inference.test.ts` | Sort order verified |
| ML-18 | Inference handles zero signal gracefully | `inference.test.ts` | Returns valid result (likely NORM, no crash) |
| ML-19 | Inference handles very short signal (resampled) | `inference.test.ts` | Signal padded/resampled to 5000, valid result |

### 3.3 Test Data Generation

**File:** `scripts/generate_test_data.py`

Generates reference data for TS↔Python parity testing:
1. 10+ synthetic ECG signals with known characteristics
2. Python feature extraction outputs for each signal
3. Python ONNX inference outputs for each signal
4. Saved as JSON fixtures in `__tests__/fixtures/`

---

## 4. Clinical Analysis Test Suite

**File:** `__tests__/analysis/`

### 4.1 Interval Calculator

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| C-1 | Heart rate from regular 75bpm signal | HR = 75 +/- 2 bpm |
| C-2 | Heart rate from 120bpm tachycardia signal | HR = 120 +/- 3 bpm |
| C-3 | Heart rate from 45bpm bradycardia signal | HR = 45 +/- 2 bpm |
| C-4 | R-peak detection on clean signal | All peaks detected (verified against known positions) |
| C-5 | R-peak detection on noisy signal | >90% peaks detected |
| C-6 | QTc calculation (Bazett) for known QT/RR | QTc matches within 10ms |
| C-7 | Handles missing leads gracefully | Returns partial results, no crash |

### 4.2 Clinical Rules

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| C-8 | Normal axis (60 degrees) | No axis deviation flagged |
| C-9 | LAD axis (-45 degrees) | LAD detected |
| C-10 | RAD axis (+110 degrees) | RAD detected |
| C-11 | Extreme axis (-120 degrees) | Extreme axis detected |
| C-12 | Sokolow-Lyon > 3.5mV → LVH voltage criteria | LVH flagged |
| C-13 | Sokolow-Lyon < 3.5mV → no LVH | LVH not flagged |
| C-14 | Low voltage all limb leads | Low voltage detected |
| C-15 | Tall T-waves in anterior leads | Hyperkalemia warning |

### 4.3 ST Territory

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| C-16 | ST elevation in V1-V4 → LAD territory | Anterior territory identified |
| C-17 | ST elevation in II, III, aVF → RCA territory | Inferior territory identified |
| C-18 | ST elevation in I, aVL, V5-V6 → LCx territory | Lateral territory identified |
| C-19 | Reciprocal changes detected | Reciprocal leads flagged |
| C-20 | No ST elevation → no territory flagged | Clean result |

---

## 5. Integration Tests

**File:** `__tests__/integration/`

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| INT-1 | Full workflow: create patient → store ECG → analyze → view → export → delete | All steps succeed, audit trail complete |
| INT-2 | Audit log covers entire workflow | Each step produces expected audit entry |
| INT-3 | No orphaned files after delete | Temp directory and app-private storage clean |
| INT-4 | Database consistency after crash simulation | DB opens cleanly after simulated crash during write |
| INT-5 | Multiple patients, records, analyses | 10 patients × 3 records each — all CRUD correct |
| INT-6 | Edge case: empty ECG file | Error displayed, no crash, audit entry for failed analysis |
| INT-7 | Edge case: very short recording (< 2 seconds) | Padded/resampled, analysis completes with warning |
| INT-8 | Edge case: all-zero signal | Analysis returns NORM with low confidence, no crash |
| INT-9 | Language switching mid-session | UI updates to new language, data unaffected |
| INT-10 | App resume after background | Re-authentication required, data intact |

---

## 6. Performance Benchmarks

**File:** `__tests__/performance/`

| # | Metric | Target | Device |
|---|--------|--------|--------|
| PERF-1 | ONNX inference latency | < 1,000 ms | Midrange (Pixel 6a / iPhone 12) |
| PERF-2 | Feature extraction latency | < 100 ms | Midrange |
| PERF-3 | ECG 12-lead rendering FPS | >= 30 fps | Midrange |
| PERF-4 | ECG scroll/zoom FPS | >= 30 fps | Midrange |
| PERF-5 | Database query (patient list, 100 patients) | < 200 ms | Midrange |
| PERF-6 | Database write (save analysis) | < 100 ms | Midrange |
| PERF-7 | App cold start to auth screen | < 3,000 ms | Midrange |
| PERF-8 | PDF generation | < 5,000 ms | Midrange |
| PERF-9 | Memory usage during inference | < 200 MB peak | Midrange |
| PERF-10 | App size (installed) | < 100 MB | Both platforms |

---

## 7. Static Analysis & Security Scanning

| # | Check | Tool | Frequency |
|---|-------|------|-----------|
| SA-1 | TypeScript strict mode — no `any` in security modules | `tsc --strict` | Every build |
| SA-2 | No hardcoded secrets/keys in source | `detect-secrets` | Pre-commit |
| SA-3 | Dependency vulnerability scan | `npm audit` | Weekly + pre-release |
| SA-4 | No `http://` URLs in source (HIPAA 1.1.4) | Grep | Pre-commit |
| SA-5 | No `console.log` with patient-related strings | ESLint custom rule | Every build |
| SA-6 | No ad SDK dependencies | Package audit | Pre-release |
| SA-7 | `allowBackup="false"` in manifest | Build check | Every build |
| SA-8 | Minimum API level enforced (Android 26+) | Build config | Every build |

---

## 8. Manual Test Procedures

These tests require physical devices and cannot be fully automated:

| # | Procedure | HIPAA Ref | Expected |
|---|-----------|-----------|----------|
| M-1 | Lock phone → unlock → app shows auth screen | 1.2.2 | Re-authentication required |
| M-2 | Kill app from recents → reopen | 1.2.1 | Auth screen shown, no cached PHI visible |
| M-3 | Take screenshot on patient detail (Android) | 1.2.3 | Screenshot is blank/black |
| M-4 | View app in recents/app switcher (iOS) | 1.2.3 | Content is blurred/hidden |
| M-5 | Connect phone to computer, browse app files | 1.1.1 | Database file is encrypted gibberish |
| M-6 | Restore from iCloud/Google backup | 2.1.4 | App data not in backup (allowBackup=false) |
| M-7 | Paper ECG capture → verify no image in Gallery | 5.3.6 | Photos app shows no new ECG images |
| M-8 | Export PDF → verify temp file deleted | 1.1.5 | No orphaned PDF/PNG in app directory |

---

## 9. Test Data Requirements

### 9.1 Synthetic ECG Fixtures (generated by `scripts/generate_test_data.py`)

| Fixture | Description | Labels |
|---------|-------------|--------|
| `normal_sinus.json` | Clean 75bpm sinus rhythm | NORM |
| `afib_irregular.json` | Irregular RR, no P waves | AFIB |
| `bradycardia_45.json` | 45bpm regular rhythm | Brady |
| `tachycardia_130.json` | 130bpm sinus tachycardia | STACH |
| `lvh_sokolow.json` | High Sokolow-Lyon voltage | LVH |
| `stemi_anterior.json` | ST elevation V1-V4 | ASMI |
| `stemi_inferior.json` | ST elevation II, III, aVF | IMI |
| `lbbb.json` | Wide QRS, LBBB morphology | CLBBB |
| `rbbb.json` | RSR' in V1, wide S in I | CRBBB |
| `zero_signal.json` | All zeros | Edge case |
| `short_signal.json` | 2 seconds only | Edge case |

### 9.2 Reference Outputs (from Python)

For each fixture, generate:
- Voltage feature vector (14-dim float32)
- RR feature vector (4-dim float32)
- Combined feature vector (18-dim float32)
- ONNX model logits (26-dim float32)
- Post-threshold detected conditions list
- Temperature-scaled probabilities

---

## 10. HIPAA Checklist → Test Mapping

Complete traceability from every Tier 1 HIPAA requirement to its test(s):

| HIPAA Item | Requirement | Test(s) |
|-----------|-------------|---------|
| 1.1.1 | Encryption at rest (AES-256) | E-1, E-3, E-4, E-5 |
| 1.1.2 | Hardware-backed keystore | E-2, E-6 |
| 1.1.3 | No PHI in logs | P-1, P-2, P-3, P-5 |
| 1.1.4 | TLS 1.2+ in transit | T-1, T-2 |
| 1.1.5 | Temp files securely deleted | TF-1, TF-2, TF-3 |
| 1.2.1 | Biometric/PIN authentication | A-1, A-2, A-3, A-9 |
| 1.2.2 | Auto-lock on inactivity | A-4, A-5, A-6 |
| 1.2.3 | Screen capture prevention | A-7, A-8 |
| 1.3.1 | Audit log of PHI access | AU-1 through AU-8 |
| 1.3.2 | Tamper-resistant audit log | AU-9, AU-10, AU-11, AU-12 |
| 1.3.3 | User can view audit log | AU-13 |
| 1.4.1 | ECG data integrity checksum | I-1, I-2 |
| 1.4.2 | Model integrity check | I-3, I-4, I-5 |
| 2.1.1 | Data wiped on factory reset | PH-4 |
| 2.1.3 | No external/SD storage writes | PH-3 |
| 2.1.4 | Excluded from backups | PH-1, PH-2 |
| 5.1.2 | Delete all data | D-1, D-2, D-3, D-4 |
| 5.1.3 | Data export | D-5, D-6 |
| 5.1.5 | No PHI in notifications | P-4 |
| 5.2.2 | No ad SDKs | D-11 |
| 5.3.1 | On-device inference | T-3, ML-11 |
| 5.3.5 | Clinical disclaimer | D-9, D-10 |
| 5.3.6 | No Gallery/Photos storage | TF-4, TF-5 |
| 5.3.7 | Minimum permissions | D-7, D-8 |

---

## 11. Test Execution Schedule

| Phase | Tests to Run | Gate |
|-------|-------------|------|
| Phase 1 (Security) | E-*, A-*, AU-*, PH-1, PH-2 | All pass before Phase 2 |
| Phase 2 (Inference) | ML-*, I-3, I-4, I-5, T-3 | All pass before Phase 3 |
| Phase 3 (Patient) | D-*, AU-3 through AU-8, AU-14 | All pass before Phase 4 |
| Phase 4 (ECG/Analysis) | C-*, D-9, D-10 | All pass before Phase 5 |
| Phase 5 (Camera/Reports) | TF-*, D-5, D-6 | All pass before Phase 6 |
| Phase 6 (Polish) | INT-*, PERF-*, SA-*, M-* | All pass before release |
| Pre-release | Full suite (all tests) | Zero failures |
