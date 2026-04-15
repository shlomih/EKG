# Copilot/Haiku Tasks — EKGMobile

These are self-contained, mechanical tasks for GitHub Copilot (Haiku) to implement.
Each task has clear inputs, expected output, and a review checklist.

**Rules for Copilot:**
- Read the referenced source files before writing anything.
- Follow the existing code style (see existing files in the same directory).
- Use the app's color palette: `#071312` (bg), `#0D1F1E` (card), `#00E5B0` (primary), `#5A8A85` (muted), `#1E3533` (border), `#FF4444` (danger), `#FF8800` (warning), `#FFCC00` (caution).
- Do NOT touch files in `src/security/` or `src/audit/` — those are HIPAA-critical.
- Do NOT add dependencies to `package.json`.
- Mark each task DONE in this file when completed.

---

## Task 1: Create RecordRepository.ts
**Status:** DONE
**File to create:** `src/db/RecordRepository.ts`
**Reference files to read first:**
- `src/db/PatientRepository.ts` (follow same patterns exactly)
- `src/db/schema.ts` (table structure for `ekg_records` and `analysis_results`)
- `src/types/ECGRecord.ts` (TypeScript interfaces)
- `src/audit/AuditLogger.ts` (for `logEvent` import)

**What to implement:**
```typescript
// All functions follow the pattern in PatientRepository.ts
// All mutations must call logEvent() for HIPAA audit trail

export async function saveRecord(record: Omit<ECGRecord, 'ekg_id' | 'captured_at'>): Promise<string>
// Generate UUID, INSERT into ekg_records, logEvent('PHI_CREATE', 'create_record', 'ekg_record', ekgId)

export async function getRecord(ekgId: string): Promise<ECGRecord | null>
// SELECT from ekg_records, logEvent('PHI_VIEW', 'view_record', 'ekg_record', ekgId)

export async function listRecordsByPatient(patientId: string): Promise<ECGRecord[]>
// SELECT WHERE patient_id = ?, ORDER BY captured_at DESC
// logEvent('PHI_VIEW', 'list_records', 'ekg_record', null, `patient:${patientId},count:${n}`)

export async function deleteRecord(ekgId: string): Promise<void>
// DELETE from ekg_records (CASCADE deletes analysis_results)
// logEvent('PHI_DELETE', 'delete_record', 'ekg_record', ekgId)
```

**Row conversion:** Create `rowToRecord()` like `rowToPatient()` in PatientRepository.ts. `signal_data` stays as-is (BLOB). JSON fields (`conditions_json`, etc.) are stored as TEXT in DB.

**Review checklist:**
- [ ] All functions call `logEvent()` with correct event types
- [ ] Uses `getDatabase()` from `./Database`
- [ ] Uses `v4 as uuidv4` from `uuid`
- [ ] No console.log with patient data (PHI leak)

---

## Task 2: Create AnalysisRepository.ts
**Status:** DONE
**File to create:** `src/db/AnalysisRepository.ts`
**Reference files to read first:**
- `src/db/PatientRepository.ts` (same patterns)
- `src/db/schema.ts` (table: `analysis_results`)
- `src/types/ECGRecord.ts` (has `AnalysisResult`, `IntervalMeasurements`, `ClinicalRuleResult`, `STTerritoryResult` interfaces)

**What to implement:**
```typescript
export async function saveAnalysis(analysis: Omit<AnalysisResult, 'analysis_id' | 'created_at'>): Promise<string>
// Generate UUID. Store conditions as JSON.stringify(analysis.conditions) in conditions_json column.
// Same for scores_json, intervals_json, clinical_rules_json, st_territory_json.
// logEvent('PHI_CREATE', 'create_analysis', 'analysis', analysisId)

export async function getAnalysis(analysisId: string): Promise<AnalysisResult | null>
// Parse JSON columns back to objects: JSON.parse(row.conditions_json) → conditions array
// logEvent('PHI_VIEW', 'view_analysis', 'analysis', analysisId)

export async function listAnalysesByRecord(ekgId: string): Promise<AnalysisResult[]>
// SELECT WHERE ekg_id = ?, ORDER BY created_at DESC
// logEvent('PHI_VIEW', 'list_analyses', 'analysis', null, `ekg:${ekgId},count:${n}`)

export async function getLatestAnalysis(ekgId: string): Promise<AnalysisResult | null>
// SELECT WHERE ekg_id = ? ORDER BY created_at DESC LIMIT 1
// logEvent('PHI_VIEW', 'view_latest_analysis', 'analysis', analysisId)
```

**Review checklist:**
- [ ] JSON serialization/deserialization for all `*_json` columns
- [ ] `intervals`, `clinical_rules`, `st_territory` can be null — handle gracefully
- [ ] All functions call `logEvent()`

---

## Task 3: Create i18n setup + English strings
**Status:** DONE
**Files to create:**
- `src/i18n/i18n.ts` — i18next initialization
- `src/i18n/en.ts` — English translation strings
- `src/i18n/es.ts` — Spanish translation strings (copy structure, translate values)
- `src/i18n/fr.ts` — French translation strings (copy structure, translate values)

**Reference:** `c:\Users\osnat\Documents\Shlomi\EKG\translations\en.py` (Python source to port)

**i18n.ts setup:**
```typescript
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import en from './en';
import es from './es';
import fr from './fr';

i18n.use(initReactI18next).init({
  resources: { en: { translation: en }, es: { translation: es }, fr: { translation: fr } },
  lng: 'en',
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
});

export default i18n;
```

**en.ts — key groups to include (port from Python source):**
```typescript
export default {
  // App
  app_title: 'EKG Intelligence',
  app_subtitle: '26-Condition ECG Analysis',

  // Navigation tabs
  tab_dashboard: 'Dashboard',
  tab_scan: 'Scan ECG',
  tab_patients: 'Patients',
  tab_settings: 'Settings',

  // Dashboard
  dashboard_scan_ecg: 'Scan ECG',
  dashboard_patients: 'Patients',
  dashboard_model_info: 'Model',
  dashboard_model_version: 'V3 Multilabel (26 conditions)',
  dashboard_model_params: 'ECGNetJoint - 1.7M parameters',
  dashboard_inference: 'On-device inference (ONNX Runtime)',

  // Patient form
  patient_first_name: 'First Name',
  patient_last_name: 'Last Name',
  patient_id_number: 'Patient ID',
  patient_age: 'Age',
  patient_sex: 'Biological Sex',
  patient_sex_male: 'Male',
  patient_sex_female: 'Female',
  patient_potassium: 'Potassium K+ (mmol/L)',
  patient_pacemaker: 'Pacemaker / ICD Present',
  patient_athlete: 'Athlete Status',
  patient_pregnant: 'Pregnancy',
  patient_save: 'Save Patient',
  patient_delete: 'Delete Patient',
  patient_no_patients: 'No patients yet',

  // Analysis
  analysis_title: 'Analysis Results',
  analysis_detected: 'Detected Conditions',
  analysis_confidence: 'Confidence',
  analysis_urgency: 'Urgency',
  analysis_urgency_critical: 'Critical',
  analysis_urgency_significant: 'Significant',
  analysis_urgency_monitor: 'Monitor',
  analysis_urgency_normal: 'Normal',
  analysis_clinical_guidance: 'Clinical Guidance',
  analysis_action: 'Action',

  // Settings
  settings_title: 'Settings',
  settings_security: 'Security',
  settings_audit_log: 'View Audit Log',
  settings_auto_lock: 'Auto-Lock Timeout',
  settings_screen_protection: 'Screen Protection',
  settings_data: 'Data',
  settings_language: 'Language',
  settings_delete_all: 'Delete All My Data',
  settings_delete_confirm_title: 'Delete All Data',
  settings_delete_confirm_msg: 'This will permanently delete all patient records, ECG data, and analysis results. The audit log will be preserved for compliance. This action cannot be undone.',
  settings_delete_cancel: 'Cancel',
  settings_delete_confirm: 'Delete Everything',
  settings_about: 'About',
  settings_model_version: 'Model Version',
  settings_inference: 'Inference',

  // Scan
  scan_title: 'Scan ECG',
  scan_capture: 'Capture',
  scan_retake: 'Retake',
  scan_analyze: 'Analyze',

  // Clinical disclaimer (MUST appear on every results screen)
  disclaimer: 'For educational purposes only. This is not a medical diagnosis. Not FDA-cleared. Always consult a qualified healthcare professional.',

  // Auth
  auth_unlock: 'Unlock EKG Intelligence',
  auth_biometric_prompt: 'Authenticate to access patient data',
  auth_pin_prompt: 'Enter PIN',
  auth_failed: 'Authentication failed',

  // Common
  common_cancel: 'Cancel',
  common_save: 'Save',
  common_delete: 'Delete',
  common_edit: 'Edit',
  common_loading: 'Loading...',
  common_error: 'Error',
  common_success: 'Success',
};
```

**For es.ts and fr.ts:** Same keys, translated values. Use natural medical Spanish/French (not machine-translate-sounding).

**Review checklist:**
- [ ] All 3 language files export `default` object with identical keys
- [ ] i18n.ts initializes correctly
- [ ] No emoji in translation strings
- [ ] Medical terms use proper localized terminology

---

## Task 4: Create Zustand store for app state
**Status:** DONE
**File to create:** `src/store/useAppStore.ts`
**Reference:** Zustand v5 docs (already in package.json)

**What to implement:**
```typescript
import { create } from 'zustand';

interface AppState {
  // Auth state
  isAuthenticated: boolean;
  lastActivity: number;
  setAuthenticated: (value: boolean) => void;
  updateLastActivity: () => void;

  // Current patient context
  currentPatientId: string | null;
  setCurrentPatient: (id: string | null) => void;

  // Model info
  modelVersion: string;
  modelLoaded: boolean;
  setModelLoaded: (loaded: boolean) => void;

  // Settings
  language: string;
  setLanguage: (lang: string) => void;
  inactivityTimeoutMs: number;
  setInactivityTimeout: (ms: number) => void;
}

const useAppStore = create<AppState>((set) => ({
  isAuthenticated: false,
  lastActivity: Date.now(),
  setAuthenticated: (value) => set({ isAuthenticated: value }),
  updateLastActivity: () => set({ lastActivity: Date.now() }),

  currentPatientId: null,
  setCurrentPatient: (id) => set({ currentPatientId: id }),

  modelVersion: 'V3',
  modelLoaded: false,
  setModelLoaded: (loaded) => set({ modelLoaded: loaded }),

  language: 'en',
  setLanguage: (lang) => set({ language: lang }),
  inactivityTimeoutMs: 300000, // 5 minutes
  setInactivityTimeout: (ms) => set({ inactivityTimeoutMs: ms }),
}));

export default useAppStore;
```

**Review checklist:**
- [ ] No PHI stored in the store (patient names, IDs, ECG data stay in encrypted DB only)
- [ ] `currentPatientId` is just a reference pointer, not the patient object
- [ ] Zustand v5 `create` import (not v4 syntax)

---

## Task 5: Build the Patient List screen (patients.tsx)
**Status:** DONE
**File to edit:** `app/(main)/patients.tsx` (replace placeholder content)
**Reference files to read first:**
- `src/db/PatientRepository.ts` (for `listPatients`, `deletePatient`)
- `src/types/Patient.ts` (Patient interface)
- `app/(main)/index.tsx` (for style reference)
- `app/(main)/settings.tsx` (for style reference)

**What to implement:**
- FlatList of patients showing: name, age, sex, last updated
- Pull-to-refresh
- Tap a patient to navigate: `router.push(\`/patient/\${patient.patient_id}\`)`
- Empty state: "No patients yet" message
- "Add Patient" floating button at bottom
- Use the app's dark theme colors (see rules at top)

**Important:**
- Wrap the data-loading call in a try/catch — the database may not be initialized yet during development
- For now, use a mock patient list if the database isn't available:
```typescript
const MOCK_PATIENTS: Patient[] = [
  {
    patient_id: 'demo-1',
    first_name: 'Demo',
    last_name: 'Patient',
    id_number: null,
    age: 65,
    sex: 'M',
    has_pacemaker: false,
    is_athlete: false,
    is_pregnant: false,
    k_level: 4.0,
  },
];
```
- Add a comment `// TODO: Replace mock data with PatientRepository.listPatients() when DB is ready`

**Review checklist:**
- [ ] FlatList (not ScrollView) for performance
- [ ] keyExtractor uses patient_id
- [ ] No PHI in console.log
- [ ] Empty state handled
- [ ] Matches existing dark theme

---

## Task 6: Build the Patient Detail screen (patient/[id].tsx)
**Status:** DONE
**File to edit:** `app/(main)/patient/[id].tsx` (replace placeholder)
**Reference files to read first:**
- `src/types/Patient.ts`
- `src/types/ECGRecord.ts` (AnalysisResult interface)
- `app/(main)/patients.tsx` (your Task 5 output — match style)

**What to implement:**
- Header: Patient name, age, sex
- Info card: ID number, pacemaker/athlete/pregnancy flags, potassium level
- ECG History section: FlatList of analysis results (placeholder for now)
- Action buttons: "New ECG Scan", "Export PDF", "Delete Patient"
- Delete button shows confirmation dialog (like settings.tsx handleDeleteAllData)

**For now:** Use mock data similar to Task 5. Add comment `// TODO: Load real data from RecordRepository`

**Review checklist:**
- [ ] Uses `useLocalSearchParams<{ id: string }>()` from expo-router
- [ ] Delete confirmation dialog before deletion
- [ ] Back navigation after deletion
- [ ] Clinical disclaimer at bottom
- [ ] Matches app theme

---

## Task 7: Create Analysis type definitions
**Status:** DONE
**File to create:** `src/types/Analysis.ts`

**What to implement:**
```typescript
/**
 * Detected condition with metadata for display.
 */
export interface DetectedCondition {
  code: string;           // e.g. 'AFIB'
  name: string;           // e.g. 'Atrial Fibrillation'
  probability: number;    // 0.0 to 1.0
  urgency: number;        // 0-3
  urgencyLabel: string;   // 'Critical' | 'Significant' | 'Monitor' | 'Normal'
  urgencyColor: string;   // hex color
  action: string;         // clinical action text
  note: string;           // clinical note
}

/**
 * Full analysis result for display.
 */
export interface AnalysisDisplay {
  conditions: DetectedCondition[];  // sorted by urgency desc, then probability desc
  modelVersion: string;
  timestamp: string;
  disclaimerShown: boolean;
}

/**
 * Convert raw model output to display format.
 * Uses ConditionMetadata for code→name, urgency, guidance lookup.
 */
export function toDetectedCondition(
  code: string,
  probability: number,
): DetectedCondition;
// Import from ConditionMetadata.ts: V3_CONDITION_DESCRIPTIONS, V3_URGENCY,
// V3_CLINICAL_GUIDANCE, getUrgencyColor, getUrgencyLabel
```

**Review checklist:**
- [ ] `toDetectedCondition` imports from `../ml/ConditionMetadata`
- [ ] Types match what the analyze.tsx screen will need

---

## Task 8: Create shared UI components
**Status:** DONE
**Files to create:**
- `src/components/DisclaimerBanner.tsx`
- `src/components/ConditionCard.tsx`

**DisclaimerBanner.tsx:**
A reusable clinical disclaimer that must appear on every results screen (HIPAA 5.3.5).
```tsx
// Props: none (text is fixed for legal reasons)
// Renders the same disclaimer view from settings.tsx / index.tsx / analyze.tsx
// Style: dark purple bg (#1a1a2e), red left border (#FF6B6B), small grey text
```

**ConditionCard.tsx:**
Displays a single detected condition.
```tsx
// Props: { condition: DetectedCondition } (from src/types/Analysis.ts)
// Shows: urgency color bar on left, condition name, probability %, action text
// Style reference: similar to settingRow in settings.tsx but with colored left border
```

**Review checklist:**
- [ ] DisclaimerBanner has identical text to existing disclaimers in the codebase
- [ ] ConditionCard urgency color uses getUrgencyColor() from ConditionMetadata.ts
- [ ] Both use StyleSheet.create (not inline styles)
- [ ] No emoji — use colored borders/backgrounds for visual urgency indicators

---

## Task 9: Create babel.config.js
**Status:** DONE
**File to create:** `babel.config.js`

**What to implement:**
```javascript
module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
  };
};
```

This is the standard Expo config. The `@/` path alias is handled by Expo's Metro bundler via `tsconfig.json` — no babel plugin needed.

**Review checklist:**
- [ ] File exists at project root (same level as `package.json`)
- [ ] Uses `babel-preset-expo` (already in Expo's dependencies)

---

## Task 10: Create .gitignore
**Status:** DONE
**File to create:** `.gitignore`

**What to implement:**
```
node_modules/
.expo/
dist/
*.jks
*.p8
*.p12
*.key
*.mobileprovision
*.orig.*
web-build/

# macOS
.DS_Store

# env files
.env
.env.local

# TypeScript
*.tsbuildinfo

# Jest
coverage/

# ONNX model (large binary — tracked separately)
assets/models/*.onnx
```

**Review checklist:**
- [ ] `node_modules/` is ignored
- [ ] `.expo/` is ignored
- [ ] Model files are ignored (they'll be added via Git LFS or downloaded at build time)

---

## Task 11: Create placeholder asset files
**Status:** DONE
**Files to create:**
- `assets/icon.png` — app icon (1024x1024)
- `assets/adaptive-icon.png` — Android adaptive icon (1024x1024)
- `assets/splash-icon.png` — splash screen icon (optional)

**Instructions:**
Since Copilot cannot create binary image files, instead create a script:

**File to create:** `scripts/generate_placeholder_assets.js`
```javascript
/**
 * Generate placeholder asset PNGs for development.
 * Run: node scripts/generate_placeholder_assets.js
 *
 * Creates minimal valid PNG files so Expo doesn't error on startup.
 * Replace with real app icons before release.
 */
const fs = require('fs');
const path = require('path');

// Minimal 1x1 pixel green (#00E5B0) PNG
// This is a valid PNG file — the smallest possible
const PNG_1x1 = Buffer.from([
  0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
  0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
  0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, // 8-bit RGB
  0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IDAT chunk
  0x54, 0x08, 0xD7, 0x63, 0x60, 0x60, 0x60, 0x00, // compressed data
  0x00, 0x00, 0x04, 0x00, 0x01, 0x27, 0x34, 0x27, //
  0x0A, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, // IEND chunk
  0x44, 0xAE, 0x42, 0x60, 0x82,
]);

const assetsDir = path.join(__dirname, '..', 'assets');
fs.mkdirSync(assetsDir, { recursive: true });

for (const name of ['icon.png', 'adaptive-icon.png', 'splash-icon.png']) {
  const dest = path.join(assetsDir, name);
  if (!fs.existsSync(dest)) {
    fs.writeFileSync(dest, PNG_1x1);
    console.log(`Created ${name}`);
  } else {
    console.log(`${name} already exists — skipping`);
  }
}
console.log('Done. Replace these with real icons before release.');
```

Then run it: `node scripts/generate_placeholder_assets.js`

**Review checklist:**
- [ ] Script creates valid PNG files
- [ ] assets/ directory has icon.png and adaptive-icon.png
- [ ] Script is idempotent (doesn't overwrite existing icons)

---

## Task 12: Replace tab bar emoji icons with text characters
**Status:** DONE
**File to edit:** `app/(main)/_layout.tsx`
**Reference:** Read the current file first.

**What to change:**
Replace emoji-based tab icons with simple Unicode text characters that render consistently across platforms:

```tsx
<Tabs.Screen
  name="index"
  options={{
    title: 'Dashboard',
    tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 18, fontWeight: '700' }}>H</Text>,
  }}
/>
<Tabs.Screen
  name="scan"
  options={{
    title: 'Scan',
    tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 18, fontWeight: '700' }}>S</Text>,
  }}
/>
<Tabs.Screen
  name="patients"
  options={{
    title: 'Patients',
    tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 18, fontWeight: '700' }}>P</Text>,
  }}
/>
<Tabs.Screen
  name="settings"
  options={{
    title: 'Settings',
    tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 18, fontWeight: '700' }}>G</Text>,
  }}
/>
```

**Why:** Emoji rendering varies across platforms and can cause build issues. Text characters are reliable. We'll add proper icons (Ionicons or custom SVG) in a future task.

**Review checklist:**
- [ ] No emoji anywhere in `_layout.tsx`
- [ ] All 4 tab icons use plain text
- [ ] Icons are visually distinguishable (different letters)

---

## Task 13: Wire i18n into the app
**Status:** DONE
**File to edit:** `app/_layout.tsx`
**Reference files to read first:**
- `src/i18n/i18n.ts` (the i18n initialization)
- `app/_layout.tsx` (current root layout)

**What to change:**
Add a single import at the top of `app/_layout.tsx` (after the existing imports, before `installSanitizer()`):

```typescript
import '../src/i18n/i18n';
```

This import triggers i18n initialization on app start. No other changes needed — the translation screens (patients.tsx, patient/[id].tsx) already use `useTranslation()`.

**Review checklist:**
- [ ] Import is at the top of the file (not inside a component)
- [ ] Import is BEFORE `installSanitizer()` — i18n should init first
- [ ] No other changes to `_layout.tsx`
- [ ] Verify: `patients.tsx` and `patient/[id].tsx` already import `useTranslation` (they do)

---

## Task 14: Create "Add Patient" form screen
**Status:** DONE
**File to create:** `app/(main)/patient/new.tsx`
**Reference files to read first:**
- `src/types/Patient.ts` (PatientInput interface)
- `src/db/PatientRepository.ts` (savePatient function signature)
- `app/(main)/patient/[id].tsx` (style reference)
- `src/i18n/en.ts` (translation keys for patient form)

**What to implement:**
A form screen for creating a new patient. Fields:
- First Name (TextInput, required)
- Last Name (TextInput, required)
- Patient ID (TextInput, optional)
- Age (TextInput, numeric keyboard)
- Sex (two buttons: Male / Female)
- Potassium K+ (TextInput, decimal keyboard, optional)
- Has Pacemaker (toggle/switch)
- Is Athlete (toggle/switch)
- Is Pregnant (toggle/switch, only show if sex=F)

**Implementation notes:**
```tsx
import { View, Text, TextInput, TouchableOpacity, Switch, ScrollView,
         StyleSheet, SafeAreaView, Alert } from 'react-native';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

// TODO: Import and call PatientRepository.savePatient() when DB is ready

export default function NewPatientScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [idNumber, setIdNumber] = useState('');
  const [age, setAge] = useState('');
  const [sex, setSex] = useState<'M' | 'F'>('M');
  const [kLevel, setKLevel] = useState('');
  const [hasPacemaker, setHasPacemaker] = useState(false);
  const [isAthlete, setIsAthlete] = useState(false);
  const [isPregnant, setIsPregnant] = useState(false);

  const handleSave = async () => {
    if (!firstName.trim() || !lastName.trim()) {
      Alert.alert(t('common_error'), 'First name and last name are required.');
      return;
    }
    try {
      // TODO: Call PatientRepository.savePatient({
      //   first_name: firstName.trim(),
      //   last_name: lastName.trim(),
      //   id_number: idNumber.trim() || null,
      //   age: parseInt(age) || null,
      //   sex,
      //   k_level: parseFloat(kLevel) || null,
      //   has_pacemaker: hasPacemaker,
      //   is_athlete: isAthlete,
      //   is_pregnant: sex === 'F' ? isPregnant : false,
      // });
      Alert.alert(t('patient_save_success'));
      router.back();
    } catch (error) {
      Alert.alert(t('common_error'), String(error));
    }
  };

  // ... render form with dark theme styling
}
```

**Style notes:**
- TextInput: bg `#0D1F1E`, border `#1E3533`, text `#E0E0E0`, placeholder `#3D6662`
- Sex selector: two side-by-side buttons, selected one has bg `#00E5B0` + text `#071312`
- Switch: trackColor `#1E3533` (off) / `#00E5B0` (on)
- Save button: bg `#00E5B0`, text `#071312`, full width

**Review checklist:**
- [ ] Form validates required fields (first + last name)
- [ ] Age input uses `keyboardType="numeric"`
- [ ] Potassium uses `keyboardType="decimal-pad"`
- [ ] Pregnancy toggle only visible when sex=F
- [ ] No PHI in console.log
- [ ] Uses i18n translation keys
- [ ] Back button at top
- [ ] Matches dark theme

---

## Task 15: Create expo-env.d.ts
**Status:** DONE
**File to create:** `expo-env.d.ts` (at project root)

**What to implement:**
```typescript
/// <reference types="expo/types" />

// NOTE: This file should not be edited and should be in your git repo.
// Expo auto-generates this reference for typed routes.
```

**Review checklist:**
- [ ] File exists at project root
- [ ] Contains the Expo types reference

---

## Task 16: Update COPILOT_TASKS.md — mark Tasks 1-8 as DONE
**Status:** TODO

**What to do:**
Go through Tasks 1-8 in this file and change `**Status:** TODO` to `**Status:** DONE` for each one.

**Review checklist:**
- [ ] Tasks 1-8 all say DONE
- [ ] Tasks 9-16 still say TODO (until completed)
