/**
 * ConditionMetadata — 26-class V3 condition codes, urgency levels,
 * descriptions, and clinical guidance.
 *
 * Ported from multilabel_v3.py:636-731 (V3_CODES, V3_URGENCY,
 * V3_CONDITION_DESCRIPTIONS, V3_CLINICAL_GUIDANCE).
 */

/**
 * 26 V3 condition codes in model output order.
 * Indices 0-13: from PTB-XL + Chapman (MERGED_CODES)
 * Indices 14-25: from PhysioNet Challenge 2021 (NEW_CONDITIONS)
 */
export const V3_CODES = [
  'NORM',   // 0
  'AFIB',   // 1
  'PVC',    // 2
  'LVH',    // 3
  'IMI',    // 4
  'ASMI',   // 5
  'CLBBB',  // 6
  'CRBBB',  // 7
  'LAFB',   // 8
  '1AVB',   // 9
  'ISC_',   // 10
  'NDT',    // 11
  'IRBBB',  // 12
  'STACH',  // 13
  'PAC',    // 14
  'Brady',  // 15
  'SVT',    // 16
  'LQTP',   // 17
  'TAb',    // 18
  'LAD',    // 19
  'RAD',    // 20
  'NSIVC',  // 21
  'AFL',    // 22
  'STc',    // 23
  'STD',    // 24
  'LAE',    // 25
] as const;

export type ConditionCode = typeof V3_CODES[number];
export const N_CLASSES = V3_CODES.length; // 26

/**
 * Urgency levels (0-3):
 *   3 = Critical / immediate action
 *   2 = Significant / prompt review
 *   1 = Mild / monitor
 *   0 = Normal
 */
export const V3_URGENCY: Record<ConditionCode, number> = {
  // Urgency 3 — critical
  AFIB: 3, AFL: 3, IMI: 3, ASMI: 3, CLBBB: 3,
  // Urgency 2 — significant
  LVH: 2, PVC: 2, CRBBB: 2, ISC_: 2, SVT: 2,
  LQTP: 2, STD: 2, STc: 2,
  // Urgency 1 — mild
  LAFB: 1, '1AVB': 1, NDT: 1, IRBBB: 1, STACH: 1,
  PAC: 1, Brady: 1, TAb: 1, LAD: 1, RAD: 1,
  NSIVC: 1, LAE: 1,
  // Urgency 0 — normal
  NORM: 0,
};

export const V3_CONDITION_DESCRIPTIONS: Record<ConditionCode, string> = {
  NORM:  'Normal ECG',
  AFIB:  'Atrial Fibrillation',
  PVC:   'Premature Ventricular Contraction',
  LVH:   'Left Ventricular Hypertrophy',
  IMI:   'Inferior Myocardial Infarction',
  ASMI:  'Anteroseptal Myocardial Infarction',
  CLBBB: 'Complete Left Bundle Branch Block',
  CRBBB: 'Complete Right Bundle Branch Block',
  LAFB:  'Left Anterior Fascicular Block',
  '1AVB': 'First-Degree AV Block',
  ISC_:  'Non-Specific Ischemic ST Changes',
  NDT:   'Non-Diagnostic T Abnormalities',
  IRBBB: 'Incomplete Right Bundle Branch Block',
  STACH: 'Sinus Tachycardia',
  PAC:   'Premature Atrial Contraction',
  Brady: 'Bradycardia',
  SVT:   'Supraventricular Tachycardia',
  LQTP:  'Prolonged QT Interval',
  TAb:   'T-Wave Abnormality',
  LAD:   'Left Axis Deviation',
  RAD:   'Right Axis Deviation',
  NSIVC: 'Non-Specific Intraventricular Conduction Delay',
  AFL:   'Atrial Flutter',
  STc:   'ST-T Change',
  STD:   'ST Depression',
  LAE:   'Left Atrial Enlargement',
};

export interface ClinicalGuidance {
  action: string;
  note: string;
}

export const V3_CLINICAL_GUIDANCE: Record<ConditionCode, ClinicalGuidance> = {
  NORM:  { action: 'No acute findings. Routine follow-up as indicated.', note: '' },
  AFIB:  { action: 'Assess stroke risk (CHA2DS2-VASc). Consider anticoagulation. Rate/rhythm control.', note: 'Irregular rhythm \u2014 no distinct P waves.' },
  AFL:   { action: 'Rate control or rhythm control. Assess for anticoagulation (similar risk to AFIB).', note: 'Atrial flutter \u2014 typically 2:1 or 3:1 block with sawtooth flutter waves.' },
  PVC:   { action: 'Assess frequency and symptoms. If >10% burden or symptomatic, refer for Holter + echo.', note: 'Isolated PVCs are common and often benign.' },
  LVH:   { action: 'Evaluate for hypertension or hypertrophic cardiomyopathy. Echo recommended.', note: 'Voltage criteria met \u2014 Cornell/Sokolow-Lyon thresholds exceeded.' },
  IMI:   { action: 'If acute: activate cath lab. Check reciprocal changes in I/aVL. Evaluate RV involvement.', note: 'Inferior territory (RCA or LCx). ST elevation in II, III, aVF.' },
  ASMI:  { action: 'If acute: activate cath lab. Anteroseptal STEMI protocol.', note: 'Anterior territory (LAD). ST elevation in V1\u2013V4.' },
  CLBBB: { action: 'New LBBB with chest pain: treat as STEMI equivalent \u2014 activate cath lab.', note: 'Complete LBBB \u2014 Sgarbossa criteria if ischaemia suspected.' },
  CRBBB: { action: 'Isolated CRBBB often benign. New CRBBB with symptoms \u2014 assess for PE or acute MI.', note: 'Complete RBBB \u2014 RSR\' in V1, wide S in I/V6.' },
  LAFB:  { action: 'Usually benign in isolation. Monitor for progression to bifascicular block.', note: 'Left anterior fascicular block \u2014 LAD with small q in I/aVL.' },
  '1AVB': { action: 'Usually benign. Review medications (beta-blockers, digoxin). Annual follow-up.', note: 'PR interval > 200ms. No treatment usually required.' },
  ISC_:  { action: 'Compare with prior ECG. If new or symptomatic, evaluate for ACS.', note: 'Non-specific ischaemic ST changes \u2014 may indicate demand ischaemia.' },
  NDT:   { action: 'Non-diagnostic. Correlate with clinical history and symptoms.', note: 'Non-diagnostic T-wave abnormalities \u2014 many potential causes.' },
  IRBBB: { action: 'Usually benign. No immediate action required.', note: 'Incomplete RBBB \u2014 RSR\' pattern in V1, QRS < 120ms.' },
  STACH: { action: 'Identify and treat underlying cause (pain, fever, hypovolaemia, anaemia).', note: 'Sinus tachycardia \u2014 rate > 100bpm with normal P waves.' },
  PAC:   { action: 'Usually benign. If frequent or symptomatic, evaluate for structural heart disease.', note: 'Premature atrial contraction \u2014 early narrow beat with abnormal P wave.' },
  Brady: { action: 'If symptomatic (syncope, hypotension), assess for pacemaker indication. Review medications.', note: 'Bradycardia \u2014 HR < 60bpm.' },
  SVT:   { action: 'Vagal manoeuvres or adenosine for acute termination. Refer for EP study if recurrent.', note: 'Supraventricular tachycardia \u2014 narrow complex tachycardia.' },
  LQTP:  { action: 'Review QT-prolonging medications. Electrolyte correction. Consider cardiology referral.', note: 'Prolonged QTc \u2014 risk of Torsades de Pointes. QTc \u2265 500ms is high risk.' },
  TAb:   { action: 'Correlate with symptoms. Compare with prior ECG. Evaluate electrolytes.', note: 'T-wave abnormality \u2014 inversion or flattening.' },
  LAD:   { action: 'Usually incidental. Rule out LAFB, inferior MI, or ventricular hypertrophy.', note: 'Left axis deviation \u2014 QRS axis between \u221230\u00b0 and \u221290\u00b0.' },
  RAD:   { action: 'Evaluate for RVH, RBBB, lateral MI, or PE if new.', note: 'Right axis deviation \u2014 QRS axis > +90\u00b0.' },
  NSIVC: { action: 'Monitor. Evaluate for structural heart disease if new or symptomatic.', note: 'Non-specific intraventricular conduction delay \u2014 QRS 110\u2013119ms.' },
  STc:   { action: 'Compare with prior ECG. If new or symptomatic, evaluate for ischaemia or pericarditis.', note: 'ST-T change \u2014 non-specific.' },
  STD:   { action: 'If new or \u22651mm in multiple leads with symptoms, evaluate urgently for ACS.', note: 'ST depression \u2014 may indicate subendocardial ischaemia or reciprocal change.' },
  LAE:   { action: 'Evaluate for mitral valve disease, hypertension, or LV dysfunction. Echo recommended.', note: 'Left atrial enlargement \u2014 broad/notched P wave in II, or biphasic in V1.' },
};

/**
 * Get urgency color for display.
 */
export function getUrgencyColor(urgency: number): string {
  switch (urgency) {
    case 3: return '#FF4444'; // Red — critical
    case 2: return '#FF8800'; // Orange — significant
    case 1: return '#FFCC00'; // Yellow — mild
    default: return '#00E5B0'; // Green — normal
  }
}

/**
 * Get urgency label for display.
 */
export function getUrgencyLabel(urgency: number): string {
  switch (urgency) {
    case 3: return 'Critical';
    case 2: return 'Significant';
    case 1: return 'Monitor';
    default: return 'Normal';
  }
}
