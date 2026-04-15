/**
 * ECG Record type — matches the ekg_records table schema.
 */
export interface ECGRecord {
  ekg_id: string;
  patient_id: string;
  signal_data: ArrayBuffer;      // Compressed float32 array (12 × 5000)
  sampling_rate: number;
  lead_count: number;
  acquisition_source: 'camera' | 'file_upload' | 'bluetooth' | 'demo';
  checksum: string;              // SHA-256 of signal_data (HIPAA 1.4.1)
  captured_at?: string;
  notes: string | null;
}

/**
 * Analysis result type — matches the analysis_results table schema.
 */
export interface AnalysisResult {
  analysis_id: string;
  ekg_id: string;
  model_version: string;
  model_hash: string;
  primary_condition: string;
  conditions: string[];          // Detected condition codes
  scores: Record<string, number>; // {code: probability}
  intervals: IntervalMeasurements | null;
  clinical_rules: ClinicalRuleResult[] | null;
  st_territory: STTerritoryResult | null;
  disclaimer_shown: boolean;
  created_at?: string;
}

export interface IntervalMeasurements {
  heart_rate: number;
  pr_interval_ms: number | null;
  qrs_duration_ms: number | null;
  qtc_ms: number | null;
  rr_variability: number | null;
}

export interface ClinicalRuleResult {
  rule: string;
  finding: string;
  severity: 'normal' | 'mild' | 'moderate' | 'severe';
}

export interface STTerritoryResult {
  territory: string | null;      // 'LAD' | 'RCA' | 'LCx' | null
  elevation_leads: string[];
  depression_leads: string[];
  reciprocal_changes: boolean;
}
