/**
 * Database schema — SQLCipher encrypted database.
 *
 * Tables mirror the existing database_setup.py schema with additions:
 * - audit_log: HIPAA 1.3.1 tamper-evident audit trail
 * - app_config: settings storage
 * - Checksum columns for data integrity (HIPAA 1.4.1)
 * - CASCADE deletes for patient data removal (HIPAA 5.1.2)
 */

export const SCHEMA_VERSION = 1;

export const CREATE_TABLES = `
-- Patients
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
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

-- EKG Records with integrity checksum (HIPAA 1.4.1)
CREATE TABLE IF NOT EXISTS ekg_records (
    ekg_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    signal_data BLOB NOT NULL,
    sampling_rate INTEGER DEFAULT 500,
    lead_count INTEGER DEFAULT 12,
    acquisition_source TEXT,
    checksum TEXT NOT NULL,
    captured_at TEXT DEFAULT (datetime('now')),
    notes TEXT
);

-- Analysis Results
CREATE TABLE IF NOT EXISTS analysis_results (
    analysis_id TEXT PRIMARY KEY,
    ekg_id TEXT NOT NULL REFERENCES ekg_records(ekg_id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    primary_condition TEXT NOT NULL,
    conditions_json TEXT NOT NULL,
    scores_json TEXT NOT NULL,
    intervals_json TEXT,
    clinical_rules_json TEXT,
    st_territory_json TEXT,
    disclaimer_shown INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now'))
);

-- HIPAA Audit Log (append-only, hash-chained)
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    action TEXT NOT NULL,
    details TEXT,
    integrity_hash TEXT NOT NULL
);

-- App Configuration
CREATE TABLE IF NOT EXISTS app_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
`;

export const CREATE_INDEXES = `
CREATE INDEX IF NOT EXISTS idx_ekg_patient ON ekg_records(patient_id);
CREATE INDEX IF NOT EXISTS idx_analysis_ekg ON analysis_results(ekg_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type);
`;

export const DEFAULT_CONFIG: Record<string, string> = {
  inactivity_timeout_ms: '300000',  // 5 minutes
  language: 'en',
  theme: 'dark',
  data_retention_days: '0',  // 0 = keep forever
  schema_version: String(SCHEMA_VERSION),
};
