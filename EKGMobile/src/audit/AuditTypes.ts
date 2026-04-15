/**
 * AuditTypes — Event type definitions for the HIPAA audit trail.
 *
 * Every PHI access event is categorized by type and action.
 * Audit entries contain resource IDs but NEVER PHI content.
 */

export type AuditEventType =
  | 'AUTH_SUCCESS'
  | 'AUTH_FAIL'
  | 'AUTH_LOCK'        // Auto-lock or manual lock
  | 'PHI_VIEW'         // Viewing patient data
  | 'PHI_CREATE'       // Creating patient/record/analysis
  | 'PHI_UPDATE'       // Updating patient data
  | 'PHI_DELETE'       // Deleting patient data
  | 'PHI_EXPORT'       // Exporting PDF or signal data
  | 'MODEL_LOAD'       // Loading ML model
  | 'MODEL_INTEGRITY_FAIL'  // Model hash mismatch
  | 'INFERENCE_RUN'    // Running ECG analysis
  | 'DATA_WIPE'        // "Delete All My Data"
  | 'APP_OPEN'         // App opened
  | 'APP_CLOSE'        // App closed/backgrounded
  | 'INTEGRITY_CHECK'  // Audit log integrity verification
  | 'SETTINGS_CHANGE'; // Security-relevant setting change

export type ResourceType =
  | 'patient'
  | 'ekg_record'
  | 'analysis'
  | 'report'
  | 'model'
  | 'settings'
  | 'audit_log';

export interface AuditEntry {
  id?: number;
  timestamp: string;      // ISO 8601 UTC
  event_type: AuditEventType;
  resource_type: ResourceType | null;
  resource_id: string | null;  // UUID of accessed resource (NOT PHI)
  action: string;              // Human-readable action description
  details: string | null;      // Additional context (NEVER contains PHI)
  integrity_hash: string;      // SHA-256 chain hash
}
