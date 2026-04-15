/**
 * RecordRepository — CRUD operations for ECG records.
 *
 * All operations are logged to the HIPAA audit trail.
 * Deletion cascades to analysis_results.
 */

import { v4 as uuidv4 } from 'uuid';
import { getDatabase } from './Database';
import { logEvent } from '../audit/AuditLogger';
import type { ECGRecord } from '../types/ECGRecord';

/**
 * Create a new ECG record.
 */
export async function saveRecord(
  record: Omit<ECGRecord, 'ekg_id' | 'captured_at'>,
): Promise<string> {
  const db = getDatabase();
  const ekgId = uuidv4();

  db.execute(
    `INSERT INTO ekg_records
     (ekg_id, patient_id, signal_data, sampling_rate, lead_count, acquisition_source, checksum, notes)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      ekgId,
      record.patient_id,
      record.signal_data,
      record.sampling_rate,
      record.lead_count,
      record.acquisition_source,
      record.checksum,
      record.notes ?? null,
    ],
  );

  await logEvent('PHI_CREATE', 'create_record', 'ekg_record', ekgId);
  return ekgId;
}

/**
 * Get a single ECG record by ID.
 */
export async function getRecord(ekgId: string): Promise<ECGRecord | null> {
  const db = getDatabase();
  const result = db.execute('SELECT * FROM ekg_records WHERE ekg_id = ?', [ekgId]);

  if (!result.rows || result.rows.length === 0) return null;

  await logEvent('PHI_VIEW', 'view_record', 'ekg_record', ekgId);
  return rowToRecord(result.rows.item(0));
}

/**
 * List all ECG records for a patient.
 */
export async function listRecordsByPatient(patientId: string): Promise<ECGRecord[]> {
  const db = getDatabase();
  const result = db.execute(
    'SELECT * FROM ekg_records WHERE patient_id = ? ORDER BY captured_at DESC',
    [patientId],
  );

  if (!result.rows) return [];

  const count = result.rows.length;
  await logEvent(
    'PHI_VIEW',
    'list_records',
    'ekg_record',
    null,
    `patient:${patientId},count:${count}`,
  );

  const records: ECGRecord[] = [];
  for (let i = 0; i < result.rows.length; i++) {
    records.push(rowToRecord(result.rows.item(i)));
  }
  return records;
}

/**
 * Delete an ECG record. Cascades to analysis_results.
 */
export async function deleteRecord(ekgId: string): Promise<void> {
  const db = getDatabase();

  db.execute('DELETE FROM ekg_records WHERE ekg_id = ?', [ekgId]);

  await logEvent('PHI_DELETE', 'delete_record', 'ekg_record', ekgId);
}

function rowToRecord(row: any): ECGRecord {
  return {
    ekg_id: row.ekg_id,
    patient_id: row.patient_id,
    signal_data: row.signal_data,
    sampling_rate: row.sampling_rate,
    lead_count: row.lead_count,
    acquisition_source: row.acquisition_source,
    checksum: row.checksum,
    captured_at: row.captured_at,
    notes: row.notes,
  };
}
