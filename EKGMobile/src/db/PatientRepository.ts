/**
 * PatientRepository — CRUD operations for patients.
 *
 * All operations are logged to the HIPAA audit trail.
 * Deletion cascades to ekg_records and analysis_results.
 */

import { v4 as uuidv4 } from 'uuid';
import { getDatabase } from './Database';
import { logEvent } from '../audit/AuditLogger';
import type { Patient } from '../types/Patient';

/**
 * Create a new patient record.
 */
export async function savePatient(patient: Omit<Patient, 'patient_id' | 'created_at' | 'updated_at'>): Promise<string> {
  const db = getDatabase();
  const patientId = uuidv4();

  db.execute(
    `INSERT INTO patients (patient_id, first_name, last_name, id_number, age, sex,
       has_pacemaker, is_athlete, is_pregnant, k_level)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      patientId,
      patient.first_name,
      patient.last_name,
      patient.id_number ?? null,
      patient.age,
      patient.sex,
      patient.has_pacemaker ? 1 : 0,
      patient.is_athlete ? 1 : 0,
      patient.is_pregnant ? 1 : 0,
      patient.k_level,
    ],
  );

  await logEvent('PHI_CREATE', 'create_patient', 'patient', patientId);
  return patientId;
}

/**
 * Get a single patient by ID.
 */
export async function getPatient(patientId: string): Promise<Patient | null> {
  const db = getDatabase();
  const result = db.execute('SELECT * FROM patients WHERE patient_id = ?', [patientId]);

  if (!result.rows || result.rows.length === 0) return null;

  await logEvent('PHI_VIEW', 'view_patient', 'patient', patientId);
  return rowToPatient(result.rows.item(0));
}

/**
 * List all patients.
 */
export async function listPatients(): Promise<Patient[]> {
  const db = getDatabase();
  const result = db.execute('SELECT * FROM patients ORDER BY updated_at DESC');

  if (!result.rows) return [];

  await logEvent('PHI_VIEW', 'list_patients', 'patient', null, `count:${result.rows.length}`);

  const patients: Patient[] = [];
  for (let i = 0; i < result.rows.length; i++) {
    patients.push(rowToPatient(result.rows.item(i)));
  }
  return patients;
}

/**
 * Update a patient record.
 */
export async function updatePatient(
  patientId: string,
  updates: Partial<Omit<Patient, 'patient_id' | 'created_at' | 'updated_at'>>,
): Promise<void> {
  const db = getDatabase();

  const fields: string[] = [];
  const values: unknown[] = [];

  for (const [key, value] of Object.entries(updates)) {
    if (key === 'has_pacemaker' || key === 'is_athlete' || key === 'is_pregnant') {
      fields.push(`${key} = ?`);
      values.push(value ? 1 : 0);
    } else {
      fields.push(`${key} = ?`);
      values.push(value);
    }
  }

  fields.push('updated_at = datetime("now")');
  values.push(patientId);

  db.execute(
    `UPDATE patients SET ${fields.join(', ')} WHERE patient_id = ?`,
    values,
  );

  await logEvent('PHI_UPDATE', 'update_patient', 'patient', patientId);
}

/**
 * Delete a patient and all associated records (CASCADE).
 * HIPAA 5.1.2: Right to deletion.
 */
export async function deletePatient(patientId: string): Promise<void> {
  const db = getDatabase();

  // Count associated records for audit detail
  const recordCount = db.execute(
    'SELECT COUNT(*) as cnt FROM ekg_records WHERE patient_id = ?',
    [patientId],
  );
  const count = recordCount.rows?.item(0)?.cnt ?? 0;

  // CASCADE delete removes ekg_records and analysis_results
  db.execute('DELETE FROM patients WHERE patient_id = ?', [patientId]);

  await logEvent(
    'PHI_DELETE',
    'delete_patient',
    'patient',
    patientId,
    `cascade_deleted_records:${count}`,
  );
}

/**
 * Delete ALL patient data. Preserves audit log.
 * HIPAA 5.1.2: "Delete All My Data" action.
 */
export async function deleteAllData(): Promise<void> {
  const db = getDatabase();

  const patientCount = db.execute('SELECT COUNT(*) as cnt FROM patients');
  const recordCount = db.execute('SELECT COUNT(*) as cnt FROM ekg_records');
  const analysisCount = db.execute('SELECT COUNT(*) as cnt FROM analysis_results');

  const pCount = patientCount.rows?.item(0)?.cnt ?? 0;
  const rCount = recordCount.rows?.item(0)?.cnt ?? 0;
  const aCount = analysisCount.rows?.item(0)?.cnt ?? 0;

  // Delete in order: analyses → records → patients
  db.execute('DELETE FROM analysis_results');
  db.execute('DELETE FROM ekg_records');
  db.execute('DELETE FROM patients');

  // Audit log is PRESERVED — this is the only table not deleted
  await logEvent(
    'DATA_WIPE',
    'delete_all_data',
    null,
    null,
    `patients:${pCount},records:${rCount},analyses:${aCount}`,
  );
}

function rowToPatient(row: any): Patient {
  return {
    patient_id: row.patient_id,
    first_name: row.first_name,
    last_name: row.last_name,
    id_number: row.id_number,
    age: row.age,
    sex: row.sex,
    has_pacemaker: !!row.has_pacemaker,
    is_athlete: !!row.is_athlete,
    is_pregnant: !!row.is_pregnant,
    k_level: row.k_level,
    created_at: row.created_at,
    updated_at: row.updated_at,
  };
}
