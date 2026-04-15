/**
 * Patient type — matches the patients table schema.
 */
export interface Patient {
  patient_id: string;
  first_name: string;
  last_name: string;
  id_number: string | null;
  age: number;
  sex: 'M' | 'F';
  has_pacemaker: boolean;
  is_athlete: boolean;
  is_pregnant: boolean;
  k_level: number;
  created_at?: string;
  updated_at?: string;
}

/**
 * Input type for creating a patient (without auto-generated fields).
 */
export type PatientInput = Omit<Patient, 'patient_id' | 'created_at' | 'updated_at'>;
