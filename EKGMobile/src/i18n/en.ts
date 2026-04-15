/**
 * English translation strings for EKG Intelligence mobile app
 */
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
  dashboard_auroc: 'AUROC: 0.9682',
  dashboard_inference: 'On-device inference (ONNX Runtime)',

  // Patient form
  patient_first_name: 'First Name',
  patient_first_name_placeholder: 'e.g. John',
  patient_last_name: 'Last Name',
  patient_last_name_placeholder: 'e.g. Doe',
  patient_id_number: 'Patient ID',
  patient_id_number_placeholder: 'e.g. 123456789',
  patient_age: 'Age',
  patient_sex: 'Biological Sex',
  patient_sex_male: 'Male',
  patient_sex_female: 'Female',
  patient_potassium: 'Potassium K+ (mmol/L)',
  patient_potassium_help: 'Normal range: 3.5–5.0 mmol/L',
  patient_pacemaker: 'Pacemaker / ICD Present',
  patient_athlete: 'Athlete Status',
  patient_pregnant: 'Pregnancy',
  patient_save: 'Save Patient',
  patient_delete: 'Delete Patient',
  patient_save_success: 'Patient saved successfully',
  patient_delete_success: 'Patient deleted',
  patient_no_patients: 'No patients yet',
  patient_add_new: 'Add Patient',
  patient_edit: 'Edit Patient',

  // Patient list
  patient_list_title: 'Patients',
  patient_list_add_button: 'Add Patient',
  patient_item_age_sex: 'Age {age}, {sex}',
  patient_item_updated: 'Updated {date}',
  patient_item_pull_refresh: 'Pull to refresh',

  // Patient detail
  patient_detail_info: 'Patient Information',
  patient_detail_records: 'ECG History',
  patient_detail_new_scan: 'New ECG Scan',
  patient_detail_export_pdf: 'Export PDF',
  patient_detail_delete_patient: 'Delete Patient',
  patient_detail_no_records: 'No ECG records yet',

  // Analysis results
  analysis_title: 'Analysis Results',
  analysis_detected_conditions: 'Detected Conditions',
  analysis_condition_name: 'Condition',
  analysis_confidence: 'Probability',
  analysis_urgency: 'Urgency',
  analysis_urgency_critical: 'Critical',
  analysis_urgency_significant: 'Significant',
  analysis_urgency_monitor: 'Monitor',
  analysis_urgency_normal: 'Normal',
  analysis_clinical_guidance: 'Clinical Guidance',
  analysis_action: 'Recommended Action',
  analysis_note: 'Clinical Note',
  analysis_timestamp: 'Analyzed {time}',
  analysis_model_version: 'Model: {version}',

  // Scan ECG
  scan_title: 'Scan ECG',
  scan_subtitle: 'Capture paper ECG image',
  scan_button: 'Take Photo',
  scan_retake: 'Retake',
  scan_analyze: 'Analyze',
  scan_analyzing: 'Analyzing...',

  // Settings
  settings_title: 'Settings',
  settings_security: 'Security',
  settings_audit_log: 'View Audit Log',
  settings_audit_log_empty: 'No activity logged',
  settings_audit_log_event: '{event} - {timestamp}',
  settings_auto_lock: 'Auto-Lock Timeout',
  settings_auto_lock_5min: '5 minutes',
  settings_auto_lock_10min: '10 minutes',
  settings_auto_lock_15min: '15 minutes',
  settings_auto_lock_30min: '30 minutes',
  settings_screen_protection: 'Screen Protection',
  settings_screen_protection_enabled: 'Active',
  settings_data: 'Data',
  settings_language: 'Language',
  settings_language_english: 'English',
  settings_language_spanish: 'Espanol',
  settings_language_french: 'Francais',
  settings_delete_all: 'Delete All My Data',
  settings_delete_confirm_title: 'Delete All Data',
  settings_delete_confirm_msg:
    'This will permanently delete all patient records, ECG data, and analysis results. The audit log will be preserved for compliance. This action cannot be undone.',
  settings_delete_cancel: 'Cancel',
  settings_delete_confirm: 'Delete Everything',
  settings_delete_success: 'All data deleted',
  settings_about: 'About',
  settings_model_version: 'Model Version',
  settings_model_info: 'V3 Multilabel (26 conditions)',
  settings_inference: 'Inference Engine',
  settings_inference_info: 'On-device ONNX Runtime',

  // Clinical disclaimer (MUST appear on every results screen)
  disclaimer:
    'For educational purposes only. This is not a medical diagnosis. Not FDA-cleared. Always consult a qualified healthcare professional.',

  // Audit log
  audit_log_title: 'Activity Log',
  audit_event_auth_success: 'Authentication successful',
  audit_event_auth_fail: 'Authentication failed',
  audit_event_auth_lock: 'Session locked',
  audit_event_phi_view: 'Data accessed',
  audit_event_phi_create: 'Data created',
  audit_event_phi_update: 'Data updated',
  audit_event_phi_delete: 'Data deleted',
  audit_event_phi_export: 'Data exported',
  audit_event_inference_run: 'Analysis run',
  audit_event_data_wipe: 'All data deleted',

  // Auth
  auth_lock_title: 'EKG Intelligence',
  auth_lock_subtitle: 'Your health data is encrypted and protected',
  auth_unlock_button: 'Authenticate',
  auth_biometric_prompt: 'Authenticate to access your health data',
  auth_failed: 'Authentication failed. Please try again.',
  auth_no_biometrics: 'This device does not have biometric authentication set up.',

  // Common
  common_cancel: 'Cancel',
  common_save: 'Save',
  common_delete: 'Delete',
  common_edit: 'Edit',
  common_close: 'Close',
  common_loading: 'Loading...',
  common_error: 'Error',
  common_success: 'Success',
  common_back: 'Back',
  common_next: 'Next',
  common_finish: 'Finish',
  common_yes: 'Yes',
  common_no: 'No',
  common_ok: 'OK',
  common_empty_state: 'No data available',
};
