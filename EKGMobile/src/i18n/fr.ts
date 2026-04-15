/**
 * French (Français) translation strings for EKG Intelligence mobile app
 */
export default {
  // App
  app_title: 'EKG Intelligence',
  app_subtitle: 'Analyse ECG de 26 conditions',

  // Navigation tabs
  tab_dashboard: 'Accueil',
  tab_scan: 'Numériser ECG',
  tab_patients: 'Patients',
  tab_settings: 'Paramètres',

  // Dashboard
  dashboard_scan_ecg: 'Numériser ECG',
  dashboard_patients: 'Patients',
  dashboard_model_info: 'Modèle',
  dashboard_model_version: 'V3 Multi-étiquettes (26 conditions)',
  dashboard_model_params: 'ECGNetJoint - 1,7M paramètres',
  dashboard_auroc: 'AUROC: 0,9682',
  dashboard_inference: 'Inférence sur le dispositif (ONNX Runtime)',

  // Patient form
  patient_first_name: 'Prénom',
  patient_first_name_placeholder: 'ex. Jean',
  patient_last_name: 'Nom',
  patient_last_name_placeholder: 'ex. Dupont',
  patient_id_number: 'ID Patient',
  patient_id_number_placeholder: 'ex. 123456789',
  patient_age: 'Âge',
  patient_sex: 'Sexe Biologique',
  patient_sex_male: 'Homme',
  patient_sex_female: 'Femme',
  patient_potassium: 'Potassium K+ (mmol/L)',
  patient_potassium_help: 'Plage normale : 3,5–5,0 mmol/L',
  patient_pacemaker: 'Pacemaker / DAI Présent',
  patient_athlete: 'Sportif de Haut Niveau',
  patient_pregnant: 'Grossesse',
  patient_save: 'Enregistrer le Patient',
  patient_delete: 'Supprimer le Patient',
  patient_save_success: 'Patient enregistré avec succès',
  patient_delete_success: 'Patient supprimé',
  patient_no_patients: 'Aucun patient pour le moment',
  patient_add_new: 'Ajouter un Patient',
  patient_edit: 'Modifier le Patient',

  // Patient list
  patient_list_title: 'Patients',
  patient_list_add_button: 'Ajouter un Patient',
  patient_item_age_sex: 'Âge {age}, {sex}',
  patient_item_updated: 'Modifié {date}',
  patient_item_pull_refresh: 'Tirez pour actualiser',

  // Patient detail
  patient_detail_info: 'Informations du Patient',
  patient_detail_records: 'Historique ECG',
  patient_detail_new_scan: 'Nouvelle Numérisation ECG',
  patient_detail_export_pdf: 'Exporter en PDF',
  patient_detail_delete_patient: 'Supprimer le Patient',
  patient_detail_no_records: 'Aucun enregistrement ECG pour le moment',

  // Analysis results
  analysis_title: 'Résultats d\'Analyse',
  analysis_detected_conditions: 'Conditions Détectées',
  analysis_condition_name: 'Condition',
  analysis_confidence: 'Probabilité',
  analysis_urgency: 'Urgence',
  analysis_urgency_critical: 'Critique',
  analysis_urgency_significant: 'Significative',
  analysis_urgency_monitor: 'Surveiller',
  analysis_urgency_normal: 'Normal',
  analysis_clinical_guidance: 'Recommandations Cliniques',
  analysis_action: 'Action Recommandée',
  analysis_note: 'Note Clinique',
  analysis_timestamp: 'Analysé {time}',
  analysis_model_version: 'Modèle : {version}',

  // Scan ECG
  scan_title: 'Numériser ECG',
  scan_subtitle: 'Capturer l\'image d\'un ECG sur papier',
  scan_button: 'Prendre une Photo',
  scan_retake: 'Reprendre',
  scan_analyze: 'Analyser',
  scan_analyzing: 'Analyse en cours...',

  // Settings
  settings_title: 'Paramètres',
  settings_security: 'Sécurité',
  settings_audit_log: 'Afficher le Registre d\'Activité',
  settings_audit_log_empty: 'Aucune activité enregistrée',
  settings_audit_log_event: '{event} - {timestamp}',
  settings_auto_lock: 'Délai de Verrouillage Automatique',
  settings_auto_lock_5min: '5 minutes',
  settings_auto_lock_10min: '10 minutes',
  settings_auto_lock_15min: '15 minutes',
  settings_auto_lock_30min: '30 minutes',
  settings_screen_protection: 'Protection d\'Écran',
  settings_screen_protection_enabled: 'Active',
  settings_data: 'Données',
  settings_language: 'Langue',
  settings_language_english: 'English',
  settings_language_spanish: 'Español',
  settings_language_french: 'Français',
  settings_delete_all: 'Supprimer Toutes Mes Données',
  settings_delete_confirm_title: 'Supprimer Toutes les Données',
  settings_delete_confirm_msg:
    'Cela supprimera définitivement tous les dossiers patients, les données ECG et les résultats d\'analyse. Le registre d\'audit sera conservé à des fins de conformité. Cette action ne peut pas être annulée.',
  settings_delete_cancel: 'Annuler',
  settings_delete_confirm: 'Supprimer Tout',
  settings_delete_success: 'Toutes les données supprimées',
  settings_about: 'À Propos',
  settings_model_version: 'Version du Modèle',
  settings_model_info: 'V3 Multi-étiquettes (26 conditions)',
  settings_inference: 'Moteur d\'Inférence',
  settings_inference_info: 'ONNX Runtime sur le dispositif',

  // Clinical disclaimer (MUST appear on every results screen)
  disclaimer:
    'À titre éducatif uniquement. Ceci n\'est pas un diagnostic médical. Non agréé par la FDA. Consultez toujours un professionnel de santé qualifié.',

  // Audit log
  audit_log_title: 'Registre d\'Activité',
  audit_event_auth_success: 'Authentification réussie',
  audit_event_auth_fail: 'Échec de l\'authentification',
  audit_event_auth_lock: 'Session verrouillée',
  audit_event_phi_view: 'Données accédées',
  audit_event_phi_create: 'Données créées',
  audit_event_phi_update: 'Données mises à jour',
  audit_event_phi_delete: 'Données supprimées',
  audit_event_phi_export: 'Données exportées',
  audit_event_inference_run: 'Analyse exécutée',
  audit_event_data_wipe: 'Toutes les données supprimées',

  // Auth
  auth_lock_title: 'EKG Intelligence',
  auth_lock_subtitle: 'Vos données de santé sont chiffrées et protégées',
  auth_unlock_button: 'S\'Authentifier',
  auth_biometric_prompt: 'Authentifiez-vous pour accéder à vos données de santé',
  auth_failed: 'Authentification échouée. Veuillez réessayer.',
  auth_no_biometrics: 'Cet appareil n\'a pas d\'authentification biométrique configurée.',

  // Common
  common_cancel: 'Annuler',
  common_save: 'Enregistrer',
  common_delete: 'Supprimer',
  common_edit: 'Modifier',
  common_close: 'Fermer',
  common_loading: 'Chargement...',
  common_error: 'Erreur',
  common_success: 'Succès',
  common_back: 'Retour',
  common_next: 'Suivant',
  common_finish: 'Terminer',
  common_yes: 'Oui',
  common_no: 'Non',
  common_ok: 'OK',
  common_empty_state: 'Aucune donnée disponible',
};
