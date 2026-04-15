/**
 * Spanish (Español) translation strings for EKG Intelligence mobile app
 */
export default {
  // App
  app_title: 'EKG Intelligence',
  app_subtitle: 'Análisis de ECG de 26 condiciones',

  // Navigation tabs
  tab_dashboard: 'Panel',
  tab_scan: 'Escanear ECG',
  tab_patients: 'Pacientes',
  tab_settings: 'Configuración',

  // Dashboard
  dashboard_scan_ecg: 'Escanear ECG',
  dashboard_patients: 'Pacientes',
  dashboard_model_info: 'Modelo',
  dashboard_model_version: 'V3 Multietiqueta (26 condiciones)',
  dashboard_model_params: 'ECGNetJoint - 1,7M parámetros',
  dashboard_auroc: 'AUROC: 0,9682',
  dashboard_inference: 'Inferencia en dispositivo (ONNX Runtime)',

  // Patient form
  patient_first_name: 'Nombre',
  patient_first_name_placeholder: 'ej. Juan',
  patient_last_name: 'Apellido',
  patient_last_name_placeholder: 'ej. García',
  patient_id_number: 'ID del Paciente',
  patient_id_number_placeholder: 'ej. 123456789',
  patient_age: 'Edad',
  patient_sex: 'Sexo Biológico',
  patient_sex_male: 'Hombre',
  patient_sex_female: 'Mujer',
  patient_potassium: 'Potasio K+ (mmol/L)',
  patient_potassium_help: 'Rango normal: 3,5–5,0 mmol/L',
  patient_pacemaker: 'Marcapasos / DAI Presente',
  patient_athlete: 'Deportista de Alto Rendimiento',
  patient_pregnant: 'Embarazo',
  patient_save: 'Guardar Paciente',
  patient_delete: 'Eliminar Paciente',
  patient_save_success: 'Paciente guardado correctamente',
  patient_delete_success: 'Paciente eliminado',
  patient_no_patients: 'Sin pacientes aún',
  patient_add_new: 'Añadir Paciente',
  patient_edit: 'Editar Paciente',

  // Patient list
  patient_list_title: 'Pacientes',
  patient_list_add_button: 'Añadir Paciente',
  patient_item_age_sex: 'Edad {age}, {sex}',
  patient_item_updated: 'Actualizado {date}',
  patient_item_pull_refresh: 'Tirone para actualizar',

  // Patient detail
  patient_detail_info: 'Información del Paciente',
  patient_detail_records: 'Historial de ECG',
  patient_detail_new_scan: 'Nuevo Escaneo ECG',
  patient_detail_export_pdf: 'Exportar PDF',
  patient_detail_delete_patient: 'Eliminar Paciente',
  patient_detail_no_records: 'Sin registros de ECG aún',

  // Analysis results
  analysis_title: 'Resultados del Análisis',
  analysis_detected_conditions: 'Condiciones Detectadas',
  analysis_condition_name: 'Condición',
  analysis_confidence: 'Probabilidad',
  analysis_urgency: 'Urgencia',
  analysis_urgency_critical: 'Crítica',
  analysis_urgency_significant: 'Significativa',
  analysis_urgency_monitor: 'Monitorear',
  analysis_urgency_normal: 'Normal',
  analysis_clinical_guidance: 'Orientación Clínica',
  analysis_action: 'Acción Recomendada',
  analysis_note: 'Nota Clínica',
  analysis_timestamp: 'Analizado {time}',
  analysis_model_version: 'Modelo: {version}',

  // Scan ECG
  scan_title: 'Escanear ECG',
  scan_subtitle: 'Capturar imagen de ECG en papel',
  scan_button: 'Tomar Foto',
  scan_retake: 'Repetir',
  scan_analyze: 'Analizar',
  scan_analyzing: 'Analizando...',

  // Settings
  settings_title: 'Configuración',
  settings_security: 'Seguridad',
  settings_audit_log: 'Ver Registro de Actividad',
  settings_audit_log_empty: 'Sin actividad registrada',
  settings_audit_log_event: '{event} - {timestamp}',
  settings_auto_lock: 'Tiempo de Bloqueo Automático',
  settings_auto_lock_5min: '5 minutos',
  settings_auto_lock_10min: '10 minutos',
  settings_auto_lock_15min: '15 minutos',
  settings_auto_lock_30min: '30 minutos',
  settings_screen_protection: 'Protección de Pantalla',
  settings_screen_protection_enabled: 'Activa',
  settings_data: 'Datos',
  settings_language: 'Idioma',
  settings_language_english: 'English',
  settings_language_spanish: 'Español',
  settings_language_french: 'Français',
  settings_delete_all: 'Eliminar Todos Mis Datos',
  settings_delete_confirm_title: 'Eliminar Todos los Datos',
  settings_delete_confirm_msg:
    'Esto eliminará permanentemente todos los registros de pacientes, datos de ECG y resultados de análisis. El registro de auditoría se conservará para cumplimiento. Esta acción no se puede deshacer.',
  settings_delete_cancel: 'Cancelar',
  settings_delete_confirm: 'Eliminar Todo',
  settings_delete_success: 'Todos los datos eliminados',
  settings_about: 'Acerca de',
  settings_model_version: 'Versión del Modelo',
  settings_model_info: 'V3 Multietiqueta (26 condiciones)',
  settings_inference: 'Motor de Inferencia',
  settings_inference_info: 'ONNX Runtime en dispositivo',

  // Clinical disclaimer (MUST appear on every results screen)
  disclaimer:
    'Solo para fines educativos. Esto no es un diagnóstico médico. No está aprobado por la FDA. Consulte siempre con un profesional de salud calificado.',

  // Audit log
  audit_log_title: 'Registro de Actividad',
  audit_event_auth_success: 'Autenticación exitosa',
  audit_event_auth_fail: 'Autenticación fallida',
  audit_event_auth_lock: 'Sesión bloqueada',
  audit_event_phi_view: 'Datos accedidos',
  audit_event_phi_create: 'Datos creados',
  audit_event_phi_update: 'Datos actualizados',
  audit_event_phi_delete: 'Datos eliminados',
  audit_event_phi_export: 'Datos exportados',
  audit_event_inference_run: 'Análisis ejecutado',
  audit_event_data_wipe: 'Todos los datos eliminados',

  // Auth
  auth_lock_title: 'EKG Intelligence',
  auth_lock_subtitle: 'Sus datos de salud están cifrados y protegidos',
  auth_unlock_button: 'Autenticar',
  auth_biometric_prompt: 'Autentíquese para acceder a sus datos de salud',
  auth_failed: 'Autenticación fallida. Por favor, intente de nuevo.',
  auth_no_biometrics: 'Este dispositivo no tiene autenticación biométrica configurada.',

  // Common
  common_cancel: 'Cancelar',
  common_save: 'Guardar',
  common_delete: 'Eliminar',
  common_edit: 'Editar',
  common_close: 'Cerrar',
  common_loading: 'Cargando...',
  common_error: 'Error',
  common_success: 'Éxito',
  common_back: 'Atrás',
  common_next: 'Siguiente',
  common_finish: 'Finalizar',
  common_yes: 'Sí',
  common_no: 'No',
  common_ok: 'Aceptar',
  common_empty_state: 'Sin datos disponibles',
};
