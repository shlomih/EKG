"""Spanish (Español) strings for EKG Intelligence."""

STRINGS = {
    # ── Página / App ─────────────────────────────────────────────────────────
    "page_title":               "EKG Intel POC",
    "app_title":                "🩺 EKG Intelligence",

    # ── Perfil del Paciente ──────────────────────────────────────────────────
    "patient_profile_header":   "Perfil del Paciente",
    "patient_profile_caption":  "El contexto cambia la interpretación, no solo las etiquetas.",
    "first_name_label":         "Nombre",
    "first_name_placeholder":   "ej. Juan",
    "last_name_label":          "Apellido",
    "last_name_placeholder":    "ej. García",
    "patient_id_label":         "ID del Paciente",
    "patient_id_placeholder":   "ej. 123456789",
    "age_label":                "Edad",
    "sex_label":                "Sexo Biológico",
    "potassium_label":          "Potasio K⁺ (mmol/L)",
    "potassium_help":           "Normal: 3,5–5,0 mmol/L",
    "logic_inverters":          "**Modificadores Clínicos**",
    "pacemaker_label":          "Marcapasos / DAI Presente",
    "athlete_label":            "Atleta de Alto Rendimiento",
    "pregnancy_label":          "Embarazo",
    "save_patient_btn":         "Guardar Paciente",
    "load_patient_label":       "Cargar paciente existente",
    "saved_msg":                "Guardado: {name} (ID: {pid})",
    "loaded_msg":               "Cargado: {first} {last} | Edad: {age} | Sexo: {sex}",

    # ── Pestañas ─────────────────────────────────────────────────────────────
    "tab_scan":                 "📸 Escáner Móvil",
    "tab_data":                 "📂 Explorador de Datos",

    # ── Pestaña de escaneo ──────────────────────────────────────────────────
    "input_method_label":       "Método de entrada",
    "camera_option":            "Cámara",
    "upload_option":            "Subir imagen",
    "scanner_label":            "Escáner",
    "upload_label":             "Subir imagen del trazado ECG",
    "input_image_caption":      "Imagen de entrada",
    "cannot_decode_image":      "No se pudo decodificar la imagen.",
    "signal_too_short_scan":    "La señal extraída mide solo {dur:.1f}s ({n} muestras). "
                                "Use una imagen más ancha u orientación horizontal para mejores resultados.",

    # Configuración de papel ECG
    "ecg_paper_settings":       "Configuración del papel ECG",
    "paper_speed_label":        "Velocidad del papel",
    "voltage_gain_label":       "Ganancia de voltaje",
    "paper_speed_standard":     "{x} mm/s (estándar)",
    "paper_speed_fast":         "{x} mm/s (rápido — UE/pediátrico)",
    "voltage_half":             "{x} mm/mV (medio estándar)",
    "voltage_standard":         "{x} mm/mV (estándar)",
    "voltage_double":           "{x} mm/mV (doble — complejos pequeños)",
    "paper_speed_help":         "25 mm/s es el estándar global. 50 mm/s se usa en algunos países europeos y ECG pediátricos.",
    "voltage_gain_help":        "10 mm/mV es el estándar. Use 5 mm/mV si los complejos están recortados. Use 20 mm/mV si la señal parece muy pequeña.",

    # Subtítulos de calibración
    "calib_detected":           "Calibrado: {speed} mm/s, {gain} mm/mV — "
                                "cuadrícula detectada ({px_x:.1f} px/mm horizontal, {px_y:.1f} px/mm vertical) — "
                                "calidad {quality:.0%}",
    "calib_no_grid":            "Señal extraída — cuadrícula no detectada, usando {speed} mm/s / "
                                "{gain} mm/mV con calibración por ancho de imagen — "
                                "calidad {quality:.0%}",
    "calib_fallback":           "Usando extracción por brillo como respaldo (unidades de píxel no calibradas)",

    # ── Pestaña Explorador de Datos ─────────────────────────────────────────
    "folder_label":             "Carpeta:",
    "record_label":             "Registro",
    "analyze_record_btn":       "Analizar Registro",
    "load_record_error":        "No se pudo cargar el registro: {error}",
    "no_signal_error":          "El registro no contiene datos de señal (p_signal ausente).",
    "record_too_short_error":   "El registro seleccionado es demasiado corto para el análisis.",

    # ── Visualización de 12 derivaciones ────────────────────────────────────
    "twelve_lead_header":       "#### ECG de 12 Derivaciones",
    "rhythm_strip_label":       "II (tira de ritmo)",
    "time_axis_label":          "Tiempo (s)",

    # ── Diagnóstico IA ───────────────────────────────────────────────────────
    "ai_diagnosis_header":      "#### Diagnóstico IA ({model})",
    "model_multilabel":         "CNN Multietiqueta (12 condiciones)",
    "model_ensemble":           "CNN Ensamble",
    "model_hybrid":             "CNN Híbrida",
    "model_cnn":                "CNN 1D",
    "model_sklearn":            "GradientBoosting",
    "normal_ecg_msg":           "**ECG Normal** — No se detectaron hallazgos significativos.",
    "findings_critical":        "**{n} crítico(s)**",
    "findings_abnormal":        "**{n} anormal(es)**",
    "findings_mild":            "{n} leve(s)",
    "findings_prefix":          "Hallazgos: {parts}",
    "all_cond_probs":           "Probabilidades de todas las condiciones",
    "urgency_critical":         "Crítico",
    "urgency_abnormal":         "Anormal",
    "urgency_mild":             "Hallazgo leve",
    "urgency_normal":           "Normal",
    "confidence_label":         "Confianza: {conf:.0%}",
    "hybrid_details":           "Detalles del clasificador híbrido",
    "hybrid_adjustment":        "Ajuste: {text}",
    "hybrid_cnn_raw":           "CNN cruda: {pred} ({conf:.0%})",
    "sokolow_lyon":             "Sokolow-Lyon",
    "cornell":                  "Cornell",
    "rvh_r_v1":                 "VHD (R en V1)",
    "met_label":                "CUMPLE",
    "not_met_label":            "no cumple",

    # ── Análisis de intervalos clínicos ─────────────────────────────────────
    "clinical_engine_missing":  "Motor clínico no disponible.",
    "signal_too_short_analysis":"La señal es demasiado corta para el análisis. Seleccione un trazado más largo.",
    "analysis_failure_error":   "Error de análisis: fallo inesperado al calcular intervalos: {error}",
    "analysis_invalid_error":   "Error de análisis: el calculador de intervalos devolvió resultados inválidos.",
    "analysis_error":           "Error de Análisis: {error}",
    "heart_rate_metric":        "Frecuencia Cardíaca",
    "heart_rate_unit":          "{val} lpm",
    "pr_interval_metric":       "Intervalo PR",
    "qrs_duration_metric":      "Duración QRS",
    "qtc_metric":               "QTc (Bazett)",
    "interval_unit":            "{val} ms",
    "clinical_findings":        "📝 Hallazgos Clínicos",

    # ── Análisis de territorio ST ────────────────────────────────────────────
    "st_territory_header":      "#### Análisis de Territorio del Segmento ST",
    "per_lead_st":              "Mediciones ST por derivación",
    "lead_col":                 "Derivación",
    "st_mv_col":                "ST (mV)",
    "status_col":               "Estado",
    "beats_col":                "Latidos",
    "st_elevated":              "Elevado",
    "st_depressed":             "Deprimido",
    "st_normal":                "Normal",

    # ── Análisis de reglas clínicas ──────────────────────────────────────────
    "clinical_rules_header":    "#### Análisis de Reglas Clínicas",
    "cardiac_axis_metric":      "Eje Cardíaco",
    "cardiac_axis_unit":        "{val:.0f}°",
    "axis_deviation_metric":    "Desviación del Eje",

    # ── Guardar / Historial del Paciente ────────────────────────────────────
    "save_analysis_btn":        "Guardar Análisis en Expediente",
    "analysis_saved_msg":       "Análisis guardado en expediente del paciente (ID ECG: {eid})",
    "patient_history":          "Historial del Paciente",
    "no_records_caption":       "No hay registros previos para este paciente.",
    "history_date_col":         "Fecha",
    "history_source_col":       "Origen",
    "history_class_col":        "Clasificación",
    "history_conf_col":         "Confianza",
    "history_hr_col":           "FC",
    "history_urgency_col":      "Urgencia",

    # ── Exportar y Compartir ────────────────────────────────────────────────
    "export_header":            "#### Exportar y Compartir",
    "pdf_btn":                  "Reporte PDF",
    "email_btn":                "Correo",
    "whatsapp_btn":             "WhatsApp",
    "copy_btn":                 "Copiar Texto",
    "copied_btn":               "¡Copiado!",
    "preview_share":            "Vista previa del texto",
    "pdf_warning":              "El generador de PDF no devolvió bytes binarios. Descarga no disponible.",

    # Texto para compartir
    "share_ecg_report":         "Informe ECG",
    "share_patient_fallback":   "Paciente",
    "share_date_label":         "Fecha",
    "share_age_sex":            "Edad: {age} | Sexo: {sex}",
    "share_ai_diagnosis":       "Diagnóstico IA",
    "share_intervals":          "FC: {hr} lpm | PR: {pr} ms | QRS: {qrs} ms | QTc: {qtc} ms",
    "share_st_analysis":        "Análisis ST",
    "share_findings_header":    "Hallazgos:",
    "share_platform":           "-- EKG Intelligence Platform",

    # ── ST análisis heredado (respaldo) ─────────────────────────────────────
    "st_analysis_header":       "#### Análisis del Segmento ST",
    "st_no_estimate":           "El análisis del segmento ST no devolvió estimación de elevación.",
    "st_omi_alert":             "🚨 ALERTA OMI: Elevación ST {val:.2f} mm",
    "st_ischemia":              "⚠️ ISQUEMIA: Depresión ST {val:.2f} mm",
    "st_stable":                "✅ Segmento ST Estable",
}
