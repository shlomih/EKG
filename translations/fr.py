"""French (Français) strings for EKG Intelligence."""

STRINGS = {
    # ── Page / App ───────────────────────────────────────────────────────────
    "page_title":               "EKG Intel POC",
    "app_title":                "🩺 EKG Intelligence",

    # ── Profil du patient ────────────────────────────────────────────────────
    "patient_profile_header":   "Profil du Patient",
    "patient_profile_caption":  "Le contexte change l'interprétation, pas seulement les étiquettes.",
    "first_name_label":         "Prénom",
    "first_name_placeholder":   "ex. Jean",
    "last_name_label":          "Nom",
    "last_name_placeholder":    "ex. Dupont",
    "patient_id_label":         "ID Patient",
    "patient_id_placeholder":   "ex. 123456789",
    "age_label":                "Âge",
    "sex_label":                "Sexe Biologique",
    "potassium_label":          "Potassium K⁺ (mmol/L)",
    "potassium_help":           "Normal : 3,5–5,0 mmol/L",
    "logic_inverters":          "**Modificateurs Cliniques**",
    "pacemaker_label":          "Pacemaker / DAI Présent",
    "athlete_label":            "Sportif de Haut Niveau",
    "pregnancy_label":          "Grossesse",
    "save_patient_btn":         "Enregistrer le Patient",
    "load_patient_label":       "Charger un patient existant",
    "saved_msg":                "Enregistré : {name} (ID : {pid})",
    "loaded_msg":               "Chargé : {first} {last} | Âge : {age} | Sexe : {sex}",

    # ── Onglets ──────────────────────────────────────────────────────────────
    "tab_scan":                 "📸 Scan Mobile",
    "tab_data":                 "📂 Explorateur de Données",

    # ── Onglet scan ──────────────────────────────────────────────────────────
    "input_method_label":       "Méthode de saisie",
    "camera_option":            "Caméra",
    "upload_option":            "Importer une image",
    "scanner_label":            "Scanner",
    "upload_label":             "Importer une image du tracé ECG",
    "input_image_caption":      "Image importée",
    "cannot_decode_image":      "Impossible de décoder l'image.",
    "signal_too_short_scan":    "Le signal extrait ne fait que {dur:.1f}s ({n} échantillons). "
                                "Utilisez une image plus large ou en orientation paysage.",

    # Paramètres papier ECG
    "ecg_paper_settings":       "Paramètres du papier ECG",
    "paper_speed_label":        "Vitesse du papier",
    "voltage_gain_label":       "Gain de tension",
    "paper_speed_standard":     "{x} mm/s (standard)",
    "paper_speed_fast":         "{x} mm/s (rapide — UE/pédiatrique)",
    "voltage_half":             "{x} mm/mV (demi-standard)",
    "voltage_standard":         "{x} mm/mV (standard)",
    "voltage_double":           "{x} mm/mV (double — petits complexes)",
    "paper_speed_help":         "25 mm/s est le standard mondial. 50 mm/s est utilisé dans certains pays européens et pour les ECG pédiatriques.",
    "voltage_gain_help":        "10 mm/mV est le standard. Utilisez 5 mm/mV si les complexes sont écrêtés. Utilisez 20 mm/mV si le signal semble trop petit.",

    # Légendes de calibration
    "calib_detected":           "Calibré : {speed} mm/s, {gain} mm/mV — "
                                "grille détectée ({px_x:.1f} px/mm horizontal, {px_y:.1f} px/mm vertical) — "
                                "qualité {quality:.0%}",
    "calib_no_grid":            "Signal extrait — grille non détectée, utilisation de {speed} mm/s / "
                                "{gain} mm/mV avec calibration par largeur d'image — "
                                "qualité {quality:.0%}",
    "calib_fallback":           "Extraction par luminosité en secours (unités pixel non calibrées)",

    # ── Onglet Explorateur de Données ────────────────────────────────────────
    "folder_label":             "Dossier :",
    "record_label":             "Enregistrement",
    "analyze_record_btn":       "Analyser l'Enregistrement",
    "load_record_error":        "Impossible de charger l'enregistrement : {error}",
    "no_signal_error":          "L'enregistrement ne contient pas de données de signal (p_signal manquant).",
    "record_too_short_error":   "L'enregistrement sélectionné est trop court pour l'analyse.",

    # ── Affichage 12 dérivations ─────────────────────────────────────────────
    "twelve_lead_header":       "#### ECG 12 Dérivations",
    "rhythm_strip_label":       "II (bande de rythme)",
    "time_axis_label":          "Temps (s)",

    # ── Diagnostic IA ────────────────────────────────────────────────────────
    "ai_diagnosis_header":      "#### Diagnostic IA ({model})",
    "model_multilabel":         "CNN Multi-étiquettes (12 conditions)",
    "model_ensemble":           "CNN Ensemble",
    "model_hybrid":             "CNN Hybride",
    "model_cnn":                "CNN 1D",
    "model_sklearn":            "GradientBoosting",
    "normal_ecg_msg":           "**ECG Normal** — Aucun résultat significatif détecté.",
    "findings_critical":        "**{n} critique(s)**",
    "findings_abnormal":        "**{n} anormal(aux)**",
    "findings_mild":            "{n} léger(s)",
    "findings_prefix":          "Résultats : {parts}",
    "all_cond_probs":           "Probabilités de toutes les conditions",
    "urgency_critical":         "Critique",
    "urgency_abnormal":         "Anormal",
    "urgency_mild":             "Résultat léger",
    "urgency_normal":           "Normal",
    "confidence_label":         "Confiance : {conf:.0%}",
    "hybrid_details":           "Détails du classifieur hybride",
    "hybrid_adjustment":        "Ajustement : {text}",
    "hybrid_cnn_raw":           "CNN brut : {pred} ({conf:.0%})",
    "sokolow_lyon":             "Sokolow-Lyon",
    "cornell":                  "Cornell",
    "rvh_r_v1":                 "HVD (R en V1)",
    "met_label":                "ATTEINT",
    "not_met_label":            "non atteint",

    # ── Analyse des intervalles cliniques ────────────────────────────────────
    "clinical_engine_missing":  "Moteur clinique non disponible.",
    "signal_too_short_analysis":"Le signal est trop court pour l'analyse. Veuillez sélectionner un tracé plus long.",
    "analysis_failure_error":   "Échec d'analyse : erreur inattendue lors du calcul des intervalles : {error}",
    "analysis_invalid_error":   "Échec d'analyse : le calculateur d'intervalles a renvoyé des résultats invalides.",
    "analysis_error":           "Erreur d'Analyse : {error}",
    "heart_rate_metric":        "Fréquence Cardiaque",
    "heart_rate_unit":          "{val} bpm",
    "pr_interval_metric":       "Intervalle PR",
    "qrs_duration_metric":      "Durée QRS",
    "qtc_metric":               "QTc (Bazett)",
    "interval_unit":            "{val} ms",
    "clinical_findings":        "📝 Résultats Cliniques",

    # ── Analyse du territoire ST ─────────────────────────────────────────────
    "st_territory_header":      "#### Analyse du Territoire du Segment ST",
    "per_lead_st":              "Mesures ST par dérivation",
    "lead_col":                 "Dérivation",
    "st_mv_col":                "ST (mV)",
    "status_col":               "Statut",
    "beats_col":                "Battements",
    "st_elevated":              "Élevé",
    "st_depressed":             "Abaissé",
    "st_normal":                "Normal",

    # ── Analyse des règles cliniques ─────────────────────────────────────────
    "clinical_rules_header":    "#### Analyse des Règles Cliniques",
    "cardiac_axis_metric":      "Axe Cardiaque",
    "cardiac_axis_unit":        "{val:.0f}°",
    "axis_deviation_metric":    "Déviation de l'Axe",

    # ── Enregistrer / Historique du patient ──────────────────────────────────
    "save_analysis_btn":        "Enregistrer l'Analyse au Dossier",
    "analysis_saved_msg":       "Analyse enregistrée dans le dossier du patient (ID ECG : {eid})",
    "patient_history":          "Historique du Patient",
    "no_records_caption":       "Aucun antécédent pour ce patient.",
    "history_date_col":         "Date",
    "history_source_col":       "Source",
    "history_class_col":        "Classification",
    "history_conf_col":         "Confiance",
    "history_hr_col":           "FC",
    "history_urgency_col":      "Urgence",

    # ── Exporter et Partager ─────────────────────────────────────────────────
    "export_header":            "#### Exporter et Partager",
    "pdf_btn":                  "Rapport PDF",
    "email_btn":                "E-mail",
    "whatsapp_btn":             "WhatsApp",
    "copy_btn":                 "Copier le Texte",
    "copied_btn":               "Copié !",
    "preview_share":            "Aperçu du texte",
    "pdf_warning":              "Le générateur PDF n'a pas renvoyé d'octets binaires. Téléchargement indisponible.",

    # Texte à partager
    "share_ecg_report":         "Rapport ECG",
    "share_patient_fallback":   "Patient",
    "share_date_label":         "Date",
    "share_age_sex":            "Âge : {age} | Sexe : {sex}",
    "share_ai_diagnosis":       "Diagnostic IA",
    "share_intervals":          "FC : {hr} bpm | PR : {pr} ms | QRS : {qrs} ms | QTc : {qtc} ms",
    "share_st_analysis":        "Analyse ST",
    "share_findings_header":    "Résultats :",
    "share_platform":           "-- EKG Intelligence Platform",

    # ── Analyse ST héritée (secours) ─────────────────────────────────────────
    "st_analysis_header":       "#### Analyse du Segment ST",
    "st_no_estimate":           "L'analyse du segment ST n'a pas renvoyé d'estimation d'élévation.",
    "st_omi_alert":             "🚨 ALERTE OMI : Élévation ST {val:.2f} mm",
    "st_ischemia":              "⚠️ ISCHÉMIE : Dépression ST {val:.2f} mm",
    "st_stable":                "✅ Segment ST Stable",
}
