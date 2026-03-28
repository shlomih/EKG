"""English strings for EKG Intelligence."""

STRINGS = {
    # ── Page / App ──────────────────────────────────────────────────────────
    "page_title":               "EKG Intel POC",
    "app_title":                "🩺 EKG Intelligence",

    # ── Patient Profile ──────────────────────────────────────────────────────
    "patient_profile_header":   "Patient Profile",
    "patient_profile_caption":  "Context changes interpretation — not just display labels.",
    "first_name_label":         "First Name",
    "first_name_placeholder":   "e.g. John",
    "last_name_label":          "Last Name",
    "last_name_placeholder":    "e.g. Doe",
    "patient_id_label":         "Patient ID",
    "patient_id_placeholder":   "e.g. 123456789",
    "age_label":                "Age",
    "sex_label":                "Biological Sex",
    "potassium_label":          "Potassium K⁺ (mmol/L)",
    "potassium_help":           "Normal: 3.5–5.0 mmol/L",
    "logic_inverters":          "**Logic Inverters**",
    "pacemaker_label":          "Pacemaker / ICD Present",
    "athlete_label":            "Athlete Status",
    "pregnancy_label":          "Pregnancy",
    "save_patient_btn":         "Save Patient",
    "load_patient_label":       "Load existing patient",
    "saved_msg":                "Saved: {name} (ID: {pid})",
    "loaded_msg":               "Loaded: {first} {last} | Age: {age} | Sex: {sex}",

    # ── Tabs ─────────────────────────────────────────────────────────────────
    "tab_scan":                 "📸 Mobile Scan",
    "tab_data":                 "📂 Dataset Explorer",

    # ── Scan tab ─────────────────────────────────────────────────────────────
    "input_method_label":       "Input method",
    "camera_option":            "Camera",
    "upload_option":            "Upload image",
    "scanner_label":            "Scanner",
    "upload_label":             "Upload an EKG strip image",
    "input_image_caption":      "Input image",
    "cannot_decode_image":      "Could not decode image.",
    "signal_too_short_scan":    "Extracted signal is only {dur:.1f}s ({n} samples). "
                                "Use a wider image or landscape orientation for best results.",

    # ECG paper settings
    "ecg_paper_settings":       "ECG paper settings",
    "paper_speed_label":        "Paper speed",
    "voltage_gain_label":       "Voltage gain",
    "paper_speed_standard":     "{x} mm/s (standard)",
    "paper_speed_fast":         "{x} mm/s (fast — EU/pediatric)",
    "voltage_half":             "{x} mm/mV (half standard)",
    "voltage_standard":         "{x} mm/mV (standard)",
    "voltage_double":           "{x} mm/mV (double — small complexes)",
    "paper_speed_help":         "25 mm/s is the global standard. 50 mm/s is used in some European countries and pediatric ECGs.",
    "voltage_gain_help":        "10 mm/mV is standard. Use 5 mm/mV if complexes are clipped. Use 20 mm/mV if the signal looks too small.",

    # Calibration captions
    "calib_detected":           "Calibrated: {speed} mm/s, {gain} mm/mV — "
                                "grid detected ({px_x:.1f} px/mm horizontal, {px_y:.1f} px/mm vertical) — "
                                "quality {quality:.0%}",
    "calib_no_grid":            "Signal extracted — grid not detected, using {speed} mm/s / "
                                "{gain} mm/mV settings with image-width calibration — "
                                "quality {quality:.0%}",
    "calib_fallback":           "Using fallback brightness-based extraction (uncalibrated pixel units)",

    # ── Dataset Explorer tab ─────────────────────────────────────────────────
    "folder_label":             "Folder:",
    "record_label":             "Record",
    "analyze_record_btn":       "Analyze Record",
    "load_record_error":        "Could not load record: {error}",
    "no_signal_error":          "Record does not contain signal data (p_signal missing).",
    "record_too_short_error":   "Selected record is too short for analysis.",

    # ── 12-Lead ECG display ──────────────────────────────────────────────────
    "twelve_lead_header":       "#### 12-Lead ECG",
    "rhythm_strip_label":       "II (rhythm strip)",
    "time_axis_label":          "Time (s)",

    # ── AI Diagnosis ─────────────────────────────────────────────────────────
    "ai_diagnosis_header":      "#### AI Diagnosis ({model})",
    "model_multilabel":         "Multi-Label CNN (12 conditions)",
    "model_ensemble":           "Ensemble CNN",
    "model_hybrid":             "Hybrid CNN",
    "model_cnn":                "1D CNN",
    "model_sklearn":            "GradientBoosting",
    "normal_ecg_msg":           "**Normal ECG** — No significant findings detected.",
    "findings_critical":        "**{n} critical**",
    "findings_abnormal":        "**{n} abnormal**",
    "findings_mild":            "{n} mild",
    "findings_prefix":          "Findings: {parts}",
    "all_cond_probs":           "All condition probabilities",
    "urgency_critical":         "Critical",
    "urgency_abnormal":         "Abnormal",
    "urgency_mild":             "Mild finding",
    "urgency_normal":           "Normal",
    "confidence_label":         "Confidence: {conf:.0%}",
    "hybrid_details":           "Hybrid classifier details",
    "hybrid_adjustment":        "Adjustment: {text}",
    "hybrid_cnn_raw":           "CNN raw: {pred} ({conf:.0%})",
    "sokolow_lyon":             "Sokolow-Lyon",
    "cornell":                  "Cornell",
    "rvh_r_v1":                 "RVH (R in V1)",
    "met_label":                "MET",
    "not_met_label":            "not met",

    # ── Clinical interval analysis ────────────────────────────────────────────
    "clinical_engine_missing":  "Clinical Engine not found.",
    "signal_too_short_analysis":"Signal is too short for analysis. Please select a longer trace.",
    "analysis_failure_error":   "Analysis failure: unexpected error while calculating intervals: {error}",
    "analysis_invalid_error":   "Analysis failure: interval calculator returned invalid results.",
    "analysis_error":           "Analysis Error: {error}",
    "heart_rate_metric":        "Heart Rate",
    "heart_rate_unit":          "{val} bpm",
    "pr_interval_metric":       "PR Interval",
    "qrs_duration_metric":      "QRS Duration",
    "qtc_metric":               "QTc (Bazett)",
    "interval_unit":            "{val} ms",
    "clinical_findings":        "📝 Clinical Findings",

    # ── ST-Segment Territory Analysis ────────────────────────────────────────
    "st_territory_header":      "#### ST-Segment Territory Analysis",
    "per_lead_st":              "Per-lead ST measurements",
    "lead_col":                 "Lead",
    "st_mv_col":                "ST (mV)",
    "status_col":               "Status",
    "beats_col":                "Beats",
    "st_elevated":              "Elevated",
    "st_depressed":             "Depressed",
    "st_normal":                "Normal",

    # ── Clinical Rules Analysis ──────────────────────────────────────────────
    "clinical_rules_header":    "#### Clinical Rules Analysis",
    "cardiac_axis_metric":      "Cardiac Axis",
    "cardiac_axis_unit":        "{val:.0f} deg",
    "axis_deviation_metric":    "Axis Deviation",

    # ── Save / Patient History ────────────────────────────────────────────────
    "save_analysis_btn":        "Save Analysis to Patient Record",
    "analysis_saved_msg":       "Analysis saved to patient record (EKG ID: {eid})",
    "patient_history":          "Patient History",
    "no_records_caption":       "No previous records for this patient.",
    "history_date_col":         "Date",
    "history_source_col":       "Source",
    "history_class_col":        "Classification",
    "history_conf_col":         "Confidence",
    "history_hr_col":           "HR",
    "history_urgency_col":      "Urgency",

    # ── Export & Share ───────────────────────────────────────────────────────
    "export_header":            "#### Export & Share",
    "pdf_btn":                  "PDF Report",
    "email_btn":                "Email",
    "whatsapp_btn":             "WhatsApp",
    "copy_btn":                 "Copy Text",
    "copied_btn":               "Copied!",
    "preview_share":            "Preview share text",
    "pdf_warning":              "PDF generator did not return binary bytes. Download unavailable.",

    # Share text (plain text content, not UI labels)
    "share_ecg_report":         "ECG Report",
    "share_patient_fallback":   "Patient",
    "share_date_label":         "Date",
    "share_age_sex":            "Age: {age} | Sex: {sex}",
    "share_ai_diagnosis":       "AI Diagnosis",
    "share_intervals":          "HR: {hr} bpm | PR: {pr} ms | QRS: {qrs} ms | QTc: {qtc} ms",
    "share_st_analysis":        "ST Analysis",
    "share_findings_header":    "Findings:",
    "share_platform":           "-- EKG Intelligence Platform",

    # ── Legacy single-lead ST (fallback) ────────────────────────────────────
    "st_analysis_header":       "#### ST-Segment Analysis",
    "st_no_estimate":           "ST segment analysis returned no elevation estimate.",
    "st_omi_alert":             "🚨 OMI ALERT: ST-Elevation {val:.2f} mm",
    "st_ischemia":              "⚠️ ISCHEMIA: ST-Depression {val:.2f} mm",
    "st_stable":                "✅ ST-Segment Stable",
}
