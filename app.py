import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import wfdb
import os
from datetime import datetime
from pathlib import Path

# ── i18n ─────────────────────────────────────────────────────────────────────
try:
    from translations import t, language_selector
    I18N_AVAILABLE = True
except ImportError:
    I18N_AVAILABLE = False
    def t(key, **kwargs):  # noqa: E302
        return key
    def language_selector():  # noqa: E302
        pass

# --- DATABASE ---
try:
    from database_setup import (
        init_db, save_patient, get_patient, list_patients,
        save_ekg_record, get_patient_records, save_analysis,
    )
    init_db()  # ensure tables exist
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# --- PDF REPORT ---
try:
    from report_generator import generate_pdf_report
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# --- CORE IMPORTS ---
try:
    from interval_calculator import (
        calculate_intervals,
        apply_clinical_context,
        URGENCY_CONFIG,
        SEVERITY_EMOJI,
        REFERENCE,
    )
    INTERVAL_ENGINE_AVAILABLE = True
except ImportError:
    INTERVAL_ENGINE_AVAILABLE = False

# Load order: multilabel > ensemble > single CNN > sklearn
_clf_model = None
_clf_type = None
CLASSIFIER_AVAILABLE = False

try:
    from multilabel_classifier import load_multilabel_cnn, predict_multilabel
    _clf_model = load_multilabel_cnn()
    if _clf_model is not None:
        CLASSIFIER_AVAILABLE = True
        _clf_type = "multilabel"
except Exception:
    pass

if not CLASSIFIER_AVAILABLE:
    try:
        from ensemble_classifier import load_ensemble, predict_ensemble
        _clf_model = load_ensemble()
        if _clf_model is not None:
            CLASSIFIER_AVAILABLE = True
            _clf_type = "ensemble"
    except Exception:
        pass

if not CLASSIFIER_AVAILABLE:
    try:
        from cnn_classifier import load_cnn_classifier, predict_cnn
        _clf_model = load_cnn_classifier()
        if _clf_model is not None:
            CLASSIFIER_AVAILABLE = True
            _clf_type = "cnn"
    except ImportError:
        pass

if not CLASSIFIER_AVAILABLE:
    try:
        from poc_classifier import load_classifier, predict as classify_ecg_sklearn
        _clf_model = load_classifier()
        if _clf_model is not None:
            CLASSIFIER_AVAILABLE = True
            _clf_type = "sklearn"
    except ImportError:
        pass


# Hybrid classifier (CNN + voltage criteria) — used for single-model fallback only
HYBRID_AVAILABLE = False
try:
    from hybrid_classifier import hybrid_predict
    HYBRID_AVAILABLE = True
except ImportError:
    pass


def classify_ecg(model, signal_12, fs, lead_names=None, sex="M", age=50):
    """Classify using multilabel (preferred) or fallback to ensemble/hybrid/CNN/sklearn."""
    if _clf_type == "multilabel":
        return predict_multilabel(model, signal_12, fs=fs, sex=sex, age=age)
    elif _clf_type == "ensemble":
        return predict_ensemble(model, signal_12, fs=fs, sex=sex, age=age)
    elif HYBRID_AVAILABLE and _clf_type == "cnn" and lead_names is not None:
        return hybrid_predict(model, signal_12, fs, lead_names, sex=sex)
    elif _clf_type == "cnn":
        return predict_cnn(model, signal_12, fs)
    else:
        return classify_ecg_sklearn(model, signal_12, fs)


try:
    from st_territory import analyze_st_territories
    ST_TERRITORY_AVAILABLE = True
except ImportError:
    ST_TERRITORY_AVAILABLE = False

# Clinical rules engine
try:
    from clinical_rules import analyze_clinical_rules
    CLINICAL_RULES_AVAILABLE = True
except ImportError:
    CLINICAL_RULES_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(page_title=t("page_title"), layout="wide")

# --- UI STYLING (Hiding Sidebar, Tab Styling) ---
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #FF6B6B; color: white; }
    [data-testid="stSidebar"] { display: none; }
    .stCamera { transform: scaleX(1); }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LANGUAGE SELECTOR (top-right compact row)
# ─────────────────────────────────────────────────────────────
_lang_spacer, _lang_col = st.columns([8, 1])
with _lang_col:
    language_selector()

# ─────────────────────────────────────────────────────────────
# MAIN PAGE: Patient Profile (Always Visible)
# ─────────────────────────────────────────────────────────────
st.title(t("app_title"))

st.header(t("patient_profile_header"))
st.caption(t("patient_profile_caption"))

# Patient identity row
col_id1, col_id2, col_id3 = st.columns([2, 2, 2])
first_name = col_id1.text_input(t("first_name_label"), value="", placeholder=t("first_name_placeholder"))
last_name = col_id2.text_input(t("last_name_label"), value="", placeholder=t("last_name_placeholder"))
id_number = col_id3.text_input(t("patient_id_label"), value="", placeholder=t("patient_id_placeholder"))

# Clinical parameters row
col_a1, col_a2, col_a3 = st.columns(3)

age = col_a1.number_input(t("age_label"), min_value=1, max_value=120, value=55)
sex = col_a2.selectbox(t("sex_label"), ["M", "F"])
k_level = col_a3.slider(
    t("potassium_label"),
    2.0, 7.0, 4.0, step=0.1,
    help=t("potassium_help")
)

st.markdown(t("logic_inverters"))
col_b1, col_b2, col_b3 = st.columns(3)

pacemaker = col_b1.toggle(t("pacemaker_label"), value=False)
athlete = col_b2.toggle(t("athlete_label"), value=False)
pregnant = col_b3.toggle(
    t("pregnancy_label"),
    value=False,
    disabled=(sex == "M")
)

# Vertical Assignment as requested
patient_profile = {
    "first_name": first_name,
    "last_name": last_name,
    "id_number": id_number,
    "age": age,
    "sex": sex,
    "has_pacemaker": pacemaker,
    "is_athlete": athlete,
    "is_pregnant": pregnant and sex == "F",
    "k_level": k_level,
}

# Save / Load patient from DB
if DB_AVAILABLE:
    col_save, col_load = st.columns(2)
    with col_save:
        if st.button(t("save_patient_btn"), disabled=not (first_name and last_name)):
            pid = save_patient(
                first_name, last_name, id_number, age, sex,
                pacemaker, athlete, pregnant and sex == "F", k_level,
            )
            st.session_state["current_patient_id"] = pid
            st.success(t("saved_msg", name=f"{first_name} {last_name}", pid=pid))
    with col_load:
        patients = list_patients()
        if patients:
            options = {
                f"{p['first_name']} {p['last_name']} ({p['id_number'] or p['patient_id']})": p
                for p in patients
            }
            selected = st.selectbox(t("load_patient_label"), [""] + list(options.keys()))
            if selected and selected in options:
                p = options[selected]
                st.session_state["current_patient_id"] = p["patient_id"]
                st.caption(t("loaded_msg",
                    first=p['first_name'], last=p['last_name'],
                    age=p['age'], sex=p['sex'],
                ))

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# 12-LEAD DISPLAY
# ─────────────────────────────────────────────────────────────

# Standard clinical 4x3 layout
TWELVE_LEAD_GRID = [
    ["I",   "AVR", "V1", "V4"],
    ["II",  "AVL", "V2", "V5"],
    ["III", "AVF", "V3", "V6"],
]
LEAD_NAMES_12 = ["I", "II", "III", "AVR", "AVL", "AVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"]


def render_12_lead(signals_12, fs, lead_names, duration_s=2.5):
    """
    Render the standard 12-lead ECG in a 4-col x 3-row grid
    plus a full-length Lead II rhythm strip at the bottom.

    signals_12: (N, 12) array
    lead_names: list of 12 lead name strings from the record
    duration_s: seconds to show per cell (standard = 2.5s)
    """
    name_to_idx = {name: i for i, name in enumerate(lead_names)}

    samples_per_cell = int(fs * duration_s)
    n_total = signals_12.shape[0]

    fig, axes = plt.subplots(
        4, 4,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [1, 1, 1, 1]},
    )
    fig.patch.set_facecolor("#071312")

    # ── Rows 0-2: 4x3 grid, each cell shows a 2.5s segment ──
    for row_idx, row_leads in enumerate(TWELVE_LEAD_GRID):
        # Each row shows a different time segment (0-2.5s, 2.5-5s, 5-7.5s offset)
        # but standard practice: each column is the SAME time window
        # Column 0: 0–2.5s, Column 1: 2.5–5s, Column 2: 5–7.5s, Column 3: 7.5–10s
        for col_idx, lead_name in enumerate(row_leads):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("#071312")

            idx = name_to_idx.get(lead_name)
            if idx is None:
                ax.set_visible(False)
                continue

            t_start = col_idx * samples_per_cell
            t_end = min(t_start + samples_per_cell, n_total)

            if t_start >= n_total:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", color="#5A8A85", fontsize=9)
            else:
                seg = signals_12[t_start:t_end, idx]
                t = np.arange(len(seg)) / fs
                ax.plot(t, seg, color="#00E5B0", linewidth=0.7, alpha=0.9)

            ax.set_title(lead_name, color="#00E5B0", fontsize=9, pad=2, loc="left")
            ax.tick_params(colors="#3D6662", labelsize=5)
            ax.grid(True, linestyle="--", alpha=0.12, color="#00E5B0")
            for spine in ax.spines.values():
                spine.set_edgecolor("#1E3533")
            ax.set_xticks([])
            ax.set_yticks([])

    # ── Row 3: Full Lead II rhythm strip ──
    for col_idx in range(4):
        axes[3][col_idx].set_visible(False)

    # Merge bottom row into one axis
    gs = axes[3][0].get_gridspec()
    for ax in axes[3]:
        ax.remove()
    ax_rhythm = fig.add_subplot(gs[3, :])
    ax_rhythm.set_facecolor("#071312")

    ii_idx = name_to_idx.get("II", 1)
    max_samples = min(n_total, int(fs * 10))
    rhythm_sig = signals_12[:max_samples, ii_idx]
    t_rhythm = np.arange(len(rhythm_sig)) / fs

    ax_rhythm.plot(t_rhythm, rhythm_sig, color="#00E5B0", linewidth=0.7, alpha=0.9)
    ax_rhythm.set_title("II (rhythm strip)", color="#00E5B0", fontsize=9, pad=2, loc="left")
    ax_rhythm.set_xlabel("Time (s)", color="#5A8A85", fontsize=8)
    ax_rhythm.tick_params(colors="#3D6662", labelsize=6)
    ax_rhythm.grid(True, linestyle="--", alpha=0.12, color="#00E5B0")
    for spine in ax_rhythm.spines.values():
        spine.set_edgecolor("#1E3533")

    fig.tight_layout(h_pad=0.3)
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# RESTORED CLINICAL HELPERS
# ─────────────────────────────────────────────────────────────

def render_urgency_banner(urgency_level):
    cfg = URGENCY_CONFIG.get(urgency_level, URGENCY_CONFIG["NORMAL"])
    st.markdown(f"""
        <div style="background-color:{cfg['color']}; padding:15px; border-radius:10px; text-align:center; margin-bottom:20px;">
            <h2 style="color:white; margin:0;">{cfg['label']}</h2>
        </div>
    """, unsafe_allow_html=True)

def analyze_st_segment(signal, fs=500):
    try:
        signal = np.nan_to_num(signal)
        signal_clean = signal - np.mean(signal)
        std_val = np.std(signal_clean)
        
        if std_val == 0:
            return None
        
        threshold = np.mean(signal_clean) + 2.5 * std_val
        peaks = np.where(signal_clean > threshold)[0]
        
        real_peaks = []
        if len(peaks) > 0:
            real_peaks.append(peaks[0])
            for i in range(1, len(peaks)):
                if peaks[i] - peaks[i-1] > fs//2: 
                    real_peaks.append(peaks[i])

        if not real_peaks:
            return None

        elevations = []
        for peak in real_peaks[:3]:
            # Vertical assignment
            b_start = max(0, peak - 50)
            b_end = max(0, peak - 25)
            baseline = np.mean(signal_clean[b_start:b_end])
            
            j_idx = peak + 30
            if j_idx < len(signal_clean):
                elevations.append(signal_clean[j_idx] - baseline)

        if not elevations:
            return None
            
        res = {
            "mm_elev": np.nanmean(elevations) * 10,
            "peak_idx": real_peaks[0],
            "j_idx": real_peaks[0] + 30,
            "baseline": np.mean(signal_clean[max(0, real_peaks[0]-50):max(0, real_peaks[0]-25)])
        }
        return res
    except:
        return None

def run_full_analysis(signal, fs, p):
    if not INTERVAL_ENGINE_AVAILABLE:
        st.warning(t("clinical_engine_missing"))
        return

    if signal is None or len(signal) < 100:
        st.error(t("signal_too_short_analysis"))
        return

    try:
        intervals = calculate_intervals(signal, fs)
    except Exception as e:
        st.error(t("analysis_failure_error", error=str(e)))
        return

    if not isinstance(intervals, dict):
        st.error(t("analysis_invalid_error"))
        return

    if intervals.get("error"):
        st.error(t("analysis_error", error=intervals['error']))
        return

    context = apply_clinical_context(intervals, p)
    render_urgency_banner(context.get("urgency", "NORMAL"))

    cols = st.columns(4)
    def fmt(key):
        val = intervals.get(key, np.nan)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        try:
            return f"{int(val)}"
        except Exception:
            return str(val)

    cols[0].metric(t("heart_rate_metric"), t("heart_rate_unit", val=fmt('hr')))
    cols[1].metric(t("pr_interval_metric"), t("interval_unit", val=fmt('pr')))
    cols[2].metric(t("qrs_duration_metric"), t("interval_unit", val=fmt('qrs')))
    cols[3].metric(t("qtc_metric"), t("interval_unit", val=fmt('qtc')))

    if context.get("flags"):
        with st.expander(t("clinical_findings"), expanded=True):
            for flag in context["flags"]:
                sev = flag.get("severity", "INFO")
                text = f"**{flag.get('finding', '')}** — {flag.get('explanation', '')}"
                if sev == "CRITICAL":
                    st.error(text)
                elif sev == "WARNING":
                    st.warning(text)
                else:
                    st.info(text)

# ─────────────────────────────────────────────────────────────
# INPUT TABS
# ─────────────────────────────────────────────────────────────
tab_scan, tab_data = st.tabs([t("tab_scan"), t("tab_data")])

with tab_scan:
    _camera_opt = t("camera_option")
    _upload_opt = t("upload_option")
    scan_mode = st.radio(t("input_method_label"), [_camera_opt, _upload_opt], horizontal=True)

    img_bytes = None

    if scan_mode == _camera_opt:
        cam_buf = st.camera_input(t("scanner_label"))
        if cam_buf:
            img_bytes = cam_buf.getvalue()
    else:
        uploaded = st.file_uploader(
            t("upload_label"),
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
        )
        if uploaded:
            img_bytes = uploaded.getvalue()

    # ── ECG paper calibration settings ──────────────────────────────
    with st.expander(t("ecg_paper_settings"), expanded=False):
        _cal_cols = st.columns(2)
        _paper_speed = _cal_cols[0].selectbox(
            t("paper_speed_label"),
            options=[25, 50],
            index=0,
            format_func=lambda x: (
                t("paper_speed_standard", x=x) if x == 25
                else t("paper_speed_fast", x=x)
            ),
            help=t("paper_speed_help"),
        )
        _mm_per_mv = _cal_cols[1].selectbox(
            t("voltage_gain_label"),
            options=[5, 10, 20],
            index=1,
            format_func=lambda x: (
                t("voltage_standard", x=x) if x == 10
                else t("voltage_half", x=x) if x == 5
                else t("voltage_double", x=x)
            ),
            help=t("voltage_gain_help"),
        )

    if img_bytes is not None:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error(t("cannot_decode_image"))
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            h = gray.shape[0]
            y_start = h // 4
            y_end = 3 * h // 4
            mid = gray[y_start:y_end, :]

            # Fallback: brightness-based extraction (pixel units, uncalibrated)
            raw_sig = (255 - np.mean(mid, axis=0)) / 100.0
            smoothed = np.convolve(raw_sig, np.ones(5) / 5, mode="same")
            _calib_info = None

            # Calibrated OpenCV digitization pipeline
            try:
                from digitization_pipeline import extract_signal_from_image
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(img_bytes)
                    tmp_path = tmp.name
                _result = extract_signal_from_image(
                    tmp_path,
                    paper_speed=_paper_speed,
                    mm_per_mv=_mm_per_mv,
                )
                os.unlink(tmp_path)
                smoothed   = _result["signal"]
                _calib_info = _result
            except Exception:
                pass  # fall back to brightness-based extraction

            # Calibrated pipeline already resamples to 500 Hz.
            # Fallback path still needs resampling (pixel units → 5000 samples).
            if _calib_info is None:
                from scipy.signal import resample as _resample
                smoothed = _resample(smoothed, 500 * 10)

            # Show calibration status
            if _calib_info:
                px_x, px_y = _calib_info["px_per_mm"]
                if px_x and px_y:
                    st.caption(t("calib_detected",
                        speed=_paper_speed, gain=_mm_per_mv,
                        px_x=px_x, px_y=px_y,
                        quality=_calib_info['quality'],
                    ))
                else:
                    st.caption(t("calib_no_grid",
                        speed=_paper_speed, gain=_mm_per_mv,
                        quality=_calib_info['quality'],
                    ))
            else:
                st.caption(t("calib_fallback"))

            st.session_state.signal = smoothed
            st.session_state.fs = 500

            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=t("input_image_caption"), width='stretch')

            sig_duration = len(smoothed) / 500
            if sig_duration < 3:
                st.warning(t("signal_too_short_scan", dur=sig_duration, n=len(smoothed)))

with tab_data:
    if 'current_path' not in st.session_state:
        st.session_state.current_path = "C:/Users/osnat/Documents/Shlomi/EKG/ekg_datasets/ptbxl/records500/00000"

    path_in = st.text_input(t("folder_label"), st.session_state.current_path)
    clean_p = path_in.replace('"', '').strip()

    if os.path.exists(clean_p):
        files = [f.replace('.dat', '') for f in os.listdir(clean_p) if f.endswith('.dat')]
        if files:
            sel = st.selectbox(t("record_label"), sorted(files))
            if st.button(t("analyze_record_btn")):
                try:
                    rec = wfdb.rdrecord(os.path.join(clean_p, sel))
                except Exception as e:
                    st.error(t("load_record_error", error=e))
                    st.stop()

                raw_signal = getattr(rec, "p_signal", None)
                if raw_signal is None:
                    st.error(t("no_signal_error"))
                    st.stop()

                lead_names = getattr(rec, "sig_name", [])
                fs = getattr(rec, "fs", 500) or 500

                if raw_signal.ndim == 1:
                    signal = raw_signal
                else:
                    # Use lead II when available; fallback to lead I
                    if "II" in lead_names:
                        signal = raw_signal[:, lead_names.index("II")]
                    elif raw_signal.shape[1] > 1:
                        signal = raw_signal[:, 1]
                    else:
                        signal = raw_signal[:, 0]

                if len(signal) < 100:
                    st.error(t("record_too_short_error"))
                    st.stop()

                st.session_state.signal = signal
                st.session_state.fs = fs

                # Store full 12-lead data for display
                if raw_signal.ndim == 2 and raw_signal.shape[1] >= 12:
                    st.session_state.signals_12 = raw_signal
                    st.session_state.lead_names = lead_names
                else:
                    st.session_state.pop("signals_12", None)
                    st.session_state.pop("lead_names", None)

# ─────────────────────────────────────────────────────────────
# GLOBAL OUTPUT
# ─────────────────────────────────────────────────────────────
if 'signal' in st.session_state:
    st.divider()

    # ── 12-Lead Display (when available) ──
    if "signals_12" in st.session_state and "lead_names" in st.session_state:
        st.markdown(t("twelve_lead_header"))
        render_12_lead(
            st.session_state.signals_12,
            st.session_state.fs,
            st.session_state.lead_names,
        )

    # ── AI Classification (when 12-lead data available) ──
    if CLASSIFIER_AVAILABLE and "signals_12" in st.session_state:
        model_label = (
            t("model_multilabel") if _clf_type == "multilabel"
            else t("model_ensemble") if _clf_type == "ensemble"
            else t("model_hybrid") if (HYBRID_AVAILABLE and _clf_type == "cnn")
            else t("model_cnn") if _clf_type == "cnn"
            else t("model_sklearn")
        )
        st.markdown(t("ai_diagnosis_header", model=model_label))
        result = classify_ecg(
            _clf_model, st.session_state.signals_12, st.session_state.fs,
            lead_names=st.session_state.get("lead_names"),
            sex=patient_profile.get("sex", "M"),
            age=patient_profile.get("age", 50),
        )

        # Apply patient context to multi-label results
        if _clf_type == "multilabel":
            from multilabel_classifier import apply_patient_context
            result = apply_patient_context(result, patient_profile)

        # ── Multi-label display ──
        if _clf_type == "multilabel":
            urgency_colors = {3: "#FF4444", 2: "#FF8C00", 1: "#FFD700", 0: "#00C49F"}
            urgency_labels = {
                3: t("urgency_critical"),
                2: t("urgency_abnormal"),
                1: t("urgency_mild"),
                0: t("urgency_normal"),
            }
            conditions = result.get("conditions", [result.get("primary", "NORM")])
            per_class  = result.get("per_class", {})

            if not conditions:
                conditions = [result.get("primary", "NORM")]

            # ── Summary header ──────────────────────────────────────────────
            from multilabel_classifier import URGENCY as _URGENCY
            _n_critical = sum(1 for c in conditions if _URGENCY.get(c, 0) == 3)
            _n_abnormal = sum(1 for c in conditions if _URGENCY.get(c, 0) == 2)
            _n_mild     = sum(1 for c in conditions if _URGENCY.get(c, 0) == 1)
            _norm_only  = all(_URGENCY.get(c, 0) == 0 for c in conditions)

            if _norm_only:
                st.success(t("normal_ecg_msg"))
            else:
                _parts = []
                if _n_critical: _parts.append(t("findings_critical", n=_n_critical))
                if _n_abnormal: _parts.append(t("findings_abnormal", n=_n_abnormal))
                if _n_mild:     _parts.append(t("findings_mild", n=_n_mild))
                _summary_msg = t("findings_prefix", parts=", ".join(_parts))
                if _n_critical:
                    st.error(_summary_msg)
                elif _n_abnormal:
                    st.warning(_summary_msg)
                else:
                    st.info(_summary_msg)

            # Skip NORM card when other findings are also present (avoids noise)
            _display_conditions = (
                [c for c in conditions if c != "NORM"] if len(conditions) > 1 else conditions
            )

            for code in _display_conditions:
                info    = per_class.get(code, {})
                prob    = info.get("prob", result.get("confidence", 0))
                urg     = info.get("urgency", 0)
                desc    = info.get("description", code)
                action  = info.get("action", "")
                note    = info.get("note", "")
                color   = urgency_colors.get(urg, "#888")
                badge   = urgency_labels.get(urg, "")
                st.markdown(f"""
                    <div style="background:{color}; padding:12px 14px; border-radius:8px;
                                margin-bottom:8px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span style="color:white; font-weight:bold; font-size:1.05em;">
                                    {code}
                                </span>
                                <span style="color:rgba(255,255,255,0.85); margin-left:10px;">
                                    {desc}
                                </span>
                            </div>
                            <div style="text-align:right;">
                                <span style="color:white; font-size:0.85em;">{badge}</span>
                                <span style="color:white; font-weight:bold; margin-left:10px;">
                                    {prob:.0%}
                                </span>
                            </div>
                        </div>
                        {f'<div style="color:rgba(255,255,255,0.92); font-size:0.88em; margin-top:6px;">⚡ {action}</div>' if action else ""}
                        {f'<div style="color:rgba(255,255,255,0.70); font-size:0.82em; margin-top:2px; font-style:italic;">{note}</div>' if note else ""}
                    </div>
                """, unsafe_allow_html=True)

            with st.expander(t("all_cond_probs")):
                sorted_codes = sorted(per_class.items(), key=lambda x: -x[1].get("prob", 0))
                prob_cols = st.columns(4)
                for i, (code, info) in enumerate(sorted_codes):
                    detected = info.get("detected", False)
                    prob     = info.get("prob", 0)
                    label    = f"{'✓ ' if detected else ''}{code}"
                    prob_cols[i % 4].metric(label, f"{prob:.0%}")

        # ── Legacy single-label display (ensemble / CNN / sklearn) ──
        else:
            pred = result["prediction"]
            desc = result["description"]
            conf = result["confidence"]
            pred_colors = {
                "NORM": "#00C49F", "MI": "#FF4444", "STTC": "#FF8C00",
                "HYP": "#FFA500", "CD": "#FFD700",
            }
            color = pred_colors.get(pred, "#888")
            st.markdown(f"""
                <div style="background-color:{color}; padding:12px; border-radius:8px;
                            text-align:center; margin-bottom:12px;">
                    <h3 style="color:white; margin:0;">{pred} — {desc}</h3>
                    <p style="color:rgba(255,255,255,0.85); margin:4px 0 0 0;">
                        Confidence: {conf:.0%}
                    </p>
                </div>
            """, unsafe_allow_html=True)

            probs = result["probabilities"]
            prob_cols = st.columns(len(probs))
            for i, (cls, prob) in enumerate(sorted(probs.items(), key=lambda x: -x[1])):
                prob_cols[i].metric(cls, f"{prob:.0%}")

            # Hybrid classifier details (voltage criteria, adjustments)
            if result.get("voltage_criteria") or result.get("adjustment_applied"):
                with st.expander(t("hybrid_details")):
                    if result.get("adjustment_applied"):
                        st.info(t("hybrid_adjustment", text=result['adjustment_applied']))
                        st.caption(t("hybrid_cnn_raw",
                            pred=result.get('cnn_raw_prediction'),
                            conf=result.get('cnn_raw_confidence', 0),
                        ))
                    vc = result.get("voltage_criteria", {})
                    if vc:
                        vc_cols = st.columns(3)
                        sl = vc.get("sokolow_lyon", {})
                        cn = vc.get("cornell", {})
                        rv = vc.get("rvh_r_v1", {})
                        vc_cols[0].metric(
                            t("sokolow_lyon"),
                            f"{sl.get('value', 0):.2f} mV",
                            t("met_label") if sl.get("met") else t("not_met_label"),
                            delta_color="inverse" if sl.get("met") else "off",
                        )
                        vc_cols[1].metric(
                            t("cornell"),
                            f"{cn.get('value', 0):.2f} mV",
                            t("met_label") if cn.get("met") else t("not_met_label"),
                            delta_color="inverse" if cn.get("met") else "off",
                        )
                        vc_cols[2].metric(
                            t("rvh_r_v1"),
                            f"{rv.get('value', 0):.2f} mV",
                            t("met_label") if rv.get("met") else t("not_met_label"),
                            delta_color="inverse" if rv.get("met") else "off",
                        )

    # ── Clinical interval analysis (Lead II) ──
    run_full_analysis(st.session_state.signal, st.session_state.fs, patient_profile)

    # ── Multi-Lead ST Territory Analysis ──
    if ST_TERRITORY_AVAILABLE and "signals_12" in st.session_state and "lead_names" in st.session_state:
        st.markdown(t("st_territory_header"))

        st_result = analyze_st_territories(
            st.session_state.signals_12,
            st.session_state.fs,
            st.session_state.lead_names,
            patient_sex=patient_profile.get("sex", "M"),
        )

        # Summary banner
        urg = st_result["urgency"]
        urg_colors = {"EMERGENCY": "#FF4444", "URGENT": "#FF8C00", "NORMAL": "#00C49F"}
        banner_color = urg_colors.get(urg, "#00C49F")
        st.markdown(f"""
            <div style="background-color:{banner_color}; padding:10px; border-radius:8px;
                        text-align:center; margin-bottom:10px;">
                <p style="color:white; margin:0; font-weight:bold;">{st_result['summary']}</p>
            </div>
        """, unsafe_allow_html=True)

        # Territory cards
        for terr_name, finding in st_result["territories"].items():
            sev = finding["severity"]
            if sev == "CRITICAL":
                st.error(f"**{terr_name}** — {finding['interpretation']}")
            elif sev == "WARNING":
                st.warning(f"**{terr_name}** — {finding['interpretation']}")
            elif sev == "INFO":
                st.info(f"**{terr_name}** — {finding['interpretation']}")
            else:
                st.caption(f"**{terr_name}** — {finding['interpretation']}")

        # Per-lead ST deviation table
        with st.expander(t("per_lead_st")):
            lead_data = []
            for lead_name, lr in st_result["lead_results"].items():
                st_mv = lr["st_mv"]
                status = (
                    t("st_elevated") if st_mv >= 0.1
                    else t("st_depressed") if st_mv <= -0.05
                    else t("st_normal")
                )
                lead_data.append({
                    t("lead_col"): lead_name,
                    t("st_mv_col"): f"{st_mv:+.3f}",
                    t("status_col"): status,
                    t("beats_col"): lr["n_beats"],
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(lead_data), width='stretch', hide_index=True)

    # ── Clinical Rules (axis, T-waves, voltage, R-progression) ──
    if CLINICAL_RULES_AVAILABLE and "signals_12" in st.session_state and "lead_names" in st.session_state:
        rules_result = analyze_clinical_rules(
            st.session_state.signals_12,
            st.session_state.fs,
            st.session_state.lead_names,
            patient_profile,
        )

        if rules_result["findings"] or rules_result["axis"] is not None:
            st.markdown(t("clinical_rules_header"))

            # Axis display
            if rules_result["axis"] is not None:
                axis_col1, axis_col2 = st.columns(2)
                axis_col1.metric(t("cardiac_axis_metric"),
                    t("cardiac_axis_unit", val=rules_result['axis']))
                axis_col2.metric(t("axis_deviation_metric"), rules_result["axis_deviation"])

            # Findings
            for finding in rules_result["findings"]:
                sev = finding["severity"]
                text = f"**{finding['finding']}** — {finding['explanation']}"
                if sev == "CRITICAL":
                    st.error(text)
                elif sev == "WARNING":
                    st.warning(text)
                else:
                    st.info(text)

    # ── Save analysis to DB ──
    if DB_AVAILABLE and st.session_state.get("current_patient_id"):
        st.markdown("---")
        if st.button(t("save_analysis_btn")):
            pid = st.session_state["current_patient_id"]
            # Save EKG record
            ekg_id = save_ekg_record(
                pid,
                acquisition_source="dataset" if "signals_12" in st.session_state else "scan",
                lead_count=12 if "signals_12" in st.session_state else 1,
                ai_model_version=_clf_type or "none",
            )
            # Gather analysis data
            clf_result = None
            if CLASSIFIER_AVAILABLE and "signals_12" in st.session_state:
                clf_result = classify_ecg(_clf_model, st.session_state.signals_12, st.session_state.fs)

            st_summary_text = None
            if ST_TERRITORY_AVAILABLE and "signals_12" in st.session_state and "lead_names" in st.session_state:
                st_res = analyze_st_territories(
                    st.session_state.signals_12, st.session_state.fs,
                    st.session_state.lead_names, patient_sex=patient_profile.get("sex", "M"),
                )
                st_summary_text = st_res.get("summary")

            hr_val = None
            if INTERVAL_ENGINE_AVAILABLE:
                try:
                    ivl = calculate_intervals(st.session_state.signal, st.session_state.fs)
                    hr_val = ivl.get("hr")
                    pr_val = ivl.get("pr")
                    qrs_val = ivl.get("qrs")
                    qtc_val = ivl.get("qtc")
                except Exception:
                    pr_val = qrs_val = qtc_val = None
            else:
                pr_val = qrs_val = qtc_val = None

            save_analysis(
                ekg_id,
                classification=clf_result["prediction"] if clf_result else None,
                confidence=clf_result["confidence"] if clf_result else None,
                probabilities=clf_result.get("probabilities") if clf_result else None,
                heart_rate=hr_val,
                pr_interval=pr_val,
                qrs_duration=qrs_val,
                qtc=qtc_val,
                st_summary=st_summary_text,
                urgency=st_res.get("urgency") if st_summary_text else None,
            )
            st.success(t("analysis_saved_msg", eid=ekg_id))

    # ── Patient History ──
    if DB_AVAILABLE and st.session_state.get("current_patient_id"):
        with st.expander(t("patient_history")):
            records = get_patient_records(st.session_state["current_patient_id"])
            if records:
                import pandas as pd
                hist_data = []
                for r in records:
                    hist_data.append({
                        t("history_date_col"):   r.get("captured_at", ""),
                        t("history_source_col"): r.get("acquisition_source", ""),
                        t("history_class_col"):  r.get("classification", "-"),
                        t("history_conf_col"):   f"{r['confidence']:.0%}" if r.get("confidence") else "-",
                        t("history_hr_col"):     r.get("heart_rate", "-"),
                        t("history_urgency_col"):r.get("urgency", "-"),
                    })
                st.dataframe(pd.DataFrame(hist_data), width='stretch', hide_index=True)
            else:
                st.caption(t("no_records_caption"))

    # ── Export & Share ──
    st.markdown("---")
    st.markdown(t("export_header"))

    # Build shareable text summary
    _share_lines = []
    _pname = f"{patient_profile.get('first_name', '')} {patient_profile.get('last_name', '')}".strip()
    _pid = patient_profile.get("id_number", "")
    _share_lines.append(
        f"{t('share_ecg_report')} - {_pname or t('share_patient_fallback')}"
        + (f" (ID: {_pid})" if _pid else "")
    )
    _share_lines.append(f"{t('share_date_label')}: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    _share_lines.append(t("share_age_sex",
        age=patient_profile.get('age', 'N/A'),
        sex=patient_profile.get('sex', 'N/A'),
    ))
    _share_lines.append("")

    _clf_for_share = None
    if CLASSIFIER_AVAILABLE and "signals_12" in st.session_state:
        _clf_for_share = classify_ecg(_clf_model, st.session_state.signals_12, st.session_state.fs)
        pred = _clf_for_share["prediction"]
        desc = _clf_for_share["description"]
        conf = _clf_for_share["confidence"]
        _share_lines.append(f"{t('share_ai_diagnosis')}: {pred} - {desc} ({conf:.0%})")

    _intervals_for_share = None
    _flags_for_share = None
    if INTERVAL_ENGINE_AVAILABLE:
        try:
            _intervals_for_share = calculate_intervals(st.session_state.signal, st.session_state.fs)
            if not _intervals_for_share.get("error"):
                ctx = apply_clinical_context(_intervals_for_share, patient_profile)
                _flags_for_share = ctx.get("flags", [])
                # Merge clinical rules findings (axis, T-waves, posterior STEMI, etc.)
                if CLINICAL_RULES_AVAILABLE and "signals_12" in st.session_state and "lead_names" in st.session_state:
                    try:
                        _rules_for_share = analyze_clinical_rules(
                            st.session_state.signals_12, st.session_state.fs,
                            st.session_state.lead_names, patient_profile,
                        )
                        _flags_for_share = _flags_for_share + _rules_for_share.get("findings", [])
                    except Exception:
                        pass
                hr = _intervals_for_share.get("hr", "N/A")
                pr = _intervals_for_share.get("pr", "N/A")
                qrs = _intervals_for_share.get("qrs", "N/A")
                qtc = _intervals_for_share.get("qtc", "N/A")
                _share_lines.append(t("share_intervals", hr=hr, pr=pr, qrs=qrs, qtc=qtc))
        except Exception:
            pass

    _st_for_share = None
    if ST_TERRITORY_AVAILABLE and "signals_12" in st.session_state and "lead_names" in st.session_state:
        _st_for_share = analyze_st_territories(
            st.session_state.signals_12, st.session_state.fs,
            st.session_state.lead_names, patient_sex=patient_profile.get("sex", "M"),
        )
        _share_lines.append(f"{t('share_st_analysis')}: {_st_for_share.get('summary', 'N/A')}")

    if _flags_for_share:
        _share_lines.append("")
        _share_lines.append(t("share_findings_header"))
        for f in _flags_for_share:
            if isinstance(f, dict):
                # Convert clinical flag dict to readable string
                finding = f.get("finding", "")
                explanation = f.get("explanation", "")
                clean = f"{finding} {explanation}".strip()
            else:
                clean = str(f)
            clean = clean.encode("ascii", "ignore").decode().strip()
            if clean:
                _share_lines.append(f"- {clean}")

    _share_lines.append("")
    _share_lines.append(t("share_platform"))

    share_text = "\n".join(_share_lines)

    # Action buttons row
    col_pdf, col_email, col_whatsapp, col_copy = st.columns(4)

    # PDF Download
    if PDF_AVAILABLE:
        with col_pdf:
            pdf_bytes = generate_pdf_report(
                patient_profile=patient_profile,
                classification=_clf_for_share,
                intervals=_intervals_for_share,
                clinical_flags=_flags_for_share,
                st_result=_st_for_share,
                signals_12=st.session_state.get("signals_12"),
                fs=st.session_state.fs,
                lead_names=st.session_state.get("lead_names"),
            )
            patient_name_file = f"{patient_profile.get('first_name', '')}_{patient_profile.get('last_name', '')}".strip("_")
            filename = f"ECG_Report_{patient_name_file or 'patient'}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            if isinstance(pdf_bytes, bytearray):
                pdf_bytes = bytes(pdf_bytes)
            if not isinstance(pdf_bytes, (bytes, bytearray)):
                st.warning(t("pdf_warning"))
            else:
                st.download_button(
                    label=t("pdf_btn"),
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                )

    # Email share
    with col_email:
        import urllib.parse
        email_subject = urllib.parse.quote(
            f"{t('share_ecg_report')} - {_pname or t('share_patient_fallback')}"
        )
        email_body = urllib.parse.quote(share_text)
        mailto_link = f"mailto:?subject={email_subject}&body={email_body}"
        st.link_button(t("email_btn"), mailto_link)

    # WhatsApp share
    with col_whatsapp:
        wa_text = urllib.parse.quote(share_text)
        wa_link = f"https://wa.me/?text={wa_text}"
        st.link_button(t("whatsapp_btn"), wa_link)

    # Copy to clipboard
    with col_copy:
        import json as _json
        safe_js_str = _json.dumps(share_text)
        copy_label = t("copy_btn")
        copied_label = t("copied_btn")
        copy_js = f"""
            <button onclick="navigator.clipboard.writeText({safe_js_str})
                .then(() => {{ this.innerText={_json.dumps(copied_label)};
                               setTimeout(() => this.innerText={_json.dumps(copy_label)}, 2000); }})"
                style="background-color:#FF6B6B; color:white; border:none;
                       padding:8px 16px; border-radius:8px; cursor:pointer;
                       font-size:14px; width:100%;">
                {copy_label}
            </button>
        """
        st.markdown(copy_js, unsafe_allow_html=True)

    # Expandable preview
    with st.expander(t("preview_share")):
        st.code(share_text, language=None)

    # ── Legacy single-lead ST Analysis (fallback) ──
    if not (ST_TERRITORY_AVAILABLE and "signals_12" in st.session_state and "lead_names" in st.session_state):
        st.markdown(t("st_analysis_header"))
        st_res = analyze_st_segment(st.session_state.signal, st.session_state.fs)
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(st.session_state.signal[:1500], color='#FF6B6B', linewidth=1.5)

        if st_res:
            j_idx = st_res.get('j_idx')
            if j_idx is not None and 0 <= j_idx < len(st.session_state.signal):
                ax.axhline(st_res.get('baseline', 0), color='blue', linestyle='--', alpha=0.3)
                ax.scatter(j_idx, st.session_state.signal[j_idx], color='black', zorder=10)

        ax.set_facecolor('#f8f9fb')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        if st_res:
            elev = st_res.get('mm_elev')
            if elev is None or np.isnan(elev):
                st.info(t("st_no_estimate"))
            elif elev >= 1.0:
                st.error(t("st_omi_alert", val=elev))
            elif elev <= -1.0:
                st.warning(t("st_ischemia", val=elev))
            else:
                st.success(t("st_stable"))