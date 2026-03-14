import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import wfdb
import os
from pathlib import Path

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

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="EKG Intel POC", layout="wide")

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
# Helpers
# ─────────────────────────────────────────────────────────────

def render_urgency_banner(context: dict):
    urgency    = context.get("urgency", "NORMAL")
    flags      = context.get("flags", [])
    suppressed = context.get("suppressed", [])
    cfg        = URGENCY_CONFIG.get(urgency, URGENCY_CONFIG["NORMAL"])
    detail     = f"{len(flags)} finding(s)"
    if suppressed:
        detail += f" · {len(suppressed)} suppressed by clinical context"
    st.markdown(f"""
        <div style="background-color:{cfg['color']}; padding:15px; border-radius:10px;
                    text-align:center; margin-bottom:20px;">
            <h2 style="color:white; margin:0;">{cfg['emoji']} {cfg['label']}</h2>
            <p style="color:rgba(255,255,255,0.85); margin:4px 0 0 0; font-size:0.9rem;">
                {detail}
            </p>
        </div>
    """, unsafe_allow_html=True)


def analyze_st_segment(signal, fs=500):
    """Your version — NaN protection + nanmean, kept exactly."""
    try:
        signal = np.nan_to_num(signal)
        signal_clean = signal - np.mean(signal)
        std_val = np.std(signal_clean)
        if std_val == 0:
            return None

        threshold  = np.mean(signal_clean) + 2.5 * std_val
        peaks      = np.where(signal_clean > threshold)[0]
        real_peaks = []
        if len(peaks) > 0:
            real_peaks.append(peaks[0])
            for i in range(1, len(peaks)):
                if peaks[i] - peaks[i-1] > fs // 2:
                    real_peaks.append(peaks[i])

        if not real_peaks:
            return None

        elevations = []
        for peak in real_peaks[:3]:
            b_start  = max(0, peak - 50)
            b_end    = max(0, peak - 25)
            baseline = np.mean(signal_clean[b_start:b_end])
            j_idx    = peak + 30
            if j_idx < len(signal_clean):
                elevations.append(signal_clean[j_idx] - baseline)

        if not elevations:
            return None

        return {
            "mm_elev":  np.nanmean(elevations) * 10,
            "peak_idx": real_peaks[0],
            "j_idx":    real_peaks[0] + 30,
            "baseline": np.mean(signal_clean[max(0, real_peaks[0]-50):max(0, real_peaks[0]-25)])
        }
    except Exception:
        return None


def render_waveform_dark(signal, fs, r_peaks=None, title="Lead II"):
    """Dark waveform with R-peak markers — restored."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor("#071312")
    ax.set_facecolor("#071312")

    max_s = min(len(signal), fs * 10)
    t     = np.arange(max_s) / fs
    ax.plot(t, signal[:max_s], color="#00E5B0", linewidth=0.9, alpha=0.95)

    if r_peaks:
        valid_r = [r for r in r_peaks if r < max_s]
        ax.scatter(
            [r / fs for r in valid_r],
            signal[valid_r],
            color="#FF6B6B", s=30, zorder=5, label="R-peaks"
        )
        ax.legend(fontsize=8, facecolor="#0D1F1E", labelcolor="#E0F7F4")

    ax.set_xlabel("Time (s)", color="#5A8A85", fontsize=9)
    ax.set_ylabel("mV",       color="#5A8A85", fontsize=9)
    ax.set_title(title,       color="#00E5B0", fontsize=10, pad=6)
    ax.tick_params(colors="#3D6662", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E3533")
    ax.grid(True, linestyle="--", alpha=0.15, color="#00E5B0")
    st.pyplot(fig)
    plt.close(fig)


def render_interval_metrics(intervals: dict, patient_sex: str = "M"):
    """Interval tiles with normal/abnormal delta colours — restored."""
    if intervals.get("error"):
        st.error(f"Interval engine: {intervals['error']}")
        return

    def safe(key):
        val = intervals.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return val

    hr   = safe("hr")
    pr   = safe("pr")
    qrs  = safe("qrs")
    qtc  = safe("qtc")
    qual = safe("quality_score")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if hr is not None:
            ref = REFERENCE["hr"]
            d = "Normal" if ref["low"] <= hr <= ref["high"] else ("⬆ High" if hr > ref["high"] else "⬇ Low")
            st.metric("Heart Rate", f"{hr:.0f} bpm", d,
                      delta_color="normal" if "Normal" in d else "inverse")
        else:
            st.metric("Heart Rate", "N/A")

    with col2:
        if pr is not None:
            ref = REFERENCE["pr"]
            d = "Normal" if ref["low"] <= pr <= ref["high"] else ("⬆ Prolonged" if pr > ref["high"] else "⬇ Short")
            st.metric("PR Interval", f"{pr:.0f} ms", d,
                      delta_color="normal" if "Normal" in d else "inverse")
        else:
            st.metric("PR Interval", "N/A", "Unavailable")

    with col3:
        if qrs is not None:
            ref = REFERENCE["qrs"]
            d = "Normal" if ref["low"] <= qrs <= ref["high"] else ("⬆ Wide" if qrs > ref["high"] else "⬇ Narrow")
            st.metric("QRS Duration", f"{qrs:.0f} ms", d,
                      delta_color="normal" if "Normal" in d else "inverse")
        else:
            st.metric("QRS Duration", "N/A", "Unavailable")

    with col4:
        if qtc is not None:
            thresh = 460 if patient_sex == "F" else 450
            if qtc >= 500:
                d, dc = "🔴 Critical",  "inverse"
            elif qtc > thresh:
                d, dc = "⬆ Prolonged", "inverse"
            elif qtc < 350:
                d, dc = "⬇ Short",     "inverse"
            else:
                d, dc = "Normal",      "normal"
            st.metric("QTc (Bazett)", f"{qtc:.0f} ms", d, delta_color=dc)
        else:
            st.metric("QTc (Bazett)", "N/A", "Unavailable")

    with col5:
        if qual is not None:
            pct = int(qual * 100)
            d  = "Good" if qual >= 0.7 else ("Fair" if qual >= 0.4 else "Poor ⚠")
            dc = "normal" if qual >= 0.7 else "inverse"
            st.metric("Signal Quality", f"{pct}%", d, delta_color=dc)
        else:
            st.metric("Signal Quality", "N/A")

    for w in intervals.get("warnings", []):
        st.warning(f"⚠ {w}")


def render_clinical_findings(context: dict):
    """
    Fixed: was printing raw dicts.
    No expanders — findings shown inline as st.info/warning/error blocks.
    Suppressed findings shown as plain captions.
    """
    flags      = context.get("flags", [])
    suppressed = context.get("suppressed", [])

    if flags:
        st.markdown("**Clinical Findings**")
        for f in flags:
            emoji = SEVERITY_EMOJI.get(f["severity"], "•")
            label = f"{emoji} **{f['finding']}** — {f['explanation']}"
            if f["severity"] == "CRITICAL":
                st.error(label)
            elif f["severity"] == "WARNING":
                st.warning(label)
            else:
                st.info(label)
    else:
        st.success("✅ No acute findings in measured intervals.")

    if suppressed:
        st.markdown("**Suppressed by clinical context:**")
        for s in suppressed:
            st.caption(f"⚪ {s['finding']} — {s['suppressed_reason']}")


def run_full_analysis(signal, fs, p):
    """Your signature kept. All rendering fixed and restored."""
    if not INTERVAL_ENGINE_AVAILABLE:
        st.warning(
            "⚠ `interval_calculator.py` not found. "
            "Run `pip install neurokit2 scipy` and place the file in this folder."
        )
        return

    with st.spinner("Calculating clinical intervals..."):
        intervals = calculate_intervals(signal, fs)
        context   = apply_clinical_context(intervals, p)

    if intervals.get("error"):
        st.error(f"Analysis Error: {intervals['error']}")
        return

    # 1. Urgency banner
    render_urgency_banner(context)

    # 2. Interval metric tiles with delta colours
    st.markdown("#### Measured Intervals")
    render_interval_metrics(intervals, patient_sex=p.get("sex", "M"))

    # 3. Patient profile summary line
    inverters = [
        label for label, active in [
            ("Pacemaker", p.get("has_pacemaker")),
            ("Athlete",   p.get("is_athlete")),
            ("Pregnant",  p.get("is_pregnant")),
        ] if active
    ]
    inv_str = ", ".join(inverters) if inverters else "No logic inverters"
    st.caption(f"**{p['age']}y {p['sex']}** | K⁺: {p['k_level']} mmol/L | {inv_str}")

    # 4. Dark waveform with R-peaks
    st.markdown("#### ECG Waveform")
    r_peaks = intervals.get("r_peaks", [])
    hr_val  = intervals.get("hr")
    title   = (f"Lead II — {context.get('urgency','?')} | HR: {hr_val:.0f} bpm"
               if hr_val else "Lead II")
    render_waveform_dark(signal, fs, r_peaks=r_peaks, title=title)

    # 5. Clinical findings — inline, no expanders
    st.markdown("#### Clinical Assessment")
    render_clinical_findings(context)


# ─────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────

st.title("🩺 EKG Intelligence Platform")

if not INTERVAL_ENGINE_AVAILABLE:
    st.warning(
        "⚠ Clinical engine not available — ST-segment analysis still works. "
        "To enable full intervals: `pip install neurokit2 scipy`"
    )

# ── 1. Patient Profile — inline grid, no expander ─────────────
st.markdown("### 👤 Patient & Clinical Context")
c1, c2, c3 = st.columns(3)
age     = c1.number_input("Age",             1,    120,  45)
sex     = c2.selectbox("Sex",                ["M", "F"])
k_level = c3.number_input("Potassium (K+)",  2.0,  9.0,  4.0, step=0.1)

c4, c5, c6 = st.columns(3)
has_pacemaker = c4.checkbox("Pacemaker / ICD")
is_athlete    = c5.checkbox("Pro Athlete")
is_pregnant   = c6.checkbox("Pregnant", disabled=(sex == "M"))

patient_data = {
    "age":          age,
    "sex":          sex,
    "k_level":      k_level,
    "has_pacemaker": has_pacemaker,
    "is_athlete":    is_athlete,
    "is_pregnant":   is_pregnant and sex == "F",
}

st.markdown("---")

# ── 2. Input tabs ──────────────────────────────────────────────
tab_scan, tab_data = st.tabs(["📸 Mobile Scan", "📂 Dataset Explorer"])

with tab_scan:
    st.header("AI Vision Scanner")
    st.caption("📱 Rotate phone to landscape for best results.")
    img_buffer = st.camera_input("Capture EKG Strip")

    if img_buffer:
        with st.spinner("Digitizing..."):
            bytes_data = img_buffer.getvalue()
            img  = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h    = gray.shape[0]
            mid_section = gray[h//4: 3*h//4, :]

            # Your scaling + smoothing
            raw_signal = (255 - np.mean(mid_section, axis=0)) / 100.0
            st.session_state.signal = np.convolve(raw_signal, np.ones(5)/5, mode="same")
            st.session_state.fs     = 500

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption="Captured scan", use_column_width=True)
        st.success("✅ Signal extracted — results below")

        # Upgrade to OpenCV pipeline if available
        try:
            from digitization_pipeline import extract_signal_from_image
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(bytes_data)
                tmp_path = tmp.name
            better_signal = extract_signal_from_image(tmp_path)
            os.unlink(tmp_path)
            st.session_state.signal = better_signal
            st.info("✓ Using OpenCV digitization pipeline (higher fidelity)")
        except Exception:
            pass  # brightness fallback already stored

with tab_data:
    st.header("PTB-XL Records")

    if "current_path" not in st.session_state:
        st.session_state.current_path = (
            "C:/Users/osnat/Documents/Shlomi/EKG/ekg_datasets/ptbxl/records500/00000"
        )

    path_in    = st.text_input("Dataset Folder:", st.session_state.current_path)
    clean_path = path_in.replace('"', "").replace("'", "").strip()
    st.session_state.current_path = clean_path

    if os.path.exists(clean_path):
        files = sorted([
            f.replace(".dat", "")
            for f in os.listdir(clean_path)
            if f.endswith(".dat")
        ])
        if files:
            selected = st.selectbox("Select Record", files)

            # Show PTB-XL metadata if labeled CSV is found
            meta_path = Path(clean_path).parents[2] / "ptbxl_labeled.csv"
            if meta_path.exists():
                try:
                    import pandas as pd
                    df_meta = pd.read_csv(meta_path, index_col="ecg_id")
                    stem    = selected.replace("_hr", "").replace("_lr", "")
                    matches = df_meta[df_meta["filename_hr"].str.contains(stem, na=False)]
                    if not matches.empty:
                        row = matches.iloc[0]
                        m1, m2, m3 = st.columns(3)
                        m1.info(f"**Diagnosis:** {row.get('primary_diagnosis', 'Unknown')}")
                        m2.info(f"**Age:** {int(row.get('age', 0))}  |  "
                                f"**Sex:** {'M' if row.get('sex', 0) == 0 else 'F'}")
                        m3.info(f"**Fold:** {int(row.get('strat_fold', 0))}")
                except Exception:
                    pass

            if st.button("Load & Analyze"):
                record_path = os.path.join(clean_path, selected)
                rec = wfdb.rdrecord(record_path)
                st.session_state.signal = rec.p_signal[:, 1]
                st.session_state.fs     = rec.fs
        else:
            st.warning("No .dat files found.")
    else:
        st.error("Invalid Path")


# ─────────────────────────────────────────────────────────────
# Results — shown after any signal is loaded
# ─────────────────────────────────────────────────────────────

if "signal" in st.session_state:
    st.divider()

    signal = st.session_state.signal
    fs     = st.session_state.get("fs", 500)

    # ── Full clinical analysis (intervals + context + dark waveform) ──
    run_full_analysis(signal, fs, patient_data)

    # ── Your original ST waveform + OMI decision (always shown) ──
    st.markdown("#### ST-Segment Analysis")
    results = analyze_st_segment(signal, fs)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(signal[:1500], color="#FF6B6B", linewidth=1.5)
    if results:
        ax.axhline(results["baseline"], color="blue",
                   linestyle="--", alpha=0.3, label="Baseline")
        j = results["j_idx"]
        if j < len(signal):
            ax.scatter(j, signal[j], color="black", zorder=10, label="J-point")
        ax.legend(fontsize=8)
    ax.set_facecolor("#f8f9fb")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    if results:
        elev = results["mm_elev"]
        c1, c2 = st.columns(2)
        c1.metric("ST-Elevation", f"{elev:.2f} mm")
        if elev >= 1.0:
            c2.error("🚨 OMI ALERT")
        elif elev <= -1.0:
            c2.warning("⚠️ DEPRESSION")
        else:
            c2.success("✅ STABLE")
    else:
        st.info("Point camera at a clear EKG trace to begin analysis.")
