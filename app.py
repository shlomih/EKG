import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import wfdb
import os
from pathlib import Path

# Graceful import — app still runs without interval_calculator,
# showing a warning instead of crashing
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

# ── CSS: your original tab styling + sidebar width ────────────
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #FF6B6B; color: white; }
    [data-testid="stSidebar"] { min-width: 250px; }
    .stCamera { transform: scaleX(1); }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Sidebar — Patient Profile
# New addition: drives clinical context engine
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🩺 EKG Intelligence")
    st.markdown("---")
    st.header("Patient Profile")
    st.caption("Context changes interpretation — not just display labels.")

    age = st.number_input("Age", min_value=1, max_value=120, value=55)
    sex = st.selectbox("Biological Sex", ["M", "F"])

    st.markdown("**Logic Inverters**")
    pacemaker = st.toggle("Pacemaker / ICD Present", value=False)
    athlete   = st.toggle("Athlete Status",           value=False)
    pregnant  = st.toggle("Pregnancy",                value=False,
                           disabled=(sex == "M"))

    st.markdown("**Electrolytes**")
    k_level = st.slider(
        "Potassium K⁺ (mmol/L)", 2.0, 7.0, 4.0, step=0.1,
        help="Normal: 3.5–5.0 mmol/L"
    )

# Patient dict passed to context engine
patient_profile = {
    "age":           age,
    "sex":           sex,
    "has_pacemaker": pacemaker,
    "is_athlete":    athlete,
    "is_pregnant":   pregnant and sex == "F",
    "k_level":       k_level,
}


# ─────────────────────────────────────────────────────────────
# Your original ST-segment analysis — kept exactly as written
# ─────────────────────────────────────────────────────────────

def analyze_st_segment(signal, fs=500):
    try:
        signal_clean = signal - np.mean(signal)
        threshold = np.mean(signal_clean) + 2.5 * np.std(signal_clean)
        peaks = np.where(signal_clean > threshold)[0]

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

        avg_elevation = np.mean(elevations)
        mm_elev = avg_elevation * 10  # convert to mm

        return {
            "mm_elev":  mm_elev,
            "peak_idx": real_peaks[0],
            "j_idx":    real_peaks[0] + 30,
            "baseline": np.mean(signal_clean[max(0, real_peaks[0]-50):max(0, real_peaks[0]-25)])
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# New: full interval analysis pipeline
# ─────────────────────────────────────────────────────────────

def run_full_analysis(signal: np.ndarray, sampling_rate: int, source_label: str = ""):
    """Run NeuroKit2 interval measurement + clinical context, store in session_state."""
    if not INTERVAL_ENGINE_AVAILABLE:
        st.session_state.intervals    = None
        st.session_state.context      = None
        st.session_state.source_label = source_label
        return

    with st.spinner("Calculating clinical intervals..."):
        intervals = calculate_intervals(signal, sampling_rate=sampling_rate)
        context   = apply_clinical_context(intervals, patient_profile)

    st.session_state.intervals    = intervals
    st.session_state.context      = context
    st.session_state.source_label = source_label


# ─────────────────────────────────────────────────────────────
# New render helpers
# ─────────────────────────────────────────────────────────────

def render_urgency_banner(context: dict):
    urgency = context.get("urgency", "NORMAL")
    cfg     = URGENCY_CONFIG[urgency]
    flags   = context.get("flags", [])
    suppressed = context.get("suppressed", [])
    colors = {
        "EMERGENCY": ("#FF4444", "#2A0000"),
        "URGENT":    ("#FF8C00", "#2A1500"),
        "ROUTINE":   ("#00C49F", "#001A14"),
        "NORMAL":    ("#00C49F", "#001A14"),
    }
    border, bg = colors[urgency]
    st.markdown(
        f"""<div style="border:2px solid {border}; background:{bg};
            border-radius:10px; padding:0.8rem 1.2rem; margin-bottom:1rem;">
            <span style="color:{border}; font-size:1.1rem; font-weight:700;">
            {cfg['emoji']} {cfg['label']}
            </span>
            <span style="color:#aaa; font-size:0.85rem; margin-left:1rem;">
            {len(flags)} finding(s) · {len(suppressed)} suppressed by clinical context
            </span>
        </div>""",
        unsafe_allow_html=True
    )


def render_interval_metrics(intervals: dict):
    if intervals.get("error"):
        st.error(f"Interval engine: {intervals['error']}")
        return

    hr   = intervals.get("hr")
    pr   = intervals.get("pr")
    qrs  = intervals.get("qrs")
    qtc  = intervals.get("qtc")
    qual = intervals.get("quality_score")

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
            thresh = 460 if patient_profile["sex"] == "F" else 450
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
    render_urgency_banner(context)
    flags      = context.get("flags", [])
    suppressed = context.get("suppressed", [])

    if flags:
        st.markdown("**Clinical Findings**")
        for f in flags:
            emoji = SEVERITY_EMOJI.get(f["severity"], "•")
            with st.expander(f"{emoji} {f['finding']}",
                             expanded=(f["severity"] == "CRITICAL")):
                st.write(f["explanation"])
    else:
        st.success("✅ No acute findings in measured intervals.")

    if suppressed:
        with st.expander(f"⚪ {len(suppressed)} finding(s) suppressed by clinical context"):
            for s in suppressed:
                st.markdown(f"**{s['finding']}** — *{s['suppressed_reason']}*")


def render_waveform_dark(signal, sampling_rate, r_peaks=None, title="Lead II"):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor("#071312")
    ax.set_facecolor("#071312")

    max_s = min(len(signal), sampling_rate * 10)
    t = np.arange(max_s) / sampling_rate
    ax.plot(t, signal[:max_s], color="#00E5B0", linewidth=0.9, alpha=0.95)

    if r_peaks:
        valid_r = [r for r in r_peaks if r < max_s]
        ax.scatter(
            [r / sampling_rate for r in valid_r],
            signal[valid_r],
            color="#FF6B6B", s=30, zorder=5, label="R-peaks"
        )
        ax.legend(fontsize=8, facecolor="#0D1F1E", labelcolor="#E0F7F4")

    ax.set_xlabel("Time (s)", color="#5A8A85", fontsize=9)
    ax.set_ylabel("mV",       color="#5A8A85", fontsize=9)
    ax.set_title(title, color="#00E5B0", fontsize=10, pad=6)
    ax.tick_params(colors="#3D6662", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E3533")
    ax.grid(True, linestyle="--", alpha=0.15, color="#00E5B0")
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Main UI — your original tab layout, preserved exactly
# ─────────────────────────────────────────────────────────────

st.title("🩺 EKG Intelligence")

if not INTERVAL_ENGINE_AVAILABLE:
    st.warning(
        "⚠ `interval_calculator.py` not found — full interval engine disabled. "
        "ST-segment analysis still works. "
        "To enable: `pip install neurokit2 scipy` and place `interval_calculator.py` here."
    )

tab_scan, tab_data = st.tabs(["📸 Mobile Scan", "📂 Dataset Explorer"])


# ── TAB 1: MOBILE SCAN ───────────────────────────────────────
with tab_scan:
    st.header("AI Vision Scanner")
    st.caption("📱 Rotate phone to landscape for best results.")

    img_buffer = st.camera_input("Align EKG strip horizontally")

    if img_buffer:
        with st.spinner("Digitizing..."):
            bytes_data = img_buffer.getvalue()
            img  = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            mid_section = gray[h//4: 3*h//4, :]

            # Your original scaling — preserved
            raw_signal = (255 - np.mean(mid_section, axis=0)) / 150.0
            st.session_state.signal        = raw_signal
            st.session_state.sampling_rate = 100

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
            pass  # brightness fallback already stored above

        run_full_analysis(
            st.session_state.signal,
            st.session_state.sampling_rate,
            source_label="Mobile scan"
        )


# ── TAB 2: DATASET EXPLORER ──────────────────────────────────
with tab_data:
    st.header("PTB-XL Records")

    # Your original default path
    if "current_path" not in st.session_state:
        st.session_state.current_path = (
            "C:/Users/osnat/Documents/Shlomi/EKG/ekg_datasets/ptbxl/records500/00000"
        )

    raw_path   = st.text_input("Folder Path:", st.session_state.current_path)
    clean_path = raw_path.replace('"', "").replace("'", "").strip()
    st.session_state.current_path = clean_path

    if os.path.exists(clean_path):
        files = sorted([
            f.replace(".dat", "")
            for f in os.listdir(clean_path)
            if f.endswith(".dat")
        ])
        if files:
            selected_record = st.selectbox("Select Record", files)

            # Show PTB-XL metadata if labeled CSV is found
            meta_path = Path(clean_path).parents[2] / "ptbxl_labeled.csv"
            record_meta = None
            if meta_path.exists():
                try:
                    import pandas as pd
                    df_meta = pd.read_csv(meta_path, index_col="ecg_id")
                    stem    = selected_record.replace("_hr", "").replace("_lr", "")
                    matches = df_meta[df_meta["filename_hr"].str.contains(stem, na=False)]
                    if not matches.empty:
                        record_meta = matches.iloc[0]
                except Exception:
                    pass

            if record_meta is not None:
                c1, c2, c3 = st.columns(3)
                c1.info(f"**Diagnosis:** {record_meta.get('primary_diagnosis', 'Unknown')}")
                c2.info(f"**Age:** {int(record_meta.get('age', 0))}  |  "
                        f"**Sex:** {'M' if record_meta.get('sex', 0) == 0 else 'F'}")
                c3.info(f"**Fold:** {int(record_meta.get('strat_fold', 0))}")

            if st.button("Analyze Record"):
                full_path = os.path.join(clean_path, selected_record)
                record    = wfdb.rdrecord(full_path)
                # Your original: column index 1 (Lead II)
                st.session_state.signal        = record.p_signal[:, 1]
                st.session_state.sampling_rate = record.fs

                run_full_analysis(
                    st.session_state.signal,
                    record.fs,
                    source_label=f"PTB-XL: {selected_record} (Lead II)"
                )
        else:
            st.warning("No .dat files found.")
    else:
        st.error("Invalid Path")


# ─────────────────────────────────────────────────────────────
# Global results — your original structure + new panels on top
# ─────────────────────────────────────────────────────────────

if "signal" in st.session_state:
    st.divider()

    signal = st.session_state.signal
    fs     = st.session_state.get("sampling_rate", 500)
    source = st.session_state.get("source_label", "")

    if source:
        st.caption(f"Source: {source}")

    # ── NEW: Clinical interval panel (shown when engine available) ──
    if (INTERVAL_ENGINE_AVAILABLE
            and "intervals" in st.session_state
            and st.session_state.intervals):

        intervals = st.session_state.intervals
        context   = st.session_state.context

        render_clinical_findings(context)
        st.markdown("---")
        st.markdown("#### Measured Intervals")
        render_interval_metrics(intervals)
        st.markdown("---")

        r_peaks = intervals.get("r_peaks", [])
        render_waveform_dark(
            signal, fs,
            r_peaks=r_peaks,
            title=f"Lead II — {context.get('urgency', '?')} | "
                  f"HR: {intervals.get('hr', '?')} bpm"
        )

        with st.expander("Patient context applied"):
            p = patient_profile
            inverters = [
                label for label, active in [
                    ("Pacemaker/ICD", p["has_pacemaker"]),
                    ("Athlete",       p["is_athlete"]),
                    ("Pregnant",      p["is_pregnant"]),
                ] if active
            ]
            st.write(
                f"**{p['age']}y {p['sex']}** | K⁺: {p['k_level']} mmol/L"
                + (f" | **{', '.join(inverters)}**" if inverters
                   else " | No logic inverters active")
            )

        st.markdown("---")

    # ── YOUR ORIGINAL: ST waveform + OMI alert (always shown) ──
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
