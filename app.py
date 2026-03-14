import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import wfdb
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="EKG Intel POC", layout="wide")

# --- CUSTOM CSS ---
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
    </style>
""", unsafe_allow_html=True)

# --- CORE INTELLIGENCE LAYER ---
def analyze_st_segment(signal, fs=500):
    try:
        signal_clean = signal - np.mean(signal)
        threshold = np.mean(signal_clean) + 2.5 * np.std(signal_clean)
        peaks = np.where(signal_clean > threshold)[0]
        
        real_peaks = []
        if len(peaks) > 0:
            real_peaks.append(peaks[0])
            for i in range(1, len(peaks)):
                if peaks[i] - peaks[i-1] > fs//2: 
                    real_peaks.append(peaks[i])

        if not real_peaks: return None

        elevations = []
        for peak in real_peaks[:3]:
            b_start, b_end = max(0, peak - 50), max(0, peak - 25)
            baseline = np.mean(signal_clean[b_start:b_end])
            j_idx = peak + 30
            if j_idx < len(signal_clean):
                elevations.append(signal_clean[j_idx] - baseline)

        if not elevations: return None
        avg_elevation = np.mean(elevations)
        mm_elev = avg_elevation * 10 

        return {
            "mm_elev": mm_elev,
            "peak_idx": real_peaks[0],
            "j_idx": real_peaks[0] + 30,
            "baseline": np.mean(signal_clean[max(0, real_peaks[0]-50):max(0, real_peaks[0]-25)])
        }
    except: return None

# --- MAIN UI ---
st.title("🩺 EKG Intelligence")

# Navigation Tabs - Always visible at the top
tab_scan, tab_data = st.tabs(["📸 Mobile Scan", "📂 Dataset Explorer"])

# --- TAB 1: MOBILE SCAN (Default) ---
with tab_scan:
    st.header("AI Vision Scanner")
    img_buffer = st.camera_input("Align EKG strip horizontally")

    if img_buffer:
        with st.spinner('Digitizing...'):
            bytes_data = img_buffer.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            mid_section = gray[h//4 : 3*h//4, :]
            # Scale scanning data
            st.session_state.signal = (255 - np.mean(mid_section, axis=0)) / 150.0

# --- TAB 2: DATASET EXPLORER ---
with tab_data:
    st.header("PTB-XL Records")
    if 'current_path' not in st.session_state:
        st.session_state.current_path = "C:/Users/osnat/Documents/Shlomi/EKG/ekg_datasets/ptbxl/records500/00000"
    
    raw_path = st.text_input("Folder Path:", st.session_state.current_path)
    clean_path = raw_path.replace('"', '').replace("'", "").strip()
    st.session_state.current_path = clean_path

    if os.path.exists(clean_path):
        files = [f.replace('.dat', '') for f in os.listdir(clean_path) if f.endswith('.dat')]
        if files:
            selected_record = st.selectbox("Select Record", sorted(files))
            if st.button("Analyze Record"):
                full_path = os.path.join(clean_path, selected_record)
                record = wfdb.rdrecord(full_path)
                st.session_state.signal = record.p_signal[:, 1]
        else:
            st.warning("No .dat files found.")
    else:
        st.error("Invalid Path")

# --- GLOBAL RESULTS SECTION ---
if 'signal' in st.session_state:
    st.divider()
    results = analyze_st_segment(st.session_state.signal)
    
    # Waveform Plot
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(st.session_state.signal[:1500], color='#FF6B6B', linewidth=1.5)
    if results:
        ax.axhline(results['baseline'], color='blue', linestyle='--', alpha=0.3)
        ax.scatter(results['j_idx'], st.session_state.signal[results['j_idx']], color='black', zorder=10)
    ax.set_facecolor('#f8f9fb')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # AI Logic
    if results:
        elev = results['mm_elev']
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