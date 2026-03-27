"""
report_generator.py
===================
Generate clinical PDF reports for ECG analyses.

Includes: patient info, 12-lead ECG plot, AI classification,
clinical intervals, ST territory analysis, and clinical findings.

Usage (from app.py):
    from report_generator import generate_pdf_report
    pdf_bytes = generate_pdf_report(patient, classification, intervals, st_result, ecg_image_path)
"""

import io
import os
import tempfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF


# Standard 12-lead grid layout
TWELVE_LEAD_GRID = [
    ["I",   "AVR", "V1", "V4"],
    ["II",  "AVL", "V2", "V5"],
    ["III", "AVF", "V3", "V6"],
]


class ECGReport(FPDF):
    """Custom PDF with header/footer for ECG reports."""

    def __init__(self, patient_name="", patient_id=""):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.patient_name = patient_name
        self.patient_id = patient_id

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 8, "EKG Intelligence - Clinical Report", ln=True, align="C")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 4, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        self.set_text_color(0, 0, 0)
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, "FOR CLINICAL DECISION SUPPORT ONLY - Not a substitute for physician interpretation", align="C")
        self.ln(3)
        self.cell(0, 5, f"Page {self.page_no()}/{{nb}}", align="C")


def _render_ecg_image(signals_12, fs, lead_names):
    """Render 12-lead ECG to a temporary PNG file. Returns path."""
    name_to_idx = {name: i for i, name in enumerate(lead_names)}
    samples_per_cell = int(fs * 2.5)
    n_total = signals_12.shape[0]

    fig, axes = plt.subplots(4, 4, figsize=(12, 7),
                             gridspec_kw={"height_ratios": [1, 1, 1, 1]})
    fig.patch.set_facecolor("white")

    for row_idx, row_leads in enumerate(TWELVE_LEAD_GRID):
        for col_idx, lead_name in enumerate(row_leads):
            ax = axes[row_idx][col_idx]
            idx = name_to_idx.get(lead_name)
            if idx is None:
                ax.set_visible(False)
                continue

            t_start = col_idx * samples_per_cell
            t_end = min(t_start + samples_per_cell, n_total)

            if t_start >= n_total:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", color="gray", fontsize=8)
            else:
                seg = signals_12[t_start:t_end, idx]
                t = np.arange(len(seg)) / fs
                ax.plot(t, seg, color="#CC0000", linewidth=0.6)

            ax.set_title(lead_name, fontsize=8, pad=2, loc="left", fontweight="bold")
            ax.grid(True, linestyle="-", alpha=0.15, color="#CC0000")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)

    # Bottom row: Lead II rhythm strip
    for col_idx in range(4):
        axes[3][col_idx].set_visible(False)
    gs = axes[3][0].get_gridspec()
    for ax in axes[3]:
        ax.remove()
    ax_rhythm = fig.add_subplot(gs[3, :])

    ii_idx = name_to_idx.get("II", 1)
    max_samples = min(n_total, int(fs * 10))
    rhythm_sig = signals_12[:max_samples, ii_idx]
    t_rhythm = np.arange(len(rhythm_sig)) / fs
    ax_rhythm.plot(t_rhythm, rhythm_sig, color="#CC0000", linewidth=0.6)
    ax_rhythm.set_title("II (rhythm strip)", fontsize=8, pad=2, loc="left", fontweight="bold")
    ax_rhythm.set_xlabel("Time (s)", fontsize=7)
    ax_rhythm.grid(True, linestyle="-", alpha=0.15, color="#CC0000")
    ax_rhythm.tick_params(labelsize=6)

    fig.tight_layout(h_pad=0.3)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return tmp.name


def generate_pdf_report(patient_profile, classification=None, intervals=None,
                        clinical_flags=None, st_result=None,
                        signals_12=None, fs=500, lead_names=None):
    """
    Generate a full clinical PDF report.

    Args:
        patient_profile: dict with first_name, last_name, id_number, age, sex, etc.
        classification: dict from classify_ecg() or None
        intervals: dict from calculate_intervals() or None
        clinical_flags: list of flag strings or None
        st_result: dict from analyze_st_territories() or None
        signals_12: (N, 12) numpy array or None
        fs: sampling rate
        lead_names: list of 12 lead name strings

    Returns:
        bytes: PDF content
    """
    name = f"{patient_profile.get('first_name', '')} {patient_profile.get('last_name', '')}".strip()
    pid = patient_profile.get("id_number", "N/A")

    pdf = ECGReport(patient_name=name, patient_id=pid)
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # -- Patient Info Section --
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Patient Information", ln=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 9)
    col_w = 63
    pdf.cell(col_w, 5, f"Name: {name or 'N/A'}")
    pdf.cell(col_w, 5, f"Patient ID: {pid}")
    pdf.cell(col_w, 5, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)

    pdf.cell(col_w, 5, f"Age: {patient_profile.get('age', 'N/A')}")
    pdf.cell(col_w, 5, f"Sex: {patient_profile.get('sex', 'N/A')}")
    pdf.cell(col_w, 5, f"K+: {patient_profile.get('k_level', 'N/A')} mmol/L", ln=True)

    modifiers = []
    if patient_profile.get("has_pacemaker"):
        modifiers.append("Pacemaker/ICD")
    if patient_profile.get("is_athlete"):
        modifiers.append("Athlete")
    if patient_profile.get("is_pregnant"):
        modifiers.append("Pregnant")
    if modifiers:
        pdf.cell(0, 5, f"Modifiers: {', '.join(modifiers)}", ln=True)

    pdf.ln(4)

    # -- AI Classification --
    if classification:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "AI Classification", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)

        # ── Multi-label format (12-condition model) ──────────────────────
        if "conditions" in classification:
            per_class  = classification.get("per_class", {})
            conditions = classification.get("conditions", [])
            primary    = classification.get("primary", "")

            urgency_rgb = {3: (220, 50, 50), 2: (220, 120, 20), 1: (180, 160, 0), 0: (0, 180, 140)}
            urgency_lbl = {3: "Critical", 2: "Abnormal", 1: "Mild", 0: "Normal"}

            if not conditions or conditions == ["NORM"]:
                pdf.set_fill_color(0, 180, 140)
                pdf.set_text_color(255, 255, 255)
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 10, "  Normal ECG -- No significant findings detected.",
                         ln=True, fill=True, align="C")
                pdf.set_text_color(0, 0, 0)
            else:
                n_critical = sum(1 for c in conditions if per_class.get(c, {}).get("urgency", 0) == 3)
                n_abnormal = sum(1 for c in conditions if per_class.get(c, {}).get("urgency", 0) == 2)
                n_mild     = sum(1 for c in conditions if per_class.get(c, {}).get("urgency", 0) == 1)
                summary_parts = []
                if n_critical: summary_parts.append(f"{n_critical} Critical")
                if n_abnormal: summary_parts.append(f"{n_abnormal} Abnormal")
                if n_mild:     summary_parts.append(f"{n_mild} Mild")
                header_color = (220, 50, 50) if n_critical else ((220, 120, 20) if n_abnormal else (100, 149, 237))
                pdf.set_fill_color(*header_color)
                pdf.set_text_color(255, 255, 255)
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 10, f"  Findings: {', '.join(summary_parts)}",
                         ln=True, fill=True, align="C")
                pdf.set_text_color(0, 0, 0)
                pdf.ln(2)

                # Per-condition rows (detected conditions only, sorted by urgency)
                display = [c for c in conditions if c != "NORM"]
                if not display:
                    display = conditions
                col_w = [18, 52, 20, 98]
                pdf.set_font("Helvetica", "B", 7)
                pdf.set_fill_color(230, 230, 230)
                for w, hdr in zip(col_w, ["Code", "Description", "Prob", "Clinical Action"]):
                    pdf.cell(w, 5, hdr, border=1, fill=True, align="C")
                pdf.ln()
                pdf.set_font("Helvetica", "", 7)
                for code in display:
                    info  = per_class.get(code, {})
                    urg   = info.get("urgency", 0)
                    desc  = info.get("description", code)
                    prob  = info.get("prob", 0)
                    action = info.get("action", "")
                    r, g, b = urgency_rgb.get(urg, (136, 136, 136))
                    pdf.set_fill_color(r, g, b)
                    pdf.set_text_color(255, 255, 255)
                    pdf.cell(col_w[0], 5, code, border=1, fill=True, align="C")
                    pdf.set_fill_color(255, 255, 255)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(col_w[1], 5, desc[:30], border=1)
                    pdf.cell(col_w[2], 5, f"{prob:.0%}", border=1, align="C")
                    action_clean = action.encode("ascii", "ignore").decode()[:80]
                    pdf.cell(col_w[3], 5, action_clean, border=1)
                    pdf.ln()

                # Clinical notes section
                pdf.ln(2)
                for code in display:
                    info = per_class.get(code, {})
                    note = info.get("note", "").encode("ascii", "ignore").decode().strip()
                    if note:
                        pdf.set_font("Helvetica", "B", 7)
                        pdf.cell(18, 4, code, ln=False)
                        pdf.set_font("Helvetica", "I", 7)
                        pdf.cell(0, 4, note[:120], ln=True)

        # ── Legacy 5-class format ────────────────────────────────────────
        else:
            pred = classification.get("prediction", "N/A")
            desc = classification.get("description", "")
            conf = classification.get("confidence", 0)

            colors = {
                "NORM": (0, 196, 159), "MI": (255, 68, 68), "STTC": (255, 140, 0),
                "HYP": (255, 165, 0), "CD": (255, 215, 0),
            }
            r, g, b = colors.get(pred, (136, 136, 136))
            pdf.set_fill_color(r, g, b)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, f"  {pred} -- {desc}  (Confidence: {conf:.0%})",
                     ln=True, fill=True, align="C")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

            probs = classification.get("probabilities", {})
            if probs:
                pdf.set_font("Helvetica", "", 8)
                for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
                    bar_w = prob * 120
                    pdf.cell(15, 5, cls)
                    pdf.set_fill_color(100, 149, 237)
                    pdf.cell(bar_w, 5, "", fill=True)
                    pdf.cell(20, 5, f"  {prob:.0%}", ln=True)
                pdf.set_fill_color(255, 255, 255)

        pdf.ln(3)

    # -- Clinical Intervals --
    if intervals and not intervals.get("error"):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Clinical Intervals", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)

        pdf.set_font("Helvetica", "", 9)

        def fmt_val(key):
            val = intervals.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            return f"{int(val)}"

        col_w = 47
        pdf.cell(col_w, 6, f"Heart Rate: {fmt_val('hr')} bpm")
        pdf.cell(col_w, 6, f"PR Interval: {fmt_val('pr')} ms")
        pdf.cell(col_w, 6, f"QRS Duration: {fmt_val('qrs')} ms")
        pdf.cell(col_w, 6, f"QTc (Bazett): {fmt_val('qtc')} ms", ln=True)
        pdf.ln(2)

    # -- Clinical Findings --
    if clinical_flags:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Clinical Findings", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)

        pdf.set_font("Helvetica", "", 9)
        for flag in clinical_flags:
            if isinstance(flag, dict):
                flag = flag.get("finding") or flag.get("text") or str(flag)
            clean = str(flag).encode("ascii", "ignore").decode().strip()
            if clean:
                pdf.cell(0, 5, f"- {clean}", ln=True)
        pdf.ln(2)

    # -- ST Territory Analysis --
    if st_result:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "ST-Segment Territory Analysis", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)

        # Summary
        summary = st_result.get("summary", "")
        urgency = st_result.get("urgency", "NORMAL")
        urg_colors = {"EMERGENCY": (255, 68, 68), "URGENT": (255, 140, 0), "NORMAL": (0, 196, 159)}
        r, g, b = urg_colors.get(urgency, (0, 196, 159))
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 7, f"  {summary}", ln=True, fill=True, align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        # Per-territory findings
        pdf.set_font("Helvetica", "", 8)
        territories = st_result.get("territories", {})
        for terr_name, finding in territories.items():
            sev = finding.get("severity", "NORMAL")
            interp = finding.get("interpretation", "")
            sev_markers = {"CRITICAL": "[!!!]", "WARNING": "[!]", "INFO": "[i]", "NORMAL": ""}
            marker = sev_markers.get(sev, "")
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(0, 5, f"{marker} {terr_name}", ln=True)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(0, 5, f"    {interp}", ln=True)

        # Per-lead ST table
        pdf.ln(2)
        lead_results = st_result.get("lead_results", {})
        if lead_results:
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_fill_color(230, 230, 230)
            col_w = 32
            pdf.cell(col_w, 5, "Lead", border=1, fill=True, align="C")
            pdf.cell(col_w, 5, "ST (mV)", border=1, fill=True, align="C")
            pdf.cell(col_w, 5, "Status", border=1, fill=True, align="C")
            pdf.cell(col_w, 5, "Beats", border=1, fill=True, align="C")
            pdf.ln()

            pdf.set_font("Helvetica", "", 7)
            for lead_name, lr in lead_results.items():
                st_mv = lr.get("st_mv", 0)
                status = "Elevated" if st_mv >= 0.1 else ("Depressed" if st_mv <= -0.05 else "Normal")
                n_beats = lr.get("n_beats", 0)

                if status == "Elevated":
                    pdf.set_text_color(200, 0, 0)
                elif status == "Depressed":
                    pdf.set_text_color(200, 100, 0)
                else:
                    pdf.set_text_color(0, 0, 0)

                pdf.cell(col_w, 4, lead_name, border=1, align="C")
                pdf.cell(col_w, 4, f"{st_mv:+.3f}", border=1, align="C")
                pdf.cell(col_w, 4, status, border=1, align="C")
                pdf.cell(col_w, 4, str(n_beats), border=1, align="C")
                pdf.ln()

            pdf.set_text_color(0, 0, 0)

        pdf.ln(3)

    # -- 12-Lead ECG Image --
    if signals_12 is not None and lead_names is not None:
        # Check if we need a new page for the ECG
        if pdf.get_y() > 160:
            pdf.add_page()

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "12-Lead ECG", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)

        img_path = _render_ecg_image(signals_12, fs, lead_names)
        try:
            pdf.image(img_path, x=10, w=190)
        finally:
            os.unlink(img_path)

    # Output
    return pdf.output()
