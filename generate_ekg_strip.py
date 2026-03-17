"""
generate_ekg_strip.py
=====================
Renders PTB-XL records as paper-style EKG strip PNGs.
Supports both single-lead strips and full 12-lead clinical layouts.

Usage:
    # Default: 12-lead layout of the first record
    python generate_ekg_strip.py

    # Single-lead strip
    python generate_ekg_strip.py --single

    # Specific record
    python generate_ekg_strip.py --record ekg_datasets/ptbxl/records500/00000/00497_hr

    # Batch: generate 10 random 12-lead PNGs
    python generate_ekg_strip.py --batch 10

    # Custom output folder
    python generate_ekg_strip.py --out strips/
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import wfdb


# ── Constants matching standard EKG paper ────────────────────
PAPER_SPEED_MM_S = 25
GAIN_MM_MV = 10
SMALL_BOX_MM = 1
LARGE_BOX_MM = 5

GRID_MAJOR_COLOR = "#D9A0A0"
GRID_MINOR_COLOR = "#F0C8C8"
TRACE_COLOR = "#111111"
BG_COLOR = "#FFF5F5"

# Standard clinical 12-lead layout (4 columns x 3 rows)
# Each column shows a 2.5s window; total = 10s
TWELVE_LEAD_GRID = [
    ["I",   "AVR", "V1", "V4"],
    ["II",  "AVL", "V2", "V5"],
    ["III", "AVF", "V3", "V6"],
]


def _style_axis(ax, duration_s):
    """Apply EKG-paper grid styling to a single axis."""
    ax.set_facecolor(BG_COLOR)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.04))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(which="minor", color=GRID_MINOR_COLOR, linewidth=0.3, alpha=0.8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.grid(which="major", color=GRID_MAJOR_COLOR, linewidth=0.6, alpha=0.9)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_strip(signal, fs, lead_name="II", duration_s=10, dpi=150):
    """Render a single-lead strip as paper-style image."""
    n_samples = min(len(signal), int(fs * duration_s))
    signal = signal[:n_samples]
    t = np.arange(n_samples) / fs

    width_mm = duration_s * PAPER_SPEED_MM_S
    height_mm = 40

    width_in = width_mm / 25.4
    height_in = height_mm / 25.4

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    fig.patch.set_facecolor(BG_COLOR)

    _style_axis(ax, duration_s)
    ax.plot(t, signal, color=TRACE_COLOR, linewidth=0.8)
    ax.set_xlim(0, duration_s)

    sig_mean = np.mean(signal)
    ax.set_ylim(sig_mean - 2.0, sig_mean + 2.0)

    ax.text(0.01, 0.95, lead_name, transform=ax.transAxes,
            fontsize=10, fontweight="bold", verticalalignment="top",
            color="#333333")

    fig.tight_layout(pad=0.1)
    return fig


def render_12_lead(signals, fs, lead_names, dpi=150):
    """
    Render the standard 12-lead clinical layout as a paper-style image.

    Layout: 4 columns x 3 rows + full Lead II rhythm strip at bottom.
    Each column covers 2.5 seconds; rhythm strip covers full 10 seconds.

    signals: (N, 12) array
    lead_names: list of 12 lead name strings
    """
    name_to_idx = {name: i for i, name in enumerate(lead_names)}
    duration_s = 10.0
    col_duration = 2.5
    samples_per_col = int(fs * col_duration)
    n_total = signals.shape[0]

    fig, axes = plt.subplots(
        4, 4,
        figsize=(14, 10),
        dpi=dpi,
        gridspec_kw={"height_ratios": [1, 1, 1, 0.9]},
    )
    fig.patch.set_facecolor(BG_COLOR)

    # ── 3 rows x 4 columns ──
    for row_idx, row_leads in enumerate(TWELVE_LEAD_GRID):
        for col_idx, lead_name in enumerate(row_leads):
            ax = axes[row_idx][col_idx]

            idx = name_to_idx.get(lead_name)
            t_start = col_idx * samples_per_col
            t_end = min(t_start + samples_per_col, n_total)

            if idx is not None and t_start < n_total:
                seg = signals[t_start:t_end, idx]
                t = np.arange(len(seg)) / fs
                _style_axis(ax, col_duration)
                ax.plot(t, seg, color=TRACE_COLOR, linewidth=0.7)
                ax.set_xlim(0, col_duration)
                sig_mean = np.mean(seg)
                ax.set_ylim(sig_mean - 1.5, sig_mean + 1.5)
            else:
                ax.set_facecolor(BG_COLOR)
                ax.set_xticks([])
                ax.set_yticks([])

            ax.text(0.02, 0.95, lead_name, transform=ax.transAxes,
                    fontsize=8, fontweight="bold", verticalalignment="top",
                    color="#333333")

    # ── Bottom row: full Lead II rhythm strip ──
    for ax in axes[3]:
        ax.remove()

    gs = axes[0][0].get_gridspec()
    ax_rhythm = fig.add_subplot(gs[3, :])
    ax_rhythm.set_facecolor(BG_COLOR)

    ii_idx = name_to_idx.get("II", 1)
    max_samples = min(n_total, int(fs * duration_s))
    rhythm_sig = signals[:max_samples, ii_idx]
    t_rhythm = np.arange(len(rhythm_sig)) / fs

    _style_axis(ax_rhythm, duration_s)
    ax_rhythm.plot(t_rhythm, rhythm_sig, color=TRACE_COLOR, linewidth=0.7)
    ax_rhythm.set_xlim(0, duration_s)
    sig_mean = np.mean(rhythm_sig)
    ax_rhythm.set_ylim(sig_mean - 1.5, sig_mean + 1.5)
    ax_rhythm.text(0.005, 0.92, "II (rhythm)", transform=ax_rhythm.transAxes,
                   fontsize=8, fontweight="bold", verticalalignment="top",
                   color="#333333")

    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    return fig


def load_record_full(record_path):
    """Load a WFDB record, return (all_signals, fs, lead_names)."""
    rec = wfdb.rdrecord(record_path)
    return rec.p_signal, rec.fs, rec.sig_name


def load_record(record_path):
    """Load a WFDB record, return (signal_lead_II, fs, lead_name)."""
    rec = wfdb.rdrecord(record_path)
    sig_names = rec.sig_name
    if "II" in sig_names:
        idx = sig_names.index("II")
        lead_name = "II"
    elif len(sig_names) > 1:
        idx = 1
        lead_name = sig_names[1]
    else:
        idx = 0
        lead_name = sig_names[0]
    return rec.p_signal[:, idx], rec.fs, lead_name


def find_all_records(base="ekg_datasets/ptbxl/records500"):
    """Find all .dat files under the PTB-XL records directory."""
    base_path = Path(base)
    if not base_path.exists():
        return []
    return sorted(
        str(p).replace(".dat", "")
        for p in base_path.rglob("*.dat")
    )


def main():
    parser = argparse.ArgumentParser(description="Generate paper-style EKG strip PNGs")
    parser.add_argument("--record", type=str, default=None,
                        help="Path to a specific WFDB record (without extension)")
    parser.add_argument("--batch", type=int, default=0,
                        help="Generate N random strips from PTB-XL")
    parser.add_argument("--out", type=str, default="ekg_strips",
                        help="Output directory for PNGs")
    parser.add_argument("--duration", type=float, default=10,
                        help="Strip duration in seconds (default: 10)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Image DPI (default: 200)")
    parser.add_argument("--single", action="store_true",
                        help="Render single-lead strip instead of 12-lead layout")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    records = []
    if args.record:
        records.append(args.record)
    elif args.batch > 0:
        all_recs = find_all_records()
        if not all_recs:
            print("No PTB-XL records found. Run download_ekg_datasets.py first.")
            return
        records = random.sample(all_recs, min(args.batch, len(all_recs)))
    else:
        all_recs = find_all_records()
        if not all_recs:
            print("No PTB-XL records found. Run download_ekg_datasets.py first.")
            return
        records = [all_recs[0]]

    for rec_path in records:
        name = Path(rec_path).stem
        print(f"  Rendering {name} ...")

        if args.single:
            signal, fs, lead_name = load_record(rec_path)
            fig = render_strip(signal, fs, lead_name,
                               duration_s=args.duration, dpi=args.dpi)
            suffix = ""
        else:
            signals, fs, lead_names = load_record_full(rec_path)
            fig = render_12_lead(signals, fs, lead_names, dpi=args.dpi)
            suffix = "_12lead"

        out_path = os.path.join(args.out, f"{name}{suffix}.png")
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        print(f"    -> {out_path}")

    print(f"\nDone. {len(records)} strip(s) saved to {args.out}/")


if __name__ == "__main__":
    main()
