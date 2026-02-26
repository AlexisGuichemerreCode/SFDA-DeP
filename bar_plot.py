#!/usr/bin/env python3
"""
MICCAI-style bar plots for WSOL results.

Generates:
  1) wsol_loc_vs_noloc_bars.pdf  (SAT / DeepMIL / PixelCAM; 4 bars per model)
  2) pixelcam_static_vs_resampling_bars.pdf (PixelCAM only; Static vs Resampling)

No explicit colors are set (uses matplotlib defaults); differentiation via hatch/labels.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def set_miccai_style():
    plt.rcParams.update({
        "font.size": 16,  
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_wsol_loc_vs_noloc(outpath: str = "wsol_loc_vs_noloc_bars.pdf"):
    models = ["SAT", "DeepMIL", "PixelCAM"]

    # Without localization (no-loc)
    pxap_noloc = np.array([29.2, 21.8, 15.8])
    cl_noloc   = np.array([73.2, 78.9, 78.4])

    # With localization (loc)
    pxap_loc = np.array([27.8, 37.7, 53.8])
    cl_loc   = np.array([77.4, 77.9, 80.6])

    x = np.arange(len(models))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    fig, ax = plt.subplots(figsize=(6.2, 2.4))  # compact & template-friendly

    # Bars: 4 per model
    bars1 = ax.bar(x + offsets[0], pxap_noloc, width,
                   label="PxAP (w/o-loc)", edgecolor="black", linewidth=0.6, hatch="///")
    bars2 = ax.bar(x + offsets[1], pxap_loc,   width,
                   label="PxAP (w/ loc)", edgecolor="black", linewidth=0.6, hatch="\\\\\\")
    bars3 = ax.bar(x + offsets[2], cl_noloc,   width,
                   label="CL (w/o-loc)", edgecolor="black", linewidth=0.6, hatch="..")
    bars4 = ax.bar(x + offsets[3], cl_loc,     width,
                   label="CL (w/ loc)", edgecolor="black", linewidth=0.6, hatch="xx")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")

    # Y range: keep enough headroom (you can tighten if you want)
    ax.set_ylim(0, 90)

    # Grid for readability (light)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    # Optional: annotate bar values (comment out if you want a cleaner look)
    def annotate(bars):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 1.0, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7)

    annotate(bars1); annotate(bars2); annotate(bars3); annotate(bars4)

    # Create legend at figure level (not axis level)
    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        loc="upper center",
        bbox_to_anchor=(0.6, 1.08)
    )

    # Adjust layout to leave space for legend
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_pixelcam_static_vs_resampling(outpath: str = "pixelcam_static_vs_resampling_bars.pdf"):
    conditions = ["Static", "Resampling"]

    pxap = np.array([39.3, 53.8])
    cl   = np.array([57.9, 80.6])

    x = np.arange(len(conditions))
    width = 0.32

    fig, ax = plt.subplots(figsize=(4.2, 2.4))

    b1 = ax.bar(x - width/2, pxap, width,
                label="PxAP", edgecolor="black", linewidth=0.6, hatch="///")
    b2 = ax.bar(x + width/2, cl,   width,
                label="CL Acc", edgecolor="black", linewidth=0.6, hatch="..")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 90)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1.0, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7)

    # Create legend at figure level (not axis level)
    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        loc="upper center",
        bbox_to_anchor=(0.6, 1.02)
    )

    # Adjust layout to leave space for legend
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    set_miccai_style()
    plot_wsol_loc_vs_noloc()
    plot_pixelcam_static_vs_resampling()
    print("Saved: wsol_loc_vs_noloc_bars.pdf")
    print("Saved: pixelcam_static_vs_resampling_bars.pdf")


if __name__ == "__main__":
    main()