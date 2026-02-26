#!/usr/bin/env python3
"""
MICCAI-style bar plots for class prediction percentages.

Generates:
  1) pred_percentages_set1.pdf  (C16, C17-0, C17-1, C17-2, C17-3, C17-4, GlaS)
  2) pred_percentages_set2.pdf  (C17-0, C17-1, C17-2, C17-3, C17-4, GlaS, C16)

No explicit colors are set (uses matplotlib defaults); differentiation via hatch/labels.
Adds a dashed horizontal reference line at 50%.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def set_miccai_style(font_size: int = 14):
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_two_class_percentages(
    datasets: list[str],
    normal: np.ndarray,
    cancer: np.ndarray,
    outpath: str,
    title: str | None = None,          # kept for compatibility, but we'll ignore it
    add_avg: bool = True,              # NEW
):
    assert len(datasets) == len(normal) == len(cancer)

    # --- NEW: append Avg as the last "dataset" ---
    if add_avg:
        avg_normal = float(np.mean(normal))
        avg_cancer = float(np.mean(cancer))
        datasets = list(datasets) + ["Avg"]
        normal = np.append(normal, avg_normal)
        cancer = np.append(cancer, avg_cancer)

    x = np.arange(len(datasets))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.6, 2.6))  # compact

    b1 = ax.bar(
        x - width/2, normal, width,
        label="Normal", edgecolor="black", linewidth=0.6, hatch="///"
    )
    b2 = ax.bar(
        x + width/2, cancer, width,
        label="Cancer", edgecolor="black", linewidth=0.6, hatch=".."
    )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("% of predictions")
    ax.set_ylim(0, 110)

    # --- NEW: remove titles entirely ---
    # (ignore `title`)

    # 50% reference line
    ax.axhline(50, linestyle="--", linewidth=0.9, alpha=0.9)

    # light y-grid
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    # annotate values
    def annotate(bars):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}",
                ha="center", va="bottom", fontsize=8  # slightly bigger if font is 14
            )

    annotate(b1)
    annotate(b2)

    # Remove axis-level legend (don't use ax.legend at all)
    handles, labels = ax.get_legend_handles_labels()

    leg = fig.legend(
        handles,
        labels,
        ncol=2,
        frameon=True,          # set False if you want MICCAI-clean
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.2)  # center above the plot
    )

    fig.tight_layout(pad=0.6)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    set_miccai_style(font_size=14)

    # -----------------------------
    # Existing plots (yours)
    # -----------------------------
    datasets1 = ["C17-0", "C17-1", "C17-2", "C17-3", "C17-4", "GlaS"]
    normal1 = np.array([55.7, 48.8, 50.2, 50.8, 56.0, 47.5], dtype=float)
    cancer1 = np.array([44.3, 51.2, 49.8, 49.2, 43.9, 52.5], dtype=float)

    mask1 = ~np.isnan(normal1) & ~np.isnan(cancer1)
    datasets1 = [d for d, m in zip(datasets1, mask1) if m]
    normal1 = normal1[mask1]
    cancer1 = cancer1[mask1]

    plot_two_class_percentages(
        datasets=datasets1,
        normal=normal1,
        cancer=cancer1,
        outpath="pred_percentages_set1.png",
        title=None,        # ignored anyway
        add_avg=True,      # NEW (default)
    )

    datasets2 = ["C17-0", "C17-1", "C17-2", "C17-3", "C17-4", "C16"]
    normal2 = np.array([14.1, 19.2, 21.6, 28.9, 21.4, 21.5], dtype=float)
    cancer2 = np.array([85.9, 80.2, 78.3, 71.0, 78.5, 78.4], dtype=float)

    mask2 = ~np.isnan(normal2) & ~np.isnan(cancer2)
    datasets2 = [d for d, m in zip(datasets2, mask2) if m]
    normal2 = normal2[mask2]
    cancer2 = cancer2[mask2]

    plot_two_class_percentages(
        datasets=datasets2,
        normal=normal2,
        cancer=cancer2,
        outpath="pred_percentages_set2.png",
        title=None,
        add_avg=True,
    )

    # -----------------------------
    # NEW plots (SFDA-DE / SFDA-GU)
    # Order assumed: [C16, C17-0, C17-1, C17-2, C17-3, C17-4]
    # -----------------------------
    datasets_sfda = ["C17-0", "C17-1", "C17-2", "C17-3", "C17-4", "C16"]

    # SFDA-DE: (Normal, Cancer)
    normal_de = np.array([100.0, 23.8, 33.5, 100.0, 21.5, 21.5], dtype=float)
    cancer_de = np.array([  0.0, 76.2, 66.5,   0.0, 78.5, 78.4], dtype=float)

    plot_two_class_percentages(
        datasets=datasets_sfda,
        normal=normal_de,
        cancer=cancer_de,
        outpath="pred_percentages_sfda_de.png",
        title=None,        # no title now
        add_avg=True,
    )

    # SFDA-GU: (Normal, Cancer)
    normal_gu = np.array([57.6, 75.0, 40.0, 56.9, 83.2, 60.8], dtype=float)
    cancer_gu = np.array([42.4, 25.0, 60.0, 43.1, 16.8, 39.2], dtype=float)

    plot_two_class_percentages(
        datasets=datasets_sfda,
        normal=normal_gu,
        cancer=cancer_gu,
        outpath="pred_percentages_sfda_gu.png",
        title=None,
        add_avg=True,
    )

    print("Saved: pred_percentages_set1.png")
    print("Saved: pred_percentages_set2.png")
    print("Saved: pred_percentages_sfda_de.png")
    print("Saved: pred_percentages_sfda_gu.png")

if __name__ == "__main__":
    main()