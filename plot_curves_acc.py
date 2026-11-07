import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# DeepMIL - C17-4
# ============================================================

epochs = np.array([
    1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
    50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100
])

# ============================================================
# IMPORTANT: INVERSION
# ============================================================
#
# Retain accuracy = original "forget acc"
# Forget accuracy = original "retain acc"
#
# ============================================================

retain_acc = np.array([
    0.5675,  # 1
    0.7765,  # 5
    0.8464,  # 10
    0.8855,  # 15
    0.8938,  # 20
    0.8948,  # 25
    0.8957,  # 30
    0.8996,  # 35
    0.8847,  # 40
    0.8946,  # 45
    0.8849,  # 50
    0.8568,  # 55
    0.8213,  # 60
    0.8584,  # 65
    0.8389,  # 70
    0.7942,  # 75
    0.8458,  # 80
    0.8387,  # 85
    0.7866,  # 90
    0.7707,  # 95
    0.7284   # 100
]) * 100


forget_acc = np.array([
    0.2715,  # 1
    0.4392,  # 5
    0.4342,  # 10
    0.4688,  # 15
    0.5231,  # 20
    0.3179,  # 25
    0.5200,  # 30
    0.5188,  # 35
    0.3046,  # 40
    0.5323,  # 45
    0.5909,  # 50
    0.3113,  # 55
    np.nan,  # 60: no forget samples
    0.3359,  # 65
    0.7444,  # 70
    0.1765,  # 75
    0.3333,  # 80
    0.7143,  # 85
    0.1622,  # 90
    0.8250,  # 95
    0.1350   # 100
]) * 100


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
})

retain_color = "#4C78A8"   # Blue
forget_color = "#F28E2B"   # Orange


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(10, 4.0))

x = np.arange(len(epochs))
bar_width = 0.40


# ============================================================
# BARS
# ============================================================

ax.bar(
    x - bar_width / 2,
    retain_acc,
    width=bar_width,
    color=retain_color,
    label="Retain",
    edgecolor="white",
    linewidth=0.3
)

ax.bar(
    x + bar_width / 2,
    forget_acc,
    width=bar_width,
    color=forget_color,
    label="Forget",
    edgecolor="white",
    linewidth=0.3
)


# ============================================================
# AXES
# ============================================================

ax.set_xticks(x)
ax.set_xticklabels(epochs)

ax.set_xlabel("Epoch")
ax.set_ylabel("Selection accuracy (%)")

ax.set_ylim(0, 100)

ax.set_yticks([
    0, 20, 40, 60, 80, 100
])


# ============================================================
# GRID
# ============================================================

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.30
)

ax.set_axisbelow(True)


# ============================================================
# CLEAN SPINES
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# LEGEND
# ============================================================

ax.legend(
    frameon=False,
    ncol=2,
    loc="upper left"
)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    "deepmil_c17_4_retain_forget_accuracy.pdf",
    bbox_inches="tight",
    dpi=300
)

plt.savefig(
    "deepmil_c17_4_retain_forget_accuracy.png",
    bbox_inches="tight",
    dpi=300
)

plt.show()