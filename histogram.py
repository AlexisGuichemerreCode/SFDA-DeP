import matplotlib.pyplot as plt
import numpy as np
import os

# === Configuration générale ===
os.makedirs("entropy_histograms", exist_ok=True)
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# === Simulation des distributions d'entropie ===
np.random.seed(42)
entropies = {
    "Class 0": np.random.normal(0.5, 0.1, 200),   # moyennes
    "Class 1": np.random.normal(0.2, 0.05, 200),  # faibles
    "Class 2": np.random.normal(0.35, 0.1, 85),   # basses à moyennes
    "Class 3": np.random.uniform(0.05, 0.95, 50)  # très variées
}

# Limiter les valeurs dans [0, 1]
entropies = {k: np.clip(v, 0, 1) for k, v in entropies.items()}

# === Couleurs cohérentes ===
colors = {
    "Class 0": "royalblue",
    "Class 1": "crimson",
    "Class 2": "seagreen",
    "Class 3": "orange"
}

# === Axe Y commun ===
bins = np.linspace(0, 1, 20)
all_counts = [np.histogram(v, bins=bins)[0].max() for v in entropies.values()]
ymax = max(all_counts) * 1.2  # marge pour éviter le clipping

# === Boucle de génération ===
for cls_name, values in entropies.items():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.hist(values, bins=bins, color=colors[cls_name],
            alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Entropy")
    ax.set_ylabel("# Images")
    ax.set_title(cls_name, pad=8)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    out_path = f"{cls_name.replace(' ', '_').lower()}.svg"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {out_path}")
