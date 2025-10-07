import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Fichiers I/O
INPUT_TXT = "/export/livia/home/vision/Aguichemerre/spearman_by_source_model_train.txt"
OUT_DIR = "/export/livia/home/vision/Aguichemerre/spearman_figures"
os.makedirs(OUT_DIR, exist_ok=True)

def parse_spearman_file(filepath):
    """
    Parse le fichier texte généré par compute_and_save_spearman
    et retourne une DataFrame avec colonnes:
    [group, metric, target, rho, N]
    """
    data = []
    current_group = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("==="):
                # Exemple: "=== GLAS__DEEPMIL ==="
                current_group = line.replace("=", "").strip()
                continue
            if "rho=" in line and ":" in line:
                # Exemple:
                # " Wasserstein_1D vs Accuracy        : rho=-0.857, p=1.370e-02, N=7"
                parts = line.split(":")
                left = parts[0].strip()
                right = parts[1].strip()

                # left = "Wasserstein_1D vs Accuracy"
                metric, target = [x.strip() for x in left.split("vs")]

                # extraire rho et N
                m = re.search(r"rho=([\-0-9\.]+),.*N=(\d+)", right)
                if not m:
                    continue
                rho = float(m.group(1))
                N = int(m.group(2))

                data.append({
                    "group": current_group,
                    "metric": metric,
                    "target": target,
                    "rho": rho,
                    "N": N
                })
    return pd.DataFrame(data)

def plot_heatmap(df, target, out_path):
    """
    Trace une heatmap pour un target donné (Accuracy ou cam_performance).
    """
    df_t = df[df["target"] == target]
    if df_t.empty:
        print(f"⚠️ Pas de données pour {target}")
        return

    # Pivot table
    pivot = df_t.pivot(index="group", columns="metric", values="rho")

    plt.figure(figsize=(12, max(6, 0.5*len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".2f", center=0, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.title(f"Corrélation Spearman (ρ) : {target}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Heatmap sauvegardée: {out_path}")

if __name__ == "__main__":
    df = parse_spearman_file(INPUT_TXT)

    # Heatmap pour Accuracy
    plot_heatmap(df, "Accuracy", os.path.join(OUT_DIR, "heatmap_accuracy_train.png"))

    # Heatmap pour cam_performance
    plot_heatmap(df, "cam_performance", os.path.join(OUT_DIR, "heatmap_cam_performance_train.png"))
