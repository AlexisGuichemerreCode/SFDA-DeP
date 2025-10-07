import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

BASE_DIR = "/export/livia/home/vision/Aguichemerre/metrics_cvpr_da/train"
OUT_DIR = "/export/livia/home/vision/Aguichemerre/figures_summary/train"
os.makedirs(OUT_DIR, exist_ok=True)


DATASET_COLORS = {
    "GLAS": "blue",
    "CAMELYON16": "red",
    "CAMELYON512": "purple",
    "CAMELYON17_512_0": "limegreen",
    "CAMELYON17_512_1": "darkgreen",
    "CAMELYON17_512_2": "orange",
    "CAMELYON17_512_3": "cyan",
    "CAMELYON17_512_4": "magenta",
}

SHIFT_LABELS = {
    "KL_1D": "Kullback–Leibler (KL)",
    "KS_1D": "Kolmogorov–Smirnov",
    "Wasserstein_1D": "Wasserstein",
    "MK-MMD": "Maximum Mean Discrepancy (MMD)",
    "CORAL_norm": "Coral",
    "SWD_proj128": "SWD",
    "CMD_K5": "CMD",
    "JSD_1D": "JSD"
}


def shorten_label(label: str):
    """Raccourcit les noms de datasets pour les plots."""
    if label.startswith("CAMELYON17_512_"):
        return label.replace("CAMELYON17_512_", "C17_")
    if label == "CAMELYON512":
        return "C16"
    return label



def get_dataset_color(label):
    """Retourne une couleur stable pour un dataset ou un fold."""
    if label in DATASET_COLORS:
        return DATASET_COLORS[label]
    # fallback: si on n’a pas prévu ce dataset
    for ds in DATASET_COLORS:
        if label.startswith(ds):
            return DATASET_COLORS[ds]
    return "black"



def load_metrics(source, method):
    """Charge tous les metrics.txt pour un (source, method)."""
    results = {}
    path = os.path.join(BASE_DIR, source, method)
    if not os.path.isdir(path):
        return results

    for target in os.listdir(path):
        target_path = os.path.join(path, target)

        # Cas 1 : metrics.txt directement dans le dossier
        metrics_path = os.path.join(target_path, "metrics.txt")
        if os.path.isfile(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)
            results[target] = data
            continue

        # Cas 2 : sous-dossiers (ex: CAMELYON17_512/0,1,2,...)
        if os.path.isdir(target_path):
            for sub in os.listdir(target_path):
                sub_path = os.path.join(target_path, sub, "metrics.txt")
                if os.path.isfile(sub_path):
                    with open(sub_path, "r") as f:
                        data = json.load(f)
                    results[f"{target}_{sub}"] = data

    return results
def plot_classification_comparison(source, method, data):
    ordered_keys = sorted(data.keys(), key=lambda k: (0 if k == source else 1, k))

    targets = []
    acc_default, f1_default = [], []
    acc_calib,  f1_calib  = [], []
    thr_calib              = []

    for k in ordered_keys:
        m = data[k]
        targets.append(shorten_label(k))

        acc_default.append(m["results"]["Default_0.5"]["Accuracy"])
        f1_default.append(m["results"]["Default_0.5"]["F1"])

        ckey = next((kk for kk in m["results"] if kk.startswith("Calibrated_")), None)
        if ckey is not None:
            acc_calib.append(m["results"][ckey]["Accuracy"])
            f1_calib.append(m["results"][ckey]["F1"])
            thr_calib.append(float(ckey.split("_")[1]))
        else:
            acc_calib.append(None)
            f1_calib.append(None)
            thr_calib.append(None)

    x = np.arange(len(targets))
    width = 0.35
    fig, ax = plt.subplots(1, 2, figsize=(9, 7))

    # Flags pour éviter doublons
    added_blue, added_orange = False, False

    # ---------- Accuracy subplot ----------
    for i, tgt in enumerate(targets):
        # barre bleue
        ax[0].bar(x[i] - width/2, acc_default[i], width,
                  color="tab:blue",
                  label="Threshold=0.5" if not added_blue else None)
        added_blue = True

        # barre orange (pas pour la source)
        if ordered_keys[i] != source and acc_calib[i] is not None:
            ax[0].bar(x[i] + width/2, acc_calib[i], width,
                      color="tab:orange",
                      label="Adjusted Threshold" if not added_orange else None)
            added_orange = True

    ax[0].set_xticks(x); ax[0].set_xticklabels(targets, rotation=45)
    ax[0].set_ylabel("Accuracy"); ax[0].set_title("Accuracy comparison")

    # ---------- F1 subplot ----------
    for i, tgt in enumerate(targets):
        ax[1].bar(x[i] - width/2, f1_default[i], width, color="tab:blue", label=None)
        if ordered_keys[i] != source and f1_calib[i] is not None:
            ax[1].bar(x[i] + width/2, f1_calib[i], width, color="tab:orange", label=None)

    ax[1].set_xticks(x); ax[1].set_xticklabels(targets, rotation=45)
    ax[1].set_ylabel("F1-score"); ax[1].set_title("F1 comparison")

    # ---------- Légende unique ----------
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), fontsize=18, frameon=True)

    plt.suptitle(f"{method} - {source}", fontsize=8, y=1.08)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    save_path = os.path.join(OUT_DIR, f"{source}_{method}_classification.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()




def plot_shift_correlations(source, method, data, metric="Accuracy"):
    all_shift_keys = list(next(iter(data.values()))["shifts"].keys())

    for shift_key in all_shift_keys:
        xs, ys, labels, colors = [], [], [], []
        for target, metrics in data.items():
            xs.append(metrics["shifts"][shift_key])
            if metric == "cam_performance":
                ys.append(metrics["cam_performance"])
            else:
                calib_keys = [k for k in metrics["results"].keys() if k.startswith("Calibrated")]
                if not calib_keys:
                    continue
                ys.append(metrics["results"][calib_keys[0]]["Accuracy"])
            labels.append(shorten_label(target))
            colors.append(get_dataset_color(target))

        plt.figure(figsize=(6, 5))

        # points
        for i in range(len(xs)):
            plt.scatter(xs[i], ys[i], c=colors[i], s=70)

        # droite
        sns.regplot(
            x=xs, y=ys, ci=None, scatter=False,
            line_kws={"color": "black", "linewidth": 0.5, "linestyle": "--"}
        )

        # légende
        unique_labels = dict()
        for lbl, col in zip(labels, colors):
            if lbl not in unique_labels:
                unique_labels[lbl] = col
        handles = [plt.Line2D([], [], marker="o", color=col, linestyle="", label=lbl)
                   for lbl, col in unique_labels.items()]
        plt.legend(handles, unique_labels.keys(), loc="best", fontsize=8, ncol=2)

        # labels axes
        ylabel = "PxAP" if metric == "cam_performance" else metric
        xlabel = SHIFT_LABELS.get(shift_key, shift_key)  # <--- ici on applique le mapping

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{method} - {source} : {ylabel} vs {xlabel}")
        plt.tight_layout()
        save_path = os.path.join(OUT_DIR, f"{source}_{method}_{ylabel}_vs_{shift_key}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()





def main():
    for source in os.listdir(BASE_DIR):
        source_path = os.path.join(BASE_DIR, source)
        if not os.path.isdir(source_path):
            continue

        # Cas 1 : source avec folds (CAMELYON17_512/0..4)
        if all(name.isdigit() for name in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, name))):
            for fold in os.listdir(source_path):
                fold_path = os.path.join(source_path, fold)
                if not os.path.isdir(fold_path):
                    continue
                for method in os.listdir(fold_path):
                    method_path = os.path.join(fold_path, method)
                    if not os.path.isdir(method_path):
                        continue

                    data = load_metrics(os.path.join(source, fold), method)
                    if not data:
                        continue

                    print(f"Processing {source}_{fold} - {method}")

                    plot_classification_comparison(f"{source}_{fold}", method, data)
                    plot_shift_correlations(f"{source}_{fold}", method, data, metric="Accuracy")
                    plot_shift_correlations(f"{source}_{fold}", method, data, metric="cam_performance")

        # Cas 2 : source “simple” (GLAS, CAMELYON512, etc.)
        else:
            for method in os.listdir(source_path):
                method_path = os.path.join(source_path, method)
                if not os.path.isdir(method_path):
                    continue

                data = load_metrics(source, method)
                if not data:
                    continue

                print(f"Processing {source} - {method}")

                plot_classification_comparison(source, method, data)
                plot_shift_correlations(source, method, data, metric="Accuracy")
                plot_shift_correlations(source, method, data, metric="cam_performance")


if __name__ == "__main__":
    main()