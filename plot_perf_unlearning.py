import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

main_dir = "/export/livia/home/vision/Aguichemerre/CALIBRATION/CAMELYON512/NEW_CL/nomemtum/cl_freeze"
save_dir = os.path.join(main_dir, "figures")
os.makedirs(save_dir, exist_ok=True)

all_data = {}

def extract_lr_from_name(folder_name):
    match_decimal = re.search(r'lr0p(\d+)', folder_name)
    if match_decimal:
        raw_lr = match_decimal.group(1)
        lr = float("0." + raw_lr)
        return f"lr {lr:.5f}".rstrip('0').rstrip('.')

    # Cas 2 : format 'lr1e-05'
    match_exp = re.search(r'lr(\d+e[-+]?\d+)', folder_name)
    if match_exp:
        lr = float(match_exp.group(1))
        return f"lr {lr:.1e}"

    return folder_name

# === Charger les données ===
for subdir in os.listdir(main_dir):
    sub_path = os.path.join(main_dir, subdir)
    if not os.path.isdir(sub_path):
        continue

    pickle_path = os.path.join(sub_path, "results_source_target_data.pickle")
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            dic = pickle.load(f)

            if isinstance(dic, dict):
                for k, v in dic.items():
                    if isinstance(v, list):
                        all_data.setdefault(k, {})[subdir] = v



# === Tracer les courbes individuelles pour 'target_train_acc_cl' uniquement ===
key_to_plot = 'target_train_acc_cl'

if key_to_plot in all_data:
    dataset = all_data[key_to_plot]

    for label, values in dataset.items():
        pretty_label = extract_lr_from_name(label)
        plt.figure(figsize=(10, 5))
        plt.plot(values, label=pretty_label)
        plt.title(f"'{key_to_plot}' – {pretty_label}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        filename = f"{key_to_plot}_{label}_individual.png".replace('/', '_')
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()
else:
    print(f"Clé '{key_to_plot}' non trouvée dans les données.")



# === Tracer les courbes individuelles + moyenne/écart-type ===
for key, dataset in all_data.items():
    # ---------- 1. Figure des courbes individuelles ----------
    plt.figure(figsize=(20, 6))
    for label, values in dataset.items():
        pretty_label = extract_lr_from_name(label)
        plt.plot(values, label=pretty_label)
    plt.title(f"Evolution of '{key}' (individual curves)")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{key}_individuals.png"))
    plt.close()

    # ---------- 2. Figure moyenne + écart-type style seaborn ----------
    fig, ax = plt.subplots(figsize=(20, 6))
    clrs = sns.color_palette("husl", len(dataset))

    # Préparation des données
    all_series = []
    labels = []
    min_len = min(len(v) for v in dataset.values())

    for i, (label, values) in enumerate(dataset.items()):
        pretty_label = extract_lr_from_name(label)
        arr = np.array(values[:min_len])
        all_series.append(arr)
        labels.append(pretty_label)

    stacked = np.stack(all_series)  # shape (n_runs, T)

    mean_curve = stacked.mean(axis=0)
    std_curve = stacked.std(axis=0)
    epochs = np.arange(min_len)

    with sns.axes_style("darkgrid"):
        ax.plot(epochs, mean_curve, label="Mean", c="black", linewidth=2)
        ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.3, facecolor="black", label="± Standard deviation")
        ax.set_title(f"Mean evolution of '{key}'")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{key}_mean_std.png"))
        plt.close()

    print(f"✅ Figures pour '{key}' enregistrées.")
