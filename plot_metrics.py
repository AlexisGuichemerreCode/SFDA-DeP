import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Path vers ton répertoire d'expériences
base_dir = "/export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/exps/CAMELYON512/resnet50/STD_CL/PixelCAM/id_target_b_cl_esfda_new_entropy_reverse_0_CAMELYON512_PixelCAM_lr0p001_5_esfdaratio0p2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50"

# Noms des fichiers pickle
files = {
    "loss": "loss_history.pickle",
    "metrics": "metrics_history.pickle",
    "accuracy": "Classification_Accuracy_results_target_data.pickle"
}

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Charger les dictionnaires
loss_history = load_pickle(os.path.join(base_dir, files["loss"]))
metrics_history = load_pickle(os.path.join(base_dir, files["metrics"]))
acc_history = load_pickle(os.path.join(base_dir, files["accuracy"]))


print(loss_history["loss"][:20]) 



loss_values = np.array(loss_history["loss"]) / 128.0  # division par batch size
loss_every_10 = loss_values[::10]  # une valeur tous les 10 batchs

plt.figure(figsize=(8,5))
plt.plot(loss_every_10, label="Loss (every 10 batches)")
plt.title("Training Loss (every 10 batches)")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "loss_every_10_batches.png"))
plt.close()

# --- Paramètres ---
images_per_epoch = 26564
batches_per_epoch = 208
num_epochs = len(loss_history["loss"]) // batches_per_epoch

# --- Calcul loss moyenne par epoch ---
epoch_losses = [
    sum(loss_history["loss"][i:i+batches_per_epoch]) / images_per_epoch
    for i in range(0, len(loss_history["loss"]), batches_per_epoch)
]

# --- Plot ---
plt.figure(figsize=(15, 5))
plt.plot(epoch_losses, marker="o", label="Epoch Loss")
plt.title("Average Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "loss_per_epoch.png"))
plt.close()

# --- Plot 2: Metrics (KL, Hm, Htilde) ---
plt.figure(figsize=(8,5))
if isinstance(metrics_history, dict):
    if "kl" in metrics_history: plt.plot(metrics_history["kl"], label="KL-consistency")
    if "hm" in metrics_history: plt.plot(metrics_history["hm"], label="Hm (marginal entropy)")
    if "h_tilde" in metrics_history: plt.plot(metrics_history["h_tilde"], label="Htilde (lack of diversity)")
plt.title("Metrics History")
plt.xlabel("Batch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "metrics_history.png"))
plt.close()

# --- Plot 3: Accuracy ---
plt.figure(figsize=(8,5))
if isinstance(acc_history, dict):
    for k, v in acc_history.items():
        plt.plot(v, label=f"Accuracy {k}")
else:
    plt.plot(acc_history, label="Accuracy")
plt.title("Classification Accuracy on Target Data")
plt.xlabel("Batch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "accuracy_history.png"))
plt.close()


def normalize(x):
    if len(x) == 0:
        return x
    xmin, xmax = min(x), max(x)
    if abs(xmax - xmin) < 1e-8:
        return [0.0 for _ in x]  # éviter division par zéro
    return [(val - xmin) / (xmax - xmin) for val in x]

    
# --- Plot 4: Superposition (normalized to [0,1]) ---
plt.figure(figsize=(10,6))

# Loss
if isinstance(loss_history, dict):
    for k, v in loss_history.items():
        plt.plot(normalize(v), label=f"Loss {k} (norm)")
else:
    plt.plot(normalize(loss_history), label="Loss (norm)")

# Metrics
if isinstance(metrics_history, dict):
    if "kl" in metrics_history:
        plt.plot(normalize(metrics_history["kl"]), label="KL-consistency (norm)")
    if "hm" in metrics_history:
        plt.plot(normalize(metrics_history["hm"]), label="Hm (norm)")
    if "h_tilde" in metrics_history:
        plt.plot(normalize(metrics_history["h_tilde"]), label="Htilde (norm)")

# Accuracy
if isinstance(acc_history, dict):
    for k, v in acc_history.items():
        plt.plot(normalize(v), label=f"Accuracy {k} (norm)")
else:
    plt.plot(normalize(acc_history), label="Accuracy (norm)")

plt.title("Superposed Curves (Normalized to [0,1])")
plt.xlabel("Epoch")
plt.ylabel("Normalized Value [0,1]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "all_metrics_superposed_normalized.png"))
plt.close()