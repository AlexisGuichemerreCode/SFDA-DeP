# import os
# import pickle
# import matplotlib.pyplot as plt

# # Dossier contenant les fichiers .pkl
# folder = "/export/livia/home/vision/Aguichemerre/local_perf/LayerCAM"

# #folder_pxcam = "/export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/exps/CAMELYON512/resnet50/STD_CL/PixelCAM/id_unlearning_LayerCAM_pxcam-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50/Localization_Accuracy_results_target_data.pickle"

# # Récupère tous les fichiers .pkl dans le dossier
# pkl_files = [f for f in os.listdir(folder) if f.endswith(".pickle")]

# plt.figure(figsize=(10, 6))

# for fname in sorted(pkl_files):
#     path = os.path.join(folder, fname)
#     with open(path, "rb") as f:
#         data = pickle.load(f)
    
#     # Si le fichier contient un dictionnaire, essaye de prendre la première clé
#     if isinstance(data, dict):
#         key = list(data.keys())[0]
#         y = data[key]
#     else:
#         y = data  # supposé être une liste ou un tableau
    
#     plt.plot(y, label=fname.replace(".pkl", ""))

# # --- Add the PixelCAM file ---
# # if os.path.exists(folder_pxcam):
# #     with open(folder_pxcam, "rb") as f:
# #         pxcam_data = pickle.load(f)
    
# #     if isinstance(pxcam_data, dict):
# #         key = list(pxcam_data.keys())[0]
# #         y_pxcam = pxcam_data[key]
# #     else:
# #         y_pxcam = pxcam_data
    
# #     plt.plot(y_pxcam, label="PixelCAM (reference)", linewidth=2.5, color="black")
# # else:
# #     print("⚠️ PixelCAM file not found:", folder_pxcam)

# plt.title("Comparison of Curves (Cosine / MSE / PixelCAM)")
# plt.xlabel("Itethresholdns / Epochs")
# plt.ylabel("Metric Value")
# plt.legend(title="Methods", loc="upper right")
# plt.grid(True)
# plt.tight_layout()

# # --- Save the figure ---
# output_path = os.path.join(folder, "comparison_losses.png")
# plt.savefig(output_path, dpi=300, bbox_inches="tight")

# plt.show()

# print(f"✅ Figure saved at: {output_path}")


# import matplotlib.pyplot as plt

# # Thresholds et accuracies correspondantes
# thresholds = [0.01,0.05,0.1, 0.2, 0.3, 0.4, 0.5]
# accuracies = [18.5,19.3,19.1,20.1,18.7,19.3,19.8]
# baseline = 23.5  # avant unlearning

# plt.figure(figsize=(8, 5))

# # Courbe principale
# plt.plot(thresholds, accuracies, marker='o', linewidth=2.5, color='royalblue', label="After Unlearning")

# # Ligne horizontale pour le modèle avant unlearning
# plt.axhline(y=baseline, color='red', linestyle='--', linewidth=2, label="Before Unlearning")

# # Légendes et titres
# plt.title("Evolution of LayerCAM Localization vs. Unlearning Threshold", fontsize=14, weight='bold')
# plt.xlabel("Unlearning Threshold", fontsize=12)
# plt.ylabel("Localization (%)", fontsize=12)
# plt.legend(loc="lower right", fontsize=11)
# plt.grid(True, linestyle='--', alpha=0.6)

# # Ajustements esthétiques
# plt.tight_layout()

# # Sauvegarde
# plt.savefig("LayerCAM_Localization_vs_threshold.png", dpi=300, bbox_inches="tight")
# plt.show()

# print("✅ Figure saved as LayerCAM_Localization_vs_threshold.png")


import matplotlib.pyplot as plt

# Seuils et nombre d'images correspondants
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
num_images = [2294,4584,6876,9168,11460]

plt.figure(figsize=(8, 5))

# Histogramme
plt.bar([str(t) for t in thresholds], num_images, color='steelblue', edgecolor='black')

# Titres et axes
plt.title("Number of images in Xforget per unlearning ratio", fontsize=14, weight='bold')
plt.xlabel("Unlearning ratio", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)

# Ajout des valeurs sur les barres
for i, v in enumerate(num_images):
    plt.text(i, v + 100, str(v), ha='center', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# Sauvegarde
plt.savefig("Deepmil xforget_images_per_ratio.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ Figure saved as xforget_images_per_threshold.png")