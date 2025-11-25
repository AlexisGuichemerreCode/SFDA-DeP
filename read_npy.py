import numpy as np
import matplotlib.pyplot as plt


arr = np.load("/export/livia/home/vision/Aguichemerre/all_preds_train_before.npy")
print(arr)


arr2 = np.load("/export/livia/home/vision/Aguichemerre/all_targets_train_before.npy")
print(arr2)


classes = np.arange(100)

pred_counts = [(arr == c).sum() for c in classes]

plt.figure(figsize=(18,6))
plt.bar(classes, pred_counts, color="orange")

plt.xlabel("Predicted class", fontsize=12)
plt.ylabel("Number of predictions", fontsize=12)
plt.title("Histogram of predictions per classe", fontsize=14)

# Label toutes les 5 classes pour lisibilité
plt.xticks(classes[::5], rotation=45)

plt.grid(axis='y', linestyle="--", alpha=0.5)

plt.axhline(145, color='red', linestyle='--', linewidth=2, label='Expected = 145')
plt.legend()

plt.savefig("prediction_histogram.png", dpi=300, bbox_inches='tight')
plt.close()


pred_counts = [(arr2 == c).sum() for c in classes]

plt.figure(figsize=(18,6))
plt.bar(classes, pred_counts, color="orange")

plt.xlabel("Real class", fontsize=12)
plt.ylabel("Number of images", fontsize=12)
plt.title("Histogram of imgs per classe", fontsize=14)

# Label toutes les 5 classes pour lisibilité
plt.xticks(classes[::5], rotation=45)

plt.grid(axis='y', linestyle="--", alpha=0.5)

plt.savefig("real_distiribution_histogram.png", dpi=300, bbox_inches='tight')
plt.close()



accuracies = []
for c in classes:
    idx = (arr2 == c)
    if idx.sum() == 0:
        # aucune image pour cette classe
        accuracies.append(0)
    else:
        acc = (arr[idx] == c).sum() / idx.sum()
        accuracies.append(acc)

plt.figure(figsize=(18, 6))
plt.bar(classes, accuracies, color="steelblue")

plt.xlabel("Class", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Accuracy per class", fontsize=14)

# Afficher un label toutes les 5 classes pour éviter la surcharge
plt.xticks(classes[::5], rotation=45)

plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle="--", alpha=0.5)

# Sauvegarder au lieu d'afficher (comme tu voulais)
plt.savefig("accuracy_per_class_clean.png", dpi=300, bbox_inches='tight')
plt.close()

