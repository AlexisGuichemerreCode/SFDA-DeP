import matplotlib.pyplot as plt
import numpy as np

model1_localization = [68.25, 66.85, 70.65, 68.51, 68.06, 67.94, 69.34, 56.69, 70.15, 64.8]
model2_localization = [81.14, 79.52, 82.55, 80.36, 80.54, 79.79, 81.69, 77.49, 82.0 , 78.89]
model1_classification = [91.25, 92.5, 91.25, 93.75, 92.5, 93.75, 93.75, 85.0 , 93.75, 91.25]
model2_classification = [100.0, 98.75, 100.0, 100.0, 100.0, 100.0, 100.0 , 97.5, 100.0, 96.25]

# Indices pour l'axe x
epochs = np.arange(1, 11)
x_labels = [str(i) for i in range(1, 11)] 

# Création de la figure
plt.figure(figsize=(8, 6))

# Tracer les courbes avec les nouvelles légendes
plt.plot(epochs, model1_classification, marker='o', linestyle='-', label='CL LayerCAM', color='#00008B')  # Bleu foncé
plt.plot(epochs, model2_classification, marker='s', linestyle='--', label='CL PixelCAM', color='#FF0000')  # Rouge
plt.plot(epochs, model1_localization, marker='d', linestyle='-', label='PXAP LayerCAM', color='#00BFFF')  # Bleu clair
plt.plot(epochs, model2_localization, marker='^', linestyle='--', label='PXAP PixelCAM', color='#FF6347')  # Rouge clair

# Labels et titre
plt.xlabel("Stain")
plt.ylabel("Accuracy")
plt.title("CL and PXAP performances")

plt.xticks(epochs, x_labels, ha="right")
plt.gca().set_xticks(epochs) 


# Personnalisation des ticks de l'axe x
#plt.xticks(epochs, x_labels, rotation=30, ha="right")

# Affichage de la légende et de la grille
plt.legend()
plt.grid(True)

# Sauvegarde de la figure
plt.savefig("performance_comparaison.png", dpi=300, bbox_inches='tight')

# Affichage du plot
plt.show()