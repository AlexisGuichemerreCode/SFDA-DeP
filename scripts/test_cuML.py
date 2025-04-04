# import numpy as np
# import time
# import matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture
# from cuml.manifold import TSNE



# # 🟢 1. Générer les features aléatoires (N échantillons, 2048 dimensions)
# N = 100000  # Tu peux modifier cette valeur
# D = 2048
# X = np.random.randn(N, D)  # Matrice (N, 2048)

# tsne = TSNE(n_components=2,n_iter=50, random_state=42)
# tsne_result = tsne.fit_transform(df)



# print(f"✅ Données générées avec {N} échantillons et {D} dimensions")

# # 🟢 2. Tester GMM avec différents nombres de clusters et mesurer le temps
# cluster_range = [3]  # Différents nombres de gaussiennes
# times = []

# for k in cluster_range:
#     print(f"🔹 Entraînement de GMM avec {k} clusters sur {N} échantillons et {D} dimensions...")
#     start_time = time.time()
    
#     # Entraîner GMM
#     gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
#     gmm.fit(X)
    
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     times.append(elapsed_time)
    
#     print(f"✅ GMM avec {k} clusters terminé en {elapsed_time:.2f} secondes")

# # 🟢 3. Afficher le temps d'exécution en fonction du nombre de clusters
# plt.figure(figsize=(8, 6))
# plt.plot(cluster_range, times, marker='o', linestyle='-', label="Temps d'exécution")
# plt.xlabel("Nombre de clusters (K)")
# plt.ylabel("Temps d'exécution (secondes)")
# plt.title(f"Temps d'exécution de GMM sans PCA (N={N}, D={D})")
# plt.legend()
# plt.grid(True)

# # 🔹 Sauvegarder l'image du graphique
# plt.savefig("gmm_execution_time_no_pca.png", dpi=300)
# plt.show()

# print("✅ Expérience terminée ! Image sauvegardée sous 'gmm_execution_time_no_pca.png'")

# import numpy as np
# import time
# import matplotlib.pyplot as plt
# from cuml.manifold import TSNE  # t-SNE GPU avec cuML

# # 🔹 Générer des features aléatoires (10000 samples, 2048 dimensions)
# num_samples = 100000  # Nombre de points
# num_features = 2048  # Nombre de dimensions
# random_features = np.random.rand(num_samples, num_features).astype(np.float32)  # Important : dtype=float32 pour cuML

# # 🔹 Exécuter t-SNE avec cuML
# perplexity_value = min(30, num_samples - 1)  # Assurer perplexité < nombre d'échantillons

# tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)

# start_time = time.time()  # Mesurer le temps d'exécution
# X_tsne = tsne.fit_transform(random_features)  # ⚡ t-SNE en GPU
# print(f"t-SNE exécuté en {time.time() - start_time:.2f} secondes.")

# # 🔹 Visualisation des points random en 2D
# plt.figure(figsize=(10, 7))
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], cmap='viridis', alpha=0.6)
# plt.title("t-SNE avec cuML sur Features Random")
# plt.xlabel("t-SNE Dim 1")
# plt.ylabel("t-SNE Dim 2")

# # 🔹 Sauvegarde de l'image
# plt.savefig("tsne_random_features.png", dpi=300)
# plt.show()

import numpy as np
import time
import matplotlib.pyplot as plt
from cuml.manifold import TSNE as cuML_TSNE  # t-SNE GPU avec cuML
from sklearn.manifold import TSNE as SK_TSNE  # t-SNE CPU avec Scikit-Learn

# 🔹 Générer des features aléatoires (100000 samples, 2048 dimensions)
num_samples = 100000  # Nombre de points
num_features = 2048  # Nombre de dimensions
random_features = np.random.rand(num_samples, num_features).astype(np.float32)  # Float32 pour compatibilité cuML

# 🔹 Définition du perplexity (éviter l'erreur "perplexity must be less than n_samples")
perplexity_value = min(30, num_samples - 1)

# ---------------------------- #
# 🔥 Test t-SNE avec Scikit-Learn (CPU)
# ---------------------------- #
print("⏳ Exécution de t-SNE (Scikit-Learn - CPU)...")
start_time = time.time()
X_tsne_sklearn = SK_TSNE(n_components=2, perplexity=perplexity_value, random_state=42).fit_transform(random_features)
cpu_time = time.time() - start_time
print(f"✅ t-SNE avec Scikit-Learn terminé en {cpu_time:.2f} secondes.")

# ---------------------------- #
# ⚡ Test t-SNE avec cuML (GPU)
# ---------------------------- #
print("⏳ Exécution de t-SNE (cuML - GPU)...")
start_time = time.time()
X_tsne_cuml = cuML_TSNE(n_components=2, perplexity=perplexity_value, random_state=42).fit_transform(random_features)
gpu_time = time.time() - start_time
print(f"✅ t-SNE avec cuML terminé en {gpu_time:.2f} secondes.")

# ---------------------------- #
# 📊 Visualisation des résultats
# ---------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# 🔹 Plot t-SNE Scikit-Learn (CPU)
axes[0].scatter(X_tsne_sklearn[:, 0], X_tsne_sklearn[:, 1], cmap='viridis', alpha=0.6)
axes[0].set_title(f"t-SNE Scikit-Learn (CPU) - {cpu_time:.2f}s")
axes[0].set_xlabel("t-SNE Dim 1")
axes[0].set_ylabel("t-SNE Dim 2")

# 🔹 Plot t-SNE cuML (GPU)
axes[1].scatter(X_tsne_cuml[:, 0], X_tsne_cuml[:, 1], cmap='viridis', alpha=0.6)
axes[1].set_title(f"t-SNE cuML (GPU) - {gpu_time:.2f}s")
axes[1].set_xlabel("t-SNE Dim 1")
axes[1].set_ylabel("t-SNE Dim 2")

# 🔹 Sauvegarde des résultats
plt.savefig("tsne_comparison_cpu_vs_gpu.png", dpi=300)
plt.show()

# 🔹 Affichage des temps d'exécution
print(f"⏱ Comparaison des temps d'exécution :")
print(f"   - Scikit-Learn (CPU) : {cpu_time:.2f} secondes")
print(f"   - cuML (GPU)        : {gpu_time:.2f} secondes")
print(f"   💡 Accélération GPU : {cpu_time / gpu_time:.1f}x plus rapide !")
