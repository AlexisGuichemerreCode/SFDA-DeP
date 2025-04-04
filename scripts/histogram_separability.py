import matplotlib.pyplot as plt
import numpy as np
import matplotlib


def plot_histograms(file_path, save_path):
    # Dictionnaire pour stocker les valeurs selon la méthode
    methods = {
        "DeepMIL": [], "PixelCAMDL": [],
        "GradCAMpp": [], "PixelCAMGC": [],
        "LayerCAM": [], "PixelCAMLC": [],
        "SAT": [], "PixelCAMSAT": []
    }
    
    # Lire le fichier
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            
            method = parts[1]
            value = float(parts[3])  # 4ème position (index 3 en 0-based index)
            
            if method in methods:
                methods[method].append(value)
    
    matplotlib.rcParams['font.family'] = 'Times New Roman'

    # Mapping des noms de méthodes pour la légende
    legend_mapping = {
        "GradCAMpp": "GradCAM++",
        "PixelCAMGC": "PixelCAM",
        "DeepMIL": "DeepMIL",
        "PixelCAMDL": "PixelCAM",
        "LayerCAM": "LayerCAM",
        "PixelCAMLC": "PixelCAM",
        "SAT": "SAT",
        "PixelCAMSAT": "PixelCAM"
    }

    # Couples de méthodes
    method_pairs = [("DeepMIL", "PixelCAMDL"), ("GradCAMpp", "PixelCAMGC"), 
                    ("LayerCAM", "PixelCAMLC"), ("SAT", "PixelCAMSAT")]

    # Créer une figure avec 4 sous-graphiques alignés en ligne
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)  # sharey=True pour aligner les axes Y

    # Tracer les histogrammes dans chaque sous-figure
    for ax, (method1, method2) in zip(axes, method_pairs):
        if methods[method1] and methods[method2]:  # Vérifier si on a des valeurs
            all_values = methods[method1] + methods[method2]
            
            # Augmenter le nombre de bins pour une meilleure résolution
            bins = np.histogram_bin_edges(all_values, bins=50)  
            
            # Calcul des histogrammes
            h1, edges1 = np.histogram(methods[method1], bins=bins)
            h2, edges2 = np.histogram(methods[method2], bins=bins)

            # Tracer uniquement la courbe des sommets des histogrammes
            ax.stairs(h1, edges1, label=legend_mapping[method1], linestyle='-', linewidth=2)
            ax.stairs(h2, edges2, label=legend_mapping[method2], linestyle='-', linewidth=2)

            ax.set_xlabel("Separability", fontsize=20)  # Ajouter un label X plus lisible
            ax.set_ylabel("Frequency", fontsize=20)
            #ax.set_xlabel("")  # Supprime le label de l'axe X
            #ax.set_ylabel("")  # Supprime le label de l'axe Y
            ax.set_title("")   # Supprime le titre
            ax.legend(fontsize=26, loc="upper right", frameon=False)  # Taille de la légende

    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout()

    # Sauvegarder l'image contenant les 4 histogrammes
    save_filename = f"{save_path}/all_histograms_target.png"
    plt.savefig(save_filename, bbox_inches='tight', dpi=300)

# Exemple d'appel : plot_histograms("chemin_vers_ton_fichier.txt")

input_file = "/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/visualization/tsne/GLAS/target_cam_performance_log.txt"  
output_file = "/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/visualization/tsne/GLAS/"
plot_histograms(input_file,output_file)

