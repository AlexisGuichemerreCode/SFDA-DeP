# Chemin des fichiers
input_file = "/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/visualization/tsne/GLAS/target_cam_performance_log.txt"  # Fichier source
output_file = "/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/visualization/tsne/GLAS/sort_target_cam_performance_log.txt"  # Fichier trié



# Lire le fichier et trier les lignes
with open(input_file, "r") as infile:
    lines = infile.readlines()

# Diviser les lignes en colonnes et trier par la 3e valeur (décroissant)
sorted_lines = sorted(
    lines, 
    key=lambda line: float(line.strip().split(",")[2]), 
    reverse=True
)

# Écrire les lignes triées dans un nouveau fichier
with open(output_file, "w") as outfile:
    outfile.writelines(sorted_lines)

print(f"Fichier trié créé : {output_file}")