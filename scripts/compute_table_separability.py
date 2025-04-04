def compute_means_from_file(input_file, output_file):
    # Dictionnaire pour stocker les données des méthodes
    method_data = {}
    methods = ['DeepMIL', 'PixelCAM DL', 'GradCAMpp', 'PixelCAM GC',
               'LayerCAM', 'PixelCAM LC', 'SAT', 'PixelCAM SAT']
    #methods = ['PixelCAM DL', 'PixelCAM GC', 'PixelCAM LC','PixelCAM SAT']

    # Lecture du fichier ligne par ligne
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Traitement des lignes
    for line in lines:
        parts = line.strip().split(",")
        # if len(parts) < 6:
        #     continue
        # method = parts[1]
        # value2 = float(parts[3])
        # class_label = int(parts[5])

        if len(parts) < 4:
            continue
        method = parts[1]
        value2 = float(parts[2])
        class_label = int(parts[4])

        if method not in method_data:
            method_data[method] = {'class_0': [], 'class_1': []}

        if class_label == 0:
            method_data[method]['class_0'].append(value2)
        elif class_label == 1:
            method_data[method]['class_1'].append(value2)

    # Calcul des moyennes
    method_stats = {}
    for method, values in method_data.items():
        mean_class_0 = sum(values['class_0']) / len(values['class_0']) if values['class_0'] else 0
        mean_class_1 = sum(values['class_1']) / len(values['class_1']) if values['class_1'] else 0
        overall_mean = (sum(values['class_0']) + sum(values['class_1'])) / \
                       (len(values['class_0']) + len(values['class_1'])) if (values['class_0'] or values['class_1']) else 0

        method_stats[method] = {
            'Mean Class 0': mean_class_0,
            'Mean Class 1': mean_class_1,
            'Overall Mean': overall_mean
        }

    # Sauvegarde des résultats dans un fichier texte
    with open(output_file, 'w') as f:
        for method, stats in method_stats.items():
            f.write(f"Method: {method}\n")
            for stat_name, value in stats.items():
                f.write(f"  {stat_name}: {value:.6f}\n")
            f.write("\n")


# Exemple d'utilisation
input_file = "/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/visualization/tsne/CAMELYON512/target_cam_performance_log.txt"  
output_file = "/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/visualization/tsne/CAMELYON512/target_average_class_separability_log.txt"
compute_means_from_file(input_file, output_file)

print(f"Les résultats ont été sauvegardés dans le fichier '{output_file}'.")
