import os
import torch
import numpy as np
import logging
import argparse
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)


def save_mean_stains(mean_he: dict, mean_maxC: dict, save_dir: str, dataset_target: str, dataset_reference: str):
    """
    Saves mean HE and maxC matrices into a specified directory.

    Args:
        mean_he (dict): Dictionary containing mean HE matrices per cluster.
        mean_maxC (dict): Dictionary containing mean maxC matrices per cluster.
        save_dir (str): Directory where the matrices will be saved.
        dataset_source (str): The dataset from which the stains originate.
        dataset_reference (str): The dataset used for distance comparison.
    """
    os.makedirs(save_dir, exist_ok=True)

    for cluster_id in mean_he:
        # Create a subdirectory for each cluster
        cluster_dir = os.path.join(save_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        # File name for the stain matrix in the cluster subdirectory
        file_name = f"{dataset_target}_wrt_{dataset_reference}.pt"
        save_path = os.path.join(cluster_dir, file_name)

        # Save the HE and maxC mean matrices
        torch.save({"mean_he": mean_he[cluster_id], "mean_maxC": mean_maxC[cluster_id]}, save_path)
        logging.info(f"Saved cluster {cluster_id} mean stain in {save_path}")

def find_closest_to_centroid(
    clustered_he, clustered_maxC, clustered_distances, centroids
):
    """
    Trouve le staining HE et la matrice maxC les plus proches du centroïde du cluster.

    Args:
        clustered_he (dict): Dictionnaire des HE matrices groupées par cluster.
        clustered_maxC (dict): Dictionnaire des maxC matrices groupées par cluster.
        clustered_distances (dict): Dictionnaire des distances aux centroïdes par cluster.
        centroids (np.ndarray): Centroïdes des clusters.

    Returns:
        closest_he (dict): Dictionnaire des HE matrices les plus proches des centroïdes.
        closest_maxC (dict): Dictionnaire des maxC matrices les plus proches des centroïdes.
    """
    closest_he = {}
    closest_maxC = {}

    for cluster_id in clustered_he.keys():
        if len(clustered_he[cluster_id]) == 0:
            logging.warning(f"Cluster {cluster_id} is empty, skipping...")
            continue

        # Récupérer les distances pré-calculées et les transformer en tensor si nécessaire
        distances = torch.tensor(clustered_distances[cluster_id], dtype=torch.float32)
        
        # Trouver l'indice du plus proche voisin au centroïde
        closest_idx = torch.argmin(torch.abs(distances - centroids[cluster_id]))

        # Récupérer les matrices HE et maxC correspondantes
        closest_he[cluster_id] = clustered_he[cluster_id][closest_idx]
        closest_maxC[cluster_id] = clustered_maxC[cluster_id][closest_idx]

    return closest_he, closest_maxC


def load_data(directory: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Loads HE matrices and maxC matrices from .pt files in a given directory.

    Args:
        directory (str): Path to the directory containing .pt files.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: Lists of HE matrices and maxC matrices.
    """
    he_matrices, maxC_matrices = [], []
    
    for filename in os.listdir(directory):
        if filename.endswith('.pt'):
            filepath = os.path.join(directory, filename)
            try:
                data = torch.load(filepath)
                if 'he_matrix' in data and 'maxC' in data:
                    he_matrices.append(data['he_matrix'])
                    maxC_matrices.append(data['maxC'])
                else:
                    logging.warning(f"File {filename} is missing 'he_matrix' or 'maxC'.")
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")

    return he_matrices, maxC_matrices




def compute_average_distances(source_stack: torch.Tensor, target_stack: torch.Tensor) -> np.ndarray:
    """
    Computes the average distance between each HE source matrix and the set of target HE matrices.

    Args:
        source_stack (torch.Tensor): HE source matrices (e.g., Camelyon).
        target_stack (torch.Tensor): HE target matrices (e.g., GLAS).

    Returns:
        np.ndarray: Array of average distances.
    """
    average_distances = []
    for source_matrix in source_stack:
        diff = source_matrix.unsqueeze(0) - target_stack
        distances = torch.norm(diff, dim=(1, 2))
        average_distances.append(distances.mean().item())
    
    return np.array(average_distances)

def apply_kmeans(average_distances: np.ndarray, num_clusters: int = 10) -> np.ndarray:
    """
    Applies K-Means clustering on the average distances.

    Args:
        average_distances (np.ndarray): Mean distances between HE matrices.
        num_clusters (int, optional): Number of clusters (default: 10).

    Returns:
        np.ndarray: Cluster labels.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(average_distances.reshape(-1, 1))  
    centroids = kmeans.cluster_centers_.flatten() 

    return labels, centroids

def group_by_cluster(
    he_matrices: List[torch.Tensor], 
    maxC_matrices: List[torch.Tensor], 
    clusters: np.ndarray, 
    average_distances: np.ndarray
) -> Tuple[Dict[int, List[torch.Tensor]], Dict[int, List[torch.Tensor]], Dict[int, List[float]]]:
    """
    Groups HE and maxC matrices by cluster, and stores their distances from the centroid.

    Args:
        he_matrices (List[torch.Tensor]): List of HE matrices.
        maxC_matrices (List[torch.Tensor]): List of maxC matrices.
        clusters (np.ndarray): Clustering labels.
        average_distances (np.ndarray): Mean distances between HE matrices.

    Returns:
        Tuple[
            Dict[int, List[torch.Tensor]],  # HE matrices grouped by cluster
            Dict[int, List[torch.Tensor]],  # maxC matrices grouped by cluster
            Dict[int, List[float]]          # Distances grouped by cluster
        ]
    """
    unique_clusters = np.unique(clusters)
    clustered_he = {i: [] for i in unique_clusters}
    clustered_maxC = {i: [] for i in unique_clusters}
    clustered_distances = {i: [] for i in unique_clusters}  # Nouveau dictionnaire pour stocker les distances

    for i, cluster in enumerate(clusters):
        clustered_he[cluster].append(he_matrices[i])
        clustered_maxC[cluster].append(maxC_matrices[i])
        clustered_distances[cluster].append(average_distances[i])  # Stocke la distance

    return clustered_he, clustered_maxC, clustered_distances

def compute_cluster_means(
    clustered_he: Dict[int, List[torch.Tensor]], 
    clustered_maxC: Dict[int, List[torch.Tensor]]
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Computes the mean HE and maxC matrices for each cluster.

    Args:
        clustered_he (Dict[int, List[torch.Tensor]]): HE matrices grouped by cluster.
        clustered_maxC (Dict[int, List[torch.Tensor]]): maxC matrices grouped by cluster.

    Returns:
        Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]: Mean HE and maxC matrices by cluster.
    """
    mean_he, mean_maxC = {}, {}

    for cluster_id in clustered_he:
        mean_he[cluster_id] = torch.stack(clustered_he[cluster_id]).numpy().mean(axis=0)
        mean_maxC[cluster_id] = torch.stack(clustered_maxC[cluster_id]).numpy().mean(axis=0)

    return mean_he, mean_maxC
    



# Define data directories
#CAMELYON_DIR = "/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/data_staining/camelyon_train_stain"
#GLAS_DIR = "/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/data_staining/glas_train_stain"

def main(dataset1_dir: str, dataset2_dir: str, num_clusters: int, dataset1: str, dataset2: str):
    logging.info("Loading data...")
    dataset1_he, dataset1_maxC = load_data(dataset1_dir)
    dataset2_he, dataset2_maxC = load_data(dataset2_dir)

    logging.info(f"{len(dataset1_he)} HE matrices loaded from {dataset1}.")
    logging.info(f"{len(dataset2_he)} HE matrices loaded from {dataset2}.")

    logging.info("Stacking data...")
    dataset1_stack = torch.stack(dataset1_he)
    dataset2_stack = torch.stack(dataset2_he)

    logging.info("Computing average distances from dataset1 to dataset2...")
    average_distances_1_to_2 = compute_average_distances(dataset1_stack, dataset2_stack)

    logging.info("Computing average distances from dataset2 to dataset1...")
    average_distances_2_to_1 = compute_average_distances(dataset2_stack, dataset1_stack)

    logging.info(f"Applying K-Means clustering with {num_clusters} clusters (dataset1 to dataset2)...")
    clusters_1_to_2, centroid_1_2 = apply_kmeans(average_distances_1_to_2, num_clusters=num_clusters)

    logging.info(f"Applying K-Means clustering with {num_clusters} clusters (dataset2 to dataset1)...")
    clusters_2_to_1, centroid_2_1 = apply_kmeans(average_distances_2_to_1, num_clusters=num_clusters)

    logging.info("Grouping matrices by cluster (dataset1 to dataset2)...")
    clustered_he_1_to_2, clustered_maxC_1_to_2, clustered_distances_1_to_2 = group_by_cluster(dataset1_he, dataset1_maxC, clusters_1_to_2, average_distances_1_to_2)

    logging.info("Grouping matrices by cluster (dataset2 to dataset1)...")
    clustered_he_2_to_1, clustered_maxC_2_to_1, clustered_distances_2_to_1 = group_by_cluster(dataset2_he, dataset2_maxC, clusters_2_to_1, average_distances_2_to_1)

    #logging.info("Computing cluster means (dataset1 to dataset2)...")
    #mean_he_1_to_2, mean_maxC_1_to_2 = compute_cluster_means(clustered_he_1_to_2, clustered_maxC_1_to_2)

    #logging.info("Computing cluster means (dataset2 to dataset1)...")
    #mean_he_2_to_1, mean_maxC_2_to_1 = compute_cluster_means(clustered_he_2_to_1, clustered_maxC_2_to_1)

    logging.info("Finding closest sample to cluster centroid (dataset1 to dataset2)...")
    closest_he_1_to_2, closest_maxC_1_to_2 = find_closest_to_centroid(clustered_he_1_to_2, clustered_maxC_1_to_2, clustered_distances_1_to_2, centroid_1_2)

    logging.info("Finding closest sample to cluster centroid (dataset2 to dataset1)...")
    closest_he_2_to_1, closest_maxC_2_to_1 = find_closest_to_centroid(clustered_he_2_to_1, clustered_maxC_2_to_1,clustered_distances_2_to_1, centroid_2_1)


    for cluster_id in closest_he_1_to_2:
        logging.info(f"Cluster {cluster_id} (dataset1 -> dataset2): HE mean shape = {closest_he_1_to_2[cluster_id].shape}, maxC mean shape = {closest_maxC_1_to_2[cluster_id].shape}")

    for cluster_id in closest_he_2_to_1:
        logging.info(f"Cluster {cluster_id} (dataset2 -> dataset1): HE mean shape = {closest_he_2_to_1[cluster_id].shape}, maxC mean shape = {closest_maxC_2_to_1[cluster_id].shape}")

    # Define save directories
    save_dir_1_to_2 = os.path.join("data_staining", "closest_stains", f"{dataset1}_wrt_{dataset2}_{num_clusters}")
    save_dir_2_to_1 = os.path.join("data_staining", "closest_stains", f"{dataset2}_wrt_{dataset1}_{num_clusters}")

    logging.info(f"Saving closest stainings for {dataset1} -> {dataset2} in {save_dir_1_to_2}...")
    save_mean_stains(closest_he_1_to_2, closest_maxC_1_to_2, save_dir_1_to_2, dataset1, dataset2)

    logging.info(f"Saving closest stainings for {dataset2} -> {dataset1} in {save_dir_2_to_1}...")
    save_mean_stains(closest_he_2_to_1, closest_maxC_2_to_1, save_dir_2_to_1, dataset2, dataset1)
    #logging.info(f"Saving mean stains for {dataset1} -> {dataset2} in {save_dir_1_to_2}...")
    #save_mean_stains(mean_he_1_to_2, mean_maxC_1_to_2, save_dir_1_to_2, dataset1, dataset2)

    #logging.info(f"Saving mean stains for {dataset2} -> {dataset1} in {save_dir_2_to_1}...")
    #save_mean_stains(mean_he_2_to_1, mean_maxC_2_to_1, save_dir_2_to_1, dataset2, dataset1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering HE matrices using K-Means")
    parser.add_argument("--dataset1", type=str, required=True, help="Name of the first dataset")
    parser.add_argument("--split_dataset1", type=str, required=True, help="Split name for the first dataset (e.g., 'train')")
    parser.add_argument("--dataset2", type=str, required=True, help="Name of the second dataset")
    parser.add_argument("--split_dataset2", type=str, required=True, help="Split name for the second dataset (e.g., 'train')")
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of clusters for K-Means (default: 10)")

    args = parser.parse_args()
    current_path = os.getcwd()
    dataset1_dir = os.path.join(current_path, "data_staining", f"{args.dataset1}_{args.split_dataset1}_stain")
    dataset2_dir = os.path.join(current_path, "data_staining", f"{args.dataset2}_{args.split_dataset2}_stain")


    main(dataset1_dir, dataset2_dir, args.num_clusters, args.dataset1, args.dataset2)











# camelyon_dir = '/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/data_staining/camelyon_train_stain'
# glas_dir = '/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/data_staining/glas_train_stain'


# camelyon_he = []
# camelyon_maxC = []
# for filename in os.listdir(camelyon_dir):
#     if filename.endswith('.pt'):
#         filepath = os.path.join(camelyon_dir, filename)
#         data = torch.load(filepath)
        
#         if 'he_matrix' and 'maxC' in data:
#             camelyon_he.append(data['he_matrix'])
#             camelyon_maxC.append(data['maxC'])
#         else:
#             print(f"HE matrix is missing in {filename}")

# glas_he = []
# glas_maxC = []
# for filename in os.listdir(glas_dir):
#     if filename.endswith('.pt'):
#         filepath = os.path.join(glas_dir, filename)
#         data = torch.load(filepath)
#         if 'he_matrix' in data:
#             glas_he.append(data['he_matrix'])
#             glas_maxC.append(data['maxC'])
#         else:
#             print(f"HE matrix is missing in {filename}")


# camelyon_stack = torch.stack(camelyon_he)  # (N_camelyon, H, W) or (N_camelyon, C, H, W)
# glas_stack = torch.stack(glas_he)  # (N_glas, H, W) or (N_glas, C, H, W)


# average_distances = []
# for camelyon_he_matrix in camelyon_stack:
#     diff = camelyon_he_matrix.unsqueeze(0) - glas_stack
#     distances = torch.norm(diff, dim=(1, 2))  
#     average_distances.append(distances.mean().item())  


# average_distances = np.array(average_distances)

# K = 10
# kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
# clusters = kmeans.fit_predict(average_distances.reshape(-1, 1))


# clustered_he = {i: [] for i in np.unique(clusters)}
# clustered_maxC = {i: [] for i in np.unique(clusters)}


# for i, cluster in enumerate(clusters):
#     clustered_he[cluster].append(camelyon_he[i])
#     clustered_maxC[cluster].append(camelyon_maxC[i])


# mean_he = {}
# mean_maxC = {}

# for cluster_id in clustered_he:
#     shapes = [tensor.shape for tensor in clustered_he[cluster_id]]
#     print(f"Cluster {cluster_id}: Shapes = {set(shapes)}")

# for cluster_id in clustered_he:
#     mean_he[cluster_id] = torch.stack(clustered_he[cluster_id]).numpy().mean(axis=0)
#     mean_maxC[cluster_id] = torch.stack(clustered_maxC[cluster_id]).numpy().mean(axis=0)


# for cluster_id in mean_he:
#     print(f"Cluster {cluster_id}: HE mean = {mean_he[cluster_id]}, maxC mean = {mean_maxC[cluster_id]}")



# camelyon_stack = torch.stack(camelyon_he)  # (N_camelyon, H, W) or (N_camelyon, C, H, W)
# glas_stack = torch.stack(glas_he)  # (N_glas, H, W) or (N_glas, C, H, W)


# average_distances = []
# # for glas_he_matrix in glas_stack:
# #     diff = glas_he_matrix.unsqueeze(0) - camelyon_stack
# #     distances = torch.norm(diff, dim=(1, 2))  
# #     average_distances.append(distances.mean().item())  


# for camelyon_he_matrix in camelyon_stack:
#     diff = camelyon_he_matrix.unsqueeze(0) - glas_stack
#     distances = torch.norm(diff, dim=(1, 2))  
#     average_distances.append(distances.mean().item())  

# plt.hist(average_distances, bins=200, edgecolor='black')
# plt.xlabel("Average distance")
# plt.ylabel("Frequency")
# plt.title("Average distance distribution between GLAS and CAMELYON")
# plt.savefig("distance histo staining.png", dpi=300, bbox_inches='tight')

# plt.show()
