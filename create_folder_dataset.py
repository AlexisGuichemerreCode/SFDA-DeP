import csv
import random
from collections import defaultdict


random.seed(42)


# chemins des fichiers
stages_file = "/export/livia/home/vision/Aguichemerre/stages.csv"
annot_file = "/export/livia/home/vision/Aguichemerre/annotated_by_center.csv"
output_file = "/export/livia/home/vision/Aguichemerre/dataset_splits_diverse_normal.csv"

random.seed(42)  

# === 1. Lire stages.csv (pour normales) ===
stages = []
with open(stages_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["patient"].endswith(".tif"):
            patient_id = row["patient"].split("_node")[0]
            stages.append({
                "center": int(row["center"]),
                "patient": row["patient"].replace(".tif", ""),
                "stage": row["stage"],
                "patient_id": patient_id
            })

# === 2. Lire annotated_by_center.csv (pour tumorales) ===
annot = []
with open(annot_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        patient_id = row["patient"].split("_node")[0]
        annot.append({
            "center": int(row["center"]),
            "patient": row["patient"],
            "stage": row["stage"],
            "patient_id": patient_id
        })

# Grouper tumorales et normales par centre
annot_by_center = defaultdict(list)
for r in annot:
    annot_by_center[r["center"]].append(r)

neg_by_center = defaultdict(list)
for r in stages:
    if r["stage"] == "negative":
        neg_by_center[r["center"]].append(r)



def find_full_negative_patients(data):
    """
    Retourne l'ensemble des patients dont les 5 nodes (0 à 4) sont négatifs.
    """
    patients = defaultdict(list)
    for row in data:
        patients[row["patient_id"]].append(row)

    full_neg = []
    for pid, rows in patients.items():
        node_names = [r["patient"].split("_")[-1] for r in rows]  # ex: "node_0"
        stages = [r["stage"] for r in rows]
        if all(s == "negative" for s in stages) and set(node_names) >= {f"node_{i}" for i in range(5)}:
            full_neg.append(pid)
    return full_neg



# === 3. Fonction de split avec priorité train → test → val ===
def split_ranked(data, n_train=7, n_val=1, n_test=2):
    # Grouper par patient
    patients = defaultdict(list)
    for row in data:
        patients[row["patient_id"]].append(row)

    # Trier les patients par nombre d'images décroissant
    sorted_patients = sorted(patients.items(), key=lambda x: len(x[1]), reverse=True)

    splits = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for pid, rows in sorted_patients:
        n = len(rows)

        # Essayer train d'abord
        if counts["train"] + n <= n_train:
            for r in rows:
                r["split"] = "train"
                splits["train"].append(r)
            counts["train"] += n
        # Puis test
        elif counts["test"] + n <= n_test:
            for r in rows:
                r["split"] = "test"
                splits["test"].append(r)
            counts["test"] += n
        # Enfin val
        elif counts["val"] + n <= n_val:
            for r in rows:
                r["split"] = "val"
                splits["val"].append(r)
            counts["val"] += n

        # Stop si quotas atteints
        if counts["train"] == n_train and counts["val"] == n_val and counts["test"] == n_test:
            break

    return splits["train"] + splits["val"] + splits["test"]


def split_ranked_diverse(data, n_train=7, n_val=1, n_test=2):
    # Grouper par patient
    patients = defaultdict(list)
    for row in data:
        patients[row["patient_id"]].append(row)

    # Identifier patients full négatifs
    full_neg = find_full_negative_patients(data)

    # Séparer patients full neg et autres
    full_patients = [(pid, patients[pid]) for pid in full_neg if pid in patients]
    other_patients = [(pid, rows) for pid, rows in patients.items() if pid not in full_neg]

    # Mélanger pour diversifier
    random.shuffle(full_patients)
    random.shuffle(other_patients)

    # Construire la liste des patients candidats (full neg d'abord, puis les autres)
    ordered_patients = full_patients + other_patients

    splits = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for pid, rows in ordered_patients:
        n = len(rows)

        # Attribuer en priorité train puis test puis val
        if counts["train"] + n <= n_train:
            split = "train"
        elif counts["test"] + n <= n_test:
            split = "test"
        elif counts["val"] + n <= n_val:
            split = "val"
        else:
            continue

        for r in rows:
            r["split"] = split
            splits[split].append(r)
        counts[split] += n

        # Si quotas atteints, on arrête
        if counts["train"] == n_train and counts["val"] == n_val and counts["test"] == n_test:
            break

    return splits["train"] + splits["val"] + splits["test"]



# === 4. Construire les splits ===
final_rows = []
for center in range(5):
    tumor_split = split_ranked(annot_by_center[center], 7, 1, 2)
    normal_split = split_ranked_diverse(neg_by_center[center], 7, 1, 2)
    final_rows.extend(tumor_split + normal_split)

# === 5. Sauvegarde CSV final ===
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["center", "patient", "stage", "split", "patient_id"])
    writer.writeheader()
    writer.writerows(final_rows)

print(f"✅ Fichier généré : {output_file}")