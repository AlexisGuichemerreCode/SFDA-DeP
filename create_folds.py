import os
import pandas as pd
import csv

# === Chemins ===
csv_path = "/export/livia/home/vision/Aguichemerre/dataset_splits_diverse_normal_new.csv"
output_root = "/export/livia/home/vision/Aguichemerre/folds"
summary_file = os.path.join(output_root, "summary.csv")

# Charger le CSV
df = pd.read_csv(csv_path)

# Déterminer le label : tumor ou normal
df["label"] = df["stage"].apply(lambda x: "normal" if x == "negative" else "tumor")

# Fonction pour générer (image, label, annotation)
def make_row(row):
    split = row["split"]
    label = row["label"]
    img_path = f"images/{'training' if split in ['train','val'] else 'testing'}/{label}/{row['patient']}.tif"
    if label == "tumor":
        ann_path = f"annotations/{'training' if split in ['train','val'] else 'testing'}/{row['patient']}.xml"
    else:
        ann_path = ""  # pas d'annotation pour normal
    return [img_path, label, ann_path]

# Créer le dossier racine
os.makedirs(output_root, exist_ok=True)

# === Créer les folds ===
for center in sorted(df["center"].unique()):
    fold_dir = os.path.join(output_root, f"fold_{center}")
    os.makedirs(fold_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        subset = df[(df["center"] == center) & (df["split"] == split)]

        rows = [make_row(r) for _, r in subset.iterrows()]
        filename = os.path.join(fold_dir, f"{split}_s_0_f_{center}.csv")

        # Écriture CSV sans quotes ni backslash
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"✅ Écrit {len(rows)} lignes dans {filename}")

# === Créer un résumé global ===
summary = (
    df.groupby(["center", "split", "label"])
      .size()
      .reset_index(name="n_images")
      .pivot_table(index=["center", "split"], columns="label", values="n_images", fill_value=0)
      .reset_index()
)

summary.to_csv(summary_file, index=False)
print(f"\n📊 Résumé global écrit dans {summary_file}")
