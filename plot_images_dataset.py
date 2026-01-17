import os
from PIL import Image
import matplotlib.pyplot as plt
import random

# ================= CONFIG =================
DATASET_ROOT = "/export/livia/home/vision/Aguichemerre/datasets/CAMELYON17_512"

SPLITS_ROOT = "/export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/folds/wsol-done-right-splits/CAMELYON17_512"


FOLDS = [0, 1, 2, 3, 4]
SPLITS = ["train", "valpx", "test"]
LABEL_FILE = "class_labels.txt"
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif")
# ==========================================


SEED = 342
random.seed(SEED)
# ==================================================


def select_random_images(label_file):
    """
    Sélectionne UNE image aléatoire par classe (0 et 1),
    avec seed fixe → reproductible.
    """
    images = {0: [], 1: []}

    with open(label_file, "r") as f:
        for line in f:
            rel_path, label = line.strip().split(",")
            label = int(label)

            if rel_path.lower().endswith(IMG_EXT):
                full_path = os.path.join(DATASET_ROOT, rel_path)
                images[label].append(full_path)

    if len(images[0]) == 0 or len(images[1]) == 0:
        raise RuntimeError(f"Missing class in {label_file}")

    return {
        0: random.choice(images[0]),
        1: random.choice(images[1]),
    }


def build_figure_for_split(split):
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))

    for col, fold in enumerate(FOLDS):
        label_file = os.path.join(
            SPLITS_ROOT,
            f"fold-{fold}",
            split,
            LABEL_FILE
        )

        selected = select_random_images(label_file)

        img_cancer = Image.open(selected[1]).convert("RGB")
        img_normal = Image.open(selected[0]).convert("RGB")

        axes[0, col].imshow(img_cancer)
        axes[0, col].set_title(f"Center {fold}", fontsize=12)
        axes[0, col].axis("off")

        axes[1, col].imshow(img_normal)
        axes[1, col].axis("off")

    # ===== LABELS DE LIGNES (ROBUSTES) =====
    fig.text(0.015, 0.72, "Cancer", va="center", ha="center",
             rotation=90, fontsize=16)

    fig.text(0.015, 0.28, "Normal", va="center", ha="center",
             rotation=90, fontsize=16)

    plt.tight_layout(rect=[0.05, 0.02, 1, 1])
    out_name = f"camelyon17_{split}_centers_random.png"
    plt.savefig(out_name, dpi=300)
    plt.close()

    print(f"[OK] Saved {out_name}")


# ===================== BOUCLE PRINCIPALE =====================
for split in SPLITS:
    build_figure_for_split(split)