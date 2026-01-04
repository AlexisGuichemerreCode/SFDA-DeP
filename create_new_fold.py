import os
import shutil
import random
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
ROOT = Path("folds/wsol-done-right-splits/CAMELYON512")
SOURCE_FOLD = ROOT / "fold-0"
OUTPUT_PREFIX = "fold-"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

FILES = [
    "class_labels.txt",
    "image_ids.txt",
    "image_sizes.txt",
    "localization.txt",
]

# ============================================================
# UTILS
# ============================================================
def read_lines(path):
    with open(path, "r") as f:
        return f.readlines()

def write_lines(path, lines):
    with open(path, "w") as f:
        f.writelines(lines)

# ============================================================
# LOAD FULL TRAIN SET
# ============================================================
train_src = SOURCE_FOLD / "train"

data = {fname: read_lines(train_src / fname) for fname in FILES}

labels = []
for line in data["class_labels.txt"]:
    parts = line.strip().split(",")
    if len(parts) != 2:
        raise ValueError(f"Malformed line in class_labels.txt: {line}")
    labels.append(int(parts[1]))

indices_normal = [i for i, y in enumerate(labels) if y == 0]
indices_cancer = [i for i, y in enumerate(labels) if y == 1]

n_total = len(labels)

print(f"Total images: {n_total}")
print(f"Normals: {len(indices_normal)} | Cancer: {len(indices_cancer)}")

# ============================================================
# CREATE FOLDS
# ============================================================

PERCENTS = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


for p in PERCENTS:
    ratio = p / 100.0
    fold_id = 100 + p
    fold_name = f"fold-{fold_id}"

    fold_dst = ROOT / fold_name

    print(f"\nCreating {fold_name} ({int(ratio*100)}%)")

    # --- Create directories
    for split in ["train", "test", "valcl", "valpx"]:
        (fold_dst / split).mkdir(parents=True, exist_ok=True)

    # --- Copy non-train splits as-is
    for split in ["test", "valcl", "valpx"]:
        src = SOURCE_FOLD / split
        dst = fold_dst / split
        for fname in FILES:
            shutil.copy(src / fname, dst / fname)

    # --- Compute per-class counts
    n_per_class = int((ratio * n_total) / 2)

    n_norm = min(n_per_class, len(indices_normal))
    n_canc = min(n_per_class, len(indices_cancer))

    sampled_norm = random.sample(indices_normal, n_norm)
    sampled_canc = random.sample(indices_cancer, n_canc)

    selected_indices = sorted(sampled_norm + sampled_canc)

    print(f"  Selected: {len(selected_indices)} "
          f"(Normal={n_norm}, Cancer={n_canc})")

    # --- Write new train files
    train_dst = fold_dst / "train"
    for fname in FILES:
        selected_lines = [data[fname][i] for i in selected_indices]
        write_lines(train_dst / fname, selected_lines)
