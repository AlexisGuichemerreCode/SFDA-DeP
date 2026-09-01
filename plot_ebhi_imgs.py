import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# Actual EBHI dataset
DATASET_ROOT = Path(
    "/projets/Aguichemerre/datasets/EBHI"
)

# WSOL folds / metadata
FOLDS_ROOT = Path(
    "/export/livia/home/vision/Aguichemerre/"
    "SFDA-DeP/folds/wsol-done-right-splits/EBHI"
)

# Output figures
OUTPUT_ROOT = Path(
    "/export/livia/home/vision/Aguichemerre/"
    "SFDA-DeP/figures/ebhi_samples"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Split directory names used in the folds
SPLITS = {
    "train": "train",
    "val": "valcl",
    "test": "test",
}

# ------------------------------------------------------------
# Fold meaning
# ------------------------------------------------------------
#
# fold-0  : source
# fold-1  : clean target
# fold-2  : stain_light
# fold-3  : stain_heavy
# fold-4  : dust
# fold-5  : air_bubble
# fold-6  : defocus_blur
# fold-7  : motion_blur
# fold-8  : gaussian_noise
# fold-9  : shot_noise
# fold-10 : brightness
# fold-11 : contrast
#
# Therefore the target-shift figure contains:
# 1 clean target + 10 corruptions = 11 image columns.
# ------------------------------------------------------------

SOURCE_FOLD = 0
CLEAN_TARGET_FOLD = 1

CORRUPTION_FOLDS = [
    (2, "stain_light", "Stain-light", "Staining"),
    (3, "stain_heavy", "Stain-heavy", "Staining"),
    (4, "dust", "Dust", "Contamination"),
    (5, "air_bubble", "Air bubble", "Contamination"),
    (6, "defocus_blur", "Defocus blur", "Blurring"),
    (7, "motion_blur", "Motion blur", "Blurring"),
    (8, "gaussian_noise", "Gaussian noise", "Noise"),
    (9, "shot_noise", "Shot noise", "Noise"),
    (10, "brightness", "Brightness", "Illumination"),
    (11, "contrast", "Contrast", "Illumination"),
]

# Source figure:
# 15 images per class, each shown with its GT mask.
N_SOURCE_IMAGES_PER_CLASS = 15

# Target-shift figure:
# 2 clean images per class, and the SAME image is shown
# under every corruption.
N_TARGET_IMAGES_PER_CLASS = 2

SEED = 42
DPI = 300


# ============================================================
# BASIC HELPERS
# ============================================================

def get_fold_dir(fold_number):
    """Return fold directory, supporting several naming styles."""

    candidates = [
        FOLDS_ROOT / f"fold-{fold_number}",
        FOLDS_ROOT / f"fold_{fold_number}",
        FOLDS_ROOT / f"fold{fold_number}",
        FOLDS_ROOT / str(fold_number),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"\nCould not find fold {fold_number}."
        "\nTried:"
        + "".join(f"\n  {p}" for p in candidates)
    )


def get_split_dir(fold_number, split_name):
    """Return train / val / test metadata directory for a fold."""

    fold_dir = get_fold_dir(fold_number)
    split_dir = fold_dir / SPLITS[split_name]

    # Fallback if validation directory is called "val"
    if not split_dir.exists() and split_name == "val":
        alternative = fold_dir / "val"
        if alternative.exists():
            split_dir = alternative

    if not split_dir.exists():
        raise FileNotFoundError(
            f"\nSplit directory not found:"
            f"\n  fold={fold_number}"
            f"\n  split={split_name}"
            f"\n  expected={split_dir}"
        )

    return split_dir


def get_dataset_path(relative_path):
    """Convert path stored in metadata to an actual dataset path."""

    relative_path = relative_path.strip()
    path = DATASET_ROOT / relative_path

    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset file not found:"
            f"\n  metadata path: {relative_path}"
            f"\n  expected path: {path}"
        )

    return path


def clean_axis(ax):
    """Remove ticks, labels and borders."""

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)


def add_text_axis(
    fig,
    gs_cell,
    text,
    fontsize=10,
    rotation=0,
    fontweight="normal",
):
    """Create a GridSpec cell containing centered text only."""

    ax = fig.add_subplot(gs_cell)
    ax.axis("off")

    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        rotation=rotation,
        fontweight=fontweight,
        transform=ax.transAxes,
    )

    return ax


def save_figure(fig, output_name):
    """Save both PNG and PDF versions."""

    png_path = OUTPUT_ROOT / f"{output_name}.png"
    pdf_path = OUTPUT_ROOT / f"{output_name}.pdf"

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        pdf_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print()


# ============================================================
# METADATA PARSING
# ============================================================

def read_class_labels(split_dir):
    """
    Read class_labels.txt.

    Expected examples:

        Adenocarcinoma/image/GTx....png,5
        Low-grade IN/image/GTx....png,2

    or for corruptions:

        corruptions/stain_light/Polyp/image/GTx....png,1
    """

    file_path = split_dir / "class_labels.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"class_labels.txt not found: {file_path}"
        )

    entries = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if "," in line:
                image_id, label = line.rsplit(",", 1)
            elif "\t" in line:
                image_id, label = line.rsplit("\t", 1)
            else:
                image_id, label = line.rsplit(maxsplit=1)

            image_id = image_id.strip()
            label = int(label.strip())

            class_name = extract_class_name(image_id)

            entries.append(
                {
                    "image_id": image_id,
                    "label": label,
                    "class_name": class_name,
                    "basename": Path(image_id).name,
                }
            )

    return entries


def read_localization(split_dir):
    """
    Read localization.txt.

    Expected EBHI format:

        Adenocarcinoma/image/file.png,
        Adenocarcinoma/label/file.png,

    i.e.:
        image_path,mask_path,

    Returns:
        dict[image_id] = mask_id
    """

    file_path = split_dir / "localization.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"localization.txt not found: {file_path}"
        )

    localization = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]

            if len(parts) < 2:
                continue

            image_id = parts[0]
            mask_id = parts[1]

            if image_id and mask_id:
                localization[image_id] = mask_id

    return localization


def extract_class_name(image_id):
    """
    Extract the EBHI class name from either clean or corrupted paths.

    Clean:
        Adenocarcinoma/image/file.png
        Low-grade IN/image/file.png

    Corrupted:
        corruptions/stain_light/Adenocarcinoma/image/file.png
        corruptions/dust/Polyp/image/file.png
    """

    parts = Path(image_id).parts

    if "image" not in parts:
        raise ValueError(
            f"Cannot infer class from image path: {image_id}"
        )

    image_idx = parts.index("image")

    if image_idx == 0:
        raise ValueError(
            f"Cannot infer class from image path: {image_id}"
        )

    return parts[image_idx - 1]


def discover_classes(split_name="train"):
    """
    Discover every EBHI class from source fold-0 and sort by
    numerical class label.

    This avoids hardcoding the six class names.
    """

    split_dir = get_split_dir(SOURCE_FOLD, split_name)
    entries = read_class_labels(split_dir)

    class_to_label = {}

    for entry in entries:
        class_to_label.setdefault(
            entry["class_name"],
            entry["label"],
        )

    classes = sorted(
        class_to_label.keys(),
        key=lambda x: class_to_label[x],
    )

    print("Discovered EBHI classes:")
    for class_name in classes:
        print(
            f"  label={class_to_label[class_name]} "
            f"-> {class_name}"
        )
    print()

    return classes


# ============================================================
# SOURCE SAMPLING: FOLD-0 + MASKS
# ============================================================

def get_source_samples(
    split_name,
    class_name,
    n_images,
    rng,
):
    """
    Sample source images for one EBHI class.

    Only images that:
      1. belong to the requested class,
      2. have an entry in localization.txt,
      3. have both image and mask files on disk,
    are considered.
    """

    split_dir = get_split_dir(
        SOURCE_FOLD,
        split_name,
    )

    entries = read_class_labels(split_dir)
    localization = read_localization(split_dir)

    candidates = []

    for entry in entries:
        if entry["class_name"] != class_name:
            continue

        image_id = entry["image_id"]

        if image_id not in localization:
            continue

        mask_id = localization[image_id]

        image_path = DATASET_ROOT / image_id
        mask_path = DATASET_ROOT / mask_id

        if image_path.exists() and mask_path.exists():
            candidates.append(
                (image_id, mask_id)
            )

    if len(candidates) == 0:
        raise RuntimeError(
            f"\nNo source image/mask pairs found."
            f"\nClass: {class_name}"
            f"\nSplit: {split_name}"
        )

    n = min(n_images, len(candidates))

    if n < n_images:
        print(
            f"WARNING: {class_name} / {split_name}: "
            f"requested {n_images}, only {n} available."
        )

    return rng.sample(candidates, n)


# ============================================================
# SOURCE FIGURE
# ============================================================

def create_source_figure(
    split_name,
    classes,
    n_images_per_class=N_SOURCE_IMAGES_PER_CLASS,
):
    """
    Create source fold-0 figure.

    Each class occupies two columns:

        Adenocarcinoma        Polyp        ...
        Image | Mask        Image | Mask

          x      GT           x      GT
          x      GT           x      GT
          ...
    """

    seed_offset = {
        "train": 0,
        "val": 100,
        "test": 200,
    }

    rng = random.Random(
        SEED + seed_offset[split_name]
    )

    samples = {}

    for class_name in classes:
        samples[class_name] = get_source_samples(
            split_name=split_name,
            class_name=class_name,
            n_images=n_images_per_class,
            rng=rng,
        )

        print(
            f"SOURCE | {split_name:5s} | "
            f"{class_name:20s} | "
            f"{len(samples[class_name])} pairs"
        )

    # Same number of rows for every class so the grid stays rectangular.
    n_rows = min(
        len(samples[class_name])
        for class_name in classes
    )

    n_classes = len(classes)
    n_cols = n_classes * 2

    fig_width = 18.0
    panel_size = fig_width / n_cols

    class_title_ratio = 0.28
    subtitle_ratio = 0.18

    fig_height = panel_size * (
        n_rows
        + class_title_ratio
        + subtitle_ratio
    )

    fig = plt.figure(
        figsize=(fig_width, fig_height)
    )

    gs = fig.add_gridspec(
        nrows=n_rows + 2,
        ncols=n_cols,
        height_ratios=(
            [class_title_ratio, subtitle_ratio]
            + [1.0] * n_rows
        ),
        wspace=0.02,
        hspace=0.02,
        left=0.002,
        right=0.998,
        bottom=0.002,
        top=0.998,
    )

    # --------------------------------------------------------
    # Class names
    # --------------------------------------------------------

    for class_idx, class_name in enumerate(classes):
        image_col = class_idx * 2
        mask_col = image_col + 1

        add_text_axis(
            fig,
            gs[0, image_col:mask_col + 1],
            class_name,
            fontsize=10,
        )

        add_text_axis(
            fig,
            gs[1, image_col],
            "Image",
            fontsize=8,
        )

        add_text_axis(
            fig,
            gs[1, mask_col],
            "Mask",
            fontsize=8,
        )

    # --------------------------------------------------------
    # Images + masks
    # --------------------------------------------------------

    for class_idx, class_name in enumerate(classes):
        image_col = class_idx * 2
        mask_col = image_col + 1

        for row in range(n_rows):
            image_id, mask_id = samples[class_name][row]

            image_path = get_dataset_path(image_id)
            mask_path = get_dataset_path(mask_id)

            ax_img = fig.add_subplot(
                gs[row + 2, image_col]
            )

            ax_mask = fig.add_subplot(
                gs[row + 2, mask_col]
            )

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                ax_img.imshow(img)

            with Image.open(mask_path) as mask:
                mask = mask.convert("L")
                mask_array = np.asarray(mask)

                # Binary visualization robust to 0/1 or 0/255 masks.
                mask_binary = (
                    mask_array > 0
                ).astype(np.float32)

                ax_mask.imshow(
                    mask_binary,
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )

            ax_img.set_aspect(
                "equal",
                adjustable="box",
            )

            ax_mask.set_aspect(
                "equal",
                adjustable="box",
            )

            ax_img.set_anchor("N")
            ax_mask.set_anchor("N")

            clean_axis(ax_img)
            clean_axis(ax_mask)

    output_name = (
        f"ebhi_source_fold0_{split_name}_image_mask"
    )

    save_figure(
        fig,
        output_name,
    )


# ============================================================
# TARGET INDICES
# ============================================================

def build_target_index(
    fold_number,
    split_name,
):
    """
    Build:
        (class_name, basename) -> image_id

    This lets us retrieve exactly the same underlying image
    in the clean target and every corruption fold.
    """

    split_dir = get_split_dir(
        fold_number,
        split_name,
    )

    entries = read_class_labels(split_dir)

    index = {}

    for entry in entries:
        key = (
            entry["class_name"],
            entry["basename"],
        )

        index[key] = entry["image_id"]

    return index


def get_common_target_samples(
    split_name,
    class_name,
    n_images,
    rng,
):
    """
    Select clean target images from fold-1 for which the SAME
    basename exists in every corruption fold 2..11.

    Returns:
        list of dicts:
        {
            "basename": ...,
            "clean": clean_image_id,
            2: fold2_image_id,
            ...
            11: fold11_image_id
        }
    """

    clean_index = build_target_index(
        CLEAN_TARGET_FOLD,
        split_name,
    )

    corruption_indices = {
        fold_number: build_target_index(
            fold_number,
            split_name,
        )
        for fold_number, _, _, _ in CORRUPTION_FOLDS
    }

    clean_candidates = [
        (key, image_id)
        for key, image_id in clean_index.items()
        if key[0] == class_name
    ]

    valid = []

    for key, clean_image_id in clean_candidates:
        present_everywhere = all(
            key in corruption_indices[fold_number]
            for fold_number, _, _, _ in CORRUPTION_FOLDS
        )

        if not present_everywhere:
            continue

        # Also verify files exist physically.
        clean_path = DATASET_ROOT / clean_image_id

        if not clean_path.exists():
            continue

        corrupt_ids = {}

        all_files_exist = True

        for fold_number, _, _, _ in CORRUPTION_FOLDS:
            image_id = corruption_indices[
                fold_number
            ][key]

            if not (DATASET_ROOT / image_id).exists():
                all_files_exist = False
                break

            corrupt_ids[fold_number] = image_id

        if not all_files_exist:
            continue

        item = {
            "basename": key[1],
            "clean": clean_image_id,
        }

        item.update(corrupt_ids)

        valid.append(item)

    if len(valid) == 0:
        raise RuntimeError(
            f"\nNo clean image shared by all corruption folds."
            f"\nClass: {class_name}"
            f"\nSplit: {split_name}"
        )

    n = min(n_images, len(valid))

    if n < n_images:
        print(
            f"WARNING: TARGET {class_name} / {split_name}: "
            f"requested {n_images}, only {n} images exist "
            f"in every corruption fold."
        )

    return rng.sample(valid, n)


# ============================================================
# TARGET SHIFT FIGURE
# ============================================================

def create_target_shift_figure(
    split_name,
    classes,
    n_images_per_class=N_TARGET_IMAGES_PER_CLASS,
):
    """
    Create a Histopatch-style figure using the same underlying
    image across clean target + all corruptions.

    Columns:

        Original
        Stain-light
        Stain-heavy
        Dust
        Air bubble
        Defocus blur
        Motion blur
        Gaussian noise
        Shot noise
        Brightness
        Contrast

    Rows:
        2 examples for Adenocarcinoma
        2 examples for Low-grade IN
        ...
        2 examples for every EBHI class
    """

    seed_offset = {
        "train": 0,
        "val": 100,
        "test": 200,
    }

    rng = random.Random(
        SEED
        + 5000
        + seed_offset[split_name]
    )

    samples = {}

    for class_name in classes:
        samples[class_name] = (
            get_common_target_samples(
                split_name=split_name,
                class_name=class_name,
                n_images=n_images_per_class,
                rng=rng,
            )
        )

        print(
            f"TARGET | {split_name:5s} | "
            f"{class_name:20s} | "
            f"{len(samples[class_name])} shared images"
        )

    # Keep the same number of examples per class.
    n_per_class = min(
        len(samples[class_name])
        for class_name in classes
    )

    # --------------------------------------------------------
    # Column definitions
    # --------------------------------------------------------

    image_columns = [
        {
            "fold": CLEAN_TARGET_FOLD,
            "key": "clean",
            "title": "Original",
            "group": "Original",
        }
    ]

    for (
        fold_number,
        corruption_dir,
        display_name,
        group_name,
    ) in CORRUPTION_FOLDS:
        image_columns.append(
            {
                "fold": fold_number,
                "key": fold_number,
                "title": display_name,
                "group": group_name,
            }
        )

    n_image_cols = len(image_columns)

    # One extra left column for the class names.
    n_cols = 1 + n_image_cols

    # Two header rows:
    #   0 -> category (Staining, Noise, ...)
    #   1 -> exact corruption name
    n_data_rows = (
        len(classes)
        * n_per_class
    )

    # Figure dimensions
    fig_width = 20.0

    class_label_width = 1.45
    image_width = 1.0

    width_ratios = (
        [class_label_width]
        + [image_width] * n_image_cols
    )

    group_header_ratio = 0.24
    column_header_ratio = 0.24

    fig_height = (
        1.45
        * (
            n_data_rows
            + group_header_ratio
            + column_header_ratio
        )
    )

    fig = plt.figure(
        figsize=(fig_width, fig_height)
    )

    gs = fig.add_gridspec(
        nrows=n_data_rows + 2,
        ncols=n_cols,
        height_ratios=(
            [group_header_ratio, column_header_ratio]
            + [1.0] * n_data_rows
        ),
        width_ratios=width_ratios,
        wspace=0.02,
        hspace=0.02,
        left=0.002,
        right=0.998,
        bottom=0.002,
        top=0.998,
    )

    # --------------------------------------------------------
    # Top-level category headers
    #
    # Original | Staining | Contamination | Blurring |
    # Noise | Illumination
    # --------------------------------------------------------

    group_ranges = []

    current_group = None
    start_col = None

    # Image columns start at GridSpec col=1
    for image_idx, col_info in enumerate(image_columns):
        grid_col = image_idx + 1
        group = col_info["group"]

        if current_group is None:
            current_group = group
            start_col = grid_col

        elif group != current_group:
            group_ranges.append(
                (
                    current_group,
                    start_col,
                    grid_col - 1,
                )
            )

            current_group = group
            start_col = grid_col

    group_ranges.append(
        (
            current_group,
            start_col,
            n_image_cols,
        )
    )

    # Blank top-left cell
    add_text_axis(
        fig,
        gs[0, 0],
        "",
    )

    for group_name, start, end in group_ranges:
        add_text_axis(
            fig,
            gs[0, start:end + 1],
            group_name,
            fontsize=11,
        )

    # --------------------------------------------------------
    # Exact corruption headers
    # --------------------------------------------------------

    add_text_axis(
        fig,
        gs[1, 0],
        "Class",
        fontsize=10,
    )

    for image_idx, col_info in enumerate(image_columns):
        grid_col = image_idx + 1

        add_text_axis(
            fig,
            gs[1, grid_col],
            col_info["title"],
            fontsize=8,
        )

    # --------------------------------------------------------
    # Rows by class
    # --------------------------------------------------------

    current_row = 2

    for class_name in classes:
        class_start = current_row
        class_end = (
            current_row
            + n_per_class
        )

        # Class label spans all examples of that class.
        add_text_axis(
            fig,
            gs[
                class_start:class_end,
                0
            ],
            class_name,
            fontsize=9,
            rotation=90,
        )

        for sample_idx in range(n_per_class):
            item = samples[
                class_name
            ][sample_idx]

            for image_idx, col_info in enumerate(
                image_columns
            ):
                grid_col = image_idx + 1

                image_id = item[
                    col_info["key"]
                ]

                image_path = get_dataset_path(
                    image_id
                )

                ax = fig.add_subplot(
                    gs[current_row, grid_col]
                )

                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    ax.imshow(img)

                ax.set_aspect(
                    "equal",
                    adjustable="box",
                )

                ax.set_anchor("N")
                clean_axis(ax)

            current_row += 1

    output_name = (
        f"ebhi_target_shifts_{split_name}"
    )

    save_figure(
        fig,
        output_name,
    )


# ============================================================
# SANITY CHECK
# ============================================================

def sanity_check():
    """Basic directory sanity check."""

    print(
        "\n"
        "===================================================="
    )
    print("EBHI SUPPLEMENTARY FIGURE GENERATION")
    print(
        "===================================================="
    )

    print(f"Dataset root   : {DATASET_ROOT}")
    print(f"Dataset exists : {DATASET_ROOT.exists()}")
    print(f"Folds root     : {FOLDS_ROOT}")
    print(f"Folds exists   : {FOLDS_ROOT.exists()}")
    print(f"Output root    : {OUTPUT_ROOT}")

    print(
        "===================================================="
        "\n"
    )

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist:"
            f"\n{DATASET_ROOT}"
        )

    if not FOLDS_ROOT.exists():
        raise FileNotFoundError(
            f"Fold directory does not exist:"
            f"\n{FOLDS_ROOT}"
        )

    # Verify every expected fold exists.
    expected_folds = [
        SOURCE_FOLD,
        CLEAN_TARGET_FOLD,
    ] + [
        fold_number
        for fold_number, _, _, _
        in CORRUPTION_FOLDS
    ]

    for fold_number in expected_folds:
        get_fold_dir(fold_number)


# ============================================================
# MAIN
# ============================================================

def main():
    sanity_check()

    # Discover classes once from source fold-0.
    classes = discover_classes(
        split_name="train"
    )

    for split_name in [
        "train",
        "val",
        "test",
    ]:
        print(
            "\n"
            "===================================================="
        )
        print(
            f"GENERATING {split_name.upper()}"
        )
        print(
            "===================================================="
        )

        # ----------------------------------------------------
        # SOURCE fold-0
        # Classes as columns, Image + Mask, 15 samples/class
        # ----------------------------------------------------

        create_source_figure(
            split_name=split_name,
            classes=classes,
            n_images_per_class=(
                N_SOURCE_IMAGES_PER_CLASS
            ),
        )

        # ----------------------------------------------------
        # TARGET fold-1 + folds 2..11
        # Same image through every corruption
        # 2 samples per class
        # ----------------------------------------------------

        create_target_shift_figure(
            split_name=split_name,
            classes=classes,
            n_images_per_class=(
                N_TARGET_IMAGES_PER_CLASS
            ),
        )

    print(
        "\n"
        "===================================================="
    )
    print("DONE")
    print(
        f"Figures saved in:\n{OUTPUT_ROOT}"
    )
    print(
        "===================================================="
        "\n"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
