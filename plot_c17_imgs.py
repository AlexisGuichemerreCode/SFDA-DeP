import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# Actual CAMELYON17 images
DATASET_ROOT = Path(
    "/projets/Aguichemerre/datasets/CAMELYON17_512"
)

# WSOL folds / metadata
FOLDS_ROOT = Path(
    "/export/livia/home/vision/Aguichemerre/"
    "SFDA-DeP/folds/wsol-done-right-splits/CAMELYON17_512"
)

# Output figures
OUTPUT_ROOT = Path(
    "/export/livia/home/vision/Aguichemerre/"
    "SFDA-DeP/figures/camelyon17_samples"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# CAMELYON17 centers
CENTERS = [0, 1, 2, 3, 4]

# Metadata split names
SPLITS = {
    "train": "train",
    "val": "valcl",
    "test": "test",
}

# Class labels
CLASS_LABELS = {
    "normal": 0,
    "cancer": 1,
}

# Number of random examples PER CENTER
N_IMAGES_PER_CENTER = 15

# Reproducibility
SEED = 42

# Figure quality
DPI = 300


# ============================================================
# FOLD DIRECTORY
# ============================================================

def get_fold_dir(center):
    """
    Return metadata directory corresponding to one center.

    Supports:
        fold-0
        fold_0
        fold0
        0
    """

    candidates = [
        FOLDS_ROOT / f"fold-{center}",
        FOLDS_ROOT / f"fold_{center}",
        FOLDS_ROOT / f"fold{center}",
        FOLDS_ROOT / str(center),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"\nCould not find fold directory for Center {center}."
        "\nTried:"
        + "".join(f"\n  {path}" for path in candidates)
    )


# ============================================================
# READ IMAGE IDS
# ============================================================

def read_image_ids(split_dir):
    """Read image_ids.txt."""

    file_path = split_dir / "image_ids.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"image_ids.txt not found: {file_path}"
        )

    with open(file_path, "r") as f:
        return [
            line.strip()
            for line in f
            if line.strip()
        ]


# ============================================================
# READ CLASS LABELS
# ============================================================

def read_class_labels(split_dir):
    """
    Read class_labels.txt.

    Supports:
        image.png,0
        image.png 0
        image.png<TAB>0
    """

    file_path = split_dir / "class_labels.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"class_labels.txt not found: {file_path}"
        )

    labels = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if "," in line:
                parts = line.rsplit(",", 1)
            elif "\t" in line:
                parts = line.rsplit("\t", 1)
            else:
                parts = line.rsplit(maxsplit=1)

            if len(parts) != 2:
                raise ValueError(
                    f"\nCannot parse line:"
                    f"\n{line}"
                    f"\nFrom:"
                    f"\n{file_path}"
                )

            image_id = parts[0].strip()
            label = int(parts[1].strip())

            labels[image_id] = label

    return labels


# ============================================================
# GET IMAGE PATH
# ============================================================

def get_image_path(image_id):
    """
    Convert the image ID stored in the WSOL metadata into the
    absolute image path.
    """

    image_id = image_id.strip()
    image_path = DATASET_ROOT / image_id

    if not image_path.exists():
        raise FileNotFoundError(
            "\nImage not found."
            f"\nImage ID:"
            f"\n  {image_id}"
            f"\nExpected path:"
            f"\n  {image_path}"
        )

    return image_path


# ============================================================
# GET CANCER MASK PATH
# ============================================================

def get_mask_path(image_path):
    """
    Find the segmentation mask corresponding to a cancer patch.

    CAMELYON17 convention used here:

        Patch_file_xxx.png
            ->
        mask_file_xxx.png
    """

    image_path = Path(image_path)
    filename = image_path.name

    replacements = [
        ("Patch_file_", "mask_file_"),
        ("Patch_file_", "Mask_file_"),
        ("Patch_", "mask_"),
        ("Patch_", "Mask_"),
        ("patch_file_", "mask_file_"),
        ("patch_", "mask_"),
    ]

    tried_paths = []

    for old, new in replacements:
        if old not in filename:
            continue

        mask_filename = filename.replace(old, new, 1)
        mask_path = image_path.parent / mask_filename

        tried_paths.append(mask_path)

        if mask_path.exists():
            return mask_path

    raise FileNotFoundError(
        "\nCorresponding cancer mask not found."
        f"\n\nImage:"
        f"\n  {image_path}"
        f"\n\nTried:"
        + "".join(f"\n  {path}" for path in tried_paths)
    )


# ============================================================
# GET SAMPLES FOR ONE CENTER
# ============================================================

def get_images_for_center(
    center,
    split_name,
    class_label,
    n_images,
    rng,
):
    """
    Randomly sample images of one class from one center.
    """

    fold_dir = get_fold_dir(center)
    metadata_split = SPLITS[split_name]
    split_dir = fold_dir / metadata_split

    # Fallback if validation is named "val"
    if not split_dir.exists() and split_name == "val":
        alternative = fold_dir / "val"

        if alternative.exists():
            split_dir = alternative

    if not split_dir.exists():
        raise FileNotFoundError(
            f"\nSplit directory not found:"
            f"\n  {split_dir}"
        )

    image_ids = read_image_ids(split_dir)
    labels = read_class_labels(split_dir)

    selected = []

    for image_id in image_ids:
        if image_id in labels:
            label = labels[image_id]
        elif Path(image_id).name in labels:
            label = labels[Path(image_id).name]
        else:
            continue

        if label == class_label:
            selected.append(image_id)

    if len(selected) == 0:
        raise RuntimeError(
            f"\nNo samples found."
            f"\nCenter: {center}"
            f"\nSplit: {split_name}"
            f"\nClass label: {class_label}"
        )

    n = min(n_images, len(selected))
    return rng.sample(selected, n)


# ============================================================
# SMALL PLOTTING HELPERS
# ============================================================

def clean_axis(ax):
    """Remove ticks, labels and spines."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)


def add_text_axis(fig, gs_cell, text, fontsize):
    """
    Create a dedicated GridSpec cell containing only centered text.
    Using actual title rows prevents the large white gap that occurs
    when titles are positioned with fig.text().
    """
    ax = fig.add_subplot(gs_cell)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        transform=ax.transAxes,
    )
    return ax


# ============================================================
# SAVE FIGURE
# ============================================================

def save_figure(fig, output_name):
    """Save PNG and PDF versions."""

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
# FIGURE NORMAL
# ============================================================

def create_normal_figure(
    split_name,
    class_label,
    n_images_per_center=3,
):
    """
    Normal samples.

    Layout:

        Center 0   Center 1   Center 2   Center 3   Center 4
        image      image      image      image      image
        image      image      image      image      image
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

    for center in CENTERS:
        samples[center] = get_images_for_center(
            center=center,
            split_name=split_name,
            class_label=class_label,
            n_images=n_images_per_center,
            rng=rng,
        )

        print(
            f"{split_name:5s} | "
            f"normal | "
            f"Center {center} | "
            f"{len(samples[center])} images"
        )

    # Use the same number of rows for all centers.
    n_rows = min(len(samples[c]) for c in CENTERS)
    n_cols = len(CENTERS)

    # Each image panel is approximately square.
    fig_width = 10.0
    panel_size = fig_width / n_cols

    # Dedicated title row instead of fig.text()/set_title whitespace.
    title_ratio = 0.22
    fig_height = panel_size * (n_rows + title_ratio)

    fig = plt.figure(
        figsize=(fig_width, fig_height)
    )

    gs = fig.add_gridspec(
        nrows=n_rows + 1,
        ncols=n_cols,
        height_ratios=[title_ratio] + [1.0] * n_rows,
        wspace=0.02,
        hspace=0.02,
        left=0.002,
        right=0.998,
        bottom=0.002,
        top=0.998,
    )

    # Center titles
    for col, center in enumerate(CENTERS):
        add_text_axis(
            fig,
            gs[0, col],
            f"Center {center}",
            fontsize=12,
        )

    # Images
    for col, center in enumerate(CENTERS):
        for row in range(n_rows):
            ax = fig.add_subplot(gs[row + 1, col])

            image_id = samples[center][row]
            image_path = get_image_path(image_id)

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                ax.imshow(img)

            ax.set_aspect("equal", adjustable="box")
            ax.set_anchor("N")
            clean_axis(ax)

    output_name = f"camelyon17_{split_name}_normal"

    save_figure(
        fig=fig,
        output_name=output_name,
    )


# ============================================================
# FIGURE CANCER
# ============================================================

def create_cancer_figure(
    split_name,
    class_label,
    n_images_per_center=3,
):
    """
    Cancer samples.

    Each cancer patch is shown next to its pixel-level mask.

    Layout:

              Center 0              Center 1
           Image   Mask          Image   Mask

           image   mask          image   mask
           image   mask          image   mask
           ...
    """

    seed_offset = {
        "train": 0,
        "val": 100,
        "test": 200,
    }

    rng = random.Random(
        SEED
        + 1000
        + seed_offset[split_name]
    )

    samples = {}

    for center in CENTERS:
        samples[center] = get_images_for_center(
            center=center,
            split_name=split_name,
            class_label=class_label,
            n_images=n_images_per_center,
            rng=rng,
        )

        print(
            f"{split_name:5s} | "
            f"cancer | "
            f"Center {center} | "
            f"{len(samples[center])} images"
        )

    # Same number of rows across all centers.
    n_rows = min(len(samples[c]) for c in CENTERS)

    n_centers = len(CENTERS)
    n_cols = n_centers * 2

    # --------------------------------------------------------
    # KEY FIX FOR THE LARGE WHITE SPACE
    # --------------------------------------------------------
    #
    # The previous implementation used:
    #   fig.text(... Center ...)
    # together with imshow's square aspect ratio.
    #
    # That makes the titles live in global figure coordinates,
    # while the image axes shrink inside their cells, producing
    # a large blank region.
    #
    # Here Center and Image/Mask each get their OWN GridSpec row.
    # Therefore the distance above the images is fixed and stays
    # compact even when N_IMAGES_PER_CENTER becomes large.
    # --------------------------------------------------------

    fig_width = 16.0
    panel_size = fig_width / n_cols

    center_title_ratio = 0.22
    sub_title_ratio = 0.17

    total_row_units = (
        center_title_ratio
        + sub_title_ratio
        + n_rows
    )

    fig_height = panel_size * total_row_units

    fig = plt.figure(
        figsize=(fig_width, fig_height)
    )

    gs = fig.add_gridspec(
        nrows=n_rows + 2,
        ncols=n_cols,
        height_ratios=(
            [center_title_ratio, sub_title_ratio]
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
    # Row 0: Center titles spanning Image + Mask
    # --------------------------------------------------------

    for center_idx, center in enumerate(CENTERS):
        image_col = center_idx * 2
        mask_col = image_col + 1

        add_text_axis(
            fig,
            gs[0, image_col:mask_col + 1],
            f"Center {center}",
            fontsize=12,
        )

    # --------------------------------------------------------
    # Row 1: Image / Mask labels
    # --------------------------------------------------------

    for center_idx, center in enumerate(CENTERS):
        image_col = center_idx * 2
        mask_col = image_col + 1

        add_text_axis(
            fig,
            gs[1, image_col],
            "Image",
            fontsize=9,
        )

        add_text_axis(
            fig,
            gs[1, mask_col],
            "Mask",
            fontsize=9,
        )

    # --------------------------------------------------------
    # Remaining rows: Image + Mask
    # --------------------------------------------------------

    for center_idx, center in enumerate(CENTERS):
        image_col = center_idx * 2
        mask_col = image_col + 1

        for row in range(n_rows):
            ax_img = fig.add_subplot(
                gs[row + 2, image_col]
            )

            ax_mask = fig.add_subplot(
                gs[row + 2, mask_col]
            )

            image_id = samples[center][row]
            image_path = get_image_path(image_id)
            mask_path = get_mask_path(image_path)

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                ax_img.imshow(img)

            with Image.open(mask_path) as mask:
                mask = mask.convert("L")
                ax_mask.imshow(
                    mask,
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )

            # Keep the 512x512 panels square.
            ax_img.set_aspect("equal", adjustable="box")
            ax_mask.set_aspect("equal", adjustable="box")

            # If any tiny mismatch remains, anchor toward the top.
            ax_img.set_anchor("N")
            ax_mask.set_anchor("N")

            clean_axis(ax_img)
            clean_axis(ax_mask)

    output_name = f"camelyon17_{split_name}_cancer"

    save_figure(
        fig=fig,
        output_name=output_name,
    )


# ============================================================
# SANITY CHECK
# ============================================================

def sanity_check():
    """Basic path sanity check before generating figures."""

    print(
        "\n"
        "============================================"
    )
    print("CAMELYON17 FIGURE GENERATION")
    print(
        "============================================"
    )

    print(f"Dataset root   : {DATASET_ROOT}")
    print(f"Dataset exists : {DATASET_ROOT.exists()}")
    print(f"Folds root     : {FOLDS_ROOT}")
    print(f"Folds exists   : {FOLDS_ROOT.exists()}")
    print(f"Output root    : {OUTPUT_ROOT}")

    print(
        "============================================"
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


# ============================================================
# MAIN
# ============================================================

def main():
    sanity_check()

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        print(
            "\n"
            "============================================"
        )
        print(f"Generating {split_name.upper()}")
        print(
            "============================================"
        )

        # Cancer: image + segmentation mask
        create_cancer_figure(
            split_name=split_name,
            class_label=CLASS_LABELS["cancer"],
            n_images_per_center=N_IMAGES_PER_CENTER,
        )

        # Normal: image only
        create_normal_figure(
            split_name=split_name,
            class_label=CLASS_LABELS["normal"],
            n_images_per_center=N_IMAGES_PER_CENTER,
        )

    print(
        "\n"
        "============================================"
    )
    print("DONE")
    print(f"Figures saved in:\n{OUTPUT_ROOT}")
    print(
        "============================================"
        "\n"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
