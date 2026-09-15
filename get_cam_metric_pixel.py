#!/usr/bin/env python3
"""
Create imbalanced CAMELYON17_512 training folds from a source fold.

Protocol
--------
ONLY the TRAIN split is modified.

    train -> imbalanced
    valcl -> unchanged
    valpx -> unchanged
    test  -> unchanged

The source fold is copied completely before replacing only the
training metadata.

Default source:
    CAMELYON17_512/fold-0

Fold naming convention
----------------------
Generated fold ID:

    <majority_percent><source_fold_id><majority_class>

For source fold-0:

    fold-6000  -> 60% normal / 40% cancer
    fold-6001  -> 60% cancer / 40% normal

    fold-7000  -> 70% normal / 30% cancer
    fold-7001  -> 70% cancer / 30% normal

    fold-8000  -> 80% normal / 20% cancer
    fold-8001  -> 80% cancer / 20% normal

    fold-9000  -> 90% normal / 10% cancer
    fold-9001  -> 90% cancer / 10% normal

    fold-10000 -> 100% normal
    fold-10001 -> 100% cancer
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple


# ============================================================
# Configuration
# ============================================================

DEFAULT_DATASET_ROOT = Path(
    "folds/wsol-done-right-splits/CAMELYON17_512"
)

DEFAULT_SOURCE_FOLD = "fold-0"

DEFAULT_RATIOS = [
    60,
    70,
    80,
    90,
    100,
]

REBALANCED_SPLIT = "train"


CLASS_NAMES = {
    0: "normal",
    1: "cancer",
}


METADATA_FILES = [
    "image_ids.txt",
    "class_labels.txt",
    "image_sizes.txt",
    "localization.txt",
]


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Create imbalanced CAMELYON17_512 training folds "
            "while keeping validation and test unchanged."
        )
    )

    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )

    parser.add_argument(
        "--source-fold",
        "--source_fold",
        type=str,
        default=DEFAULT_SOURCE_FOLD,
    )

    parser.add_argument(
        "--ratios",
        nargs="+",
        type=int,
        default=DEFAULT_RATIOS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--n-samples",
        "--n_samples",
        type=int,
        default=None,
        help=(
            "Fixed training-set size. "
            "Default: largest common feasible size."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Metadata
# ============================================================

def read_nonempty_lines(
    path: Path,
) -> List[str]:

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing metadata file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return [
            line.rstrip("\n")
            for line in f
            if line.strip()
        ]


def line_image_id(
    line: str,
) -> str:

    return line.split(
        ",",
        1,
    )[0].strip()


def read_labels(
    split_dir: Path,
) -> Tuple[List[str], Dict[str, int]]:

    image_ids = read_nonempty_lines(
        split_dir
        / "image_ids.txt"
    )

    class_lines = read_nonempty_lines(
        split_dir
        / "class_labels.txt"
    )

    labels: Dict[str, int] = {}

    for line in class_lines:

        parts = line.rsplit(
            ",",
            1,
        )

        if len(parts) != 2:
            raise ValueError(
                f"Invalid class_labels line: {line}"
            )

        image_id = (
            parts[0]
            .strip()
        )

        label = int(
            parts[1]
            .strip()
        )

        if label not in (
            0,
            1,
        ):
            raise ValueError(
                f"Expected binary label 0/1, "
                f"got {label} for {image_id}"
            )

        labels[
            image_id
        ] = label

    missing = [
        image_id
        for image_id in image_ids
        if image_id not in labels
    ]

    if missing:

        raise RuntimeError(
            f"{len(missing)} image IDs have no class label "
            f"in {split_dir}.\n"
            + "\n".join(
                missing[:20]
            )
        )

    return (
        image_ids,
        labels,
    )


# ============================================================
# Sampling
# ============================================================

def class_pools(
    image_ids: Sequence[str],
    labels: Dict[str, int],
    seed: int,
) -> Dict[int, List[str]]:

    pools = {

        0: [
            image_id
            for image_id in image_ids
            if labels[image_id] == 0
        ],

        1: [
            image_id
            for image_id in image_ids
            if labels[image_id] == 1
        ],
    }

    rng0 = random.Random(
        seed + 100
    )

    rng1 = random.Random(
        seed + 101
    )

    rng0.shuffle(
        pools[0]
    )

    rng1.shuffle(
        pools[1]
    )

    return pools


def required_counts(
    n_total: int,
    majority_percent: int,
    majority_class: int,
) -> Dict[int, int]:

    frac = (
        majority_percent
        / 100.0
    )

    n_majority = int(
        round(
            n_total
            * frac
        )
    )

    n_majority = min(
        max(
            n_majority,
            0,
        ),
        n_total,
    )

    n_minority = (
        n_total
        - n_majority
    )

    if majority_class == 0:

        return {
            0: n_majority,
            1: n_minority,
        }

    return {
        0: n_minority,
        1: n_majority,
    }


def max_common_feasible_n(
    n_class0: int,
    n_class1: int,
    ratios: Sequence[int],
) -> int:

    available = {
        0: n_class0,
        1: n_class1,
    }

    upper_bounds = []

    for ratio in ratios:

        frac_major = (
            ratio
            / 100.0
        )

        frac_minor = (
            1.0
            - frac_major
        )

        for majority_class in (
            0,
            1,
        ):

            minority_class = (
                1
                - majority_class
            )

            if frac_major > 0:

                upper_bounds.append(
                    available[
                        majority_class
                    ]
                    / frac_major
                )

            if frac_minor > 0:

                upper_bounds.append(
                    available[
                        minority_class
                    ]
                    / frac_minor
                )

    candidate = int(
        math.floor(
            min(
                upper_bounds
            )
        )
    )

    while candidate > 0:

        ok = True

        for ratio in ratios:

            for majority_class in (
                0,
                1,
            ):

                counts = required_counts(
                    n_total=candidate,
                    majority_percent=ratio,
                    majority_class=majority_class,
                )

                if (
                    counts[0]
                    > n_class0
                    or
                    counts[1]
                    > n_class1
                ):

                    ok = False
                    break

            if not ok:
                break

        if ok:
            return candidate

        candidate -= 1

    return 0


def choose_ids(
    pools: Dict[int, List[str]],
    n_total: int,
    majority_percent: int,
    majority_class: int,
) -> Tuple[Set[str], Dict[int, int]]:

    counts = required_counts(
        n_total=n_total,
        majority_percent=majority_percent,
        majority_class=majority_class,
    )

    if counts[0] > len(
        pools[0]
    ):

        raise ValueError(
            f"Need {counts[0]} normal samples, "
            f"only {len(pools[0])} available."
        )

    if counts[1] > len(
        pools[1]
    ):

        raise ValueError(
            f"Need {counts[1]} cancer samples, "
            f"only {len(pools[1])} available."
        )

    selected = set(

        pools[0][
            :counts[0]
        ]

        +

        pools[1][
            :counts[1]
        ]
    )

    if len(
        selected
    ) != n_total:

        raise RuntimeError(
            f"Expected {n_total} unique images, "
            f"got {len(selected)}."
        )

    return (
        selected,
        counts,
    )


# ============================================================
# Metadata filtering
# ============================================================

def filter_metadata_file(
    source_path: Path,
    destination_path: Path,
    selected_ids: Set[str],
):

    lines = read_nonempty_lines(
        source_path
    )

    kept = [

        line

        for line in lines

        if line_image_id(
            line
        ) in selected_ids
    ]

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for line in kept:

            f.write(
                line
                + "\n"
            )


# ============================================================
# Validation
# ============================================================

def validate_rebalanced_split(
    split_dir: Path,
    expected_ids: Set[str],
    expected_counts: Dict[int, int],
):

    image_ids, labels = read_labels(
        split_dir
    )

    actual_ids = set(
        image_ids
    )

    if actual_ids != expected_ids:

        raise RuntimeError(
            f"ID mismatch in {split_dir}: "
            f"expected {len(expected_ids)}, "
            f"got {len(actual_ids)}"
        )

    actual_counts = {

        0: sum(
            labels[x] == 0
            for x in image_ids
        ),

        1: sum(
            labels[x] == 1
            for x in image_ids
        ),
    }

    if actual_counts != expected_counts:

        raise RuntimeError(
            f"Class-count mismatch in {split_dir}: "
            f"expected={expected_counts}, "
            f"actual={actual_counts}"
        )

    for filename in METADATA_FILES:

        path = (
            split_dir
            / filename
        )

        if not path.is_file():

            raise FileNotFoundError(
                f"Generated metadata missing: "
                f"{path}"
            )

        referenced = {

            line_image_id(
                line
            )

            for line
            in read_nonempty_lines(
                path
            )
        }

        unexpected = (
            referenced
            - expected_ids
        )

        if unexpected:

            raise RuntimeError(
                f"{filename} contains "
                f"unexpected IDs in {split_dir}: "
                f"{list(unexpected)[:10]}"
            )


# ============================================================
# Fold naming
# ============================================================

def extract_source_fold_id(
    source_fold: str,
) -> str:

    source_fold = str(
        source_fold
    )

    if source_fold.startswith(
        "fold-"
    ):

        source_fold_id = (
            source_fold[
                len("fold-"):
            ]
        )

    else:

        source_fold_id = (
            source_fold
        )

    if not source_fold_id:

        raise ValueError(
            f"Could not extract fold ID "
            f"from '{source_fold}'."
        )

    if not source_fold_id.isdigit():

        raise ValueError(
            f"Source fold ID must be numeric. "
            f"Got '{source_fold}' -> "
            f"'{source_fold_id}'."
        )

    return source_fold_id


def fold_number(
    majority_percent: int,
    source_fold: str,
    majority_class: int,
) -> str:

    source_fold_id = (
        extract_source_fold_id(
            source_fold
        )
    )

    return (
        f"{majority_percent}"
        f"{source_fold_id}"
        f"{majority_class}"
    )


# ============================================================
# Copy fold
# ============================================================

def copy_source_fold(
    source_fold: Path,
    destination_fold: Path,
    overwrite: bool,
):

    if destination_fold.exists():

        if not overwrite:

            raise FileExistsError(
                f"Destination already exists: "
                f"{destination_fold}\n"
                "Use --overwrite to replace it."
            )

        shutil.rmtree(
            destination_fold
        )

    shutil.copytree(
        source_fold,
        destination_fold,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )

    source_fold = (
        dataset_root
        / args.source_fold
    )

    if not source_fold.is_dir():

        raise FileNotFoundError(
            f"Source fold not found: "
            f"{source_fold}"
        )

    source_fold_id = (
        extract_source_fold_id(
            args.source_fold
        )
    )

    ratios = sorted(
        set(
            args.ratios
        )
    )

    for ratio in ratios:

        if not (
            50
            <= ratio
            <= 100
        ):

            raise ValueError(
                f"Invalid ratio {ratio}. "
                "Expected values between "
                "50 and 100."
            )

    # --------------------------------------------------------
    # Source splits
    # --------------------------------------------------------

    source_splits = sorted(

        p.name

        for p
        in source_fold.iterdir()

        if p.is_dir()
    )

    if REBALANCED_SPLIT not in source_splits:

        raise FileNotFoundError(
            f"'{REBALANCED_SPLIT}' "
            f"does not exist in "
            f"{source_fold}"
        )

    # --------------------------------------------------------
    # Print protocol
    # --------------------------------------------------------

    print(
        "=" * 90
    )

    print(
        "CAMELYON17_512 "
        "TRAIN IMBALANCE GENERATION"
    )

    print(
        "=" * 90
    )

    print(
        f"Dataset root       : "
        f"{dataset_root}"
    )

    print(
        f"Source fold        : "
        f"{source_fold}"
    )

    print(
        f"Source fold ID     : "
        f"{source_fold_id}"
    )

    print(
        f"Rebalanced split   : "
        f"{REBALANCED_SPLIT}"
    )

    print(
        "Unchanged splits   : "
        + ", ".join(
            split
            for split
            in source_splits
            if split
            != REBALANCED_SPLIT
        )
    )

    print(
        f"Ratios             : "
        f"{ratios}"
    )

    print(
        f"Seed               : "
        f"{args.seed}"
    )

    print()

    # ========================================================
    # Source training set
    # ========================================================

    train_dir = (
        source_fold
        / REBALANCED_SPLIT
    )

    image_ids, labels = (
        read_labels(
            train_dir
        )
    )

    n0 = sum(
        labels[x] == 0
        for x in image_ids
    )

    n1 = sum(
        labels[x] == 1
        for x in image_ids
    )

    pools = class_pools(
        image_ids=image_ids,
        labels=labels,
        seed=args.seed,
    )

    feasible_n = (
        max_common_feasible_n(
            n_class0=n0,
            n_class1=n1,
            ratios=ratios,
        )
    )

    if feasible_n <= 0:

        raise RuntimeError(
            "No feasible common "
            "training-set size."
        )

    if args.n_samples is None:

        n_total = feasible_n

    else:

        n_total = (
            args.n_samples
        )

        if (
            n_total
            > feasible_n
        ):

            raise ValueError(
                f"--n-samples={n_total} "
                f"is too large. "
                f"Maximum={feasible_n}"
            )

    print(
        f"[SOURCE TRAIN] "
        f"N={len(image_ids)} | "
        f"normal={n0} | "
        f"cancer={n1}"
    )

    print(
        f"[GENERATED] "
        f"common training N="
        f"{n_total}"
    )

    # ========================================================
    # Manifest
    # ========================================================

    manifest = {

        "dataset":
            "CAMELYON17_512",

        "source_fold":
            str(
                source_fold
            ),

        "source_fold_name":
            args.source_fold,

        "source_fold_id":
            source_fold_id,

        "seed":
            args.seed,

        "ratios":
            ratios,

        "rebalanced_split":
            "train",

        "unchanged_splits": [

            split

            for split
            in source_splits

            if split != "train"
        ],

        "source_train": {

            "n_total":
                len(
                    image_ids
                ),

            "normal_count":
                n0,

            "cancer_count":
                n1,
        },

        "generated_train_size":
            n_total,

        "generated_folds": {},
    }

    # ========================================================
    # Generate
    # ========================================================

    for ratio in ratios:

        for majority_class in (
            0,
            1,
        ):

            fold_id = (
                fold_number(
                    majority_percent=ratio,
                    source_fold=args.source_fold,
                    majority_class=majority_class,
                )
            )

            destination_fold = (
                dataset_root
                / f"fold-{fold_id}"
            )

            majority_name = (
                CLASS_NAMES[
                    majority_class
                ]
            )

            minority_class = (
                1
                - majority_class
            )

            minority_name = (
                CLASS_NAMES[
                    minority_class
                ]
            )

            selected_ids, counts = (
                choose_ids(
                    pools=pools,
                    n_total=n_total,
                    majority_percent=ratio,
                    majority_class=majority_class,
                )
            )

            print()

            print(
                f"[fold-{fold_id}] "
                f"TRAIN: "
                f"{ratio}% "
                f"{majority_name} / "
                f"{100-ratio}% "
                f"{minority_name}"
            )

            print(
                f"  N={n_total} | "
                f"normal={counts[0]} | "
                f"cancer={counts[1]}"
            )

            print(
                "  val/test: unchanged"
            )

            fold_manifest = {

                "fold_id":
                    fold_id,

                "source_fold":
                    args.source_fold,

                "majority_percent":
                    ratio,

                "majority_class":
                    majority_class,

                "majority_name":
                    majority_name,

                "train": {

                    "n_total":
                        n_total,

                    "normal_count":
                        counts[0],

                    "cancer_count":
                        counts[1],

                    "normal_fraction":
                        counts[0]
                        / n_total,

                    "cancer_fraction":
                        counts[1]
                        / n_total,

                    "selected_image_ids":
                        sorted(
                            selected_ids
                        ),
                },
            }

            manifest[
                "generated_folds"
            ][
                f"fold-{fold_id}"
            ] = fold_manifest

            if args.dry_run:

                print(
                    f"  would create: "
                    f"{destination_fold}"
                )

                continue

            # ------------------------------------------------
            # Copy complete fold-0.
            #
            # train / valcl / valpx / test all copied.
            # ------------------------------------------------

            copy_source_fold(
                source_fold=source_fold,
                destination_fold=destination_fold,
                overwrite=args.overwrite,
            )

            # ------------------------------------------------
            # Modify TRAIN only.
            # ------------------------------------------------

            source_train = (
                source_fold
                / "train"
            )

            destination_train = (
                destination_fold
                / "train"
            )

            for filename in METADATA_FILES:

                filter_metadata_file(
                    source_path=(
                        source_train
                        / filename
                    ),
                    destination_path=(
                        destination_train
                        / filename
                    ),
                    selected_ids=selected_ids,
                )

            validate_rebalanced_split(
                split_dir=destination_train,
                expected_ids=selected_ids,
                expected_counts=counts,
            )

            # ------------------------------------------------
            # Save info
            # ------------------------------------------------

            with (
                destination_fold
                / "imbalance_info.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    fold_manifest,
                    f,
                    indent=2,
                )

    # ========================================================
    # Global manifest
    # ========================================================

    if not args.dry_run:

        manifest_path = (
            dataset_root
            / "generated_imbalanced_folds.json"
        )

        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                manifest,
                f,
                indent=2,
            )

        print()
        print(
            "=" * 90
        )

        print(
            "DONE"
        )

        print(
            f"Manifest: "
            f"{manifest_path}"
        )

        print()
        print(
            "TRAIN : imbalanced"
        )

        print(
            "VALCL : unchanged"
        )

        print(
            "VALPX : unchanged"
        )

        print(
            "TEST  : unchanged"
        )

        print(
            "=" * 90
        )

    else:

        print()
        print(
            "DRY RUN complete."
        )


if __name__ == "__main__":
    main()