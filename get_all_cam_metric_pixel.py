from copy import deepcopy
from os.path import join
import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FONT_PATH = os.path.expanduser("~/fonts/times.ttf")

fm.fontManager.addfont(FONT_PATH)
font_name = fm.FontProperties(fname=FONT_PATH).get_name()

plt.rcParams["font.family"] = font_name



import yaml
from PIL import Image
from tqdm import tqdm

from dlib.configure import constants, config
from dlib.dllogger import ArbStdOutBackend
from dlib.dllogger import ArbTextStreamBackend
from dlib.dllogger import Verbosity
import dlib.dllogger as DLLogger
from dlib.datasets.wsol_loader import get_data_loader
from dlib.learning.inference_wsol import CAMComputer
from dlib.process.instantiators import get_model
from dlib.process.parseit import str2bool
from dlib.utils.reproducibility import set_seed
from dlib.utils.tools import Dict2Obj, get_cpu_device, get_tag


# ============================================================
# YAML loader compatible with the checkpoints in your repo
# ============================================================

class IgnoreKeyLoader(yaml.SafeLoader):
    pass


def ignore_keys(loader, node):
    ignore_key = "best_valid_tau_cl"
    if isinstance(node, yaml.MappingNode):
        i = 0
        while i < len(node.value):
            if node.value[i][0].value == ignore_key:
                del node.value[i]
            else:
                i += 1
    return loader.construct_mapping(node)


def ignore_numpy_scalars(loader, node):
    try:
        value = loader.construct_scalar(node)
        return float(value)
    except Exception:
        return None


IgnoreKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    ignore_keys,
)
IgnoreKeyLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:numpy._core.multiarray.scalar",
    ignore_numpy_scalars,
)
IgnoreKeyLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar",
    ignore_numpy_scalars,
)


# ============================================================
# Load ONE source model
# ============================================================

def load_source_model(
    exp_path,
    checkpoint_type,
    cudaid,
    pixel_wise_classification=False,
    tmp_outd="tmp_outd",
):
    """
    Load one source model (PixelCAM/EnergyCAM, SAT, or DeepMIL).

    Returns
    -------
    args:
        Model/config arguments loaded from the source checkpoint.
    args_dict:
        Dict corresponding to config_model.yaml. We reuse it to create
        target dataloaders.
    model:
        Loaded source model.
    method_name:
        WSOL method stored in the checkpoint configuration.
    """
    # --------------------------------------------------------
    # Read global experiment config
    # --------------------------------------------------------
    with open(join(exp_path, "config_obj_final.yaml"), "r") as f:
        initial_dict = yaml.load(f, Loader=IgnoreKeyLoader)

    initial_dict["model"]["freeze_encoder"] = False
    initial_dict["pixel_wise_classification"] = False

    initial_args = Dict2Obj(initial_dict)
    initial_args.outd = tmp_outd
    initial_args.distributed = False
    initial_args.eval_checkpoint_type = checkpoint_type

    os.makedirs(tmp_outd, exist_ok=True)

    set_seed(seed=initial_args.MYSEED, verbose=False)
    os.environ["MYSEED"] = str(initial_args.MYSEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # --------------------------------------------------------
    # Resolve checkpoint folder
    # --------------------------------------------------------
    tag = get_tag(initial_args, checkpoint_type=checkpoint_type)
    path_cl = join(exp_path, tag)

    # --------------------------------------------------------
    # Read model config from selected checkpoint
    # --------------------------------------------------------
    with open(join(path_cl, "config_model.yaml"), "r") as f:
        args_dict = yaml.load(f, Loader=IgnoreKeyLoader)

    args_dict["pixel_wise_classification"] = pixel_wise_classification
    args_dict["multiple_layer_pixel_classifier"] = False
    args_dict["anchors_ortogonal"] = False
    args_dict["detach_pixel_classifier"] = False
    args_dict["batch_norm_pixel_classifier"] = False
    args_dict["one_layer_pixel_classifier"] = False
    args_dict["cpt_cam_entropy"] = False
    args_dict["ttda"] = False

    args = Dict2Obj(args_dict)
    args.outd = tmp_outd
    args.distributed = False
    args.eval_checkpoint_type = checkpoint_type
    args.model["folder_pre_trained_cl"] = path_cl
    args.sf_uda = False

    encoder_name = args.model["encoder_name"]
    method_name = args.method

    # --------------------------------------------------------
    # Instantiate + load model
    # --------------------------------------------------------
    model = get_model(args)[0]

    is_pixelcam_family = (
        "PixelCAM" in method_name
        or "EnergyCAM" in method_name
    )

    if is_pixelcam_family:
        if "deit" in encoder_name:
            model_state = torch.load(
                join(path_cl, "model.pt"),
                map_location=get_cpu_device(),
            )
            model.load_state_dict(model_state, strict=True)
        else:
            encoder_w = torch.load(
                join(path_cl, "encoder.pt"),
                map_location=get_cpu_device(),
            )
            model.encoder.super_load_state_dict(encoder_w, strict=True)

            header_w = torch.load(
                join(path_cl, "classification_head.pt"),
                map_location=get_cpu_device(),
            )
            model.classification_head.load_state_dict(header_w, strict=True)

            pixel_head_path = join(
                path_cl,
                "pixel_wise_classification_head.pt",
            )

            # The auxiliary pixel classifier is optional. It is only
            # instantiated/loaded for the PixelCAM family. The hasattr
            # guard prevents crashes for checkpoints/models that do not
            # expose this head.
            if (
                pixel_wise_classification
                and hasattr(model, "pixel_wise_classification_head")
                and os.path.isfile(pixel_head_path)
            ):
                header_p = torch.load(
                    pixel_head_path,
                    map_location=get_cpu_device(),
                )
                model.pixel_wise_classification_head.load_state_dict(
                    header_p,
                    strict=True,
                )
                print(
                    "Loaded pixel-wise classification head from: "
                    f"{pixel_head_path}"
                )
            elif pixel_wise_classification:
                print(
                    "Pixel-wise classification requested for "
                    f"{method_name}, but no compatible saved pixel head "
                    "was found. Continuing with the classification model."
                )

    elif method_name == "NEGEV":
        encoder_w = torch.load(
            join(path_cl, "encoder.pt"),
            map_location=get_cpu_device(),
        )
        model.encoder.super_load_state_dict(encoder_w, strict=True)

        header_w = torch.load(
            join(path_cl, "classification_head.pt"),
            map_location=get_cpu_device(),
        )
        model.classification_head.load_state_dict(header_w, strict=True)

        decoder_w = torch.load(
            join(path_cl, "decoder.pt"),
            map_location=get_cpu_device(),
        )
        model.decoder.super_load_state_dict(decoder_w, strict=True)

        seg_head_w = torch.load(
            join(path_cl, "segmentation_head.pt"),
            map_location=get_cpu_device(),
        )
        model.segmentation_head.load_state_dict(seg_head_w, strict=True)

    else:
        # Covers SAT/DeiT and DeepMIL/ResNet in your setup.
        if "deit" in encoder_name:
            model_state = torch.load(
                join(path_cl, "model.pt"),
                map_location=get_cpu_device(),
            )
            model.load_state_dict(model_state, strict=False)
        else:
            encoder_w = torch.load(
                join(path_cl, "encoder.pt"),
                map_location=get_cpu_device(),
            )
            model.encoder.super_load_state_dict(encoder_w, strict=True)

            header_w = torch.load(
                join(path_cl, "classification_head.pt"),
                map_location=get_cpu_device(),
            )
            model.classification_head.load_state_dict(header_w, strict=True)

    device = torch.device(f"cuda:{cudaid}")
    model.to(device)
    model.eval()

    print(
        f"Loaded source model: method={method_name}, "
        f"encoder={encoder_name}, checkpoint={path_cl}"
    )

    return args, args_dict, model, method_name


# ============================================================
# Build a target CAMELYON17 loader for one fold
# ============================================================

def build_target_loader(
    source_args,
    source_args_dict,
    target_dataset,
    target_fold,
    split,
    data_root,
):
    """
    Reuse the source model's image preprocessing, but load a target
    CAMELYON17_512 fold.
    """
    target_dict = deepcopy(source_args_dict)

    # Important: switch dataset/fold to TARGET domain.
    target_dict["dataset"] = target_dataset
    target_dict["fold"] = target_fold
    target_dict["data_root"] = data_root

    target_data_paths = config.configure_data_paths(
        target_dict,
        target_dataset,
    )

    metadata_root = join(
        constants.RELATIVE_META_ROOT,
        target_dataset,
        f"fold-{target_fold}",
    )

    # Copy source args because CAMComputer needs args, but make its dataset
    # information correspond to the TARGET domain.
    eval_args = deepcopy(source_args)
    eval_args.dataset = target_dataset
    eval_args.fold = target_fold

    target_basic_config = config.get_config(
        ds=target_dataset,
        fold=target_fold,
        magnification=eval_args.magnification,
    )

    eval_args.data_paths = target_data_paths
    eval_args.metadata_root = metadata_root
    eval_args.mask_root = target_basic_config["mask_root"]
    eval_args.cam_curve_interval = target_basic_config["cam_curve_interval"]

    loaders = get_data_loader(
        data_roots=target_data_paths,
        metadata_root=metadata_root,
        batch_size=32,
        workers=eval_args.num_workers,
        resize_size=eval_args.resize_size,
        crop_size=eval_args.crop_size,
        proxy_training_set=eval_args.proxy_training_set,
        num_val_sample_per_class=eval_args.num_val_sample_per_class,
        std_cams_folder=eval_args.std_cams_folder,
        get_splits_eval=[split],
        eval_batch_size=32,
    )

    return loaders[split], eval_args, metadata_root


# ============================================================
# Pixel metric for one (source model, target fold)
# ============================================================

def compute_foreground_precision_curve(
    model,
    eval_args,
    loader,
    metadata_root,
    target_dataset,
    data_root,
    split,
    cudaid,
    positive_class=1,
):
    """
    Same metric as in your current script:

      1. Keep cancer images only.
      2. Compute the CAM for class 1.
      3. Rank pixels from largest CAM score to smallest.
      4. For Top 1%, ..., Top 100%, compute the fraction of selected
         pixels that belong to the GT foreground.

    This is Precision@Top-k% of CAM pixels, not global pixel accuracy.
    """
    device = torch.device(f"cuda:{cudaid}")

    cam_computer = CAMComputer(
        args=deepcopy(eval_args),
        model=model,
        loader=loader,
        metadata_root=os.path.join(metadata_root, split),
        mask_root=eval_args.mask_root,
        iou_threshold_list=eval_args.iou_threshold_list,
        dataset_name=target_dataset,
        split=split,
        cam_curve_interval=eval_args.cam_curve_interval,
        multi_contour_eval=eval_args.multi_contour_eval,
        out_folder=eval_args.outd,
    )

    percentages = np.arange(1, 101)
    all_precision_curves = []

    for batch in tqdm(loader, ncols=100, total=len(loader)):
        # Use indices rather than exact tuple unpacking so this remains
        # compatible if your loader returns 9 or 10 objects.
        images = batch[0]
        targets = batch[1]
        image_ids = batch[3]

        image_size = images.shape[2:]

        images = images.to(device)
        targets = targets.to(device)

        for image, target, image_id in zip(images, targets, image_ids):
            # Positive/cancer class only.
            if target.item() != positive_class:
                continue

            with torch.set_grad_enabled(cam_computer.req_grad):
                cam, _ = cam_computer.get_cam_one_sample(
                    image=image.unsqueeze(0),
                    target=positive_class,
                )

            cam = cam.detach().float().squeeze()

            # ----------------------------------------------------
            # Load GT mask using CAMComputer's mask mapping.
            # ----------------------------------------------------
            mask_rel_path = cam_computer.evaluator.mask_paths[image_id][0]
            mask_path = os.path.join(
                data_root,
                target_dataset,
                mask_rel_path,
            )

            gt_annotation = Image.open(mask_path)
            gt_annotation = gt_annotation.resize(image_size)
            gt_annotation = np.asarray(gt_annotation)
            gt_annotation = (gt_annotation > 0).astype(np.uint8)

            gt_mask = torch.from_numpy(gt_annotation).float().to(device)
            gt_mask = gt_mask.squeeze()

            if cam.shape[-2:] != gt_mask.shape[-2:]:
                cam = F.interpolate(
                    cam.unsqueeze(0).unsqueeze(0),
                    size=gt_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

            gt_mask = (gt_mask > 0).float()

            # ----------------------------------------------------
            # Same ranking logic as your existing script.
            # Normalization does not change the rank as long as the
            # denominator is positive.
            # ----------------------------------------------------
            eps = 1e-6
            cam_prob = cam + eps

            denom = cam_prob.sum()
            if torch.abs(denom) > eps:
                cam_prob = cam_prob / denom

            probs_flat = cam_prob.flatten()
            gt_flat = gt_mask.flatten()

            n_pixels = probs_flat.numel()

            sorted_indices = torch.argsort(
                probs_flat,
                descending=True,
            )
            sorted_gt = gt_flat[sorted_indices]
            cumulative_fg = torch.cumsum(sorted_gt, dim=0)

            precision_curve = []

            for percentage in percentages:
                k = int(
                    np.ceil(
                        (percentage / 100.0) * n_pixels
                    )
                )
                k = max(1, min(k, n_pixels))

                true_fg = cumulative_fg[k - 1]
                precision_at_k = true_fg / float(k)

                precision_curve.append(
                    precision_at_k.item()
                )

            all_precision_curves.append(precision_curve)

    if len(all_precision_curves) == 0:
        raise RuntimeError(
            f"No cancer image was evaluated for "
            f"{target_dataset} fold {eval_args.fold}."
        )

    all_precision_curves = np.asarray(
        all_precision_curves,
        dtype=np.float32,
    )

    mean_precision = np.mean(all_precision_curves, axis=0)
    std_precision = np.std(all_precision_curves, axis=0)

    return percentages, mean_precision, std_precision, all_precision_curves.shape[0]



# ============================================================
# Generic plotting
# ============================================================

def display_dataset_name(dataset_name):
    if dataset_name == "CAMELYON17_512":
        return "CAMELYON17"
    if dataset_name == "CAMELYON512":
        return "CAMELYON16"
    return dataset_name


def display_domain(dataset_name, fold=None):
    name = display_dataset_name(dataset_name)

    if dataset_name == "CAMELYON17_512" and fold is not None:
        return f"{name} center {fold}"

    return name


def plot_grid(
    results,
    method_order,
    target_folds,
    output_path,
    source_dataset,
    source_fold,
    target_dataset,
):
    """
    Generic grid:
        rows    = WSOL methods
        columns = target folds

    Examples:
        GLAS -> C17 0..4       : 3 x 5
        C17-0 -> C17 1..4      : 3 x 4
        C16 -> C17 0..4        : 3 x 5
        C16 -> GLAS            : 3 x 1
    """
    n_rows = len(method_order)
    n_cols = len(target_folds)

    fig_width = max(5.0, 4.4 * n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, 10),
        sharex=True,
        sharey=True,
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, method in enumerate(method_order):
        for col_idx, fold in enumerate(target_folds):
            ax = axes[row_idx, col_idx]

            res = results[method][fold]
            x = res["percentages"]
            mean = res["mean"]
            std = res["std"]

            ax.plot(
                x,
                mean,
                linewidth=2,
            )

            ax.fill_between(
                x,
                np.clip(mean - std, 0, 1),
                np.clip(mean + std, 0, 1),
                alpha=0.18,
            )

            ax.grid(alpha=0.20)
            ax.set_xlim(1, 100)
            ax.set_ylim(0, 1)

            if row_idx == 0:
                ax.set_title(
                    display_domain(target_dataset, fold),
                    fontsize=14,
                )

            if col_idx == 0:
                ax.set_ylabel(
                    f"{method}\nForeground precision",
                    fontsize=13,
                )

            if row_idx == n_rows - 1:
                ax.set_xlabel(
                    "Top-ranked CAM pixels (%)",
                    fontsize=13,
                )

            ax.text(
                0.98,
                0.04,
                f"n={res['n_images']}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=13,
            )

    source_display = display_domain(
        source_dataset,
        source_fold if source_dataset == "CAMELYON17_512" else None,
    )

    if len(target_folds) == 1:
        target_display = display_domain(
            target_dataset,
            target_folds[0]
            if target_dataset == "CAMELYON17_512"
            else None,
        )
    else:
        target_display = display_dataset_name(target_dataset)

    fig.suptitle(
        f"{source_display} → {target_display}",
        fontsize=17,
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.975])

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved figure to: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cudaid",
        type=str,
        default="0",
    )

    parser.add_argument(
        "--checkpoint_type",
        type=str,
        default="best_localization",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
    )

    # --------------------------------------------------------
    # Source domain
    # --------------------------------------------------------
    parser.add_argument(
        "--source_dataset",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--source_fold",
        type=int,
        default=0,
    )

    # --------------------------------------------------------
    # Target domain
    # --------------------------------------------------------
    parser.add_argument(
        "--target_dataset",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--target_folds",
        nargs="+",
        type=int,
        required=True,
        help=(
            "Target folds/centers to evaluate. "
            "Example: --target_folds 1 2 3 4"
        ),
    )

    # If source and target are C17, this automatically removes
    # source_fold from target_folds.
    parser.add_argument(
        "--skip_same_c17_center",
        type=str2bool,
        default=True,
    )

    # --------------------------------------------------------
    # Same model inputs as your original script
    # --------------------------------------------------------
    parser.add_argument(
        "--pixelcam_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--sat_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--deepmil_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--pixel_wise_classification",
        type=str2bool,
        default=False,
        help=(
            "Kept only for backward compatibility with existing shell "
            "scripts. Pixel-wise classification is now selected "
            "automatically: PixelCAM/EnergyCAM=True, SAT/DeepMIL=False."
        ),
    )

    parser.add_argument(
        "--positive_class",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=(
            "Folder containing GLAS/, CAMELYON17_512/, CAMELYON512/, etc. "
            "Default: $DATASETSH/datasets"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--results_pickle",
        type=str,
        required=True,
    )

    parsedargs = parser.parse_args()

    # Expand ~ and environment variables in model paths.
    parsedargs.pixelcam_path = os.path.expandvars(
        os.path.expanduser(parsedargs.pixelcam_path)
    )
    parsedargs.sat_path = os.path.expandvars(
        os.path.expanduser(parsedargs.sat_path)
    )
    parsedargs.deepmil_path = os.path.expandvars(
        os.path.expanduser(parsedargs.deepmil_path)
    )

    if parsedargs.data_root is None:
        if "DATASETSH" not in os.environ:
            raise RuntimeError(
                "DATASETSH is not defined. Either export DATASETSH "
                "or pass --data_root explicitly."
            )

        data_root = os.path.join(
            os.environ["DATASETSH"],
            "datasets",
        )
    else:
        data_root = parsedargs.data_root

    os.makedirs("tmp_outd", exist_ok=True)

    output_parent = os.path.dirname(parsedargs.output)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    pickle_parent = os.path.dirname(parsedargs.results_pickle)
    if pickle_parent:
        os.makedirs(pickle_parent, exist_ok=True)

    # --------------------------------------------------------
    # Remove source center from C17 -> C17 evaluation.
    # --------------------------------------------------------
    target_folds = list(parsedargs.target_folds)

    if (
        parsedargs.skip_same_c17_center
        and parsedargs.source_dataset == "CAMELYON17_512"
        and parsedargs.target_dataset == "CAMELYON17_512"
    ):
        target_folds = [
            fold
            for fold in target_folds
            if fold != parsedargs.source_fold
        ]

    if len(target_folds) == 0:
        raise ValueError(
            "No target folds remain after filtering."
        )

    print("\n" + "#" * 90)
    print(
        "SOURCE:",
        display_domain(
            parsedargs.source_dataset,
            parsedargs.source_fold
            if parsedargs.source_dataset == "CAMELYON17_512"
            else None,
        ),
    )
    print(
        "TARGET:",
        display_dataset_name(parsedargs.target_dataset),
        "| folds:",
        target_folds,
    )
    print("#" * 90)

    # ========================================================
    # INIT DLLOGGER
    # ========================================================
    log_backends = [
        ArbTextStreamBackend(
            Verbosity.VERBOSE,
            join("tmp_outd", "pixel_precision_grid_log.txt"),
        ),
        ArbStdOutBackend(Verbosity.VERBOSE),
    ]

    DLLogger.GLOBAL_LOGGER = DLLogger.NotInitializedObject()
    DLLogger.init_arb(
        backends=log_backends,
        master_pid=os.getpid(),
    )

    model_paths = {
        "PixelCAM": parsedargs.pixelcam_path,
        "SAT": parsedargs.sat_path,
        "DeepMIL": parsedargs.deepmil_path,
    }

    method_order = [
        "PixelCAM",
        "SAT",
        "DeepMIL",
    ]

    results = {
        method: {}
        for method in method_order
    }

    # ========================================================
    # Loop over PixelCAM / SAT / DeepMIL
    # ========================================================
    for display_method in method_order:
        print("\n" + "=" * 90)
        print(f"METHOD: {display_method}")
        print("=" * 90)

        # ----------------------------------------------------
        # Automatic pixel-classifier configuration
        # ----------------------------------------------------
        # The first CLI slot is the PixelCAM-family model. In your C16
        # experiments this slot contains EnergyCAM, which is treated in
        # the same family here. SAT and DeepMIL do not instantiate the
        # auxiliary pixel classifier.
        use_pixel_wise_classification = (
            display_method == "PixelCAM"
        )

        print(
            "pixel_wise_classification = "
            f"{use_pixel_wise_classification}"
        )

        (
            source_args,
            source_args_dict,
            model,
            loaded_method_name,
        ) = load_source_model(
            exp_path=model_paths[display_method],
            checkpoint_type=parsedargs.checkpoint_type,
            cudaid=parsedargs.cudaid,
            pixel_wise_classification=use_pixel_wise_classification,
            tmp_outd="tmp_outd",
        )

        print(
            f"Requested label: {display_method} | "
            f"method stored in checkpoint: {loaded_method_name}"
        )

        checkpoint_dataset = getattr(
            source_args,
            "dataset",
            None,
        )

        checkpoint_fold = getattr(
            source_args,
            "fold",
            None,
        )

        if (
            checkpoint_dataset is not None
            and checkpoint_dataset != parsedargs.source_dataset
        ):
            print(
                "WARNING: --source_dataset="
                f"{parsedargs.source_dataset}, "
                f"but checkpoint dataset={checkpoint_dataset}"
            )

        if (
            checkpoint_fold is not None
            and int(checkpoint_fold) != int(parsedargs.source_fold)
        ):
            print(
                "WARNING: --source_fold="
                f"{parsedargs.source_fold}, "
                f"but checkpoint fold={checkpoint_fold}"
            )

        # ----------------------------------------------------
        # Evaluate source model on all requested target folds
        # ----------------------------------------------------
        for fold in target_folds:
            print("\n" + "-" * 80)
            print(
                f"{display_method}: "
                f"{display_domain(parsedargs.source_dataset, parsedargs.source_fold if parsedargs.source_dataset == 'CAMELYON17_512' else None)} "
                f"-> "
                f"{display_domain(parsedargs.target_dataset, fold if parsedargs.target_dataset == 'CAMELYON17_512' else None)}"
            )
            print("-" * 80)

            loader, eval_args, metadata_root = build_target_loader(
                source_args=source_args,
                source_args_dict=source_args_dict,
                target_dataset=parsedargs.target_dataset,
                target_fold=fold,
                split=parsedargs.split,
                data_root=data_root,
            )

            (
                percentages,
                mean_precision,
                std_precision,
                n_images,
            ) = compute_foreground_precision_curve(
                model=model,
                eval_args=eval_args,
                loader=loader,
                metadata_root=metadata_root,
                target_dataset=parsedargs.target_dataset,
                data_root=data_root,
                split=parsedargs.split,
                cudaid=parsedargs.cudaid,
                positive_class=parsedargs.positive_class,
            )

            results[display_method][fold] = {
                "percentages": percentages,
                "mean": mean_precision,
                "std": std_precision,
                "n_images": n_images,
            }

            print(f"Number of positive/cancer images: {n_images}")

            for p in [1, 2, 5, 10, 20, 30, 50, 75, 100]:
                idx = p - 1
                print(
                    f"Top {p:3d}%: "
                    f"{mean_precision[idx]:.4f} "
                    f"+/- {std_precision[idx]:.4f}"
                )

            # Save progressively.
            with open(parsedargs.results_pickle, "wb") as f:
                pickle.dump(results, f)

        del model
        torch.cuda.empty_cache()

    # ========================================================
    # Final save + plot
    # ========================================================
    with open(parsedargs.results_pickle, "wb") as f:
        pickle.dump(results, f)

    print(
        f"\nSaved raw results to: "
        f"{parsedargs.results_pickle}"
    )

    plot_grid(
        results=results,
        method_order=method_order,
        target_folds=target_folds,
        output_path=parsedargs.output,
        source_dataset=parsedargs.source_dataset,
        source_fold=parsedargs.source_fold,
        target_dataset=parsedargs.target_dataset,
    )


if __name__ == "__main__":
    main()
