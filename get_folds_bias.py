from calendar import EPOCH
from copy import deepcopy
import os
import sys
from os.path import join, dirname, abspath, basename
import subprocess
from pathlib import Path
import datetime as dt
import argparse
# import more_itertools as mit
from dlib.sf_uda import adadsa
import torch.nn as nn

import json
import numpy as np
import numpy
from tqdm import tqdm
# import pretrainedmodels.utils
import yaml
import munch
import pickle
# from texttable import Texttable
from dlib.datasets.wsol_data_core import get_mask
from PIL import Image
import torch
from torch.cuda.amp import autocast

import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score

#import cuml
#print("cuML version:", cuml.__version__)

# root_dir = dirname(dirname(dirname(abspath(__file__))))
root_dir = dirname(abspath(__file__))
sys.path.append(root_dir)
import torch.nn.functional as F

from dlib.utils.shared import find_files_pattern
from dlib.utils.shared import announce_msg
from  dlib.configure import constants

from dlib.dllogger import ArbJSONStreamBackend
from dlib.dllogger import Verbosity
from dlib.dllogger import ArbStdOutBackend
from dlib.dllogger import ArbTextStreamBackend
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.tools import get_tag
from dlib.utils.tools import Dict2Obj
# from dlib.utils.tools import log_device
from dlib.configure import config

from dlib.learning.inference_wsol import CAMComputer
from dlib.cams import build_std_cam_extractor
from dlib.utils.reproducibility import set_seed
from dlib.process.instantiators import get_model, get_pretrainde_classifier

from dlib.datasets.wsol_loader import get_data_loader
from dlib.datasets.wsol_loader import get_eval_transforms_global
from dlib.datasets.wsol_loader import configure_metadata
from dlib.datasets.wsol_loader import get_class_labels
from dlib.datasets.wsol_loader import get_image_ids
from dlib.learning.train_wsol import Basic, PerformanceMeter
from dlib.utils.tools import get_cpu_device
from dlib.process.parseit import str2bool
from dlib.utils.tools import t2n
import cv2
import json
from glob import glob
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
import shutil
import random



def cl_forward(args, model, images):

    output = model(images)

    if args.task == constants.STD_CL:
        cl_logits = output

    elif args.task == constants.F_CL:
        cl_logits, fcams, im_recon = output
    else:
        raise NotImplementedError

    return cl_logits
    
def _compute_accuracy(args, model, loader):
    num_correct = 0
    num_images = 0

    for i, (images, targets, _, _, _, _, _, _) in enumerate(loader):
        images = images.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            cl_logits = cl_forward(args, model, images)
            pred = cl_logits.argmax(dim=1)

        num_correct += (pred == targets).sum().item()
        num_images += images.size(0)

    classification_acc = num_correct / float(num_images) * 100
    return classification_acc




def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class IgnoreKeyLoader(yaml.SafeLoader):
    pass

# --- Ignore une clé spécifique ---
def ignore_keys(loader, node):
    ignore_key = 'best_valid_tau_cl'
    if isinstance(node, yaml.MappingNode):
        i = 0
        while i < len(node.value):
            if node.value[i][0].value == ignore_key:
                del node.value[i]
            else:
                i += 1
    return loader.construct_mapping(node)

# --- Ignore les scalaires NumPy (float32, int64, etc.) ---
def ignore_numpy_scalars(loader, node):
    try:
        value = loader.construct_scalar(node)
        return float(value)
    except Exception:
        return None

# --- Enregistre les constructeurs ---
IgnoreKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    ignore_keys
)
IgnoreKeyLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy._core.multiarray.scalar',
    ignore_numpy_scalars
)
IgnoreKeyLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar',
    ignore_numpy_scalars
)


def _compute_accuracy(args, model, loader):
    num_correct = 0
    num_images = 0

    for i, (images, targets, _, _, _, _, _, _, _) in enumerate(loader):
        images = images.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            cl_logits = cl_forward(args, model, images)
            pred = cl_logits.argmax(dim=1)

        num_correct += (pred == targets).sum().item()
        num_images += images.size(0)

    classification_acc = num_correct / float(num_images) * 100
    return classification_acc



def compute_kl_uniform(preds, num_classes):
    # p(y) estimé empiriquement
    hist = torch.bincount(preds, minlength=num_classes).float()
    p = hist / hist.sum()          # distribution empirique des classes
    u = torch.ones_like(p) / num_classes  # prior uniforme
    kl = (p * (p / u).log()).sum()
    return float(kl.item())



def build_biased_image_ids(imgid_to_pred, bias_percent, seed=42):
    np.random.seed(seed)

    # split par prédiction
    ids_0 = [k for k, v in imgid_to_pred.items() if v == 0]
    ids_1 = [k for k, v in imgid_to_pred.items() if v == 1]

    bias = bias_percent / 100.0
    p_major = 0.5 + bias
    p_minor = 0.5 - bias

    # classe dominante
    if len(ids_1) >= len(ids_0):
        major_ids, minor_ids = ids_1, ids_0
        major_label = 1
    else:
        major_ids, minor_ids = ids_0, ids_1
        major_label = 0

    N = len(imgid_to_pred)
    n_major = int(N * p_major)
    n_minor = int(N * p_minor)

    sel_major = np.random.choice(
        major_ids, size=min(n_major, len(major_ids)), replace=False
    )
    sel_minor = np.random.choice(
        minor_ids, size=min(n_minor, len(minor_ids)), replace=False
    )

    selected_ids = list(sel_major) + list(sel_minor)
    random.shuffle(selected_ids)

    return selected_ids, major_label


def write_biased_fold(
    selected_image_ids,
    src_fold_root,
    dst_fold_root,
    files=("class_labels.txt", "image_ids.txt", "image_sizes.txt", "localization.txt"),
):
    import shutil


    for split in ["valcl", "valpx", "test"]:
        (dst_fold_root / split).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(src_fold_root / split / f, dst_fold_root / split / f)


    train_src = src_fold_root / "train"
    data = {f: open(train_src / f).readlines() for f in files}


    image_ids = [l.strip() for l in data["image_ids.txt"]]

    id_to_idx = {img_id: i for i, img_id in enumerate(image_ids)}


    missing = set(selected_image_ids) - set(id_to_idx.keys())
    assert len(missing) == 0, f"Missing image_ids: {list(missing)[:5]}"

    selected_indices = [id_to_idx[i] for i in selected_image_ids]

    # écrire train biaisé
    train_dst = dst_fold_root / "train"
    train_dst.mkdir(parents=True, exist_ok=True)

    for f in files:
        lines = [data[f][i] for i in selected_indices]
        with open(train_dst / f, "w") as fw:
            fw.writelines(lines)

def build_balanced_biased_dataset_auto(
    imgid_to_gt,
    imgid_to_pred,
    bias,
    seed=42
):
    random.seed(seed)

    classes = sorted(set(imgid_to_gt.values()))
    assert len(classes) == 2, "Binary classification only"

    # --------------------------------------------------
    # 1. Détecter automatiquement la classe biaisée
    # --------------------------------------------------
    err_rate = {}
    for c in classes:
        ids_c = [i for i in imgid_to_gt if imgid_to_gt[i] == c]
        err_rate[c] = sum(imgid_to_pred[i] != c for i in ids_c) / len(ids_c)

    biased_class = max(err_rate, key=err_rate.get)
    clean_class  = [c for c in classes if c != biased_class][0]

    # --------------------------------------------------
    # 2. Pools disponibles
    # --------------------------------------------------
    clean_good = [
        i for i in imgid_to_gt
        if imgid_to_gt[i] == clean_class and imgid_to_pred[i] == clean_class
    ]

    bias_good = [
        i for i in imgid_to_gt
        if imgid_to_gt[i] == biased_class and imgid_to_pred[i] == biased_class
    ]

    bias_bad = [
        i for i in imgid_to_gt
        if imgid_to_gt[i] == biased_class and imgid_to_pred[i] != biased_class
    ]

    # --------------------------------------------------
    # 3. 🔥 CALCUL AUTOMATIQUE n_per_class (ICI)
    # --------------------------------------------------
    max_pc_good = int(len(bias_good) / max(1e-6, (1 - bias)))
    max_pc_bad  = int(len(bias_bad)  / max(1e-6, bias)) if bias > 0 else 10**9

    n_per_class = min(
        len(clean_good),
        max_pc_good,
        max_pc_bad
    )

    assert n_per_class > 0, "No feasible balanced biased dataset"

    # --------------------------------------------------
    # 4. Sélection finale
    # --------------------------------------------------
    n_bad  = int(round(bias * n_per_class))
    n_good = n_per_class - n_bad

    sel_clean = random.sample(clean_good, n_per_class)
    sel_bias  = (
        random.sample(bias_good, n_good) +
        random.sample(bias_bad,  n_bad)
    )

    selected_ids = sel_clean + sel_bias
    random.shuffle(selected_ids)

    stats = {
        "biased_class": biased_class,
        "clean_class": clean_class,
        "bias": bias,
        "n_per_class": n_per_class,
        "counts": {
            "clean_correct": len(sel_clean),
            "biased_correct": n_good,
            "biased_wrong": n_bad,
        }
    }

    return selected_ids, stats





def get_features(exp_path, sf_uda_source_folder,image_ids_to_draw,image_ids_to_draw_target, checkpoint_type, dataset, cudaid, split, tmp_outd='tmp_outd', parsedargs=None, target_method=None):


    with open(join(exp_path, 'config_obj_final.yaml'), 'r') as fy:
    #with open(join(exp_path, 'OpenImagesSrc-0-deit_sat_base_patch16_224-SAT-GAP-cp_best_classification', 'config_model.yaml'), 'r') as fy:
        args_dict = yaml.load(fy, Loader=IgnoreKeyLoader)
        # args_dict = yaml.safe_load(fy)
        args_dict['model']['freeze_encoder'] = False
        args_dict['pixel_wise_classification'] = False
        args = Dict2Obj(args_dict)
        args.outd = tmp_outd
        args.distributed = False
        args.eval_checkpoint_type = checkpoint_type

    os.makedirs(args.outd, exist_ok=True)

    _DEFAULT_SEED = args.MYSEED
    os.environ['MYSEED'] = str(args.MYSEED)

    tag = get_tag(args, checkpoint_type=checkpoint_type)

    msg = 'Task: {} \t box_v2_metric: {} \t' \
        'Dataset: {} \t Method: {} \t ' \
        'Encoder: {} \t'.format(args.task, args.box_v2_metric, args.dataset,
                                args.method, args.model['encoder_name'])
    encoder_name = args.model['encoder_name']
    method_name = args.method
    source_dataset = dataset
    
    # DLLogger.log(fmsg("Start time: {}".format(t0)))
    DLLogger.log(fmsg(msg))

    set_seed(seed=_DEFAULT_SEED, verbose=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:{}'.format(cudaid))
    
    #tag = get_tag(args, checkpoint_type=checkpoint_type)
    # path_cl = join(exp_path, tag)
    # args.sf_uda_source_folder = path_cl
    # config_model.yaml
    tag = get_tag(args, checkpoint_type=checkpoint_type)
    path_cl = join(exp_path, tag)
    with open(join(path_cl, 'config_model.yaml'), 'r') as fy:
        args_dict = yaml.load(fy, Loader=IgnoreKeyLoader)
        # args_dict = yaml.safe_load(fy)
        # args_dict['model']['freeze_encoder'] = False
        if 'PixelCAM' in args.method:
            args_dict['pixel_wise_classification'] = True
            args_dict['anchors_ortogonal'] = False
            args_dict['batch_norm_pixel_classifier'] = False
            args_dict['low_res'] = False
            args_dict['multiple_layer_pixel_classifier'] = False
            args_dict['detach_pixel_classifier'] = False
            args_dict['one_layer_pixel_classifier'] = False
            args_dict['cpt_cam_entropy'] = False
        else:
            args_dict['pixel_wise_classification'] = False
            args_dict['anchors_ortogonal'] = False
            args_dict['batch_norm_pixel_classifier'] = False
            args_dict['low_res'] = False
            args_dict['multiple_layer_pixel_classifier'] = False
            args_dict['detach_pixel_classifier'] = False
            args_dict['one_layer_pixel_classifier'] = False
        args = Dict2Obj(args_dict)
        args.outd = tmp_outd
        args.distributed = False
        args.eval_checkpoint_type = checkpoint_type
        args.model['folder_pre_trained_cl'] = None

    args.sf_uda = False

    model = get_model(args)[0]

    print(f'Loading model for {method_name}-{encoder_name} from {path_cl}')

    if 'PixelCAM' in method_name:
        if "deit" in encoder_name:
            model_sat = torch.load(join(path_cl, 'model.pt'),map_location=get_cpu_device())

            model.load_state_dict(model_sat, strict=True)
        else:
            encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                                map_location=get_cpu_device())
            model.encoder.super_load_state_dict(encoder_w, strict=True)

            header_w = torch.load(join(path_cl, 'classification_head.pt'),
                                map_location=get_cpu_device())
            model.classification_head.load_state_dict(header_w, strict=True)

            header_p = torch.load(join(path_cl, 'pixel_wise_classification_head.pt'),
                                map_location=get_cpu_device())
            model.pixel_wise_classification_head.load_state_dict(header_p, strict=True)

    elif method_name == 'NEGEV':
        encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                            map_location=get_cpu_device())
        model.encoder.super_load_state_dict(encoder_w, strict=True)

        header_w = torch.load(join(path_cl, 'classification_head.pt'),
                            map_location=get_cpu_device())
        model.classification_head.load_state_dict(header_w, strict=True)

        decoder_w = torch.load(join(path_cl, 'decoder.pt'),
                            map_location=get_cpu_device())
        model.decoder.super_load_state_dict(decoder_w, strict=True)

        seg_head_w = torch.load(join(path_cl, 'segmentation_head.pt'),
                            map_location=get_cpu_device())
        model.segmentation_head.load_state_dict(seg_head_w, strict=True)

    else:
        if "deit" in encoder_name:
            model_sat = torch.load(join(path_cl, 'model.pt'),map_location=get_cpu_device())

            model.load_state_dict(model_sat, strict=False)
        else:
            encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                                map_location=get_cpu_device())
            model.encoder.super_load_state_dict(encoder_w, strict=True)

            header_w = torch.load(join(path_cl, 'classification_head.pt'),
                                map_location=get_cpu_device())
            model.classification_head.load_state_dict(header_w, strict=True)



    DLLogger.log(fmsg("Model checkpoint Loaded from {}".format(path_cl)))
        
    model.to(device)
    model.eval()

    basic_config = config.get_config(ds=source_dataset, fold=args.fold, magnification=args.magnification)

    args.data_paths = basic_config['data_paths']
    args.metadata_root = basic_config['metadata_root']
    args.mask_root = basic_config['mask_root']
    args.cam_curve_interval = basic_config['cam_curve_interval']

    ####################################################################################
    ###############load performance log file from orignal exp checkpoint ###############
    ####################################################################################
    assert split == constants.TESTSET or split == constants.VALIDSET or split == constants.TRAINSET
    #split == constants.VALIDSET
    #split = constants.VALIDSET
    log_file_path_best_loc = os.path.join(exp_path, f'performance_log_{checkpoint_type}.pickle')
    if os.path.isfile(log_file_path_best_loc):
        with open(log_file_path_best_loc, 'rb') as f:
            results = pickle.load(f)
        # if split == constants.TESTSET:
        #     clas_acc_from_orginal_exp_path = results[split]['classification']['value_per_epoch'][-1]
        #     loc_acc_from_orginal_exp_path = results[split]['localization_IOU_50']['value_per_epoch'][-1]
        # else:
        best_epoch = -1 if split == constants.TESTSET else results[split][checkpoint_type.replace('best_', '')]['best_epoch']
        #clas_acc_from_orginal_exp_path = results[split]['classification']['value_per_epoch'][best_epoch]
        #loc_acc_from_orginal_exp_path = results[split]['localization']['value_per_epoch'][best_epoch]


    ####################################################################################
    ####################################################################################
    DLLogger.flush()
    
    metadata_root = join(constants.RELATIVE_META_ROOT, dataset, f"fold-{args.fold}")
    #read sys var DATASETSH
    args_dict['data_root'] = os.path.join(os.environ['DATASETSH'], 'datasets')
    target_domain_data_paths = config.configure_data_paths(args_dict, dataset)

    metadata_root_CAME = join('./folds/wsol-done-right-splits', 'CAMELYON512', f"fold-{parsedargs.fold_dataset}")
    args_dict['data_root'] = '/export/gauss/vision/Aguichemerre/datasets'
    target_domain_data_paths_CAME = config.configure_data_paths(args_dict, 'CAMELYON512')

    
    loaders = get_data_loader(
            data_roots=target_domain_data_paths,
            metadata_root=metadata_root,
            batch_size=32,#args.batch_size,
            workers=args.num_workers,
            resize_size=args.resize_size,
            crop_size=args.crop_size,
            proxy_training_set=args.proxy_training_set,
            num_val_sample_per_class=args.num_val_sample_per_class,
            std_cams_folder=args.std_cams_folder,
            # distributed_eval=False,
            get_splits_eval=[parsedargs.split],
            #constants.TRAINSET
            eval_batch_size = 32#args.eval_batch_size,
        )
    
    
    
    overlay_images = {}
    input_images = {}
    gt_masks = {}


    image_features_all = []
    image_labels_all = []
    J2_px = []
    J2_img = []
    DBI_img = []

    cam_computer = CAMComputer(
                        args=deepcopy(args),
                        model=model,
                        loader=loaders[parsedargs.split],
                        metadata_root=os.path.join(metadata_root, parsedargs.split),
                        mask_root=args.mask_root,
                        iou_threshold_list=args.iou_threshold_list,
                        dataset_name=args.dataset,
                        split= parsedargs.split,
                        cam_curve_interval=args.cam_curve_interval,
                        multi_contour_eval=args.multi_contour_eval,
                        out_folder=args.outd,
                    )
    

    acc_cl = _compute_accuracy(args, model, loaders[parsedargs.split])

 
    entropies = []
    preds = []
    gts = []     

    logits_all = []
    probs_all  = []
    preds_all  = []


    for batch_idx, (images, targets, p_glabel, index, raw_imgs, std_cams, _, views, _) in tqdm(
        enumerate(loaders[split]), ncols=constants.NCOLS,
        total=len(loaders[split])):
        image_size = images.shape[2:]
        images = images.to(device)
        targets = targets.to(device)
        

        GroundTruth = []

        with torch.no_grad():
           out = model(images.cuda())
        GroundTruth = []
        for image, target, image_id in zip(images, targets, index):
            image_size = images.shape[2:]

        with torch.no_grad():
            out = model(images.cuda())
            img_features = model.lin_ft.detach().cpu()
            pixel_features = model.encoder_last_features.detach().cpu()  # [1, C, H, W]


            image_features_all.append(img_features)
            image_labels_all.append(targets)

            logits = model(images.cuda())
            probs  = torch.softmax(logits, dim=1)

            logits_all.append(logits.cpu())
            probs_all.append(probs.cpu())
            preds_all.append(torch.argmax(probs, dim=1).cpu())


    logits_all = torch.cat(logits_all, dim=0)
    probs_all  = torch.cat(probs_all,  dim=0)
    preds_all  = torch.cat(preds_all,  dim=0)
    labels_all = torch.cat(image_labels_all, dim=0)

    # ============================================================
    # Build image_id -> prediction mapping (TRAIN ONLY)
    # ============================================================

    image_ids_all = []

    for _, _, _, index, *_ in loaders[split]:
        image_ids_all.extend(index)

    assert len(image_ids_all) == len(preds_all), \
        f"Mismatch: {len(image_ids_all)} image_ids vs {len(preds_all)} preds"

    imgid_to_pred = {
        img_id: int(pred)
        for img_id, pred in zip(image_ids_all, preds_all.cpu().numpy())
    }

    # ============================================================
    # Build image_id -> GT label mapping
    # ============================================================

    imgid_to_gt = {}

    for _, targets, _, index, *_ in loaders[split]:
        for img_id, gt in zip(index, targets):
            imgid_to_gt[img_id] = int(gt.item())

    assert set(imgid_to_gt.keys()) == set(imgid_to_pred.keys()), \
        "GT / Pred image_id mismatch"


    # ============================================================
    # === CREATE BIASED TARGET FOLDS (TRAIN ONLY)
    # ============================================================

    if split == constants.TRAINSET:
        from pathlib import Path
        import shutil

        ROOT = Path("folds/wsol-done-right-splits") / dataset
        SRC_FOLD = ROOT / f"fold-{parsedargs.fold_dataset}"

        BIAS_LEVELS = [1, 10, 20, 50]

        image_ids_all = []
        for _, _, _, index, *_ in loaders[split]:
            image_ids_all.extend(index)

        assert len(image_ids_all) == len(preds_all)

        for b in BIAS_LEVELS:
            fold_id = 300 + b
            fold_name = f"fold-{fold_id}"
            dst_fold = ROOT / fold_name

            selected_ids, stats = build_balanced_biased_dataset_auto(
                imgid_to_gt=imgid_to_gt,
                imgid_to_pred=imgid_to_pred,
                bias=b / 100.0,      
                seed=fold_id
            )

            ratio = np.mean([imgid_to_pred[i] for i in selected_ids])

            # ===== DEBUG / VERIFICATION DU BIAIS =====
            n0 = sum(imgid_to_pred[i] == 0 for i in selected_ids)
            n1 = sum(imgid_to_pred[i] == 1 for i in selected_ids)

            print(
                f"[BIAS-STATS] fold={fold_name} | "
                f"total={len(selected_ids)} | "
                f"normal={n0} | cancer={n1} | "
                f"bias_ratio={abs(n1 - n0) / (n1 + n0):.4f}"
            )

            gt0 = sum(imgid_to_gt[i] == 0 for i in selected_ids)
            gt1 = sum(imgid_to_gt[i] == 1 for i in selected_ids)

            print(
                f"[GT-STATS] fold={fold_name} | "
                f"gt_normal={gt0} | gt_cancer={gt1}"
            )


            # (optionnel) sanity check
            assert n0 + n1 == len(selected_ids)

            write_biased_fold(
                selected_image_ids=selected_ids,
                src_fold_root=SRC_FOLD,
                dst_fold_root=dst_fold,
    )


    return 0
        
def fast_eval():
    t0 = dt.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cudaid", type=str, default=None, help="cuda id.")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--checkpoint_type", type=str, default=None)
    parser.add_argument("--encoder_name", type=str, default=None)
    parser.add_argument("--dataset_type", type=str, default=None)
    # parser.add_argument("--exp_path", type=str, default=None)
    parser.add_argument("--tmp_outd", type=str, default='tmp_outd')
    parser.add_argument('--noise_level_for_eval_with_noisy_bbox', nargs='+',
                        type=int, default=[5, 10, 15, 20, 25, 30, 35 ,40, 45, 50])
    #parser.add_argument("--target_dataset", type=str, default=None,
    #                    help="Name of the dataset.", required=True, choices=[constants.CAMELYON512, constants.GLAS])
    parser.add_argument('--image_ids_to_draw', nargs='+', type=str, default=None)
    parser.add_argument('--image_ids_to_draw_target', nargs='+', type=str, default=None)
    parser.add_argument("--source_dataset", type=str, default=None, help="Source dataset")
    #parser.add_argument("--path_pre_trained_source", type=str, default=None, help="Path to the pre-trained source model.")
    parser.add_argument("--path_pre_trained_source", type=str, default=None, help="Path to the pre-trained source model.")
    parser.add_argument("--fold_dataset", type=int, default=0, help="fold.")
    parser.add_argument("--sfda_method", type=str, default=0, help="fold.")
    parser.add_argument("--wsol_method", type=str, default='wsol', help="fold.")
    

    parsedargs = parser.parse_args()
    
    os.makedirs('tmp_outd', exist_ok=True)  
    log_backends = [
                # ArbJSONStreamBackend(Verbosity.VERBOSE,
                #                     join(args.outd, "log.json")),
                ArbTextStreamBackend(Verbosity.VERBOSE,
                                    join('tmp_outd', f"split_{parsedargs.split}_log.txt")),
            ]
    
    _VERBOSE = True
    if _VERBOSE:
        log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))
        
    DLLogger.GLOBAL_LOGGER = DLLogger.NotInitializedObject()
        
    DLLogger.init_arb(backends=log_backends, master_pid=os.getpid())
    ##########

    base_checkpoint_types = [parsedargs.checkpoint_type]
        
    for checkpoint_type_extended in base_checkpoint_types:
        checkpoint_type = checkpoint_type_extended
        
        #split = parsedargs.split
        split = parsedargs.split
        # exp_path = parsedargs.exp_path
        # # checkpoint_type = parsedargs.checkpoint_type
        # # tmp_outd = join(parsedargs.tmp_outd, os.path.split(exp_path)[-1])#, 'split_'+split+'_'+checkpoint_type)
        # tmp_outd = join(exp_path, '0_re-eval_log')#, 'split_'+split+'_'+checkpoint_type)
        # # tmp_outd = parsedargs.tmp_outd
        # assert os.path.isdir(exp_path)
        assert split == constants.TESTSET or split == constants.VALIDSET or split == constants.TRAINSET, split
        
        _CODE_FUNCTION = 'fast_eval_{}'.format(split)

        #target_methods = ['SOURCE', 'SFDE', 'SHOT', 'CDCL', 'ADADSA']
        #target_methods = ['DeepMIL', 'PixelCAM DL', 'GradCAMpp', 'PixelCAM GC', 'LayerCAM', 'PixelCAM LC', 'SAT', 'PixelCAM SAT']
        #target_methods = ['PixelCAM DL','PixelCAM GC','PixelCAM LC','PixelCAM SAT']
        #target_methods = ['DeepMIL','GradCAMpp','LayerCAM','SAT']
        #target_methods = ['GradCAMpp']

        target_methods = ['SOURCE']

        method_name_lst = []
        for ind_method, target_method in enumerate(target_methods):
            ind_method+= 2
                
            # Get model path   
            if target_method == 'SOURCE':
                exp_path = parsedargs.path_pre_trained_source
            else:
                exp_path = parsedargs.path_pre_trained_source[target_method]
                #exp_path = parsedargs.target_domain_exp_path[target_method]


            #Get features at the pixel level
            get_features(exp_path=exp_path, sf_uda_source_folder=parsedargs.path_pre_trained_source,image_ids_to_draw=parsedargs.image_ids_to_draw,image_ids_to_draw_target=parsedargs.image_ids_to_draw_target, checkpoint_type=checkpoint_type, dataset=parsedargs.source_dataset, cudaid=parsedargs.cudaid, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs, target_method=target_method)

if __name__ == '__main__':
    fast_eval()