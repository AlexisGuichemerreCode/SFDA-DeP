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
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import seaborn as sns



from collections import defaultdict



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



def get_resized_gt_mask(mask_path, ignore_path, dataset, label, size, device):
    if dataset == constants.GLAS or (dataset == constants.CAMELYON512 and label == 1):
        gt_mask = get_mask(f'/export/gauss/vision/Aguichemerre/datasets/{dataset}',mask_path,ignore_path)
        #gt_mask = get_mask(dataset, mask_path, ignore_path)
        gt_tensor = torch.tensor(gt_mask, dtype=torch.float32).to(device)
        gt_resize = F.interpolate(gt_tensor[None, None], size=size, mode='bilinear', align_corners=False)
        return (gt_resize.squeeze(0).squeeze(0) > 0.5)
    else:
        gt_resize = torch.zeros(size, dtype=torch.float32, device=device)
        
    return torch.zeros(size, dtype=torch.bool, device=device)


def compute_energy_distributions(model, loader, cam_computer, dataset_name, energy_fn, split, device):

    energy_data = {
    'images': [],
    'pixels': [],
    'per_class': defaultdict(list),
    'foreground': [],
    'background': []}

    for batch_idx, (images, targets, p_glabel, index, raw_imgs, std_cams, _, views) in tqdm(
        enumerate(loader[split]), ncols=constants.NCOLS,
        total=len(loader[split])):

        images = images.to(device)
        targets = targets.to(device)

        # Per-image
        with torch.no_grad():
            lgt_imgs = model(images)
            px_lin_ft = model.encoder_last_features
            lgt_pxs = model.pixel_wise_classification_head(px_lin_ft)[0]

            energy_images = energy_fn(lgt_imgs)
            energy_pixels = energy_fn(lgt_pxs)

            energy_data['images'].append(energy_images.cpu())
            energy_data['pixels'].append(energy_pixels.flatten(start_dim=1).cpu())

            for energy, label in zip(energy_images.cpu(), targets.cpu()):
                energy_data['per_class'][int(label)].append(energy.item())


        # Per-pixel foreground / background
        for i, (label, image_id) in enumerate(zip(targets, index)):
            _, _, h, w = px_lin_ft.shape
            gt_bin = get_resized_gt_mask(
                cam_computer.evaluator.mask_paths[image_id],
                cam_computer.evaluator.ignore_paths[image_id],
                dataset_name,
                label.item(),
                size=(h, w),
                device=device
            )

            energy_map = energy_pixels[i]  # (H, W)
            energy_data['foreground'].append(energy_map[gt_bin].detach().cpu())
            energy_data['background'].append(energy_map[~gt_bin].detach().cpu())

    # Concatenation
    energy_data['images'] = torch.cat(energy_data['images']).numpy()
    energy_data['pixels'] = torch.cat(energy_data['pixels']).view(-1).numpy()
    energy_data['foreground'] = torch.cat(energy_data['foreground']).numpy()
    energy_data['background'] = torch.cat(energy_data['background']).numpy()

    return energy_data


def plot_energy_histograms_by_class(
    source_dict, target_dict, out_dir, title_prefix="",
    source_dataset=None, target_dataset=None
):

    os.makedirs(out_dir, exist_ok=True)
    all_classes = sorted(set(source_dict.keys()) | set(target_dict.keys()))

    dataset_label = ""
    title_suffix = ""
    if source_dataset and target_dataset:
        dataset_label = f"source_{source_dataset}_vs_target_{target_dataset}"
        title_suffix = f" (Source: {source_dataset} | Target: {target_dataset})"
    else:
        dataset_label = "source_vs_target"

    for cls in all_classes:
        source_vals = source_dict.get(cls, [])
        target_vals = target_dict.get(cls, [])

        if not source_vals and not target_vals:
            continue

        plt.figure(figsize=(8, 6))

        if source_vals:
            plt.hist(source_vals, bins=50, alpha=0.5, density=True,
                     color='blue', edgecolor='black', label='Source')

        if target_vals:
            plt.hist(target_vals, bins=50, alpha=0.5, density=True,
                     color='red', edgecolor='black', label='Target')

        title = f"{title_prefix} - Class {cls}{title_suffix}"
        file_name = f"hist_class_{cls}_{dataset_label}.png"

        plt.title(title)
        plt.xlabel("Energy")
        plt.ylabel("Density")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, file_name), dpi=300)
        plt.close()


def plot_global_energy_histogram(source_vals, target_vals, out_dir, title, xlim=None, type = None, label_src="Source", label_tgt="Target", source_dataset=None, target_dataset=None):

    os.makedirs(out_dir, exist_ok=True)


    if source_dataset and target_dataset and type:
        figure_name = f"hist_{type}_pixels_source_{source_dataset}_vs_target_{target_dataset}.png"


    
    plt.figure(figsize=(8, 6))
    plt.hist(source_vals, bins=100, alpha=0.5, density=True,
             label=label_src, color='blue', edgecolor='black')
    plt.hist(target_vals, bins=100, alpha=0.5, density=True,
             label=label_tgt, color='red', edgecolor='black')
    plt.title(title)
    plt.xlabel("Energy")
    plt.ylabel("Density")
    plt.legend(loc="upper right", fontsize=12)

    if xlim:
        plt.xlim(*xlim)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, figure_name), dpi=300)
    plt.close()



def cl_forward(args, model, images):

    output = model(images)

    if args.task == constants.STD_CL:
        cl_logits = output

    elif args.task == constants.F_CL:
        cl_logits, fcams, im_recon = output
    else:
        raise NotImplementedError

    return cl_logits

def sgld_sample(x_init, energy_fn, n_iter=10, step_size=1e-2, noise_scale=1e-2):
    x = x_init.clone().detach().requires_grad_(True)

    for _ in range(n_iter):
        energy_map = energy_fn(x)  # shape: (H, W)

        # Backward pixel-wise sans agrégation
        energy_map.backward(torch.ones_like(energy_map))

        noise = torch.randn_like(x) * noise_scale
        x.data += 0.5 * step_size * x.grad.data + noise

        x.grad.detach_()
        x.grad.zero_()

    return x.detach()

def sgld_sample_ebm(init_sample, energy_fn, n_steps, sgld_lr, sgld_std,
                    y=None, model=None, gt_bin=None, track_accuracy=False, track_every=50):
    """
    Perform SGLD sampling for energy-based models.

    Args:
        init_sample (torch.Tensor): Initial sample (e.g., random noise or prior sample).
        energy_fn (callable): Energy function that takes `x` (and optionally `y`) and returns the energy.
        n_steps (int): Number of SGLD steps to perform.
        sgld_lr (float): Learning rate for SGLD updates.
        sgld_std (float): Standard deviation of the noise added during SGLD.
        y (torch.Tensor, optional): Optional label or conditioning variable for the energy function.

    Returns:
        torch.Tensor: Final samples after SGLD.
    """

    x_k = torch.autograd.Variable(init_sample.clone(), requires_grad=True)
    tracked_accuracies = []

    for k in range(n_steps):
       
        energy = energy_fn(x_k)

        fake_grad = torch.ones_like(energy)
        
        f_prime = torch.autograd.grad(energy, [x_k], grad_outputs=fake_grad, retain_graph=True)[0]
        #f_prime = torch.autograd.grad(energy.sum(), [x_k], retain_graph=True)[0]

        x_k.data = x_k.data + sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)

        if track_accuracy and (k + 1) % track_every == 0:
            with torch.no_grad():
                logits = model.pixel_wise_classification_head(x_k)
                pred_labels = logits[0].argmax(dim=1).squeeze(0)  # (H, W)

                gt_bin = gt_bin.to(pred_labels.device).view_as(pred_labels)
                acc = (pred_labels == gt_bin).float().mean().item()

                #acc = accuracy_score(
                #    gt_bin.view(-1).cpu().numpy(),
                #    pred_labels.view(-1).cpu().numpy()
                #)
                tracked_accuracies.append((k + 1, acc))

    return x_k.detach(), tracked_accuracies if track_accuracy else None

def energy_fn(logits):
    """
    Compute the energy map for the given features using the model.

    Args:
        feat (torch.Tensor): Input features (e.g., pixel features).
        model (torch.nn.Module): The model containing the pixel-wise classification head.

    Returns:
        torch.Tensor: Energy map (H, W).
    """
    energy_map = -torch.logsumexp(logits, dim=1)  # logsumexp over classes
    return energy_map  # (H, W)

    
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



def extract_features_source_target(mask_source, feature_source, label_source, cam_source, image_id_source,mask_target, feature_target, label_target, cam_target, image_id_target, anchors):
    cancer_features = np.empty((0, 2048))
    non_cancer_features = np.empty((0, 2048))
    background_features = np.empty((0, 2048))

    #source
    mask_source = torch.from_numpy(mask_source)
    non_zero_indices_source = torch.nonzero(mask_source, as_tuple=True)
    non_zero_feature_source = feature_source[0, :, non_zero_indices_source[0], non_zero_indices_source[1]]
    zero_indices_source = torch.nonzero(mask_source == 0, as_tuple=True)
    zero_feature_source = feature_source[0, :, zero_indices_source[0], zero_indices_source[1]]
    features1_source = non_zero_feature_source.detach().cpu().numpy().T
    features2_source = zero_feature_source.detach().cpu().numpy().T

    #target
    mask_target = torch.from_numpy(mask_target)
    non_zero_indices_target = torch.nonzero(mask_target, as_tuple=True)
    non_zero_feature_target = feature_target[0, :, non_zero_indices_target[0], non_zero_indices_target[1]]
    zero_indices_target = torch.nonzero(mask_target == 0, as_tuple=True)
    zero_feature_target = feature_target[0, :, zero_indices_target[0], zero_indices_target[1]]
    features1_target = non_zero_feature_target.detach().cpu().numpy().T
    features2_target= zero_feature_target.detach().cpu().numpy().T

    anchors_resize = anchors.squeeze().detach().cpu().numpy()
    all_features = np.concatenate([features1_source, features2_source, features1_target, features2_target])
    tsne = TSNE(n_components=2)
    embedded_features = tsne.fit_transform(all_features)

    # Diviser les features intégrées en fonction de la taille des listes de features originales
    embedded_features1_source = embedded_features[:features1_source.shape[0]]
    embedded_features2_source = embedded_features[features1_source.shape[0]:features1_source.shape[0]+features2_source.shape[0]]
    embedded_features1_target = embedded_features[features1_source.shape[0]+features2_source.shape[0]:features1_source.shape[0]+features2_source.shape[0]+features1_target.shape[0]]
    embedded_features2_target = embedded_features[features1_source.shape[0]+features2_source.shape[0]+features1_target.shape[0]:]

    # Visualiser les résultats avec un code couleur
    plt.scatter(embedded_features1_source[:, 0], embedded_features1_source[:, 1], color='blue', label='Foreground GLAS')
    plt.scatter(embedded_features2_source[:, 0], embedded_features2_source[:, 1], color='red', label='Background GLAS')
    plt.scatter(embedded_features1_target[:, 0], embedded_features1_target[:, 1], color='black')
    plt.scatter(embedded_features2_target[:, 0], embedded_features2_target[:, 1], color='orange', label='Background CAMELYON')

  
    plt.legend()
    if label_source == 1:
        classe = 'cancer'
    else:
        classe = 'normal'


    parts = image_id_source.split('/')
    clean_image_id = parts[1].replace('.bmp', '')
    plt.title(f"T-SNE visualization between source and target features \n from CAMELYON to GLAS at the pixel level \n for a {classe} image ", fontsize=10)
    # Sauvegarder l'image
    plt.savefig('tsne_plot_shift_cam_to_glas_normal.png')
    plt.close()
    return cancer_features, non_cancer_features, background_features



def extract_features(mask, feature, label, image_id):
    mask = torch.from_numpy(mask)

    # (foreground)
    non_zero_indices = torch.nonzero(mask, as_tuple=True)
    non_zero_feature = feature[0, :, non_zero_indices[0], non_zero_indices[1]]

    # (background)
    zero_indices = torch.nonzero(mask == 0, as_tuple=True)
    zero_feature = feature[0, :, zero_indices[0], zero_indices[1]]

    foreground_features = non_zero_feature.detach().cpu().numpy().T  # shape: (N_pixels, 2048)
    background_features = zero_feature.detach().cpu().numpy().T

    return foreground_features, background_features



def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class IgnoreKeyLoader(yaml.SafeLoader):
    def ignore_keys(self, node):
        ignore_key = 'best_valid_tau_cl'
        if isinstance(node, yaml.MappingNode):
            i = 0
            while i < len(node.value):
                if node.value[i][0].value == ignore_key:
                    del node.value[i]
                else:
                    i += 1
        return self.construct_yaml_map(node)

    def ignore_numpy_scalars(self, node):
        return None  # or any other dummy value

IgnoreKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, IgnoreKeyLoader.ignore_keys)
IgnoreKeyLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', IgnoreKeyLoader.ignore_numpy_scalars)


def get_features(exp_path, sf_uda_source_folder,image_ids_to_draw,image_ids_to_draw_target, checkpoint_type, source_dataset,target_dataset, cudaid, split, tmp_outd='tmp_outd', parsedargs=None):


    with open(join(exp_path, 'config_obj_final.yaml'), 'r') as fy:
        args_dict = yaml.load(fy, Loader=IgnoreKeyLoader)
        # args_dict = yaml.safe_load(fy)
        args_dict['model']['freeze_encoder'] = False
        args_dict['model']['folder_pre_trained_cl'] = None
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
    source_dataset = source_dataset
    
    # DLLogger.log(fmsg("Start time: {}".format(t0)))
    DLLogger.log(fmsg(msg))

    set_seed(seed=_DEFAULT_SEED, verbose=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:{}'.format(cudaid))

    tag = get_tag(args, checkpoint_type=checkpoint_type)
    path_cl = join(exp_path, tag)
    with open(join(path_cl, 'config_model.yaml'), 'r') as fy:
        args_dict = yaml.load(fy, Loader=IgnoreKeyLoader)
        # args_dict = yaml.safe_load(fy)
        # args_dict['model']['freeze_encoder'] = False
        args_dict['model']['folder_pre_trained_cl'] = None
        args_dict['multiple_layer_pixel_classifier'] = False
        args_dict['anchors_ortogonal'] = False
        args_dict['detach_pixel_classifier'] = False
        args_dict['batch_norm_pixel_classifier'] = False
        args_dict['one_layer_pixel_classifier'] = False
        args = Dict2Obj(args_dict)
        args.outd = tmp_outd
        args.distributed = False
        args.eval_checkpoint_type = checkpoint_type

    args.sf_uda = False

    model = get_model(args)[0]

    print(f'Loading model for {method_name}-{encoder_name} from {path_cl}')
    if "tscam" in encoder_name:
        model_tscam = torch.load(join(path_cl, 'model.pt'),map_location=get_cpu_device())

        model.load_state_dict(model_tscam, strict=True)
    else:
        encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                            map_location=get_cpu_device())
        model.encoder.super_load_state_dict(encoder_w, strict=True)

        header_w = torch.load(join(path_cl, 'classification_head.pt'),
                            map_location=get_cpu_device())
        model.classification_head.load_state_dict(header_w, strict=True)

        if method_name == constants.METHOD_PIXELCAM:    #'EnergyCAM': constants.METHOD_ENERGY:
            header_p = torch.load(join(path_cl, 'pixel_wise_classification_head.pt'),
                            map_location=get_cpu_device())
            model.pixel_wise_classification_head.load_state_dict(header_p, strict=True)




    DLLogger.log(fmsg("Model checkpoint Loaded from {}".format(path_cl)))
        
    model.to(device)
    model.eval()

    # with torch.no_grad():
    #     anchors = model.pixel_wise_classification_head.conv4.weight

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
    
    source_metadata_root = join(constants.RELATIVE_META_ROOT, source_dataset, f"fold-{args.fold}")
    #read sys var DATASETSH
    args_dict['data_root'] = os.path.join(os.environ['DATASETSH'], 'datasets')
    source_domain_data_paths = config.configure_data_paths(args_dict, source_dataset)

    target_metadata_root = join('./folds/wsol-done-right-splits', target_dataset, f"fold-{args.fold}")
    # args_dict['data_root'] = '/export/gauss/vision/Aguichemerre/datasets'
    target_domain_data_paths = config.configure_data_paths(args_dict, target_dataset)

    source_loaders = get_data_loader(
            data_roots=source_domain_data_paths,
            metadata_root=source_metadata_root,
            batch_size=32,#args.batch_size,
            workers=args.num_workers,
            resize_size=args.resize_size,
            crop_size=args.crop_size,
            proxy_training_set=args.proxy_training_set,
            num_val_sample_per_class=args.num_val_sample_per_class,
            std_cams_folder=args.std_cams_folder,
            # distributed_eval=False,
            get_splits_eval=[split],
            #constants.TRAINSET
            eval_batch_size = 32#args.eval_batch_size,
        )

    
    source_cam_computer = CAMComputer(
            args=deepcopy(args),
            model=model,
            loader=source_loaders['train'],
            metadata_root=os.path.join(source_metadata_root, 'train'),
            mask_root=args.mask_root,
            iou_threshold_list=args.iou_threshold_list,
            dataset_name=source_dataset,
            split= 'train',
            cam_curve_interval=args.cam_curve_interval,
            multi_contour_eval=args.multi_contour_eval,
            out_folder=args.outd,
        )

    target_loaders = get_data_loader(
            data_roots=target_domain_data_paths,
            metadata_root=target_metadata_root,
            batch_size=32,#args.batch_size,
            workers=args.num_workers,
            resize_size=args.resize_size,
            crop_size=args.crop_size,
            proxy_training_set=args.proxy_training_set,
            num_val_sample_per_class=args.num_val_sample_per_class,
            std_cams_folder=args.std_cams_folder,
            # distributed_eval=False,
            get_splits_eval=[split],
            #constants.TRAINSET
            eval_batch_size = 32#args.eval_batch_size,
        )
    
    target_cam_computer = CAMComputer(
            args=deepcopy(args),
            model=model,
            loader=target_loaders['train'],
            metadata_root=os.path.join(target_metadata_root, 'train'),
            mask_root=args.mask_root,
            iou_threshold_list=args.iou_threshold_list,
            dataset_name=target_dataset,
            split= 'train',
            cam_curve_interval=args.cam_curve_interval,
            multi_contour_eval=args.multi_contour_eval,
            out_folder=args.outd,
        )

    #base_dir = os.path.join("visualization", "Energy_results", source_dataset, "SGLD_curves_per_pixel")
    #os.makedirs(base_dir, exist_ok=True)

    # all_energies_pixels_source = []
    # all_energies_pixels_target = []

    # all_energies_images_source = []
    # all_energies_images_target = []

    # energies_images_per_class_source = defaultdict(list)
    # energies_images_per_class_target = defaultdict(list)

    # energy_pixels_foreground_source = []
    # energy_pixels_background_source = []

    # energy_pixels_foreground_target = []
    # energy_pixels_background_target = []

    energy_data = {
    'images': [],
    'pixels': [],
    'per_class': defaultdict(list),
    'foreground': [],
    'background': []}


    source_energy = compute_energy_distributions(model, source_loaders, source_cam_computer, source_dataset, energy_fn, split, device)
    target_energy = compute_energy_distributions(model, target_loaders, target_cam_computer, target_dataset, energy_fn, split, device)

    #plot_energy_histograms_by_class(source_energy, target_energy, out_dir, title_prefix="")
    out_dir = "plots_energy"
    os.makedirs(out_dir, exist_ok=True)

    plot_energy_histograms_by_class(source_energy['per_class'],target_energy['per_class'],out_dir=out_dir,title_prefix="Energy Distribution", source_dataset=source_dataset, target_dataset=target_dataset)
    
    # 2. Foreground pixels
    plot_global_energy_histogram(source_energy['foreground'],target_energy['foreground'],out_dir,title="Pixel Energy Distribution (Foreground)", type = "foreground", source_dataset=source_dataset, target_dataset=target_dataset)

    # 3. Background pixels
    plot_global_energy_histogram(source_energy['background'],target_energy['background'],out_dir,title="Pixel Energy Distribution (Background)", type = "background", source_dataset=source_dataset, target_dataset=target_dataset)

    # 4. All pixels
    plot_global_energy_histogram(source_energy['pixels'],target_energy['pixels'],out_dir,title="Pixel Energy Distribution (All)",xlim=(-5, 5), type = "global", source_dataset=source_dataset, target_dataset=target_dataset)
    
    
    # for batch_idx, (images, targets, p_glabel, index, raw_imgs, std_cams, _, views) in tqdm(
    #     enumerate(source_loaders[split]), ncols=constants.NCOLS,
    #     total=len(source_loaders[split])):

    #     images = images.to(device)
    #     targets = targets.to(device)
        
    #     with torch.no_grad():
    #         lgt_imgs = model(images)
    #         px_lin_ft = model.encoder_last_features
    #         lgt_pxs = model.pixel_wise_classification_head(px_lin_ft)[0]

    #         energy_images = energy_fn(lgt_imgs)
    #         energy_pixels = energy_fn(lgt_pxs)

    #         energy_data['images'].append(energy_images.cpu())
    #         energy_data['pixels'].append(energy_pixels.flatten(start_dim=1).cpu())

    #         for energy, label in zip(energy_images.cpu(), targets.cpu()):
    #             energy_data['per_class'][int(label)].append(energy.item())

    #         #all_energies_images_source.append(energy_images.flatten( ).cpu())
    #         #all_energies_pixels_source.append(energy_pixels.flatten().cpu())

    #         #for energy, label in zip(energy_images.cpu(), targets.cpu()):
    #             #energies_images_per_class_source[int(label)].append(energy.item())


    #     # Per-pixel foreground / background
    #     for i, (label, image_id) in enumerate(zip(targets, index)):
    #         _, _, h, w = pixel_features.shape
    #         gt_bin = get_resized_gt_mask(
    #             cam_computer.evaluator.mask_paths[image_id],
    #             cam_computer.evaluator.ignore_paths[image_id],
    #             dataset_name,
    #             label.item(),
    #             size=(h, w),
    #             device=device
    #         )

    #         energy_map = energy_pixels[i]  # (H, W)
    #         energy_data['foreground'].append(energy_map[gt_bin].detach().cpu())
    #         energy_data['background'].append(energy_map[~gt_bin].detach().cpu())

    #     # Concatenation
    #     energy_data['images'] = torch.cat(energy_data['images']).numpy()
    #     energy_data['pixels'] = torch.cat(energy_data['pixels']).numpy()
    #     energy_data['foreground'] = torch.cat(energy_data['foreground']).numpy()
    #     energy_data['background'] = torch.cat(energy_data['background']).numpy()

    #     for i, (image, label, image_id) in enumerate(zip(images, targets, index)):
    #         h, l, m, n = px_lin_ft.shape  # [B, C, H, W]
    #         if source_dataset == constants.GLAS or (source_dataset == constants.CAMELYON512 and label ==1):
    #             gt_mask = get_mask(f'/export/gauss/vision/Aguichemerre/datasets/{source_dataset}',
    #                             source_cam_computer.evaluator.mask_paths[image_id],
    #                             source_cam_computer.evaluator.ignore_paths[image_id])
                
    #             gt_mask_tensor = torch.tensor(gt_mask, dtype=torch.float32).to(device)

    #             gt_resize = F.interpolate(gt_mask_tensor.unsqueeze(0).unsqueeze(0), size=(m, n),
    #                                     mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    #         else:
    #             gt_resize = torch.zeros((m, n), dtype=torch.float32, device=device)


    #         gt_bin = (gt_resize > 0.5).to(torch.bool)  # foreground: True


    #         energy_map = energy_pixels[i]  # shape: [H, W]

    #         # Split foreground / background
    #         fg_energies = energy_map[gt_bin].detach().cpu()
    #         bg_energies = energy_map[~gt_bin].detach().cpu()


    #         energy_pixels_foreground_source.append(fg_energies)
    #         energy_pixels_background_source.append(bg_energies)
            
    # all_energies_pixels_source = torch.cat(all_energies_pixels_source).numpy()
    # all_energies_images_source = torch.cat(all_energies_images_source).numpy()

    # energy_pixels_foreground_source = torch.cat(energy_pixels_foreground_source).numpy()
    # energy_pixels_background_source = torch.cat(energy_pixels_background_source).numpy()

    # for batch_idx, (images, targets, p_glabel, index, raw_imgs, std_cams, _, views) in tqdm(
    #     enumerate(target_loaders[split]), ncols=constants.NCOLS,
    #     total=len(target_loaders[split])):
    #     image_size = images.shape[2:]
    #     images = images.to(device)
    #     targets = targets.to(device)
        
    #     with torch.no_grad():
    #         lgt_imgs = model(images)
    #         px_lin_ft = model.encoder_last_features
    #         lgt_pxs = model.pixel_wise_classification_head(px_lin_ft)[0]

    #         energy_images = energy_fn(lgt_imgs)
    #         energy_pixels = energy_fn(lgt_pxs)

    #         all_energies_images_target.append(energy_images.flatten().cpu())
    #         all_energies_pixels_target.append(energy_pixels.flatten().cpu())

    #         for energy, label in zip(energy_images.cpu(), targets.cpu()):
    #             energies_images_per_class_target[int(label)].append(energy.item())

    #     for i, (image, label, image_id) in enumerate(zip(images, targets, index)):
    #         h, l, m, n = px_lin_ft.shape  # [B, C, H, W]
    #         if target_dataset == constants.GLAS or (target_dataset == constants.CAMELYON512 and label ==1):
    #             gt_mask = get_mask(f'/export/gauss/vision/Aguichemerre/datasets/{target_dataset}',
    #                             target_cam_computer.evaluator.mask_paths[image_id],
    #                             target_cam_computer.evaluator.ignore_paths[image_id])
                
    #             gt_mask_tensor = torch.tensor(gt_mask, dtype=torch.float32).to(device)

    #             gt_resize = F.interpolate(gt_mask_tensor.unsqueeze(0).unsqueeze(0), size=(m, n),
    #                                     mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    #         else:
    #             gt_resize = torch.zeros((m, n), dtype=torch.float32, device=device)


    #         gt_bin = (gt_resize > 0.5).to(torch.bool)  # foreground: True


    #         energy_map = energy_pixels[i]  # shape: [H, W]

    #         # Split foreground / background
    #         fg_energies = energy_map[gt_bin].detach().cpu()
    #         bg_energies = energy_map[~gt_bin].detach().cpu()


    #         energy_pixels_foreground_target.append(fg_energies)
    #         energy_pixels_background_target.append(bg_energies)
            
    # all_energies_pixels_target = torch.cat(all_energies_pixels_target).numpy()
    # all_energies_images_target = torch.cat(all_energies_images_target).numpy()

    # energy_pixels_foreground_target = torch.cat(energy_pixels_foreground_target).numpy()
    # energy_pixels_background_target = torch.cat(energy_pixels_background_target).numpy()

    # all_classes = sorted(set(energies_images_per_class_source.keys()) | set(energies_images_per_class_target.keys()))

    # for cls in all_classes:
    #     source_energies = energies_images_per_class_source.get(cls, [])
    #     target_energies = energies_images_per_class_target.get(cls, [])

    #     if not source_energies and not target_energies:
    #         continue  

    #     plt.figure(figsize=(8, 6))

    #     if source_energies:
    #         plt.hist(source_energies, bins=50, alpha=0.5, density=True,
    #                 color='blue', edgecolor='black', label='Source')

    #     if target_energies:
    #         plt.hist(target_energies, bins=50, alpha=0.5, density=True,
    #                 color='red', edgecolor='black', label='Target')

    #     plt.title(f"Energy Distribution Class {cls}")
    #     plt.xlabel("Energy")
    #     plt.ylabel("Density")
    #     plt.legend(loc="upper right")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f"histogram_energy_class_{cls}_source_vs_target.png", dpi=300)
    #     plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.hist(energy_pixels_foreground_source, bins=100, alpha=0.5, density=True,
    #         label='Foreground (Source)', color='blue', edgecolor='black')
    # plt.hist(energy_pixels_foreground_target, bins=100, alpha=0.5, density=True,
    #         label='Foreground (Target)', color='red', edgecolor='black')
    # plt.title("Pixel Energy Distribution (Foreground : Source vs Target)")
    # plt.xlabel("Energy")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("histogram_energy_pixel_fg_source_vs_fg_target.png", dpi=300)
    # plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.hist(energy_pixels_background_source, bins=100, alpha=0.5, density=True,
    #         label='Background (Source)', color='blue', edgecolor='black')
    # plt.hist(energy_pixels_background_target, bins=100, alpha=0.5, density=True,
    #         label='Background (Target)', color='red', edgecolor='black')
    # plt.title("Pixel Energy Distribution (Background : Source vs Target)")
    # plt.xlabel("Energy")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("histogram_energy_pixel_bg_source_vs_bg_target.png", dpi=300)
    # plt.show()

 

    # plt.figure(figsize=(8, 6))
    # plt.hist(all_energies_pixels_source, bins=100, density=True, alpha=0.5, label="Source", edgecolor='black', color='blue')
    # plt.hist(all_energies_pixels_target, bins=100, density=True, alpha=0.5, label="Target", edgecolor='black', color='red')
    # plt.title("Energy distribution")
    # plt.xlabel("Energy")
    # plt.ylabel("Density")
    # plt.legend(loc="upper right", fontsize=12)

    # plt.xlim(-5, 5)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("histogram_energy_pixels_target_glas.png", dpi=300)
    # plt.show()


    # # Affiche l'histogramme
    # plt.figure(figsize=(8, 6))
    # plt.hist(all_energies_images_source, bins=100, density=True, alpha=0.5, label="Source", edgecolor='black', color='blue')
    # plt.hist(all_energies_images_target, bins=100, density=True, alpha=0.5, label="Target", edgecolor='black', color='red')
    # plt.title("Energy distribution")
    # plt.xlabel("Energy")
    # plt.ylabel("Density")
    # plt.legend(loc="upper right", fontsize=12)

    # plt.xlim(-10, 10)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("histogram_energy_images_target_glas.png", dpi=300)
    # plt.show()

    
    return 0
    
def fast_eval():
    t0 = dt.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cudaid", type=str, default=None, help="cuda id.")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--checkpoint_type", type=str, default=None)
    parser.add_argument("--encoder_name", type=str, default=None)
    # parser.add_argument("--exp_path", type=str, default=None)
    parser.add_argument("--tmp_outd", type=str, default='tmp_outd')
    parser.add_argument('--noise_level_for_eval_with_noisy_bbox', nargs='+',
                        type=int, default=[5, 10, 15, 20, 25, 30, 35 ,40, 45, 50])
    parser.add_argument("--target_dataset", type=str, default=None, help="Source dataset")
    parser.add_argument('--image_ids_to_draw', nargs='+', type=str, default=None)
    parser.add_argument('--image_ids_to_draw_target', nargs='+', type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--source_dataset", type=str, default=None, help="Source dataset")
    parser.add_argument("--path_pre_trained_source", type=str, default=None, help="Path to the pre-trained source model.")

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
        
        split = parsedargs.split
        assert split == constants.TESTSET or split == constants.VALIDSET or split == constants.TRAINSET
        
        _CODE_FUNCTION = 'fast_eval_{}'.format(split)

        #target_methods = ['SOURCE', 'SFDE', 'SHOT', 'CDCL', 'ADADSA']
        target_methods = ['SOURCE']

        method_name_lst = []
        for ind_method, target_method in enumerate(target_methods):
            ind_method+= 2
                
            # Get model path   
            if target_method == 'SOURCE':
                exp_path = parsedargs.path_pre_trained_source
            else:
                exp_path = parsedargs.target_domain_exp_path[target_method]


            #Get features at the pixel level
            overlay_images, input_images, method_name, gt_masks = get_features(exp_path=exp_path, sf_uda_source_folder=parsedargs.path_pre_trained_source,image_ids_to_draw=parsedargs.image_ids_to_draw,image_ids_to_draw_target=parsedargs.image_ids_to_draw_target, checkpoint_type=checkpoint_type, source_dataset=parsedargs.source_dataset,target_dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs)

if __name__ == '__main__':
    fast_eval()