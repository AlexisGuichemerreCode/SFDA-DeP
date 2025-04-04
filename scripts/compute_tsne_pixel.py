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

def energy_fn(feat, model):
    """
    Compute the energy map for the given features using the model.

    Args:
        feat (torch.Tensor): Input features (e.g., pixel features).
        model (torch.nn.Module): The model containing the pixel-wise classification head.

    Returns:
        torch.Tensor: Energy map (H, W).
    """
    logits = model.pixel_wise_classification_head(feat)  # (1, C, H, W)
    logits = logits[0].squeeze(0)  # (C, H, W)
    energy_map = -torch.logsumexp(logits, dim=0)  # logsumexp over classes
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

        



# def extract_features(mask, feature, label, image_id):
#     cancer_features = np.empty((0, 2048))
#     non_cancer_features = np.empty((0, 2048))
#     background_features = np.empty((0, 2048))

#     mask = torch.from_numpy(mask)
#     #mask_cam = cam > 0.2

#     # Interpolate feature to image size
#     #feature = F.interpolate(feature.unsqueeze(0), size=mask.shape, mode='bilinear', align_corners=False).squeeze(0)
#     # Get position of pixels that are not zero
#     # non_zero_indices = torch.nonzero(mask_cam, as_tuple=True)
#     non_zero_indices = torch.nonzero(mask, as_tuple=True)
#     # Get corresponding features
#     non_zero_feature = feature[0, :, non_zero_indices[0], non_zero_indices[1]]
#     # if label == 1:
#     #     cancer_features= [cancer_features, non_zero_feature.cpu().numpy().T]
#     # else:
#     #     non_cancer_features = [non_cancer_features, non_zero_feature.cpu().numpy().T]

#     # Get position of pixels that are zero
#     #mask_cam = cam < 0.2
#     zero_indices = torch.nonzero(mask == 0, as_tuple=True)
#     #zero_indices = torch.nonzero(mask == 0, as_tuple=True)
#     # Get corresponding features
#     zero_feature = feature[0, :, zero_indices[0], zero_indices[1]]
#     #background_features = [background_features, zero_feature.cpu().numpy().T]

#     features1 = non_zero_feature.detach().cpu().numpy().T
#     features2 = zero_feature.detach().cpu().numpy().T

#     all_features = np.concatenate([features1, features2])
#     tsne = TSNE(n_components=2)
#     embedded_features = tsne.fit_transform(all_features)
#     embedded_features1 = embedded_features[:features1.shape[0]]
#     embedded_features2 = embedded_features[features1.shape[0]:]

#     # Visualiser les résultats avec un code couleur
#     plt.scatter(embedded_features1[:, 0], embedded_features1[:, 1], color='blue', label='Foreground')
#     plt.scatter(embedded_features2[:, 0], embedded_features2[:, 1], color='red', label='Background')

#     plt.legend()
#     if label == 1:
#         classe = 'cancer'
#     else:
#         classe = 'non-cancer'


#     parts = image_id.split('/')
#     clean_image_id = parts[1].replace('.bmp', '')
#     plt.title(f"T-SNE visualization of source features from GLAS\n at the pixel level with CAM for a {classe} image ({clean_image_id})")
#     # Sauvegarder l'image
#     plt.savefig(f'tsne_plot_cam_{clean_image_id}.png')
#     plt.close()
#     return features1,features2

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


def get_features(exp_path, sf_uda_source_folder,image_ids_to_draw,image_ids_to_draw_target, checkpoint_type, dataset, cudaid, split, tmp_outd='tmp_outd', parsedargs=None):


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
    
    metadata_root = join(constants.RELATIVE_META_ROOT, dataset, f"fold-{args.fold}")
    #read sys var DATASETSH
    args_dict['data_root'] = os.path.join(os.environ['DATASETSH'], 'datasets')
    target_domain_data_paths = config.configure_data_paths(args_dict, dataset)

    # metadata_root_CAME = join('./folds/wsol-done-right-splits', 'CAMELYON512', f"fold-{args.fold}")
    # args_dict['data_root'] = '/export/gauss/vision/Aguichemerre/datasets'
    # target_domain_data_paths_CAME = config.configure_data_paths(args_dict, 'CAMELYON512')

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
            get_splits_eval=[split],
            #constants.TRAINSET
            eval_batch_size = 32#args.eval_batch_size,
        )

    # loaders_came = get_data_loader(
    #             data_roots=target_domain_data_paths_CAME,
    #             metadata_root=metadata_root_CAME,
    #             batch_size=32,#args.batch_size,
    #             workers=args.num_workers,
    #             resize_size=args.resize_size,
    #             crop_size=args.crop_size,
    #             proxy_training_set=args.proxy_training_set,
    #             num_val_sample_per_class=args.num_val_sample_per_class,
    #             std_cams_folder=args.std_cams_folder,
    #             # distributed_eval=False,
    #             get_splits_eval=['valpx'],
    #             eval_batch_size = 32#args.eval_batch_size,
    #         )     
    
    cam_computer = CAMComputer(
            args=deepcopy(args),
            model=model,
            loader=loaders['train'],
            metadata_root=os.path.join(metadata_root, 'train'),
            mask_root=args.mask_root,
            iou_threshold_list=args.iou_threshold_list,
            dataset_name=args.dataset,
            split= 'train',
            cam_curve_interval=args.cam_curve_interval,
            multi_contour_eval=args.multi_contour_eval,
            out_folder=args.outd,
        )
    # cam_computer_came = CAMComputer(
    #         args=deepcopy(args),
    #         model=model,
    #         loader=loaders_came['valpx'],
    #         metadata_root=os.path.join(metadata_root_CAME, 'valpx'),
    #         mask_root=args.mask_root,
    #         iou_threshold_list=args.iou_threshold_list,
    #         dataset_name='CAMELYON512',
    #         split='valpx',
    #         cam_curve_interval=args.cam_curve_interval,
    #         multi_contour_eval=args.multi_contour_eval,
    #         out_folder=args.outd,
    #     )        
    overlay_images = {}
    input_images = {}
    gt_masks = {}

    all_flattened_cam_values = []
    all_flattened_gt_values = []
    all_flattened_softmax_values = []

    all_foreground_features = []
    all_background_features = []

    all_classifier_preds = []
    all_classifier_gt = []

    all_kmeans_conf = []
    all_kmeans_preds = []

    per_image_pixel_acc = []
    per_image_kmeans_acc = []

    image_class_correct = []
    image_prediction_confidence = []

    weights = model.pixel_wise_classification_head.conv4.weight.data  # shape: (2, 2048, 1, 1)
    centroids_init = weights.squeeze(-1).squeeze(-1).cpu().numpy()  # shape: (2, 2048)

    # Dictionnaire pour stocker les accuracies par itération SGLD
    sgld_acc_by_iter = defaultdict(list)

    base_dir = os.path.join("visualization", "Energy_results", dataset, "SGLD_curves_per_pixel")
    os.makedirs(base_dir, exist_ok=True)

    for batch_idx, (images, targets, p_glabel, index, raw_imgs, std_cams, _, views) in tqdm(
        enumerate(loaders[split]), ncols=constants.NCOLS,
        total=len(loaders[split])):
        image_size = images.shape[2:]
        images = images.to(device)
        targets = targets.to(device)
        
        #with torch.no_grad():
        #    out = model(images.cuda())
        #    pixel_features = model.encoder_last_features
        GroundTruth = []
        for image, target, image_id in zip(images, targets, index):
            # if image_id not in image_ids_to_draw:
            #     continue

            def energy_fn_with_model(x):
                return energy_fn(x, model)


            with torch.set_grad_enabled(cam_computer.req_grad):
                cam, cl_logits = cam_computer.get_cam_one_sample(
                    image=image.unsqueeze(0), target=target.item())

            probs = torch.softmax(cl_logits, dim=1)
            pred_class = cl_logits.argmax().item()
            true_class = target.item()
            image_correct = int(pred_class == true_class)
            image_class_correct.append(image_correct)
            image_prediction_confidence.append(probs[0, pred_class].item())
                
            with torch.no_grad():
                out = model(image.cuda().unsqueeze(0))
                pixel_features = model.encoder_last_features
                logits = model.pixel_wise_classification_head(pixel_features)  # (1, 2, H, W)
                probs = torch.softmax(logits[0], dim=1)[0]  # (2, H, W)
                confidences, _ = probs.max(dim=0)  # (H, W)
                predicted_labels = logits[0].argmax(dim=1).squeeze(0)  # (H, W)

            
            

            gt_mask = get_mask(f'/export/gauss/vision/Aguichemerre/datasets/{dataset}',
                            cam_computer.evaluator.mask_paths[image_id],
                            cam_computer.evaluator.ignore_paths[image_id])
                
            gt_mask_tensor = torch.tensor(gt_mask, dtype=torch.float32)

            


            h,l,m,n = pixel_features.shape
            gt_resize=F.interpolate(gt_mask_tensor.unsqueeze(0).unsqueeze(0),
                                (m,n),
                                mode='bilinear',
                                align_corners=False).squeeze(0).squeeze(0)
            
            resized_mask_array = (gt_resize.detach().cpu().numpy() *255).astype('uint8')

            gt_bin = (gt_resize > 0.5).to(torch.uint8)  # (H, W)

            # # Fonction d'énergie = négatif de la confiance du classifieur
            # def energy_fn(feat):
            #     logits = model.pixel_wise_classification_head(feat)  # (1, C, H, W)
            #     logits = logits[0].squeeze(0)  # (C, H, W)

            #     energy_map = -torch.logsumexp(logits, dim=0)  # logsumexp sur les classes
            #     return energy_map  # (H, W)


            # Initialisation
            #feat_init = pixel_features.detach()
            #sgld_accuracies = []
            #n_sgld_iter = 10 

            sgld_lrs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
            colors = plt.cm.viridis_r(np.linspace(0, 1, len(sgld_lrs)))  

            plt.figure(figsize=(10, 6))

            for lr, color in zip(sgld_lrs, colors):
                _, accs_during_sgld = sgld_sample_ebm(
                    init_sample=pixel_features,
                    energy_fn=energy_fn_with_model,
                    model=model,
                    gt_bin=gt_bin,
                    track_accuracy=True,
                    track_every=1,
                    n_steps=500,
                    sgld_lr=lr,
                    sgld_std=1e-2
                )

                steps, accs = zip(*accs_during_sgld)
                plt.plot(steps, accs, label=f"SGLD LR={lr}", color=color)

            plt.xlabel("SGLD Step")
            plt.ylabel("Accuracy")
            plt.title(f"SGLD Accuracy – Image {image_id}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            filename = os.path.basename(image_id)          
            filename_no_ext = os.path.splitext(filename)[0]
            save_path = os.path.join(base_dir, f"sgld_curve_image_{filename_no_ext}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()


            # logits = model.pixel_wise_classification_head(sampled_feats)
            # pred_labels = logits[0].argmax(dim=1).squeeze(0)  # (H, W)

            # acc_sgld = accuracy_score(
            #         gt_bin.view(-1).cpu().numpy(),
            #         pred_labels.view(-1).cpu().numpy()
            #     )
            
            # acc_ref = accuracy_score(
            #         gt_bin.view(-1).cpu().numpy(),
            #         predicted_labels.view(-1).cpu().numpy()
            #     )

            # logits = model.pixel_wise_classification_head(sampled_feats)
            # pred_labels = logits[0].argmax(dim=1).squeeze(0)  # (H, W)

            # acc_sgld = accuracy_score(
            #         gt_bin.view(-1).cpu().numpy(),
            #         pred_labels.view(-1).cpu().numpy()
            #     )

            # sgld_acc_by_iter[i].append(acc_sgld)

            # for i in range(n_sgld_iter):
            #     #sampled_feat = sgld_sample(feat_init, energy_fn, n_iter=i+1, step_size=1e-2, noise_scale=1e-2)
            #     sampled_feats

            #     with torch.no_grad():
            #         logits = model.pixel_wise_classification_head(sampled_feat)
            #         pred_labels = logits[0].argmax(dim=1).squeeze(0)  # (H, W)

            #     acc_sgld = accuracy_score(
            #         gt_bin.view(-1).cpu().numpy(),
            #         pred_labels.view(-1).cpu().numpy()
            #     )

            #     sgld_acc_by_iter[i].append(acc_sgld)


            # #Pixel Classifier Accuracy
            # gt_bin = (gt_resize > 0.5).to(torch.uint8)
            # acc_pixel = accuracy_score(
            #     gt_bin.view(-1).cpu().numpy(),
            #     predicted_labels.view(-1).cpu().numpy()
            # )
            # per_image_pixel_acc.append(acc_pixel)

            # #K-means Accuracy
            # pixel_features_flat = pixel_features.squeeze(0).permute(1, 2, 0).reshape(-1, pixel_features.shape[1])  # (H*W, C)
            # gt_flat = gt_bin.view(-1).cpu().numpy()

            # #centroids_init = weights.squeeze(-1).squeeze(-1).cpu().numpy()  # (2, C)

            # kmeans = KMeans(
            #     n_clusters=2,
            #     init=centroids_init,
            #     n_init=1,
            #     max_iter=200,
            #     random_state=42
            # )

            # # Exécution du clustering
            # predicted_clusters = kmeans.fit_predict(pixel_features_flat.detach().cpu().numpy())  # (H*W,)

            # # Comparaison avec le GT (prendre le mapping optimal)
            # kmeans_acc = accuracy_score(gt_flat, predicted_clusters)

            # # Stockage (tu peux faire une nouvelle liste si tu veux les garder à part)
            # per_image_kmeans_acc.append(kmeans_acc)

            # all_classifier_gt.append(gt_bin.view(-1).cpu().numpy())
            # all_classifier_preds.append(predicted_labels.view(-1).cpu().numpy())

            # #cancer_features, non_cancer_features, background_features = extract_features_source_target(gt_resize, pixel_features, target, cam, image_id, gt_resize_came, pixel_features_target, target_came, cam_came, image_id_came, anchors)
            # foreground_features, background_features = extract_features(resized_mask_array, pixel_features, target,image_id)
            # all_foreground_features.append(foreground_features)
            # all_background_features.append(background_features)
            # features_np = pixel_features_flat.detach().cpu().numpy()  # (H*W, C)
            # assigned_clusters = predicted_clusters  # (H*W,)
            # centroids = kmeans.cluster_centers_  # shape: (2, C)

            # dists = np.linalg.norm(features_np - centroids[assigned_clusters], axis=1)  # (H*W,)
            # kmeans_confidence = -dists  # Plus proche = plus confiant

            # # Stockage global
            # all_kmeans_conf.append(kmeans_confidence)
            # all_kmeans_preds.append(assigned_clusters)

            # cam_flat = cam.view(-1).detach().cpu().numpy()          # shape: (784,)
            # gt_flat = (gt_resize > 0.5).view(-1).to(torch.uint8).cpu().numpy()  # shape: (784,)
            # all_flattened_cam_values.append(cam_flat)
            # all_flattened_gt_values.append(gt_flat)
            # all_flattened_softmax_values.append(confidences.view(-1).cpu().numpy())

        print("pass")


    iters = list(sgld_acc_by_iter.keys())
    means = [np.mean(sgld_acc_by_iter[i]) for i in iters]
    stds = [np.std(sgld_acc_by_iter[i]) for i in iters]

    plt.figure(figsize=(8, 5))
    plt.plot(iters, means, marker='o', label='SGLD accuracy')
    plt.xlabel('SGLD Iteration')
    plt.ylabel('Pixel-wise Accuracy')
    plt.title('Accuracy vs SGLD iterations')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sgld_accuracy_by_iter.png")
    plt.close()


    kmeans_acc = np.array(per_image_kmeans_acc)
    classifier_acc = np.array(per_image_pixel_acc)
    x = np.arange(len(kmeans_acc))
    width = 0.4
    plt.figure(figsize=(15, 6))
    plt.bar(x - width/2, kmeans_acc, width=width, label='KMeans', alpha=0.8)
    plt.bar(x + width/2, classifier_acc, width=width, label='Classifier', alpha=0.8)
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random baseline (50%)')
    plt.xlabel('Image Index')
    plt.ylabel('Pixel Accuracy')
    plt.title('Per-Image Pixel Accuracy: KMeans vs Classifier')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("histogram_pixel_vs_kmeans.png")
    plt.close()

    all_cam = np.concatenate(all_flattened_cam_values)  # CAMs
    all_px_class = np.concatenate(all_classifier_preds) 
    all_gt = np.concatenate(all_flattened_gt_values)    # GTs (0 ou 1)
    all_softmax_conf = np.concatenate(all_flattened_softmax_values)  # Classifier softmax max
    all_kmeans_conf = np.concatenate(all_kmeans_conf)  # KMeans: -distance
    all_kmeans_preds_concat = np.concatenate(all_kmeans_preds)


    # Normalisation des distances pour KMeans
    kmeans_dists = -all_kmeans_conf  # on revient aux distances positives
    kmeans_conf_norm = 1 - (kmeans_dists - kmeans_dists.min()) / (kmeans_dists.max() - kmeans_dists.min() + 1e-8)

    # Trie des indices des pixels par confiance décroissante
    sorted_idx_softmax = np.argsort(-all_softmax_conf)
    sorted_idx_kmeans = np.argsort(-kmeans_conf_norm)

    # Prédictions triées
    sorted_preds_softmax = all_px_class[sorted_idx_softmax]
    sorted_preds_kmeans = all_kmeans_preds_concat[sorted_idx_kmeans]

    # GT trié (même pourcentage à comparer)
    sorted_gt_softmax = all_gt[sorted_idx_softmax]
    sorted_gt_kmeans = all_gt[sorted_idx_kmeans]

    n_bins = 100  # 1% à 100%
    bin_size = len(all_gt) // n_bins

    acc_softmax_bins = []
    acc_kmeans_bins = []

    for i in range(1, n_bins + 1):
        k = i * bin_size
        # Classifier
        acc_s = (sorted_preds_softmax[:k] == sorted_gt_softmax[:k]).mean()
        acc_softmax_bins.append(acc_s)
        
        # KMeans — test des deux permutations (0/1 <-> 1/0)
        acc_k1 = (sorted_preds_kmeans[:k] == sorted_gt_kmeans[:k]).mean()
        acc_kmeans_bins.append(acc_k1)  # choisir le meilleur mappage

    # Courbe comparative
    plt.figure(figsize=(10, 6))
    x = np.arange(1, n_bins + 1)

    plt.plot(x, acc_softmax_bins, label='Classifier (softmax)', linewidth=2, marker='o')
    plt.plot(x, acc_kmeans_bins, label='KMeans (distance)', linewidth=2, marker='^')

    plt.xlabel('Top-k% most confident pixels')
    plt.ylabel('Prediction accuracy')
    plt.title('Relationship between Prediction Confidence and Accuracy at Pixel Level')

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("accuracy_vs_confidence_softmax_kmeans_glas.png")
    plt.close()
    #sorted_indices = np.argsort(-all_cam)
    #sorted_gt = all_gt[sorted_indices]

    # Évalue en top-n %
    # n_bins = 10
    # bin_size = len(sorted_gt) // n_bins
    # acc_per_bin = []

    # for i in range(n_bins):
    #     start = i * bin_size
    #     end = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_gt)
    #     bin_gt = sorted_gt[start:end]
    #     acc = bin_gt.mean()  # Proportion de foreground dans ce bin
    #     acc_per_bin.append(acc)

    # # Affichage
    # plt.figure(figsize=(8, 5))
    # plt.plot(np.arange(1, n_bins + 1) * 10, acc_per_bin, marker='o')
    # plt.xlabel('Top-n % pixels (selon activation CAM)')
    # plt.ylabel('Proportion de pixels foreground (selon GT)')
    # plt.title('Précision des CAMs en fonction de l’activation')
    # plt.grid(True)
    # plt.tight_layout()

    # plt.savefig("histogram_pixel_accuracy_based on intensity.png")
    # plt.close()

    foreground_features = np.concatenate(all_foreground_features, axis=0)
    background_features = np.concatenate(all_background_features, axis=0)

    all_features = np.concatenate([foreground_features, background_features])


    per_image_pixel_acc = np.array(per_image_pixel_acc)
    image_class_correct = np.array(image_class_correct)

    acc_when_correct = per_image_pixel_acc[image_class_correct == 1]
    acc_when_wrong = per_image_pixel_acc[image_class_correct == 0]

    print(f"✅ Pixel accuracy when image correctly classified: {acc_when_correct.mean():.4f}")
    print(f"✅ Pixel accuracy when image misclassified:       {acc_when_wrong.mean():.4f}")

    # === Analyze bins of confidence ===
    per_image_pixel_acc = np.array(per_image_pixel_acc)
    image_prediction_confidence = np.array(image_prediction_confidence)
    image_class_correct = np.array(image_class_correct)

    # Bins 0 to 1 
    bins = np.linspace(0, 1.0, 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    nb_bins = len(bin_centers)

    # === Containers
    accs_when_correct = [[] for _ in range(nb_bins)]
    accs_when_wrong = [[] for _ in range(nb_bins)]
    counts_correct = [0 for _ in range(nb_bins)]
    counts_wrong = [0 for _ in range(nb_bins)]

    for acc, conf, correct in zip(per_image_pixel_acc, image_prediction_confidence, image_class_correct):
        bin_idx = np.digitize(conf, bins) - 1
        bin_idx = np.clip(bin_idx, 0, nb_bins - 1)
        if correct:
            accs_when_correct[bin_idx].append(acc)
            counts_correct[bin_idx] += 1
        else:
            accs_when_wrong[bin_idx].append(acc)
            counts_wrong[bin_idx] += 1

    # === Moyenne par bin
    avg_acc_correct = [np.mean(accs) if accs else np.nan for accs in accs_when_correct]
    avg_acc_wrong = [np.mean(accs) if accs else np.nan for accs in accs_when_wrong]

    # === Plot en histogrammes (bar plot)
    plt.figure(figsize=(12, 6))
    bar_width = 4
    x = bin_centers * 100

    # Barres
    bars1 = plt.bar(x - bar_width/2, avg_acc_correct, width=bar_width, color='green', alpha=0.7, label='Correct classification')
    bars2 = plt.bar(x + bar_width/2, avg_acc_wrong, width=bar_width, color='red', alpha=0.7, label='Wrong classification')

    # Annotations : nombre d’images par bin
    for i in range(nb_bins):
        # Pour les barres "correct"
        if not np.isnan(avg_acc_correct[i]):
            plt.text(x[i] - bar_width/2, avg_acc_correct[i] + 0.01,
                    f"{counts_correct[i]}", ha='center', va='bottom', fontsize=9)
        # Pour les barres "wrong"
        if not np.isnan(avg_acc_wrong[i]):
            plt.text(x[i] + bar_width/2, avg_acc_wrong[i] + 0.01,
                    f"{counts_wrong[i]}", ha='center', va='bottom', fontsize=9)

    # Axe et légende
    plt.xticks(np.arange(0, 110, 10))
    plt.ylim(0, 1.05)
    plt.xlabel("Image prediction confidence (%)")
    plt.ylabel("Pixel-wise classifier accuracy")
    plt.title("Pixel accuracy vs image confidence (histogram by bin)")
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("histogram_pixel_accuracy_vs_certainty.png")
    plt.close()

    all_classifier_preds_flat = np.concatenate(all_classifier_preds)
    all_classifier_gt_flat = np.concatenate(all_classifier_gt)

    classifier_acc = accuracy_score(all_classifier_gt_flat, all_classifier_preds_flat)
    print(f"✅ Pixel-wise classifier accuracy (all pixels): {classifier_acc:.4f}")

    k = 2

    weights = model.pixel_wise_classification_head.conv4.weight.data  # shape: (2, 2048, 1, 1)
    centroids_init = weights.squeeze(-1).squeeze(-1).cpu().numpy()  # shape: (2, 2048)



    kmeans = KMeans(
    n_clusters=2,
    init=centroids_init,
    n_init=1,  # important quand on fournit les centres
    max_iter=200,
    random_state=42
    )
    cluster_ids = kmeans.fit_predict(all_features)  # shape: (N,)



    centroids = kmeans.cluster_centers_ 

    tsne = TSNE(n_components=2, random_state=42)
    embedded_features = tsne.fit_transform(all_features)

    n_fg = foreground_features.shape[0]
    e_fg = embedded_features[:n_fg]
    e_bg = embedded_features[n_fg:]

    true_labels = np.concatenate([
    np.ones(foreground_features.shape[0]),  # foreground = 1
    np.zeros(background_features.shape[0])  # background = 0
    ])

    # Cas 1 : cluster_ids tels quels
    acc1 = accuracy_score(true_labels, cluster_ids)

    # Cas 2 : labels inversés
    acc2 = accuracy_score(true_labels, 1 - cluster_ids)

    # Meilleur des deux
    best_acc = max(acc1, acc2)
    print(f"Best clustering accuracy (foreground/background): {best_acc:.4f}")


    # plt.figure(figsize=(10, 8))
    # plt.scatter(e_fg[:, 0], e_fg[:, 1], color='blue', alpha=0.1, label='Foreground')
    # plt.scatter(e_bg[:, 0], e_bg[:, 1], color='red', alpha=0.1, label='Background')
    # plt.legend()
    # plt.title("T-SNE of pixel-level features (Foreground vs Background)")
    # plt.savefig("tsne_all_features_target.png")
    # plt.close()
        
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
    #parser.add_argument("--target_dataset", type=str, default=None,
    #                    help="Name of the dataset.", required=True, choices=[constants.CAMELYON512, constants.GLAS])
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
        #split = "train"
        # exp_path = parsedargs.exp_path
        # # checkpoint_type = parsedargs.checkpoint_type
        # # tmp_outd = join(parsedargs.tmp_outd, os.path.split(exp_path)[-1])#, 'split_'+split+'_'+checkpoint_type)
        # tmp_outd = join(exp_path, '0_re-eval_log')#, 'split_'+split+'_'+checkpoint_type)
        # # tmp_outd = parsedargs.tmp_outd
        # assert os.path.isdir(exp_path)
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
            overlay_images, input_images, method_name, gt_masks = get_features(exp_path=exp_path, sf_uda_source_folder=parsedargs.path_pre_trained_source,image_ids_to_draw=parsedargs.image_ids_to_draw,image_ids_to_draw_target=parsedargs.image_ids_to_draw_target, checkpoint_type=checkpoint_type, dataset=parsedargs.source_dataset, cudaid=parsedargs.cudaid, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs)

if __name__ == '__main__':
    fast_eval()