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
#from sklearn.mixture import GaussianMixture
from dlib.gmm.gmm import GaussianMixture
import seaborn as sns

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

def extract_pixel_features(mask_source, feature_source, label_source, cam_source, image_id_source, target_method, dataset, parsedargs):
    _,l,m,n=feature_source.shape

    cancer_features = np.empty((0, l))
    non_cancer_features = np.empty((0, l))
    background_features = np.empty((0, l))

    # cancer_features = np.empty((0, 256))
    # non_cancer_features = np.empty((0, 256))
    # background_features = np.empty((0, 256))

    #source
    mask_source = torch.from_numpy(mask_source)
    non_zero_indices_source = torch.nonzero(mask_source, as_tuple=True)
    non_zero_feature_source = feature_source[0, :, non_zero_indices_source[0], non_zero_indices_source[1]]
    zero_indices_source = torch.nonzero(mask_source == 0, as_tuple=True)
    zero_feature_source = feature_source[0, :, zero_indices_source[0], zero_indices_source[1]]
    foreground_features = non_zero_feature_source.detach().cpu().numpy().T
    background_features = zero_feature_source.detach().cpu().numpy().T

    return foreground_features, background_features

def extract_features_source_target(mask_source, feature_source, label_source, cam_source, image_id_source, target_method, dataset, parsedargs):
    _,l,m,n=feature_source.shape

    cancer_features = np.empty((0, l))
    non_cancer_features = np.empty((0, l))
    background_features = np.empty((0, l))

    # cancer_features = np.empty((0, 256))
    # non_cancer_features = np.empty((0, 256))
    # background_features = np.empty((0, 256))

    #source
    mask_source = torch.from_numpy(mask_source)
    non_zero_indices_source = torch.nonzero(mask_source, as_tuple=True)
    non_zero_feature_source = feature_source[0, :, non_zero_indices_source[0], non_zero_indices_source[1]]
    zero_indices_source = torch.nonzero(mask_source == 0, as_tuple=True)
    zero_feature_source = feature_source[0, :, zero_indices_source[0], zero_indices_source[1]]
    features1_source = non_zero_feature_source.detach().cpu().numpy().T
    features2_source = zero_feature_source.detach().cpu().numpy().T



    #target
    # mask_target = torch.from_numpy(mask_target)
    # non_zero_indices_target = torch.nonzero(mask_target, as_tuple=True)
    # non_zero_feature_target = feature_target[0, :, non_zero_indices_target[0], non_zero_indices_target[1]]
    # zero_indices_target = torch.nonzero(mask_target == 0, as_tuple=True)
    # zero_feature_target = feature_target[0, :, zero_indices_target[0], zero_indices_target[1]]
    # features1_target = non_zero_feature_target.detach().cpu().numpy().T
    # features2_target= zero_feature_target.detach().cpu().numpy().T

    # anchors_resize = anchors.squeeze().detach().cpu().numpy()
    all_features = np.concatenate([features1_source, features2_source])
    tsne = TSNE(n_components=2)
    embedded_features = tsne.fit_transform(all_features)

    # Diviser les features intégrées en fonction de la taille des listes de features originales
    embedded_features1_source = embedded_features[:features1_source.shape[0]]
    embedded_features2_source = embedded_features[features1_source.shape[0]:features1_source.shape[0]+features2_source.shape[0]]
    # embedded_features1_target = embedded_features[features1_source.shape[0]+features2_source.shape[0]:features1_source.shape[0]+features2_source.shape[0]+features1_target.shape[0]]
    # embedded_features2_target = embedded_features[features1_source.shape[0]+features2_source.shape[0]+features1_target.shape[0]:]

    # Visualiser les résultats avec un code couleur
    #plt.scatter(embedded_features1_source[:, 0], embedded_features1_source[:, 1], color='blue', label='Foreground GLAS')
    #plt.scatter(embedded_features2_source[:, 0], embedded_features2_source[:, 1], color='red', label='Background GLAS')
    
    #plt.scatter(embedded_features1_source[:, 0], embedded_features1_source[:, 1], color='blue', label='Foreground GLAS')
    #plt.scatter(embedded_features2_source[:, 0], embedded_features2_source[:, 1], color='red', label='Background GLAS')
    
    plt.scatter(embedded_features1_source[:, 0], embedded_features1_source[:, 1], color='blue')
    plt.scatter(embedded_features2_source[:, 0], embedded_features2_source[:, 1], color='red')
    
    
    # plt.scatter(embedded_features1_target[:, 0], embedded_features1_target[:, 1], color='black')
    # plt.scatter(embedded_features2_target[:, 0], embedded_features2_target[:, 1], color='orange', label='Background CAMELYON')

  
    plt.legend()
    if label_source.item() == 1:
        classe = 'cancer'
    else:
        classe = 'normal'


    parts = image_id_source.split('/')
    clean_image_id = parts[-1].replace('.bmp', '').replace('.png', '')
    #.replace('.png', '')
    #plt.title(f"T-SNE visualization for a {classe} image ", fontsize=10)
    # Sauvegarder l'image
    output_dir = os.path.join('visualization', 'tsne', dataset, classe, parsedargs.dataset_type, clean_image_id)
    os.makedirs(output_dir, exist_ok=True)

    #method_name = target_method.replace(' ', '_')
    filename = os.path.join(output_dir, f"{target_method}.png")
    plt.xticks([]) 
    plt.yticks([])
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    return cancer_features, non_cancer_features, background_features


def plot_pixel_prob(mask_source, probabilities, label_source, cam_source, image_id_source, target_method, dataset, parsedargs):
    _,l,m,n=probabilities.shape

    cancer_features = np.empty((0, l))
    non_cancer_features = np.empty((0, l))
    background_features = np.empty((0, l))

    # cancer_features = np.empty((0, 256))
    # non_cancer_features = np.empty((0, 256))
    # background_features = np.empty((0, 256))

    #source
    mask_source = torch.from_numpy(mask_source)

    non_zero_indices = torch.nonzero(mask_source, as_tuple=True)
    zero_indices = torch.nonzero(mask_source == 0, as_tuple=True)

    non_zero_probs = probabilities[0, :, non_zero_indices[0], non_zero_indices[1]]
    zero_probs = probabilities[0, :, zero_indices[0], zero_indices[1]]

    non_zero_coords = non_zero_probs.detach().cpu().numpy()
    zero_coords = zero_probs.detach().cpu().numpy()


    if label_source.item() == 1:
        classe = 'cancer'
    else:
        classe = 'normal'

    parts = image_id_source.split('/')
    clean_image_id = parts[-1].replace('.bmp', '').replace('.png', '')

    output_dir = os.path.join('visualization', 'tsne', dataset, classe, parsedargs.dataset_type, clean_image_id)
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"{target_method}_prob_plot.png")

    # Créer le scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(non_zero_coords[0], non_zero_coords[1], c='blue', alpha=0.5)
    plt.scatter(zero_coords[0], zero_coords[1], c='red', alpha=0.5)
    plt.xlabel('Background Probability ')
    plt.ylabel('Foreground Probability ')

    plt.grid(True)
    plt.show()
    plt.legend()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    return non_zero_probs, non_zero_probs, zero_probs


def plot_pixel_logits(mask_source, logits, label_source, cam_source, image_id_source, target_method, dataset, parsedargs):
    _,l,m,n=logits.shape

    cancer_features = np.empty((0, l))
    non_cancer_features = np.empty((0, l))
    background_features = np.empty((0, l))

    mask_source = torch.from_numpy(mask_source)

    non_zero_indices = torch.nonzero(mask_source, as_tuple=True)
    zero_indices = torch.nonzero(mask_source == 0, as_tuple=True)

    non_zero_probs = logits[0, :, non_zero_indices[0], non_zero_indices[1]]
    zero_probs = logits[0, :, zero_indices[0], zero_indices[1]]

    non_zero_coords = non_zero_probs.detach().cpu().numpy()
    zero_coords = zero_probs.detach().cpu().numpy()


    if label_source.item() == 1:
        classe = 'cancer'
    else:
        classe = 'normal'

    parts = image_id_source.split('/')
    clean_image_id = parts[-1].replace('.bmp', '').replace('.png', '')
    #plt.title(f"T-SNE visualization for a {classe} image ", fontsize=10)
    # Sauvegarder l'image
    output_dir = os.path.join('visualization', 'tsne', dataset, classe, parsedargs.dataset_type, clean_image_id)
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"{target_method}_logits_plot.png")

    # Créer le scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(non_zero_coords[0], non_zero_coords[1], c='blue', alpha=0.5)
    plt.scatter(zero_coords[0], zero_coords[1], c='red', alpha=0.5)
    plt.xlabel('Background')
    plt.ylabel('Foreground')
    #plt.legend()
    #plt.title('2D Scatter Plot of Probabilities')
    plt.grid(True)
    plt.show()
    plt.legend()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    return non_zero_probs, non_zero_probs, zero_probs

def calculate_scatter_matrices(features, labels):
    """
    Calculate scatter matrices (SW, SB, ST) based on features and labels.
    
    Parameters:
        features (ndarray): Feature matrix of shape (n_samples, n_features).
        labels (ndarray): Class labels of shape (n_samples,).
        
    Returns:
        SW (ndarray): Within-class scatter matrix.
        SB (ndarray): Between-class scatter matrix.
        ST (ndarray): Total scatter matrix.
    """
    # Compute the overall mean vector
    overall_mean = np.mean(features, axis=0)
    
    # Get unique classes and their sizes
    classes = np.unique(labels)
    n_features = features.shape[1]
    
    # Initialize scatter matrices
    SW = np.zeros((n_features, n_features))
    SB = np.zeros((n_features, n_features))
    
    for cls in classes:
        # Extract samples of the current class
        class_samples = features[labels == cls]
        class_mean = np.mean(class_samples, axis=0)
        n_samples_in_class = class_samples.shape[0]
        
        # Compute SW (within-class scatter)
        SW += np.dot((class_samples - class_mean).T, (class_samples - class_mean))
        
        # Compute SB (between-class scatter)
        mean_diff = (class_mean - overall_mean).reshape(-1, 1)
        SB += n_samples_in_class * np.dot(mean_diff, mean_diff.T)
    
    # Compute ST (total scatter matrix)
    ST = np.dot((features - overall_mean).T, (features - overall_mean))
    
    return SW, SB, ST

def class_separability_measure(SW, SB, ST):
    """
    Compute the class separability measure J.
    
    Parameters:
        SW (ndarray): Within-class scatter matrix.
        SB (ndarray): Between-class scatter matrix.
        ST (ndarray): Total scatter matrix.
        
    Returns:
        J (float): Separability measure.
    """
    trace_SW = np.trace(SW)
    trace_SB = np.trace(SB)
    trace_ST = np.trace(ST)
    
    # Compute J using both definitions
    J1 = trace_SB / trace_SW
    J2 = trace_SB / trace_ST
    
    return J1, J2


def class_separability(mask_source, features, label_source, cam_source, image_id_source, target_method, dataset, parsedargs):
    _,l,m,n=features.shape

    cancer_features = np.empty((0, l))
    non_cancer_features = np.empty((0, l))
    background_features = np.empty((0, l))

    #source
    mask_source = torch.from_numpy(mask_source)

    non_zero_indices = torch.nonzero(mask_source, as_tuple=True)
    zero_indices = torch.nonzero(mask_source == 0, as_tuple=True)

    non_zero_probs = features[0, :, non_zero_indices[0], non_zero_indices[1]]
    zero_probs = features[0, :, zero_indices[0], zero_indices[1]]

    # Convert your PyTorch tensors to NumPy arrays (if necessary)
    non_zero_probs_np = non_zero_probs.detach().cpu().numpy()
    zero_probs_np = zero_probs.detach().cpu().numpy()

    # Combine features into a single array
    features = np.concatenate([non_zero_probs_np.T, zero_probs_np.T], axis=0)

    # Create corresponding class labels
    labels = np.concatenate([
        np.zeros(non_zero_probs_np.shape[1]),  # Class 0 for zero_probs
        np.ones(zero_probs_np.shape[1])       # Class 1 for non_zero_probs
    ])

    # Calculate scatter matrices
    SW, SB, ST = calculate_scatter_matrices(features, labels)

    # Compute separability measures
    J1, J2 = class_separability_measure(SW, SB, ST)

    # Print the results
    #print(f"Within-class scatter matrix (SW):\n{SW}")
    #print(f"Between-class scatter matrix (SB):\n{SB}")
    #print(f"Total scatter matrix (ST):\n{ST}")
    #print(f"Separability measure J (SB/SW): {J1}")
    #print(f"Separability measure J (SB/ST): {J2}")


   
    return J1, J2


def extract_features(mask, feature, label, cam, image_id):
    cancer_features = np.empty((0, 2048))
    non_cancer_features = np.empty((0, 2048))
    background_features = np.empty((0, 2048))

    mask = torch.from_numpy(mask)
    #mask_cam = cam > 0.2

    # Interpolate feature to image size
    #feature = F.interpolate(feature.unsqueeze(0), size=mask.shape, mode='bilinear', align_corners=False).squeeze(0)
    # Get position of pixels that are not zero
    # non_zero_indices = torch.nonzero(mask_cam, as_tuple=True)
    non_zero_indices = torch.nonzero(mask, as_tuple=True)
    # Get corresponding features
    non_zero_feature = feature[0, :, non_zero_indices[0], non_zero_indices[1]]
    # if label == 1:
    #     cancer_features= [cancer_features, non_zero_feature.cpu().numpy().T]
    # else:
    #     non_cancer_features = [non_cancer_features, non_zero_feature.cpu().numpy().T]

    # Get position of pixels that are zero
    #mask_cam = cam < 0.2
    #zero_indices = torch.nonzero(mask_cam, as_tuple=True)
    zero_indices = torch.nonzero(mask == 0, as_tuple=True)
    # Get corresponding features
    zero_feature = feature[0, :, zero_indices[0], zero_indices[1]]
    #background_features = [background_features, zero_feature.cpu().numpy().T]

    features1 = non_zero_feature.detach().cpu().numpy().T
    features2 = zero_feature.detach().cpu().numpy().T

    all_features = np.concatenate([features1, features2])
    tsne = TSNE(n_components=2)
    embedded_features = tsne.fit_transform(all_features)
    embedded_features1 = embedded_features[:features1.shape[0]]
    embedded_features2 = embedded_features[features1.shape[0]:]

    # Visualiser les résultats avec un code couleur
    plt.scatter(embedded_features1[:, 0], embedded_features1[:, 1], color='blue', label='Foreground')
    plt.scatter(embedded_features2[:, 0], embedded_features2[:, 1], color='red', label='Background')

    plt.legend()
    if label == 1:
        classe = 'cancer'
    else:
        classe = 'non-cancer'


    parts = image_id.split('/')
    clean_image_id = parts[1].replace('.bmp', '')
    plt.title(f"T-SNE visualization of source features from GLAS\n at the pixel level for a {classe} image ({clean_image_id})")
    # Sauvegarder l'image
    plt.savefig(f'tsne_plot_{clean_image_id}.png')
    plt.close()
    return cancer_features, non_cancer_features, background_features

def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def extract_features_cam(mask, feature, label, cam, image_id):
    cancer_features = np.empty((0, 2048))
    non_cancer_features = np.empty((0, 2048))
    background_features = np.empty((0, 2048))

    mask = torch.from_numpy(mask)
    mask_cam = cam > 0.2

    # Interpolate feature to image size
    #feature = F.interpolate(feature.unsqueeze(0), size=mask.shape, mode='bilinear', align_corners=False).squeeze(0)
    # Get position of pixels that are not zero
    # non_zero_indices = torch.nonzero(mask_cam, as_tuple=True)
    non_zero_indices = torch.nonzero(mask_cam, as_tuple=True)
    # Get corresponding features
    non_zero_feature = feature[0, :, non_zero_indices[0], non_zero_indices[1]]
    # if label == 1:
    #     cancer_features= [cancer_features, non_zero_feature.cpu().numpy().T]
    # else:
    #     non_cancer_features = [non_cancer_features, non_zero_feature.cpu().numpy().T]

    # Get position of pixels that are zero
    mask_cam = cam < 0.2
    zero_indices = torch.nonzero(mask_cam, as_tuple=True)
    #zero_indices = torch.nonzero(mask == 0, as_tuple=True)
    # Get corresponding features
    zero_feature = feature[0, :, zero_indices[0], zero_indices[1]]
    #background_features = [background_features, zero_feature.cpu().numpy().T]

    features1 = non_zero_feature.detach().cpu().numpy().T
    features2 = zero_feature.detach().cpu().numpy().T

    all_features = np.concatenate([features1, features2])
    tsne = TSNE(n_components=2)
    embedded_features = tsne.fit_transform(all_features)
    embedded_features1 = embedded_features[:features1.shape[0]]
    embedded_features2 = embedded_features[features1.shape[0]:]

    # Visualiser les résultats avec un code couleur
    plt.scatter(embedded_features1[:, 0], embedded_features1[:, 1], color='blue', label='Foreground')
    plt.scatter(embedded_features2[:, 0], embedded_features2[:, 1], color='red', label='Background')

    plt.legend()
    if label == 1:
        classe = 'cancer'
    else:
        classe = 'non-cancer'


    parts = image_id.split('/')
    clean_image_id = parts[1].replace('.bmp', '')
    plt.title(f"T-SNE visualization of source features from GLAS\n at the pixel level with CAM for a {classe} image ({clean_image_id})")
    # Sauvegarder l'image
    plt.savefig(f'tsne_plot_cam_{clean_image_id}.png')
    plt.close()
    return cancer_features, non_cancer_features, background_features

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

def compute_energy(logits):
    return -torch.logsumexp(logits, dim=1)

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


def get_features(exp_path, sf_uda_source_folder,image_ids_to_draw,image_ids_to_draw_target, checkpoint_type, source_dataset, target_dataset, cudaid, split, tmp_outd='tmp_outd', parsedargs=None, target_method=None, n_components=2):


    with open(join(exp_path, 'config_obj_final.yaml'), 'r') as fy:
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
        if 'PixelCAM' in args.method:
            args_dict['pixel_wise_classification'] = True
            args_dict['anchors_ortogonal'] = False
            args_dict['batch_norm_pixel_classifier'] = False
            args_dict['low_res'] = False
            args_dict['multiple_layer_pixel_classifier'] = False
            args_dict['detach_pixel_classifier'] = False
            args_dict['one_layer_pixel_classifier'] = False
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

    # elif target_method == 'NEGEV':
    #     encoder_w = torch.load(join(path_cl, 'encoder.pt'),
    #                         map_location=get_cpu_device())
    #     model.encoder.super_load_state_dict(encoder_w, strict=True)

    #     header_w = torch.load(join(path_cl, 'classification_head.pt'),
    #                         map_location=get_cpu_device())
    #     model.classification_head.load_state_dict(header_w, strict=True)

    #     decoder_w = torch.load(join(path_cl, 'decoder.pt'),
    #                         map_location=get_cpu_device())
    #     model.decoder.super_load_state_dict(decoder_w, strict=True)

    #     seg_head_w = torch.load(join(path_cl, 'segmentation_head.pt'),
    #                         map_location=get_cpu_device())
    #     model.segmentation_head.load_state_dict(seg_head_w, strict=True)




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
    

    overlay_images = {}
    input_images = {}
    gt_masks = {}
    #all_pixel_features = []
    source_img_features = []
    target_img_features = []
    

    model.eval()
    for batch_idx, (images, targets, p_glabel, index, raw_imgs, std_cams, _, views) in tqdm(
        enumerate(source_loaders[split]), ncols=constants.NCOLS,
        total=len(source_loaders[split])):
        image_size = images.shape[2:]
        images = images.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            out = model(images)
            lin_ft = model.lin_ft.detach().cpu()
            source_img_features.append(lin_ft)

    source_img_features = torch.cat(source_img_features, dim=0)
    
    
    h,w = source_img_features.shape 
    gmm = GaussianMixture(n_components, w)
    gmm.fit(source_img_features,delta=1e-8, n_iter=500)      

    for batch_idx, (images, targets, p_glabel, index, raw_imgs, std_cams, _, views) in tqdm(
    enumerate(target_loaders[split]), ncols=constants.NCOLS,
    total=len(target_loaders[split])):
        image_size = images.shape[2:]
        images = images.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            out = model(images)
            lin_ft = model.lin_ft.detach().cpu()
            target_img_features.append(lin_ft)

    target_img_features = torch.cat(target_img_features, dim=0)

    
    source_log_likelihood = gmm.score_samples(source_img_features)
    target_log_likelihood = gmm.score_samples(target_img_features)
    
    proba_source = gmm.predict_proba(source_img_features) 
    max_probas_source = proba_source.max(dim=1).values   # shape: [N]

    proba_target = gmm.predict_proba(target_img_features) 
    max_probas_target = proba_target.max(dim=1).values   # shape: [N]

    max_probas_source = max_probas_source.cpu().numpy()
    max_probas_target = max_probas_target.cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(max_probas_source, bins=30, alpha=0.5, density=True,  label='Source', color='blue', edgecolor='black', linewidth=0.5)
    plt.hist(max_probas_target, bins=30, alpha=0.5, density=True,  label='Target', color='red', edgecolor='black', linewidth=0.5)

    plt.title("Histo max probability GMM")
    plt.xlabel("Max proba")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gmm_max_proba_density_fixed.png")
    plt.show()  
                
    # output_dir = os.path.join('gmm_model', dataset, 'source', f"n_components_{n_components}")
    # file_path = os.path.join(output_dir, f"gmm_params_{dataset}_{n_components}.pt")

    # os.makedirs(output_dir, exist_ok=True)

    # torch.save({'mu': gmm.mu.detach().cpu(), 'var': gmm.var.detach().cpu(), 'pi': gmm.pi.detach().cpu()}, file_path)

    # indices = torch.randint(0, 50000, (100,))

    # selected_features = all_pixel_features[indices]
    # log_likelihood = gmm.score_samples(selected_features)
    # proba = gmm.predict_proba(selected_features)

    # density = torch.exp(log_likelihood)

    # # Plot des résultats
    # log_likelihood_np = log_likelihood.cpu().numpy()

    # plt.figure(figsize=(8, 4))
    # plt.hist(log_likelihood_np, bins=30, color='blue', alpha=0.6)
    # plt.xlabel("Log-Likelihood")
    # plt.ylabel("Nombre de features")
    # plt.title("Distribution des log-likelihoods du GMM")
    # plt.axvline(log_likelihood_np.mean(), color='red', linestyle='dashed', linewidth=2, label="Moyenne")
    # plt.legend()
    # plt.savefig("density_gmm_2D.png", dpi=300, bbox_inches="tight")
    # plt.show()



    #torch.save({'mu': gmm.mu.detach().cpu(), 'var': gmm.var.detach().cpu()}, file_path)

    # tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    # X_tsne = tsne.fit_transform(all_pixel_features_np)  # (N, 2)

    # # Prédiction des clusters GMM
    # labels = gmm.predict(all_pixel_features_np)

    # # Visualisation des clusters GMM après t-SNE
    # plt.figure(figsize=(10, 7))
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6)
    # #plt.scatter(tsne.fit_transform(gmm.means_)[:, 0], tsne.fit_transform(gmm.means_)[:, 1], 
    # #            c='red', marker='x', s=200, label="Centres GMM")
    # plt.title("Clustering GMM avec Réduction t-SNE")
    # plt.legend()
 
    # # Sauvegarde de l'image
    # plt.savefig("clustering_gmm_tsne.png", dpi=300)


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
    parser.add_argument("--target_dataset", type=str, default=None, help="Source dataset")
    parser.add_argument('--image_ids_to_draw', nargs='+', type=str, default=None)
    parser.add_argument('--image_ids_to_draw_target', nargs='+', type=str, default=None)
    parser.add_argument("--n_components", type=int, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--source_dataset", type=str, default=None, help="Source dataset")
    parser.add_argument("--path_pre_trained_source", type=str, default=None, help="Path to the pre-trained source model.")
    #parser.add_argument("--path_pre_trained_source", type=json.loads, default={})
    

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

    base_checkpoint_types = [constants.BEST_LOC]
        
    for checkpoint_type_extended in base_checkpoint_types:
        checkpoint_type = checkpoint_type_extended
        
        #split = parsedargs.split
        split = "train"
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
            get_features(exp_path=exp_path, sf_uda_source_folder=parsedargs.path_pre_trained_source,image_ids_to_draw=parsedargs.image_ids_to_draw,image_ids_to_draw_target=parsedargs.image_ids_to_draw_target, checkpoint_type=checkpoint_type, source_dataset=parsedargs.source_dataset,target_dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs, target_method=target_method, n_components = parsedargs.n_components)

if __name__ == '__main__':
    fast_eval()