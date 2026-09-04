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
import umap

from collections import Counter

import re
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from dlib.learning.inference_wsol import CAMComputer

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
from collections import Counter





def cl_forward(args, model, images):

    output = model(images)

    if args.task == constants.STD_CL:
        cl_logits = output

    elif args.task == constants.F_CL:
        cl_logits, fcams, im_recon = output
    else:
        raise NotImplementedError

    return cl_logits
    
# def _compute_accuracy(args, model, loader):
#     num_correct = 0
#     num_images = 0

#     for i, (images, targets, _, _, _, _, _, _) in enumerate(loader):
#         images = images.cuda()
#         targets = targets.cuda()
#         with torch.no_grad():
#             cl_logits = cl_forward(args, model, images)
#             pred = cl_logits.argmax(dim=1)

#         num_correct += (pred == targets).sum().item()
#         num_images += images.size(0)

#     classification_acc = num_correct / float(num_images) * 100
#     return classification_acc

def _compute_accuracy(args, model, loader, num_classes=2):
    num_correct = 0
    num_images = 0

    pred_counter = Counter()

    for i, (images, targets, _, _, _, _, _, _, _) in enumerate(loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        with torch.no_grad():
            cl_logits = cl_forward(args, model, images)
            pred = cl_logits.argmax(dim=1)

        # accuracy
        num_correct += (pred == targets).sum().item()
        num_images += images.size(0)

        # count predictions
        pred_counter.update(pred.cpu().tolist())

    # accuracy
    classification_acc = num_correct / float(num_images) * 100

    # counts per predicted class
    pred_counts = torch.zeros(num_classes, dtype=torch.float32)
    for k, v in pred_counter.items():
        if 0 <= k < num_classes:
            pred_counts[k] = float(v)

    # prediction percentages (bias)
    total_preds = pred_counts.sum().clamp_min(1.0)  # avoid div by 0
    pred_percents = (pred_counts / total_preds) * 100.0

    # over / under predicted classes
    over_pred_class = int(pred_counts.argmax().item())
    under_pred_class = int(pred_counts.argmin().item())

    max_pred = pred_counts.max().item()
    min_pred = pred_counts.min().item()
    ratio_pred = max_pred / (min_pred + 1e-6)

    return (
        classification_acc,
        ratio_pred,
        pred_counts,
        pred_percents,      
        over_pred_class,
        under_pred_class,
    )


def cl_compute_logits(args, model, b_cl_logits, patch_embdgs, patch_embdgs_without_proj, train=False, return_cam=False, image=None):
    
    # if not args.use_conv_binary_classifier:
    b_cl_logits_norm = b_cl_logits.softmax(dim=1)
    if return_cam:
        # assert len(b_cl_logits) == 1
        # if args.use_decoder_to_upscale_features:
        #     cam = b_cl_logits[:,:,1].reshape(args.crop_size, args.crop_size)
        # elif args.use_pixel_shuffle_to_upscale_features:
        #     cam = b_cl_logits[:,1,:,:].squeeze()#[:,:,1].reshape(56,56)
        # else:
        assert len(b_cl_logits) == 1
        cam = b_cl_logits[0,1,:,:]#.reshape(14,14)
            
    #check if best_cl_model exists in self.model.best_cl_model
    if hasattr(model, 'best_cl_model') and model.best_cl_model is not None:
        cl_logits = model.best_cl_model(image)
        
        if return_cam:
            return cl_logits, cam

        return cl_logits
        
    # if args.aug_ran_erase_patch_embd_p and train:
    #     random_tensor_p_to_erase = torch.rand(patch_embdgs.size(1))
    #     mask = random_tensor_p_to_erase <= args.aug_ran_erase_patch_embd_p
    #     mask_b_cl_logits_norm = mask.clone().unsqueeze(0).unsqueeze(2).expand_as(b_cl_logits_norm[:, :, 1:2])
    #     mask = mask.unsqueeze(0).unsqueeze(2).expand_as(patch_embdgs)
    #     patch_embdgs_modified = patch_embdgs.clone()
    #     patch_embdgs_modified[mask] = 0
        
    #     b_cl_logits_norm_modified = b_cl_logits_norm[:, :, 1:2].clone()
    #     b_cl_logits_norm_modified[mask_b_cl_logits_norm] = 0

    #     patch_embdgs_wa = (b_cl_logits_norm_modified * patch_embdgs_modified).sum(1) / (torch.sum(b_cl_logits_norm_modified, dim=1)+1e-8)
    # elif args.aug_wildcat_pool_patch_embd:
    #     patch_embdgs_wa = cl_forward_wild_cat_aug_patchs(args=args, patch_embdgs=patch_embdgs, b_cl_logits_norm=b_cl_logits_norm)
        
    # else:
    patch_embdgs_wa = (b_cl_logits_norm[:, 1:2, :, :] * patch_embdgs).sum((2,3)) / (torch.sum(b_cl_logits_norm[:, 1:2, :, :], dim=(2,3))+1e-8)
    # else:
    #     if return_cam:
    #         cam = b_cl_logits[:,1,:,:]
                
    #     b_cl_logits_norm = b_cl_logits.softmax(dim=1)
        
    #     b = b_cl_logits_norm.shape[0]
    #     w = b_cl_logits_norm.shape[2]
    #     h = b_cl_logits_norm.shape[3]
    #     b_cl_logits_norm = b_cl_logits_norm[:, 1, :, :].reshape(b, w*h).unsqueeze(2)
    #     if args.aug_ran_erase_patch_embd_p and train:
    #         raise NotImplementedError
    #     elif args.aug_wildcat_pool_patch_embd:
    #         raise NotImplementedError
    #     else:
    #         patch_embdgs_wa = (b_cl_logits_norm * patch_embdgs).sum(1) / (torch.sum(b_cl_logits_norm, dim=1)+1e-8)
       
    # cl_logits = model.visual.forward_cl(patch_embdgs_wa)
    if hasattr(args, 'inference_with_shared_backbone') and args.inference_with_shared_backbone:
        cl_logits = model.forward_ext_cl(patch_embdgs_without_proj)
    if hasattr(model, 'use_default_cl_forward') and model.use_default_cl_forward:
        cl_logits = model.encoder.forward_head(patch_embdgs_without_proj)
    else:
        cl_logits = model.forward_cl(patch_embdgs_wa)
         
    if return_cam:
        return cl_logits, cam

    return cl_logits


def cl_forward_CLIPDISTILL_TXTENC(args, model, images: torch.Tensor, blured_imgs: torch.Tensor=None):
        output = model(images)

        if args.task == constants.STD_CL:
            cl_logits = output

        elif args.task in [constants.F_CL, constants.TCAM,
                                constants.COLOCAM]:
            cl_logits, fcams, im_recon = output

        elif args.task == constants.CLIPDISTILL_TXTENC:
            output, output_without_proj = output
            patch_embdgs = output#[:,1:,:]
            patch_embdgs_without_proj = output_without_proj
            b_cl_logits = model.forward_binary(patch_embdgs)
            cl_logits = cl_compute_logits(args=args, model=model, b_cl_logits=b_cl_logits, patch_embdgs=patch_embdgs, patch_embdgs_without_proj=patch_embdgs_without_proj, train=False, image=images)

        else:
            raise NotImplementedError
        

        return cl_logits

def _compute_accuracy_CLIPDISTILL_TXTENC(args, model, loader):
        num_correct = 0
        num_images = 0

        for i, (images, targets, _, _, _, _, _, _, _, _) in enumerate(loader):
            images = images.cuda(args.c_cudaid)
            targets = targets.cuda(args.c_cudaid)
            with torch.no_grad():
                blured_imgs = None
                with autocast(enabled=args.amp_eval):


                    cl_logits = cl_forward_CLIPDISTILL_TXTENC(args=args, model=model, images=images,
                                                blured_imgs=blured_imgs
                                                ).detach()

                pred = cl_logits.argmax(dim=1)
                num_correct += (pred == targets).sum().detach()
                num_images += images.size(0)

        # sync
        # if self.args.distributed:
        #     num_correct = sync_tensor_across_gpus(num_correct.view(1, )).sum()
        #     nx = torch.tensor([num_images], dtype=torch.float,
        #                       requires_grad=False, device=torch.device(
        #             self.args.c_cudaid)).view(1, )
        #     num_images = sync_tensor_across_gpus(nx).sum().item()
        #     dist.barrier()

        classification_acc = num_correct / float(num_images) * 100
        # if self.args.distributed:
        #     dist.barrier()

        torch.cuda.empty_cache()
        return classification_acc.item()



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
    #plt.title(f"T-SNE visualization for a {classe} image ", fontsize=10)
    # Sauvegarder l'image
    output_dir = os.path.join('visualization', 'tsne', dataset, classe, parsedargs.dataset_type, clean_image_id)
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"{target_method}_prob_plot.png")

    # Créer le scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(non_zero_coords[0], non_zero_coords[1], c='blue', alpha=0.5)
    plt.scatter(zero_coords[0], zero_coords[1], c='red', alpha=0.5)
    plt.xlabel('Background Probability ')
    plt.ylabel('Foreground Probability ')
    #plt.legend()
    #plt.title('2D Scatter Plot of Probabilities')
    plt.grid(True)
    plt.show()
    plt.legend()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    # if label_source.item() == 1:
    #     classe = 'cancer'
    # else:
    #     classe = 'normal'


    # parts = image_id_source.split('/')
    # clean_image_id = parts[-1].replace('.bmp', '').replace('.png', '')
    # #plt.title(f"T-SNE visualization for a {classe} image ", fontsize=10)
    # # Sauvegarder l'image
    # output_dir = os.path.join('visualization', 'tsne', dataset, classe, parsedargs.dataset_type, clean_image_id)
    # os.makedirs(output_dir, exist_ok=True)

    # filename = os.path.join(output_dir, f"{target_method}.png")
    # plt.xticks([]) 
    # plt.yticks([])
    # plt.axis('off')
    # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    # plt.close()
    return non_zero_probs, non_zero_probs, zero_probs


def plot_pixel_logits(mask_source, logits, label_source, cam_source, image_id_source, target_method, dataset, parsedargs):
    _,l,m,n=logits.shape

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
    # if label_source.item() == 1:
    #     classe = 'cancer'
    # else:
    #     classe = 'normal'


    # parts = image_id_source.split('/')
    # clean_image_id = parts[-1].replace('.bmp', '').replace('.png', '')
    # #plt.title(f"T-SNE visualization for a {classe} image ", fontsize=10)
    # # Sauvegarder l'image
    # output_dir = os.path.join('visualization', 'tsne', dataset, classe, parsedargs.dataset_type, clean_image_id)
    # os.makedirs(output_dir, exist_ok=True)

    # filename = os.path.join(output_dir, f"{target_method}.png")
    # plt.xticks([]) 
    # plt.yticks([])
    # plt.axis('off')
    # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    # plt.close()
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


def class_separability(mask_source, features, label_source, image_id_source, target_method, dataset, parsedargs):
    l,m,n=features.shape

    cancer_features = np.empty((0, l))
    non_cancer_features = np.empty((0, l))
    background_features = np.empty((0, l))

    #source
    mask_source = torch.from_numpy(mask_source)

    non_zero_indices = torch.nonzero(mask_source, as_tuple=True)
    zero_indices = torch.nonzero(mask_source == 0, as_tuple=True)

    non_zero_probs = features[:, non_zero_indices[0], non_zero_indices[1]]
    zero_probs = features[:, zero_indices[0], zero_indices[1]]

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


# def _compute_accuracy(args, model, loader):
#     num_correct = 0
#     num_images = 0

#     for i, (images, targets, _, _, _, _, _, _, _) in enumerate(loader):
#         images = images.cuda()
#         targets = targets.cuda()
#         with torch.no_grad():
#             cl_logits = cl_forward(args, model, images)
#             pred = cl_logits.argmax(dim=1)

#         num_correct += (pred == targets).sum().item()
#         num_images += images.size(0)

#     classification_acc = num_correct / float(num_images) * 100
#     return classification_acc

def compute_ece(probs, labels, n_bins=15):
    bin_boundaries = torch.linspace(0, 1, n_bins+1)
    ece = 0.0

    confidences, predictions = torch.max(probs, dim=1)
    accuracies = (predictions == labels).float()

    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i+1]
        mask = (confidences > low) & (confidences <= high)
        if mask.sum() > 0:
            accuracy_in_bin  = accuracies[mask].mean()
            avg_conf_in_bin  = confidences[mask].mean()
            ece += (mask.float().mean() * torch.abs(avg_conf_in_bin - accuracy_in_bin))
    return float(ece)


def compute_nll(logits, labels):
    return float(F.cross_entropy(logits, labels).item())


def compute_brier(probs, labels, num_classes):
    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    return float(((probs - one_hot)**2).mean().item())



def compute_kl_uniform(preds, num_classes):
    # p(y) estimé empiriquement
    hist = torch.bincount(preds, minlength=num_classes).float()
    p = hist / hist.sum()          # distribution empirique des classes
    u = torch.ones_like(p) / num_classes  # prior uniforme
    kl = (p * (p / u).log()).sum()
    return float(kl.item())


def compute_margin(probs):
    top2 = torch.topk(probs, k=2, dim=1).values
    margin = top2[:,0] - top2[:,1]
    return float(margin.mean().item()), float(margin.median().item())



def compute_knn_acc(features, labels, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features, labels)
    pred = knn.predict(features)
    return float((pred == labels).mean())


def get_clip_text_embeddings_and_names(args, model):
    text_features = None
    if hasattr(model, "fg_text_features"):
        text_features = model.fg_text_features
    elif hasattr(model, "text_features"):
        text_features = model.text_features
    elif hasattr(model, "class_anchors"):
        text_features = model.class_anchors
    elif hasattr(model, "decoder") and hasattr(model.decoder, "class_anchors"):
        text_features = model.decoder.class_anchors

    if text_features is None:
        return None, []

    if isinstance(text_features, torch.Tensor):
        text_features = text_features.detach().float().cpu()
    else:
        text_features = torch.as_tensor(text_features, dtype=torch.float32)
    text_features = F.normalize(text_features, dim=-1)

    class_names = getattr(model, "class_names", None)
    if class_names is None and hasattr(args, "name_classes"):
        name_classes = args.name_classes
        if isinstance(name_classes, dict):
            class_names = [name for name, _ in sorted(name_classes.items(), key=lambda kv: kv[1])]

    if class_names is None:
        class_names = [f"class {i}" for i in range(text_features.shape[0])]
    class_names = [str(name) for name in class_names]

    return text_features.numpy(), class_names


def get_clip_image_embeddings_for_umap(args, model, images):
    if args.task == constants.CLIPDISTILL_TXTENC:
        output, output_without_proj = model(images)
        patch_embdgs = output
        patch_embdgs_without_proj = output_without_proj

        if hasattr(model, "encoder") and hasattr(model.encoder, "visual") and hasattr(model.encoder.visual, "forward_project"):
            image_features = model.encoder.visual.forward_project(patch_embdgs_without_proj)
        else:
            b_cl_logits = model.forward_binary(patch_embdgs)
            b_cl_logits_norm = b_cl_logits.softmax(dim=1)
            image_features = (
                b_cl_logits_norm[:, 1:2, :, :] * patch_embdgs
            ).sum((2, 3)) / (torch.sum(b_cl_logits_norm[:, 1:2, :, :], dim=(2, 3)) + 1e-8)

            if hasattr(args, "inference_with_shared_backbone") and args.inference_with_shared_backbone:
                image_features = patch_embdgs_without_proj
        return F.normalize(image_features.detach().float(), dim=-1)

    image_features = getattr(model, "encoder_last_features", None)
    if image_features is None:
        return None
    if image_features.ndim > 2:
        image_features = image_features.flatten(2).mean(dim=-1)
    return F.normalize(image_features.detach().float(), dim=-1)


def subsample_umap_embeddings(features, labels, max_per_class, seed=42):
    if max_per_class is None or max_per_class <= 0:
        return features, labels

    rng = np.random.default_rng(seed)
    keep_indices = []
    labels = np.asarray(labels)
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        if len(indices) > max_per_class:
            indices = rng.choice(indices, size=max_per_class, replace=False)
        keep_indices.extend(indices.tolist())

    keep_indices = np.array(sorted(keep_indices), dtype=np.int64)
    return features[keep_indices], labels[keep_indices]


def plot_clip_embeddings_umap(
    image_features,
    image_labels,
    text_features,
    class_names,
    out_dir,
    split,
    target_dataset,
    n_neighbors=15,
    min_dist=0.2,
    metric="cosine",
    max_images_per_class=500,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_features = np.asarray(image_features, dtype=np.float32)
    image_labels = np.asarray(image_labels, dtype=np.int64)
    text_features = np.asarray(text_features, dtype=np.float32)

    if image_features.ndim != 2 or text_features.ndim != 2:
        print("[WARN] UMAP skipped: image/text embeddings must be 2D arrays.")
        return None
    if image_features.shape[1] != text_features.shape[1]:
        print(
            "[WARN] UMAP skipped: image embedding dim "
            f"{image_features.shape[1]} != text embedding dim {text_features.shape[1]}."
        )
        return None

    image_features, image_labels = subsample_umap_embeddings(
        image_features, image_labels, max_per_class=max_images_per_class
    )
    if len(image_features) < 2:
        print("[WARN] UMAP skipped: not enough image embeddings.")
        return None

    all_features = np.vstack([image_features, text_features])
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, max(2, len(all_features) - 1)),
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    embedding = reducer.fit_transform(all_features)
    image_embedding = embedding[:len(image_features)]
    text_embedding = embedding[len(image_features):]

    plt.figure(figsize=(8, 6))
    dot_size = 24
    for label in np.unique(image_labels):
        idx = image_labels == label
        label_int = int(label)
        label_name = (
            class_names[label_int]
            if 0 <= label_int < len(class_names)
            else f"class {label_int}"
        )
        plt.scatter(
            image_embedding[idx, 0],
            image_embedding[idx, 1],
            label=f"images: {label_name}",
            alpha=0.65,
            s=dot_size,
        )

    plt.scatter(
        text_embedding[:, 0],
        text_embedding[:, 1],
        marker="*",
        s=260,
        c="black",
        edgecolors="white",
        linewidths=0.8,
        label="class prompts",
        zorder=5,
    )
    for i, name in enumerate(class_names[:len(text_embedding)]):
        plt.text(
            text_embedding[i, 0],
            text_embedding[i, 1],
            f" {name}",
            fontsize=10,
            ha="left",
            va="center",
            zorder=6,
        )

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=True)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"UMAP image embeddings vs class prompts - {target_dataset} ({split})")
    plt.tight_layout()

    out_path = out_dir / f"umap_clip_image_vs_prompt_embeddings_{target_dataset}_{split}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def get_tedloc_pixel_entropy_maps(args, model, images):
    if args.task != constants.CLIPDISTILL_TXTENC:
        return None

    output, _ = model(images)
    b_cl_logits = model.forward_binary(output)
    pixel_probs = torch.softmax(b_cl_logits, dim=1)
    pixel_entropy = -torch.sum(
        pixel_probs * torch.log(pixel_probs.clamp_min(1e-12)),
        dim=1,
    )
    return pixel_entropy.detach()


def collect_tedloc_fg_bg_pixel_entropies(
    pixel_entropy,
    image_ids,
    mask_root,
    mask_paths,
    ignore_paths,
):
    fg_entropies = []
    bg_entropies = []

    for entropy_map, image_id in zip(pixel_entropy, image_ids):
        if image_id not in mask_paths:
            continue

        gt_mask = get_mask(mask_root, mask_paths[image_id], ignore_paths[image_id])
        entropy_np = entropy_map.float().cpu().numpy()

        if gt_mask.shape != entropy_np.shape:
            gt_mask = cv2.resize(
                gt_mask.astype(np.uint8),
                (entropy_np.shape[1], entropy_np.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        valid_mask = gt_mask != 255
        fg_mask = (gt_mask > 0) & valid_mask
        bg_mask = (gt_mask == 0) & valid_mask

        if fg_mask.any():
            fg_entropies.append(entropy_np[fg_mask])
        if bg_mask.any():
            bg_entropies.append(entropy_np[bg_mask])

    return fg_entropies, bg_entropies


def plot_tedloc_pixel_entropy_distribution(
    fg_entropies,
    bg_entropies,
    out_dir,
    split,
    target_dataset,
    bins=80,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fg_vals = np.concatenate(fg_entropies) if len(fg_entropies) > 0 else np.array([])
    bg_vals = np.concatenate(bg_entropies) if len(bg_entropies) > 0 else np.array([])

    if len(fg_vals) == 0 and len(bg_vals) == 0:
        print("[WARN] TEDLOC pixel entropy skipped: no FG/BG entropy values collected.")
        return None

    saved_paths = []
    for vals, region_name, color in [
        (fg_vals, "fg", "tab:blue"),
        (bg_vals, "bg", "tab:orange"),
    ]:
        if len(vals) == 0:
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(vals, bins=bins, alpha=0.85, density=True, color=color)
        plt.xlabel("Pixel entropy")
        plt.ylabel("Density")
        plt.title(
            f"TEDLOC {region_name.upper()} pixel entropy - "
            f"{target_dataset} ({split}, n={len(vals)})"
        )
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_path = out_dir / f"tedloc_pixel_entropy_{region_name}_{target_dataset}_{split}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
        saved_paths.append(out_path)

    stats_path = out_dir / f"tedloc_pixel_entropy_fg_bg_{target_dataset}_{split}.json"
    stats = {
        "fg": {
            "count": int(len(fg_vals)),
            "mean": float(np.mean(fg_vals)) if len(fg_vals) > 0 else float("nan"),
            "std": float(np.std(fg_vals)) if len(fg_vals) > 0 else float("nan"),
            "median": float(np.median(fg_vals)) if len(fg_vals) > 0 else float("nan"),
        },
        "bg": {
            "count": int(len(bg_vals)),
            "mean": float(np.mean(bg_vals)) if len(bg_vals) > 0 else float("nan"),
            "std": float(np.std(bg_vals)) if len(bg_vals) > 0 else float("nan"),
            "median": float(np.median(bg_vals)) if len(bg_vals) > 0 else float("nan"),
        },
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")

    return saved_paths


def save_or_show(filename=None, save=False, save_dir=None):
    if save and filename is not None and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[saved] {path}")
    else:
        plt.show()


def save_results(
    output_dir,
    source_dataset,
    source_fold,
    target_dataset,
    target_fold,
    classification_acc,
    cam_perf_pxap,
    filename="conch_result_target.txt",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # create folder if needed

    file_path = output_dir / filename

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Source dataset: {source_dataset} | fold: {source_fold}\n")
        f.write(f"Target dataset: {target_dataset} | fold: {target_fold}\n")
        f.write(f"Classification Accuracy: {classification_acc:.2f}%\n")
        f.write(f"CAM Performance PxAP: {cam_perf_pxap:.2f}\n")
        f.write("=" * 60 + "\n")

    return file_path


def parse_camelyon_patch_info(image_id: str):
    """
    Parse robuste pour les noms CAMELYON, y compris les variantes avec:
      - reg_<id>
      - row_<id>
      - tissue_<score>
      - metastatic_<score>
    """

    filename = os.path.basename(image_id)

    m_x = re.search(r"_x_(\d+)", filename)
    m_y = re.search(r"_y_(\d+)", filename)
    m_w = re.search(r"_w_(\d+)", filename)
    m_h = re.search(r"_h_(\d+)\.png$", filename)

    if not all([m_x, m_y, m_w, m_h]):
        return None

    x = int(m_x.group(1))
    y = int(m_y.group(1))
    w = int(m_w.group(1))
    h = int(m_h.group(1))

    # tout ce qui est avant _x_<...>
    prefix = filename[:m_x.start()]

    # enlever le dernier suffixe _patch_<id> s'il existe
    prefix_wo_patch = re.sub(r"_patch_\d+$", "", prefix)

    # enlever le préfixe Patch_file_
    slide_id = re.sub(r"^Patch_file_", "", prefix_wo_patch)

    # patch id = dernier _patch_<id> avant _x_
    m_patch = re.search(r"_patch_(\d+)$", prefix)
    patch_id = int(m_patch.group(1)) if m_patch else -1

    return {
        "slide_id": slide_id,
        "patch_id": patch_id,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "cx": x + w / 2.0,
        "cy": y + h / 2.0,
    }

def enrich_metrics_with_spatial_info(per_image_metrics):
    enriched = []
    num_failed = 0

    for m in per_image_metrics:
        info = parse_camelyon_patch_info(m["image_id"])
        if info is None:
            num_failed += 1
            print(f"[WARN] parse failed for image_id: {m['image_id']}")
            continue

        mm = dict(m)
        mm.update(info)
        enriched.append(mm)

    print(f"[INFO] enrich_metrics_with_spatial_info: kept={len(enriched)}, failed={num_failed}")
    return enriched


def compute_neighbor_stats(per_image_metrics, ks=(3, 5, 7)):
    """
    Pour chaque image, calcule les k plus proches voisins dans la même slide.
    Retourne:
      - augmented_metrics: liste enrichie
      - summary: stats globales pour chaque k
    """
    data = enrich_metrics_with_spatial_info(per_image_metrics)

    by_slide = defaultdict(list)
    for idx, item in enumerate(data):
        by_slide[item["slide_id"]].append(idx)

    # initialiser champs
    for item in data:
        for k in ks:
            item[f"neighbor_ids_{k}"] = []
            item[f"neighbor_distances_{k}"] = []
            item[f"neighbor_mean_cl_{k}"] = float("nan")
            item[f"neighbor_std_cl_{k}"] = float("nan")
            item[f"neighbor_mean_pxap_{k}"] = float("nan")
            item[f"neighbor_std_pxap_{k}"] = float("nan")
            item[f"neighbor_valid_pxap_count_{k}"] = 0

    # calcul voisins slide par slide
    for slide_id, indices in by_slide.items():
        if len(indices) < 2:
            continue

        coords = np.array([[data[i]["cx"], data[i]["cy"]] for i in indices], dtype=np.float32)

        # matrice des distances euclidiennes
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))

        # ignorer self
        np.fill_diagonal(dist, np.inf)

        for local_i, global_i in enumerate(indices):
            order = np.argsort(dist[local_i])

            for k in ks:
                kk = min(k, len(indices) - 1)
                neigh_local = order[:kk]
                neigh_global = [indices[j] for j in neigh_local]
                neigh_dist = dist[local_i, neigh_local].tolist()

                data[global_i][f"neighbor_ids_{k}"] = [data[j]["image_id"] for j in neigh_global]
                data[global_i][f"neighbor_distances_{k}"] = neigh_dist

                # CL par image = is_correct -> 0/1
                cl_vals = np.array([1.0 if data[j]["is_correct"] else 0.0 for j in neigh_global], dtype=np.float32)
                data[global_i][f"neighbor_mean_cl_{k}"] = float(cl_vals.mean()) if len(cl_vals) > 0 else float("nan")
                data[global_i][f"neighbor_std_cl_{k}"] = float(cl_vals.std()) if len(cl_vals) > 0 else float("nan")

                # PxAP seulement si dispo
                pxap_vals = [data[j]["pxap"] for j in neigh_global if data[j]["pxap"] is not None]
                data[global_i][f"neighbor_valid_pxap_count_{k}"] = len(pxap_vals)

                if len(pxap_vals) > 0:
                    pxap_vals = np.array(pxap_vals, dtype=np.float32)
                    data[global_i][f"neighbor_mean_pxap_{k}"] = float(pxap_vals.mean())
                    data[global_i][f"neighbor_std_pxap_{k}"] = float(pxap_vals.std())

    # résumé global
    summary = {}
    for k in ks:
        all_cl = [m[f"neighbor_mean_cl_{k}"] for m in data if not np.isnan(m[f"neighbor_mean_cl_{k}"])]
        all_pxap = [m[f"neighbor_mean_pxap_{k}"] for m in data if not np.isnan(m[f"neighbor_mean_pxap_{k}"])]

        summary[str(k)] = {
            "num_images": len(data),
            "cl_neighbor_mean_mean": float(np.mean(all_cl)) if len(all_cl) else float("nan"),
            "cl_neighbor_mean_std": float(np.std(all_cl)) if len(all_cl) else float("nan"),
            "pxap_neighbor_mean_mean": float(np.mean(all_pxap)) if len(all_pxap) else float("nan"),
            "pxap_neighbor_mean_std": float(np.std(all_pxap)) if len(all_pxap) else float("nan"),
        }

    return data, summary


def plot_cancer_only_neighbor_histograms(augmented_metrics, out_dir, split, ks=(3, 5, 7)):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # garder uniquement les images cancer
    cancer_metrics = [m for m in augmented_metrics if m["true_class"] == 1]

    if len(cancer_metrics) == 0:
        print("No cancer images found for neighbor histograms.")
        return

    # =========================================================
    # Histogrammes PxAP voisins
    # =========================================================
    plt.figure(figsize=(15, 4))
    for i, k in enumerate(ks, start=1):
        vals = [
            m[f"neighbor_mean_pxap_{k}"]
            for m in cancer_metrics
            if not np.isnan(m[f"neighbor_mean_pxap_{k}"])
        ]

        plt.subplot(1, len(ks), i)
        if len(vals) > 0:
            plt.hist(vals, bins=100, alpha=0.8)
        plt.xlabel(f"Mean PxAP of {k} neighbors")
        plt.ylabel("Number of cancer images")
        plt.title(f"k={k}")
        plt.grid(alpha=0.3)

    plt.suptitle(f"Cancer only - neighbor mean PxAP histograms ({split})")
    plt.tight_layout()
    plt.savefig(out_dir / f"cancer_only_neighbor_pxap_histograms_{split}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================
    # Histogrammes CL voisins
    # =========================================================
    plt.figure(figsize=(15, 4))
    for i, k in enumerate(ks, start=1):
        vals = [
            m[f"neighbor_mean_cl_{k}"]
            for m in cancer_metrics
            if not np.isnan(m[f"neighbor_mean_cl_{k}"])
        ]

        plt.subplot(1, len(ks), i)
        if len(vals) > 0:
            plt.hist(vals, bins=100, alpha=0.8)
        plt.xlabel(f"Mean CL of {k} neighbors")
        plt.ylabel("Number of cancer images")
        plt.title(f"k={k}")
        plt.grid(alpha=0.3)

    plt.suptitle(f"Cancer only - neighbor mean CL histograms ({split})")
    plt.tight_layout()
    plt.savefig(out_dir / f"cancer_only_neighbor_cl_histograms_{split}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

def plot_neighbor_cl_vs_center_accuracy(augmented_metrics, out_dir, split, ks=(3, 5, 7)):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(15, 4))

    for i, k in enumerate(ks, start=1):
        x_vals = []
        y_vals = []

        for m in augmented_metrics:
            x = m.get(f"neighbor_mean_cl_{k}", float("nan"))
            y = 1.0 if m["is_correct"] else 0.0
            if not np.isnan(x):
                x_vals.append(x)
                y_vals.append(y)

        x_vals = np.array(x_vals, dtype=np.float32)
        y_vals = np.array(y_vals, dtype=np.float32)

        uniq = np.sort(np.unique(x_vals))
        mean_acc = []
        counts = []

        for u in uniq:
            mask = x_vals == u
            mean_acc.append(y_vals[mask].mean())
            counts.append(mask.sum())

        plt.subplot(1, len(ks), i)
        plt.plot(uniq, mean_acc, marker='o')
        for x, y, c in zip(uniq, mean_acc, counts):
            plt.text(x, y, str(int(c)), fontsize=8, ha='center', va='bottom')

        plt.ylim(0.0, 1.05)
        plt.xlabel(f"neighbor_mean_cl_{k}")
        plt.ylabel("Center accuracy")
        plt.title(f"k={k}")
        plt.grid(alpha=0.3)

    plt.suptitle(f"Neighbor CL vs center correctness ({split})")
    plt.tight_layout()
    plt.savefig(out_dir / f"neighbor_cl_vs_center_accuracy_{split}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

def plot_neighbor_pxap_vs_center_pxap(augmented_metrics, out_dir, split, ks=(3, 5, 7)):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(15, 4))

    for i, k in enumerate(ks, start=1):
        x_vals = []
        y_vals = []

        for m in augmented_metrics:
            if m["true_class"] != 1:
                continue

            x = m.get(f"neighbor_mean_pxap_{k}", float("nan"))
            y = m.get("pxap", None)

            if y is not None and not np.isnan(x):
                x_vals.append(x)
                y_vals.append(y)

        x_vals = np.array(x_vals, dtype=np.float32)
        y_vals = np.array(y_vals, dtype=np.float32)

        plt.subplot(1, len(ks), i)
        if len(x_vals) > 0:
            plt.scatter(x_vals, y_vals, alpha=0.6)
            rho, pval = spearmanr(x_vals, y_vals)
            plt.title(f"k={k} | rho={rho:.3f}")
        else:
            plt.title(f"k={k} | no data")

        plt.xlabel(f"neighbor_mean_pxap_{k}")
        plt.ylabel("center pxap")
        plt.grid(alpha=0.3)

    plt.suptitle(f"Neighbor PxAP vs center PxAP ({split})")
    plt.tight_layout()
    plt.savefig(out_dir / f"neighbor_pxap_vs_center_pxap_{split}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

def compute_random_neighbor_stats(per_image_metrics, ks=(3, 5, 7), seed=0):
    rng = np.random.default_rng(seed)
    data = enrich_metrics_with_spatial_info(per_image_metrics)

    by_slide = defaultdict(list)
    for idx, item in enumerate(data):
        by_slide[item["slide_id"]].append(idx)

    for item in data:
        for k in ks:
            item[f"random_neighbor_mean_cl_{k}"] = float("nan")
            item[f"random_neighbor_mean_pxap_{k}"] = float("nan")

    for slide_id, indices in by_slide.items():
        if len(indices) < 2:
            continue

        for global_i in indices:
            candidates = [j for j in indices if j != global_i]

            for k in ks:
                kk = min(k, len(candidates))
                if kk <= 0:
                    continue

                sampled = rng.choice(candidates, size=kk, replace=False)

                cl_vals = np.array(
                    [1.0 if data[j]["is_correct"] else 0.0 for j in sampled],
                    dtype=np.float32
                )
                data[global_i][f"random_neighbor_mean_cl_{k}"] = float(cl_vals.mean())

                pxap_vals = [data[j]["pxap"] for j in sampled if data[j]["pxap"] is not None]
                if len(pxap_vals) > 0:
                    data[global_i][f"random_neighbor_mean_pxap_{k}"] = float(np.mean(pxap_vals))

    return data

def evaluate_neighbor_predictiveness(augmented_metrics, out_dir, split, ks=(3, 5, 7)):
    rows = []

    for k in ks:
        # CL -> center correctness
        x_cl = []
        y_cl = []

        # PxAP -> center PxAP
        x_pxap = []
        y_pxap = []

        for m in augmented_metrics:
            x1 = m.get(f"neighbor_mean_cl_{k}", float("nan"))
            if not np.isnan(x1):
                x_cl.append(x1)
                y_cl.append(1 if m["is_correct"] else 0)

            if m["true_class"] == 1:
                x2 = m.get(f"neighbor_mean_pxap_{k}", float("nan"))
                y2 = m.get("pxap", None)
                if y2 is not None and not np.isnan(x2):
                    x_pxap.append(x2)
                    y_pxap.append(y2)

        x_cl = np.array(x_cl, dtype=np.float32)
        y_cl = np.array(y_cl, dtype=np.int32)

        x_pxap = np.array(x_pxap, dtype=np.float32)
        y_pxap = np.array(y_pxap, dtype=np.float32)

        auc = float("nan")
        ap = float("nan")
        if len(np.unique(y_cl)) == 2:
            auc = roc_auc_score(y_cl, x_cl)
            ap = average_precision_score(y_cl, x_cl)

        rho_pxap = float("nan")
        p_pxap = float("nan")
        if len(x_pxap) > 1:
            rho_pxap, p_pxap = spearmanr(x_pxap, y_pxap)

        rows.append({
            "k": k,
            "auc_neighbor_cl_to_center_correct": auc,
            "ap_neighbor_cl_to_center_correct": ap,
            "spearman_neighbor_pxap_to_center_pxap": rho_pxap,
            "spearman_pvalue_pxap": p_pxap,
            "n_cl": int(len(x_cl)),
            "n_pxap": int(len(x_pxap)),
        })

    out_path = Path(out_dir) / f"neighbor_predictiveness_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved: {out_path}")
    return rows

def plot_neighbor_std_cl_by_center_correctness(augmented_metrics, out_dir, split, ks=(3, 5, 7)):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(15, 4))
    for i, k in enumerate(ks, start=1):
        wrong_vals = []
        correct_vals = []

        for m in augmented_metrics:
            x = m.get(f"neighbor_std_cl_{k}", float("nan"))
            if np.isnan(x):
                continue
            if m["is_correct"]:
                correct_vals.append(x)
            else:
                wrong_vals.append(x)

        plt.subplot(1, len(ks), i)
        plt.boxplot([wrong_vals, correct_vals], labels=["wrong", "correct"])
        plt.ylabel(f"neighbor_std_cl_{k}")
        plt.title(f"k={k}")
        plt.grid(alpha=0.3)

    plt.suptitle(f"Neighbor CL heterogeneity vs center correctness ({split})")
    plt.tight_layout()
    plt.savefig(out_dir / f"neighbor_std_cl_by_center_correctness_{split}.png",
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_classification_vs_entropy(per_image_metrics, out_dir, split, n_bins=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entropies = np.array([m["entropy"] for m in per_image_metrics], dtype=np.float32)
    correct = np.array([1.0 if m["is_correct"] else 0.0 for m in per_image_metrics], dtype=np.float32)

    if len(entropies) == 0:
        print("No data for entropy vs classification plot.")
        return

    bin_edges = np.linspace(entropies.min(), entropies.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    accs = []
    counts = []

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (entropies >= left) & (entropies <= right)
        else:
            mask = (entropies >= left) & (entropies < right)

        if mask.sum() > 0:
            accs.append(correct[mask].mean())
            counts.append(int(mask.sum()))
        else:
            accs.append(np.nan)
            counts.append(0)

    plt.figure(figsize=(7, 5))
    plt.plot(bin_centers, accs, marker='o')
    for x, y, c in zip(bin_centers, accs, counts):
        if not np.isnan(y):
            plt.text(x, y, str(c), fontsize=8, ha='center', va='bottom')

    plt.xlabel("Entropy")
    plt.ylabel("Classification accuracy")
    plt.title(f"Classification accuracy vs entropy ({split})")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)

    out_path = out_dir / f"classification_vs_entropy_{split}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_entropy_hist_by_correctness(per_image_metrics, out_dir, split, bins=30):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    correct_entropies = [m["entropy"] for m in per_image_metrics if m["is_correct"]]
    wrong_entropies = [m["entropy"] for m in per_image_metrics if not m["is_correct"]]

    plt.figure(figsize=(7, 5))
    if len(wrong_entropies) > 0:
        plt.hist(wrong_entropies, bins=bins, alpha=0.6, label="Wrong")
    if len(correct_entropies) > 0:
        plt.hist(correct_entropies, bins=bins, alpha=0.6, label="Correct")

    plt.xlabel("Entropy")
    plt.ylabel("Number of images")
    plt.title(f"Entropy distribution by classification correctness ({split})")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = out_dir / f"entropy_hist_by_correctness_{split}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_entropy_boxplot_by_correctness(per_image_metrics, out_dir, split):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    correct_entropies = [m["entropy"] for m in per_image_metrics if m["is_correct"]]
    wrong_entropies = [m["entropy"] for m in per_image_metrics if not m["is_correct"]]

    plt.figure(figsize=(6, 5))
    plt.boxplot([wrong_entropies, correct_entropies], labels=["wrong", "correct"])
    plt.ylabel("Entropy")
    plt.title(f"Entropy by classification correctness ({split})")
    plt.grid(alpha=0.3)

    out_path = out_dir / f"entropy_boxplot_by_correctness_{split}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_neighbor_distance_histograms(
    augmented_metrics,
    out_dir,
    split,
    ks=(1, 3, 5, 7),
    cancer_only=False,
    normalize_by_patch=True,
    bins=50
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = augmented_metrics
    if cancer_only:
        metrics = [m for m in metrics if m["true_class"] == 1]

    if len(metrics) == 0:
        print("No images found for distance histograms.")
        return

    plt.figure(figsize=(5 * len(ks), 4))

    for i, k in enumerate(ks, start=1):
        vals = []

        for m in metrics:
            dists = m.get(f"neighbor_distances_{k}", [])
            if len(dists) == 0:
                continue

            # on peut soit prendre toutes les distances,
            # soit uniquement la moyenne des k voisins
            mean_dist = float(np.mean(dists))

            if normalize_by_patch:
                patch_scale = max(float(m["w"]), float(m["h"]), 1.0)
                mean_dist = mean_dist / patch_scale

            vals.append(mean_dist)

        plt.subplot(1, len(ks), i)

        if len(vals) > 0:
            plt.hist(vals, bins=bins, alpha=0.8)
            plt.axvline(np.median(vals), linestyle="--", linewidth=2, label=f"median={np.median(vals):.2f}")
            plt.axvline(np.mean(vals), linestyle=":", linewidth=2, label=f"mean={np.mean(vals):.2f}")
            plt.legend()

        xlabel = f"Mean distance to {k} neighbors"
        if normalize_by_patch:
            xlabel += " / patch_size"

        plt.xlabel(xlabel)
        plt.ylabel("Number of images")
        plt.title(f"k={k}")
        plt.grid(alpha=0.3)

    prefix = "cancer_only_" if cancer_only else ""
    suffix = "normalized" if normalize_by_patch else "absolute"

    plt.suptitle(f"{prefix}Neighbor distance histograms ({split}, {suffix})")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{prefix}neighbor_distance_histograms_{suffix}_{split}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def get_features(exp_path, sf_uda_source_folder,image_ids_to_draw,image_ids_to_draw_target, checkpoint_type, dataset, cudaid, split, tmp_outd='tmp_outd', parsedargs=None, target_method=None):


    with open(join(exp_path, 'config_obj_final.yaml'), 'r') as fy:
    #with open(join(exp_path, 'OpenImagesSrc-0-deit_sat_base_patch16_224-SAT-GAP-cp_best_classification', 'config_model.yaml'), 'r') as fy:
        args_dict = yaml.load(fy, Loader=IgnoreKeyLoader)
        # args_dict = yaml.safe_load(fy)
        args_dict['model']['freeze_encoder'] = False
        args_dict['pixel_wise_classification'] = False

       
        args_dict['crop_size'] = constants.CROP_SIZE
        args_dict['resize_size'] = constants.RESIZE_SIZE


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
    target_dataset = dataset
    
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
        
        # If evaluating with CLIP-ES but trained with CLIP standard, override method and arch
        if parsedargs.wsol_method == 'CLIPES' and args_dict.get('method') == constants.METHOD_CLIP:
            args_dict['method'] = constants.METHOD_CLIPES
            args_dict['model']['arch'] = constants.CLIPESCLASSIFIER  # Change architecture to CLIP-ES
            args_dict['task'] = constants.STD_CL
            
            # Extract encoder name from config (could be conch-ViT-B-16 or similar)
            original_encoder = args_dict['model'].get('encoder_name', 'ViT-B/16')
            # Set CLIP-ES model name based on original encoder
            if 'conch' in original_encoder.lower():
                args_dict['clipes_model_name'] = constants.CLIPDISTILL_TXTENC_CONCH_VIT_B_16
                args_dict['clipes_model_path'] = None  # Will use default CONCH checkpoint path
            else:
                args_dict['clipes_model_name'] = 'ViT-B/16'
                args_dict['clipes_model_path'] = None
            args_dict['clipes_freeze'] = True
            # Reuse the exact prompt the model was trained with so text features stay aligned
            args_dict['clipes_prompt_template'] = args_dict.get(
                'clip_prompt_template', "a histopathology image of {}")
            args_dict['clipes_bg_prompt'] = "a histopathology image of background tissue"
            print(f"[EVAL] Switching from METHOD_CLIP to METHOD_CLIPES for evaluation")
            print(f"[EVAL] Architecture: {args_dict['model']['arch']}")
            print(f"[EVAL] CLIP-ES model: {args_dict.get('clipes_model_name')}")
        
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

        args_dict['ttda'] = False
        args = Dict2Obj(args_dict)
        args.outd = tmp_outd
        args.distributed = False
        args.eval_checkpoint_type = checkpoint_type
        args.model['folder_pre_trained_cl'] = None

    args.sf_uda = False

    model = get_model(args)[0]

    print(f'Loading model for {method_name}-{encoder_name} from {path_cl}')
    # if "tscam" in encoder_name:
    #     model_tscam = torch.load(join(path_cl, 'model.pt'),map_location=get_cpu_device())

    #     model.load_state_dict(model_tscam, strict=True)
    # else:
    #     encoder_w = torch.load(join(path_cl, 'encoder.pt'),
    #                         map_location=get_cpu_device())
    #     model.encoder.super_load_state_dict(encoder_w, strict=True)

    #     header_w = torch.load(join(path_cl, 'classification_head.pt'),
    #                         map_location=get_cpu_device())
    #     model.classification_head.load_state_dict(header_w, strict=True)

    #     if method_name == constants.METHOD_Pixel:
    #         header_p = torch.load(join(path_cl, 'pixel_wise_classification_head.pt'),
    #                         map_location=get_cpu_device())
    #         model.pixel_wise_classification_head.load_state_dict(header_p, strict=True)


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

    # with torch.no_grad():
    #     anchors = model.pixel_wise_classification_head.conv4.weight

    basic_config = config.get_config(ds=target_dataset, fold=args.fold, magnification=args.magnification)

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
    #################################################################################### parsedargs.fold_dataset
    DLLogger.flush()
    
    metadata_root = join(constants.RELATIVE_META_ROOT, parsedargs.target_dataset, f"fold-{parsedargs.fold_target_dataset}")
    #read sys var DATASETSH
    args_dict['data_root'] = os.path.join(os.environ['DATASETSH'], 'datasets')
    target_domain_data_paths = config.configure_data_paths(args_dict, parsedargs.target_dataset)

    metadata_root_CAME = join('./folds/wsol-done-right-splits', 'CAMELYON512', f"fold-{parsedargs.fold_target_dataset}")
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
            eval_batch_size = 32
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
    

    cam_performance = cam_computer.compute_and_evaluate_cams()
    if str(parsedargs.wsol_method).upper() == "TEDLOC":
        classification_acc = _compute_accuracy_CLIPDISTILL_TXTENC(
            args, model, loaders[parsedargs.split]
        )
    else:
        classification_acc = _compute_accuracy(
            args, model, loaders[parsedargs.split], num_classes=args.num_classes
        )[0]
    
    
   
    print(f"Classification Accuracy on {parsedargs.split} set: {classification_acc:.2f}%")
    print(f"CAM Performance PxAP on {parsedargs.split} set: {cam_performance:.2f}")

    output_root = (
        parsedargs.output_dir
        if parsedargs.output_dir is not None
        else args.outd
    )

    per_image_metrics_dir = (
        Path(output_root)
        / f"{parsedargs.source_dataset}_fold{parsedargs.fold_source_dataset}"
        / f"{args_dict['method']}"
        / f"{parsedargs.target_dataset}_fold{parsedargs.fold_target_dataset}"
    )

    per_image_metrics_dir.mkdir(parents=True, exist_ok=True)

    per_image_metrics_path = (
        per_image_metrics_dir
        / f"results_metrics_{parsedargs.split}.json"
    )

    results = {
        "classification_accuracy": float(classification_acc),
        "pxap": float(cam_performance),
    }

    with open(per_image_metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    results_path = per_image_metrics_dir / f"results_{parsedargs.split}.txt"

    with open(results_path, "w") as f:
        f.write(f"Classification Accuracy: {classification_acc:.2f}%\n")
        f.write(f"CAM Performance PxAP: {cam_performance:.2f}\n")

    print(f"Results saved to: {results_path}")


    print(f"Saved results to {per_image_metrics_path}")

    return overlay_images, input_images, method_name, gt_masks

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
    parser.add_argument("--fold_source_dataset", type=int, default=0, help="fold.")
    parser.add_argument("--sfda_method", type=str, default=0, help="fold.")
    parser.add_argument("--wsol_method", type=str, default='wsol', help="fold.")

    parser.add_argument("--target_dataset", type=str, default=None, help="Source dataset")
    parser.add_argument("--fold_target_dataset", type=int, default=0, help="fold.")
    parser.add_argument("--output_dir", type=str, default=None, help="Source dataset")
    parser.add_argument("--plot_umap_clip_embeddings", type=str2bool, default=False)
    parser.add_argument("--umap_max_images_per_class", type=int, default=500)
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.2)
    parser.add_argument("--umap_metric", type=str, default="cosine")
    parser.add_argument("--plot_tedloc_pixel_entropy", type=str2bool, default=False)
    parser.add_argument("--tedloc_pixel_entropy_bins", type=int, default=80)
    
    
    import torch

    path = "/export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/class_anchors_cache/hist_GLAS_anchor_cache_q_by_qr_decom.pt"

    data = torch.load(path, map_location="cpu")

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
            overlay_images, input_images, method_name, gt_masks = get_features(exp_path=exp_path, sf_uda_source_folder=parsedargs.path_pre_trained_source,image_ids_to_draw=parsedargs.image_ids_to_draw,image_ids_to_draw_target=parsedargs.image_ids_to_draw_target, checkpoint_type=checkpoint_type, dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs, target_method=target_method)

if __name__ == '__main__':
    fast_eval()
