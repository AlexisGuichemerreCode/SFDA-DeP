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

from math import ceil
from scipy.optimize import linear_sum_assignment

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

import umap


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


from dlib.datasets.wsol_loader_natural import get_data_loader_natural

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


def to_cuda(x):
    return x.cuda()

def to_onehot(label, num_classes):
    identity = to_cuda(torch.eye(num_classes))
    onehot = torch.index_select(identity, 0, label)
    return onehot


class DIST(object):
    def __init__(self, dist_type='cos'):
        self.dist_type = dist_type

    def get_dist(self, pointA, pointB, cross=False):
        return getattr(self, self.dist_type)(
            pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)
        if not cross:
            return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
        else:
            # NA = pointA.size(0)
            # NB = pointB.size(0)
            assert (pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))
        



class Clustering(object):
    def __init__(self, eps, feat_key, model_trg, max_len=1000, dist_type='cos'):
        self.eps = eps
        self.Dist = DIST(dist_type)
        self.samples = {}
        self.path2label = {}
        self.center_change = None
        self.stop = False
        self.feat_key = feat_key
        self.max_len = max_len
        self.model = model_trg

    def set_init_centers(self, init_centers):
        self.centers = init_centers
        self.init_centers = init_centers
        self.num_classes = self.centers.size(0)

    def clustering_stop(self, centers):
        if centers is None:
            self.stop = False
        else:
            dist = self.Dist.get_dist(centers, self.centers)
            dist = torch.mean(dist, dim=0)
            #print('dist %.4f' % dist.item())
            self.stop = dist.item() < self.eps

    def assign_labels(self, feats):
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)
        return dists, labels
    


    def align_centers(self):
        cost = self.Dist.get_dist(self.centers, self.init_centers, cross=True)
        cost = cost.data.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)
        return col_ind

    def collect_samples(self, net, loader):
        data_feat, data_gt, data_paths, data_truth = [], [], [], []
        #layer_to_extract_feat = 'classification_head.avgpool'
        #self.feature_extractor = FeatureExtractor_for_source_code(model=net, layers=[layer_to_extract_feat])
        for sample in iter(loader):
            #data = sample['Img'].cuda()
            data = sample[0].cuda()
            data_truth += sample[1]
            #data_paths += sample['Path']
            data_paths += sample[3]
            # if 'Label' in sample.keys():
            #     data_gt += [to_cuda(sample['Label'])]

            # output = net.forward(data)
            # feature = output[self.feat_key].data
            #feature = net.forward(data, get_feature=True)[-1].data
            #feature = self.feature_extractor(data)[layer_to_extract_feat].squeeze(2).squeeze(2)
            out = self.model(data)
            feature = self.model.lin_ft
            data_feat += [feature]

        self.samples['data'] = data_paths
        # self.samples['gt'] = torch.cat(data_gt, dim=0) \
        #     if len(data_gt) > 0 else None
        self.samples['gt'] = None
        self.samples['feature'] = torch.cat(data_feat, dim=0)
        self.samples['data_truth_label'] = torch.tensor([t.item() for t in data_truth])

    def feature_clustering(self, net, loader):
        centers = None
        self.stop = False

        self.collect_samples(net, loader)
        feature = self.samples['feature']

        refs = to_cuda(torch.LongTensor(range(self.num_classes)).unsqueeze(1))
        num_samples = feature.size(0)
        num_split = ceil(1.0 * num_samples / self.max_len)

        while True:
            self.clustering_stop(centers)
            if centers is not None:
                self.centers = centers
            if self.stop:
                break

            centers = 0
            count = 0

            start = 0
            for N in range(num_split):
                cur_len = min(self.max_len, num_samples - start)
                cur_feature = feature.narrow(0, start, cur_len)
                dist2center, labels = self.assign_labels(cur_feature)
                labels_onehot = to_onehot(labels, self.num_classes)
                count += torch.sum(labels_onehot, dim=0)
                labels = labels.unsqueeze(0)
                mask = (labels == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
                reshaped_feature = cur_feature.unsqueeze(0)
                # update centers
                centers += torch.sum(reshaped_feature * mask, dim=1)
                start += cur_len

            mask = (count.unsqueeze(1) > 0).type(torch.cuda.FloatTensor)
            centers = mask * centers + (1 - mask) * self.init_centers

        dist2center, labels = [], []
        start = 0
        count = 0
        for N in range(num_split):
            cur_len = min(self.max_len, num_samples - start)
            cur_feature = feature.narrow(0, start, cur_len)
            cur_dist2center, cur_labels = self.assign_labels(cur_feature)

            labels_onehot = to_onehot(cur_labels, self.num_classes)
            count += torch.sum(labels_onehot, dim=0)

            dist2center += [cur_dist2center]
            labels += [cur_labels]
            start += cur_len

        self.samples['label'] = torch.cat(labels, dim=0)
        self.samples['dist2center'] = torch.cat(dist2center, dim=0)

        cluster2label = self.align_centers()
        # reorder the centers
        self.centers = self.centers[cluster2label, :]
        # re-label the data according to the index
        num_samples = len(self.samples['feature'])
        for k in range(num_samples):
            self.samples['label'][k] = cluster2label[self.samples['label'][k]].item()


        acc = (self.samples['label'].detach().cpu() == self.samples['data_truth_label']).float().mean() * 100.
        msg = f"SFDE - ACC pseudo-label image-class -- : {acc} %"
        DLLogger.log(fmsg(msg))

        self.center_change = torch.mean(self.Dist.get_dist(self.centers,self.init_centers))

        for i in range(num_samples):
            self.path2label[self.samples['data'][i]] = self.samples['label'][i].item()

        del self.samples['feature']





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

def generate_pixel_features(model, loader, dataset_name, cam_computer, split, device):
    """
    Generate pixel features from the model for a given dataset and split.
    Args:
        model (torch.nn.Module): The model to use for feature extraction.
        loader (DataLoader): DataLoader for the dataset.
        dataset_name (str): Name of the dataset.
        split (str): Split of the dataset (e.g., 'train', 'val').
        device (torch.device): Device to run the model on.
    Returns:
        tuple: Foreground and background features.
    """
    fg_features = []
    bg_features = []

    for batch_idx, (images, targets, _, index, _, _, _, _) in tqdm(
            enumerate(loader[split]), ncols=constants.NCOLS,
            total=len(loader[split])):

        images = images.to(device)
        targets = targets.to(device)
        
        for image, target, image_id in zip(images, targets, index):

            gt_mask = get_mask(f'/export/livia/home/vision/Aguichemerre/datasets/{dataset_name}',
                           cam_computer.evaluator.mask_paths[image_id],
                           cam_computer.evaluator.ignore_paths[image_id])
            
            gt_mask_tensor = torch.tensor(gt_mask, dtype=torch.float32).to(device)
            gt_resize = F.interpolate(gt_mask_tensor.unsqueeze(0).unsqueeze(0), size=(28, 28),
                                        mode='bilinear', align_corners=False).squeeze(0).squeeze(0).cpu().numpy()

            with torch.set_grad_enabled(cam_computer.req_grad):
                cam, cl_logits = cam_computer.get_cam_one_sample(
                    image=image.unsqueeze(0), target=target.item())
                
            gt_resize = gt_resize > 0.5
            cam = cam > 0.5

            same_mask = gt_resize == cam

            gt_resize_int = gt_resize.to(torch.int)
            result = torch.where(same_mask, gt_resize_int, torch.full_like(gt_resize_int, -255))

            print("a")


    return fg_features, bg_features

def generate_synthetic_features(gt_bin_batch, pixel_anchor_weights, noise_std=0.05):
    """
    Generate synthetic features based on the ground truth binary mask and pixel anchor weights.
    Args:
        gt_bin_batch (torch.Tensor): Ground truth binary mask of shape (B, 1, H, W).
        pixel_anchor_weights (torch.Tensor): Pixel anchor weights of shape (C, 1, 1).
        noise_std (float): Standard deviation of the noise to be added.
    Returns:
        torch.Tensor: Generated features of shape (B, C, H, W).
    """

    B, _, H, W = gt_bin_batch.shape
    C = pixel_anchor_weights.shape[1]  # C = 2048
    device = gt_bin_batch.device

   
    fg_weight = pixel_anchor_weights[1].view(1, C, 1, 1)
    bg_weight = pixel_anchor_weights[0].view(1, C, 1, 1)

    # Expand the weights to match the batch size and spatial dimensions
    fg_feature = fg_weight.expand(B, C, H, W)
    bg_feature = bg_weight.expand(B, C, H, W)

    # Add noise to the features
    fg_noise = torch.randn_like(fg_feature) * noise_std
    bg_noise = torch.randn_like(bg_feature) * noise_std

    features = torch.where(gt_bin_batch == 1, fg_feature + fg_noise, bg_feature + bg_noise)
    return features 


def evaluate_noise_impact(model, features, gt_bin_batch, pixel_classifier, targets, steps=100):
 
    B, C, H, W = features.shape
    device = features.device
    results = []


    gt_bin_flat = gt_bin_batch.view(B, -1)

    for error_rate in range(1, steps + 1):
 
        corrupted_mask = gt_bin_flat.clone()

        for b in range(B):
            n_pixels = H * W
            n_errors = int(n_pixels * (error_rate / 100))
            perm = torch.randperm(n_pixels, device=device)[:n_errors]
            corrupted_mask[b, perm] = ~corrupted_mask[b, perm]  

        corrupted_mask = corrupted_mask.view(B, 1, H, W)


        fg_weight = pixel_classifier[1].view(1, C, 1, 1)
        bg_weight = pixel_classifier[0].view(1, C, 1, 1)
        fg_noise = torch.randn_like(features) * 0.05
        bg_noise = torch.randn_like(features) * 0.05
        modified_features = torch.where(corrupted_mask == 1, fg_weight + fg_noise, bg_weight + bg_noise)


        with torch.no_grad():
            logits = model.classification_head(modified_features)  # [B, 2, H, W]
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1) 

        
        acc = (preds == targets).float().mean().item()
        results.append((error_rate, acc))

    return results



def pixel_feature_errors(model,loader,cam_computer, dataset_name, split, device):

    fg_features, bg_features = generate_pixel_features(model=model, loader=loader, dataset_name=dataset_name, cam_computer=cam_computer, split=split, device=device)


    for batch_idx, (images, targets, p_glabel, index, raw_imgs, std_cams, _, views) in tqdm(
        enumerate(loader[split]), ncols=constants.NCOLS,
        total=len(loader[split])):

        with torch.no_grad():
            pixel_anchor_weights = model.pixel_wise_classification_head.conv4.weight

        images = images.to(device)
        targets = targets.to(device)

        h, w = 28, 28
        gt_bin_list = []
        all_results = []

        # Per-image
        for i, (label, image_id) in enumerate(zip(targets, index)):
            gt_bin = get_resized_gt_mask(
                cam_computer.evaluator.mask_paths[image_id],
                cam_computer.evaluator.ignore_paths[image_id],
                dataset_name,
                label.item(),
                size=(h, w),
                device=device
            )  # [H, W] ou [1, H, W]

            if gt_bin.dim() == 2:
                gt_bin = gt_bin.unsqueeze(0)  # [1, H, W]

            gt_bin_list.append(gt_bin)

        gt_bin_batch = torch.stack(gt_bin_list, dim=0)
        features = generate_synthetic_features(gt_bin_batch, pixel_anchor_weights)
        results = evaluate_noise_impact(model, features, gt_bin_batch, pixel_anchor_weights, targets)

        all_results.append(results)

    error_rates, accuracies = zip(*results)

    plt.figure(figsize=(8, 5))
    plt.plot(error_rates, accuracies, marker='o')
    plt.xlabel('Pourcentage of errors')
    plt.ylabel('Accuracy of image classifier')
    plt.title('Impact of errors on classification')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("error_vs_accuracy.png", dpi=300)
    plt.show()

    return 0


def plot_weights_model(model, support_background, out_dir, title_prefix=""):
    with torch.no_grad():
        pixel_anchor_weights = model.pixel_wise_classification_head.conv4.weight

    if support_background:
        image_anchor_weights = model.classification_head.fc.weight[1:] 
    else:   
        image_anchor_weights = model.classification_head.fc.weight

    pixel_anchor_weights = pixel_anchor_weights.view(pixel_anchor_weights.size(0), -1)
    image_anchor_weights = image_anchor_weights.view(image_anchor_weights.size(0), -1)

    pixel_anchor_weights = pixel_anchor_weights.detach().cpu().numpy()
    image_anchor_weights = image_anchor_weights.detach().cpu().numpy()

    all_weights = np.concatenate([image_anchor_weights,pixel_anchor_weights])
    tsne = TSNE(n_components=2, perplexity=2.0)

    embedded_weights = tsne.fit_transform(all_weights)

    embedded_img_weights = embedded_weights[:image_anchor_weights.shape[0]]
    embedded_pxl_weights = embedded_weights[image_anchor_weights.shape[0]:image_anchor_weights.shape[0]+pixel_anchor_weights.shape[0]]

    plt.scatter(embedded_img_weights[:, 0], embedded_img_weights[:, 1], color='blue', label='Img weights')
    plt.scatter(embedded_pxl_weights[:, 0], embedded_pxl_weights[:, 1], color='red', label='Pxl weights')

    plt.title(f"test ", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    # Sauvegarder l'image
    plt.savefig('test_weights_tsne.png')
    plt.close()

    return 0



def compute_energy_distributions_cub(model, loader, dataset_name, energy_fn, split, device):

    energy_data = {
    'images': [],
    'pixels': [],
    'per_class': defaultdict(list),
    'foreground': [],
    'background': []}

    for batch_idx, (images, targets, index, _, _) in tqdm(
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


    # Concatenation
    energy_data['images'] = torch.cat(energy_data['images']).numpy()
    energy_data['pixels'] = torch.cat(energy_data['pixels']).view(-1).numpy()

    return energy_data



def compute_energy_distributions(model, loader, cam_computer, dataset_name, energy_fn, split, device,args=None, metadata_root=None, cam_performance = False):

    energy_data = {
    'images': [],
    'pixels': [],
    'per_class': defaultdict(list),
    'foreground': [],
    'background': [],
    'probs_images': [],
    'pred_images': [],
    'label_images': [],
    'cam_performance': [],
    'min_logits_img': [],
    'max_logits_img': [],
    'min_logits_pxs': [],
    'max_logits_pxs': [],
    'lin_ft': [],
    'img_ft': [],
    'entropy_px': [],
    'metrics_px_bg': [],
    'metrics_px_fg': [],
    'n_fg': [],
    'n_bg': [],
    'pct_fg': [],
    }

    model.eval()
    for batch_idx, (images, targets, p_glabel, index, raw_imgs, std_cams, _, views) in tqdm(
        enumerate(loader[split]), ncols=constants.NCOLS,
        total=len(loader[split])):

        images = images.to(device)
        targets = targets.to(device)
        image_size = images.shape[2:]

        # Per-image
        with torch.no_grad():
            lgt_imgs = model(images)
            px_lin_ft = model.encoder_last_features
            lgt_pxs = model.pixel_wise_classification_head(px_lin_ft)[0]

            probs_img = F.softmax(lgt_imgs, dim=1)
            preds = probs_img.argmax(dim=1) 

            probs_px = torch.softmax(lgt_pxs, dim=1)   
            entropy_px = -(probs_px * probs_px.log()).sum(dim=1)
            mean_H_img = entropy_px.mean(dim=(1, 2))

            pred_px  = probs_px.argmax(1)                          # [B, H, W] 0=normal,1=cancer
            mask_c   = (pred_px == 1)
            mask_n   = (pred_px == 0)

            area_img = mask_c.shape[1] * mask_c.shape[2]
            n_fg = mask_c.sum(dim=(1, 2))                  # [B]  (tensor int64)

            # 3. Compter les pixels prédits "normal" (background)
            n_bg = mask_n.sum(dim=(1, 2))                  # [B]

            # 4. (option) convertir en pourcentage
            pct_fg = n_fg.float() / area_img              # [B]  entre 0 – 1
            pct_bg = n_bg.float() / area_img

            energy_data["n_fg"].append(n_fg.cpu())        # liste de tensors [B]
            energy_data["n_bg"].append(n_bg.cpu())
            energy_data["pct_fg"].append(pct_fg.cpu())

            eps = 1e-6
            H_c = (entropy_px * mask_c).sum((1,2)) / (mask_c.sum((1,2)).float() + eps)  # [B]
            H_n = (entropy_px * mask_n).sum((1,2)) / (mask_n.sum((1,2)).float() + eps) 

            energy_data['metrics_px_bg'].append(H_n.detach().cpu())
            energy_data['metrics_px_fg'].append(H_c.detach().cpu())

            energy_data['entropy_px'].append(mean_H_img.detach().cpu())

            energy_data['lin_ft'].append(px_lin_ft.flatten(start_dim=1).detach().cpu())
            energy_data['img_ft'].append(model.lin_ft.flatten(start_dim=1).detach().cpu())


            energy_data['probs_images'].append(probs_img.max(dim=1).values.detach().cpu())
            energy_data['label_images'].append(targets.detach().cpu())
            energy_data['pred_images'].append(preds.detach().cpu())

            energy_images = energy_fn(lgt_imgs)
            energy_pixels = energy_fn(lgt_pxs)

            max_vals_img, _ = lgt_imgs.max(dim=1) 
            min_vals_img, _ = lgt_imgs.min(dim=1)

            max_vals_pxs, _ = lgt_pxs.max(dim=1) 
            min_vals_pxs, _ = lgt_pxs.min(dim=1)

            energy_data['images'].append(energy_images.cpu())
            energy_data['pixels'].append(energy_pixels.flatten(start_dim=1).cpu())

            energy_data['max_logits_img'].append(max_vals_img.cpu())
            energy_data['min_logits_img'].append(min_vals_img.cpu())

            energy_data['max_logits_pxs'].append(max_vals_pxs.flatten(start_dim=1).cpu())
            energy_data['min_logits_pxs'].append(min_vals_pxs.flatten(start_dim=1).cpu())



            for energy, label in zip(energy_images.cpu(), targets.cpu()):
                energy_data['per_class'][int(label)].append(energy.item())

    # Concatenation
    energy_data['images'] = torch.cat(energy_data['images']).numpy()
    energy_data['pixels'] = torch.cat(energy_data['pixels']).view(-1).numpy()
    #energy_data['foreground'] = torch.cat(energy_data['foreground']).numpy()
    #energy_data['background'] = torch.cat(energy_data['background']).numpy()
    energy_data['probs_images'] = torch.cat(energy_data['probs_images']).numpy()
    energy_data['label_images'] = torch.cat(energy_data['label_images']).numpy()
    energy_data['pred_images'] = torch.cat(energy_data['pred_images']).numpy()
    energy_data['max_logits_img'] = torch.cat(energy_data['max_logits_img']).view(-1).numpy()
    energy_data['min_logits_img'] = torch.cat(energy_data['min_logits_img']).view(-1).numpy()
    energy_data['max_logits_pxs'] = torch.cat(energy_data['max_logits_pxs']).view(-1).numpy()
    energy_data['min_logits_pxs'] = torch.cat(energy_data['min_logits_pxs']).view(-1).numpy()
    energy_data['lin_ft'] = torch.cat(energy_data['lin_ft'], dim=0).cpu().numpy()
    energy_data['img_ft'] = torch.cat(energy_data['img_ft'], dim=0).cpu().numpy()
    energy_data['entropy_px'] = torch.cat(energy_data['entropy_px']).cpu().numpy()
    energy_data['metrics_px_bg'] = torch.cat(energy_data['metrics_px_bg']).cpu().numpy()
    energy_data['metrics_px_fg'] = torch.cat(energy_data['metrics_px_fg']).cpu().numpy()
    energy_data["n_fg"]   = torch.cat(energy_data["n_fg"]).numpy()      # [N]
    energy_data["n_bg"]   = torch.cat(energy_data["n_bg"]).numpy()
    energy_data["pct_fg"] = torch.cat(energy_data["pct_fg"]).numpy()

    return energy_data


def plot_energy_for_source_images(source_vals, out_dir, title, xlim=None, label_src="Source", label_tgt="Target", source_dataset=None, target_dataset = None, source_model_name = None, target_model_name = None, external_pixel_classifier=None):

    output_dir = os.path.join('visualization', 'Energy_results', source_dataset)
    os.makedirs(output_dir, exist_ok=True)



    if external_pixel_classifier is None:
        figure_name = f"hist_pixels_source_{source_dataset}_with_{source_model_name}_on_target_{target_dataset}_with_{target_model_name}.png"
    else:
        figure_name = f"hist_pixels_source_{source_dataset}_with_{source_model_name}_on_target_{target_dataset}_with_{target_model_name}_with_external_px_classifier.png"

    
    plt.figure(figsize=(8, 6))
    plt.hist(source_vals, bins=100, alpha=0.5, density=True,
             label=label_src, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Energy")
    plt.ylabel("Density")
    plt.legend(loc="upper right", fontsize=12)

    if xlim:
        plt.xlim(*xlim)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, figure_name), dpi=300)
    plt.close()


def plot_logits(source_pxs_logits, target_pxs_logits, out_dir, title, type = None, xlim=None, label_src="Source", label_tgt="Target", source_dataset=None, target_dataset = None, source_model_name = None, target_model_name = None, external_pixel_classifier=None):

    output_dir = os.path.join('visualization', 'Logits_results', source_dataset)
    os.makedirs(output_dir, exist_ok=True)



    if external_pixel_classifier is None:
        figure_name = f"hist_{type}_logits_{source_dataset}_with_{source_model_name}_on_target_{target_dataset}_with_{target_model_name}.png"
    else:
        figure_name = f"hist_{type}_logits_source_{source_dataset}_with_{source_model_name}_on_target_{target_dataset}_with_{target_model_name}_with_external_px_classifier.png"

    
    plt.figure(figsize=(8, 6))
    plt.hist(source_pxs_logits, bins=100, alpha=0.5, density=True,
             label=label_src, color='blue', edgecolor='black')
    
    plt.hist(target_pxs_logits, bins=100, alpha=0.5, density=True,
             label=label_src, color='red', edgecolor='black')
    plt.title(title)
    plt.xlabel("Logits")
    plt.ylabel("Density")
    plt.legend(loc="upper right", fontsize=12)

    if xlim:
        plt.xlim(*xlim)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, figure_name), dpi=300)
    plt.close()


def plot_histograms(data_list, labels, colors=None, bins=30, 
                    xlabel='Value', ylabel='Count',
                    title='Histogram', save_path=None, show=True, alpha=0.6):
    """
    Trace plusieurs histogrammes sur une même figure.

    Args:
        data_list (list of arrays): listes des tableaux de données à tracer.
        labels (list of str): noms des courbes pour la légende.
        colors (list of str, optional): couleurs à utiliser (doit être de même longueur que data_list).
        bins (int or sequence): nombre de bins ou les bins exacts.
        xlabel (str): nom de l'axe x.
        ylabel (str): nom de l'axe y.
        title (str): titre de la figure.
        save_path (str or None): chemin pour sauvegarder la figure. Aucun fichier n'est enregistré si None.
        show (bool): si True, affiche la figure.
        alpha (float): transparence des courbes (entre 0 et 1).
    """
    plt.figure(figsize=(8, 4))
    
    for i, data in enumerate(data_list):
        color = colors[i] if colors else None
        plt.hist(data, bins=bins, alpha=alpha, label=labels[i], color=color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close()



def plot_hist_energy_based_on_target_image_acc(target_dict, out_dir, save_path=None, title_prefix=None,target_dataset=None, model = None):
    os.makedirs(out_dir, exist_ok=True)

    img_classes = target_dict['label_images']
    img_predict = target_dict['pred_images']
    img_confidence = target_dict['probs_images']
    accuracy = (img_classes == img_predict)

    img_energy_correct = target_dict['images'][accuracy == 1]
    img_energy_incorrect = target_dict['images'][accuracy == 0]

    img_confidence_correct = img_confidence[accuracy == 1]
    img_confidence_incorrect = img_confidence[accuracy == 0]



    mask_class_1 = (img_classes == 1)
    mask_class_1_correct = mask_class_1 & (accuracy == 1)
    mask_class_1_incorrect = mask_class_1 & (accuracy == 0)

    mask_class_0 = (img_classes == 0)
    mask_class_0_correct = mask_class_0 & (accuracy == 1)
    mask_class_0_incorrect = mask_class_0 & (accuracy == 0)


    img_energy_class1_correct = target_dict['images'][mask_class_1_correct]
    img_energy_class1_incorrect = target_dict['images'][mask_class_1_incorrect]
    img_confidence_1_correct = img_confidence[mask_class_1_correct]
    img_confidence_1_incorrect = img_confidence[mask_class_1_incorrect]
    img_ft_1_correct = target_dict['img_ft'][mask_class_1_correct]
    img_ft_1_incorrect = target_dict['img_ft'][mask_class_1_incorrect]
    


    img_energy_class0_correct = target_dict['images'][mask_class_0_correct]
    img_energy_class0_incorrect = target_dict['images'][mask_class_0_incorrect]
    img_confidence_0_correct = img_confidence[mask_class_0_correct]
    img_confidence_0_incorrect = img_confidence[mask_class_0_incorrect]
    img_ft_0_correct = target_dict['img_ft'][mask_class_0_correct]
    img_ft_0_incorrect = target_dict['img_ft'][mask_class_0_incorrect]


    H = target_dict["entropy_px"] 

    H_c_correct = H[mask_class_1_correct] 
    H_n_incorrect      = H[mask_class_0_incorrect]  

    H_c_incorrect = H[mask_class_1_incorrect]
    H_n_correct     = H[mask_class_0_correct]

    H_c = target_dict["metrics_px_fg"]
    H_n = target_dict["metrics_px_bg"]

    H_c_c1_correct = H_c[mask_class_1_correct]
    H_n_c1_correct     = H_n[mask_class_1_correct]

    H_c_c0_incorrect = H_c[mask_class_0_incorrect]
    H_n_c0_incorrect     = H_n[mask_class_0_incorrect]


    H_c_c1_incorrect = H_c[mask_class_1_incorrect]
    H_n_c1_incorrect     = H_n[mask_class_1_incorrect]

    H_c_c0_correct = H_c[mask_class_0_correct]
    H_n_c0_correct     = H_n[mask_class_0_correct]







    out_dir = f"visualization/bcl/target/{target_dataset}/"
    os.makedirs(out_dir, exist_ok=True)

    plot_histograms([H_c_correct, H_n_incorrect],
                 labels=['Cancer correctly classified', 'Normal misclassified'],
                 colors=['green', 'orange'],
                 xlabel='Average entropy of pixels per image',
                 ylabel='Number of images',
                 title='Distribution of pixel entropy \n(cancer correct vs normal misclassified)',
                 save_path= os.path.join(out_dir, 'hist_entropy_tp_vs_fp_cancer_class.png'))


    plot_histograms([H_c_incorrect, H_n_correct],
                 labels=['Cancer misclassified', 'Normal correctly classified'],
                 colors=['green', 'orange'],
                 xlabel='Average entropy of pixels per image',
                 ylabel='Number of images',
                 title='Distribution of pixel entropy \n(cancer misclassified vs normal correct)',
                 save_path= os.path.join(out_dir, 'hist_entropy_tp_vs_fp_normal_class.png'))


    plot_histograms([H_c_c1_correct, H_c_c0_incorrect],
                 labels=['Cancer correctly classified', 'Normal misclassified'],
                 colors=['green', 'orange'],
                 xlabel='Average entropy of pixels (fg) per image',
                 ylabel='Number of images',
                 title='Distribution of (fg) pixel entropy \n(cancer correctly classified vs normal misclassifier)',
                 save_path= os.path.join(out_dir, 'hist_entropy_tp_vs_fp_cancer_class_fg.png'))


    plot_histograms([H_n_c1_correct, H_n_c0_incorrect],
                 labels=['Cancer correctly classified', 'Normal misclassified'],
                 colors=['green', 'orange'],
                 xlabel='Average entropy of pixels (bg) per image',
                 ylabel='Number of images',
                 title='Distribution of (bg) pixel entropy \n(cancer correctly classified vs normal misclassifier)',
                 save_path= os.path.join(out_dir, 'hist_entropy_tp_vs_fp_cancer_class_bg.png'))
    

    plot_histograms([H_c_c1_incorrect, H_c_c0_correct],
                 labels=['Cancer misclassified', 'Normal correctly classified'],
                 colors=['green', 'orange'],
                 xlabel='Average entropy of pixels (fg) per image',
                 ylabel='Number of images',
                 title='Distribution of (fg) pixel entropy \n(cancer misclassified vs normal correctly classified)',
                 save_path= os.path.join(out_dir, 'hist_entropy_tp_vs_fp_normal_class_fg.png'))
    

    plot_histograms([H_n_c1_incorrect, H_n_c0_correct],
                 labels=['Cancer misclassified', 'Normal correctly classified'],
                 colors=['green', 'orange'],
                 xlabel='Average entropy of pixels (bg) per image',
                 ylabel='Number of images',
                 title='Distribution of (bg) pixel entropy \n(cancer misclassified vs normal correctly classified)',
                 save_path= os.path.join(out_dir, 'hist_entropy_tp_vs_fp_normal_class_bg.png'))
    



    

    pct_fg = target_dict["pct_fg"]
    pct_fg_c1_correct = pct_fg[mask_class_1_correct]
    pct_fg_c0_incorrect = pct_fg[mask_class_0_incorrect]

    pct_bg = 1 - pct_fg
    pct_bg_c1_correct = pct_bg[mask_class_1_correct]
    pct_bg_c0_incorrect = pct_bg[mask_class_0_incorrect]


    pct_fg_c1_incorrect = pct_fg[mask_class_1_incorrect]
    pct_fg_c0_correct = pct_fg[mask_class_0_correct]

    pct_bg_c1_incorrect = pct_bg[mask_class_1_incorrect]
    pct_bg_c0_correct = pct_bg[mask_class_0_correct]


    entropy_px = target_dict["entropy_px"]
    mask_bg_90 = pct_bg > 0.9
    mask_bg_60_90 = (pct_bg > 0.6) & (pct_bg <= 0.9)
    mask_bg_00_60 = pct_bg <= 0.6

    mask_fg_90 = pct_fg <= 0.1
    mask_fg_60_90 = (pct_fg > 0.1) & (pct_fg <= 0.3)
    mask_fg_00_60 = pct_fg > 0.3

    ################################# BG #######################################

    mask_cancer_correct_bg_90 = mask_class_1_correct & mask_bg_90
    mask_normal_incorrect_bg_90 = mask_class_0_incorrect & mask_bg_90

    entropy_cancer_correct_bg_90 = entropy_px[mask_cancer_correct_bg_90]
    entropy_normal_incorrect_bg_90 = entropy_px[mask_normal_incorrect_bg_90]

    mask_cancer_correct_bg_60_90 = mask_class_1_correct & mask_bg_60_90
    mask_normal_incorrect_bg_60_90 = mask_class_0_incorrect & mask_bg_60_90

    entropy_cancer_correct_bg_60_90 = entropy_px[mask_cancer_correct_bg_60_90]
    entropy_normal_incorrect_bg_60_90 = entropy_px[mask_normal_incorrect_bg_60_90]

    mask_cancer_correct_bg_00_60 = mask_class_1_correct & mask_bg_00_60
    mask_normal_incorrect_bg_00_60 = mask_class_0_incorrect & mask_bg_00_60

    entropy_cancer_correct_bg_00_60 = entropy_px[mask_cancer_correct_bg_00_60]
    entropy_normal_incorrect_bg_00_60 = entropy_px[mask_normal_incorrect_bg_00_60]


    ##################################### FG #################################

    mask_cancer_correct_fg_90 = mask_class_1_correct & mask_fg_90
    mask_normal_incorrect_fg_90 = mask_class_0_incorrect & mask_fg_90

    entropy_cancer_correct_fg_90 = entropy_px[mask_cancer_correct_fg_90]
    entropy_normal_incorrect_fg_90 = entropy_px[mask_normal_incorrect_fg_90]

    mask_cancer_correct_fg_60_90 = mask_class_1_correct & mask_fg_60_90
    mask_normal_incorrect_fg_60_90 = mask_class_0_incorrect & mask_fg_60_90

    entropy_cancer_correct_fg_60_90 = entropy_px[mask_cancer_correct_fg_60_90]
    entropy_normal_incorrect_fg_60_90 = entropy_px[mask_normal_incorrect_fg_60_90]

    mask_cancer_correct_fg_00_60 = mask_class_1_correct & mask_fg_00_60
    mask_normal_incorrect_fg_00_60 = mask_class_0_incorrect & mask_fg_00_60

    entropy_cancer_correct_fg_00_60 = entropy_px[mask_cancer_correct_fg_00_60]
    entropy_normal_incorrect_fg_00_60 = entropy_px[mask_normal_incorrect_fg_00_60]






    #  BG
    

    plot_histograms([entropy_cancer_correct_bg_90, entropy_normal_incorrect_bg_90],
                 labels=['Cancer correctly classified ', 'Normal misclassified'],
                 colors=['green', 'orange'],
                 xlabel='Average entropy of pixels (bg) per image',
                 ylabel='Number of images',
                 title='Distribution of (bg) pixel entropy \n(cancer correctly classified vs normal misclassified with bg > 90%)',
                 save_path= os.path.join(out_dir, 'hist_entropy_px_bg90_cancer_correct_vs_normal_misclassified.png'))
    
    
    plot_histograms([entropy_cancer_correct_bg_60_90, entropy_normal_incorrect_bg_60_90],
                 labels=['Cancer correctly classified ', 'Normal misclassified'],
                 colors=['green', 'orange'],
                 xlabel='Average entropy of pixels (bg) per image',
                 ylabel='Number of images',
                 title='Distribution of (bg) pixel entropy \n(cancer correctly classified vs normal misclassified with 90% >  bg > 60%)',
                 save_path= os.path.join(out_dir, 'hist_entropy_px_bg_60_90_cancer_correct_vs_normal_misclassified.png'))
    

    plot_histograms([entropy_cancer_correct_bg_00_60, entropy_normal_incorrect_bg_00_60],
                 labels=['Cancer correctly classified ', 'Normal misclassified'],
                 colors=['green', 'orange'],
                 xlabel='Average entropy of pixels (bg) per image',
                 ylabel='Number of images',
                 title='Distribution of (bg) pixel entropy \n(cancer correctly classified vs normal misclassified with 60% >  bg > 00%)',
                 save_path= os.path.join(out_dir, 'hist_entropy_px_bg_00_60_cancer_correct_vs_normal_misclassified.png'))
    



    #  FG


    plot_histograms([entropy_cancer_correct_fg_90, entropy_normal_incorrect_fg_90],
                 labels=['Cancer correctly classified ', 'Normal misclassified'],
                 colors=['green', 'orange'],
                 xlabel='Average entropy of pixels (fg) per image',
                 ylabel='Number of images',
                 title='Distribution of (fg) pixel entropy \n(cancer correctly classified vs normal misclassified with bg > 90%)',
                 save_path= os.path.join(out_dir, 'hist_entropy_px_fg_bg90_cancer_correct_vs_normal_misclassified.png'))
    
    
    plot_histograms([entropy_cancer_correct_fg_60_90, entropy_normal_incorrect_fg_60_90],
                 labels=['Cancer correctly classified ', 'Normal misclassified'],
                 colors=['green', 'orange'],
                 xlabel='Average entropy of pixels (fg) per image',
                 ylabel='Number of images',
                 title='Distribution of (fg) pixel entropy \n(cancer correctly classified vs normal misclassified with 90% >  bg > 60%)',
                 save_path= os.path.join(out_dir, 'hist_entropy_px_fg_bg_60_90_cancer_correct_vs_normal_misclassified.png'))
    

    plot_histograms([entropy_cancer_correct_fg_00_60, entropy_normal_incorrect_fg_00_60],
                 labels=['Cancer correctly classified ', 'Normal misclassified'],
                 colors=['green', 'orange'],
                 xlabel='Average entropy of pixels (fg) per image',
                 ylabel='Number of images',
                 title='Distribution of (fg) pixel entropy \n(cancer correctly classified vs normal misclassified with 60% >  bg > 00%)',
                 save_path= os.path.join(out_dir, 'hist_entropy_px_fg_bg_00_60_cancer_correct_vs_normal_misclassified.png'))
    

    # plt.figure(figsize=(8, 4))
    # plt.hist(pct_fg_c1_correct, bins=bins, alpha=0.6, color='green', label='Cancer bien classé')
    # plt.hist(pct_fg_c0_incorrect, bins=bins, alpha=0.6, color='orange', label='Normal mal classé (FP cancer)')
    # plt.xlabel('Pourcentage de pixels prédits cancer')
    # plt.ylabel('Nombre d’images')
    # plt.legend()
    # plt.title('Distribution du pourcentage de pixels prédits cancer\n(cancer correct vs normal mal classé)')
    # plt.tight_layout()
    # plt.savefig('hist_pct_fg_cancer_correct_vs_normal_misclassified_bloc.png', dpi=300)
    # plt.show()       

    # plt.hist(pct_bg_c1_correct, bins=bins, alpha=0.6, color='blue', label='Cancer bien classé (background)')
    # plt.hist(pct_bg_c0_incorrect, bins=bins, alpha=0.6, color='red', label='Normal mal classé (background)')

    # plt.figure(figsize=(8, 4))
    # plt.hist(pct_bg_c1_correct, bins=bins, alpha=0.6, color='green', label='Cancer bien classé')
    # plt.hist(pct_bg_c0_incorrect, bins=bins, alpha=0.6, color='orange', label='Normal mal classé (FP cancer)')
    # plt.xlabel('Pourcentage de pixels prédits bg')
    # plt.ylabel('Nombre d’images')
    # plt.legend()
    # plt.title('Distribution du pourcentage de pixels prédits cancer\n(cancer correct vs normal mal classé)')
    # plt.tight_layout()
    # plt.savefig('hist_pct_bg_cancer_correct_vs_normal_misclassified_bloc.png', dpi=300)
    # plt.show()  


    # pct_fg_c1_incorrect = pct_fg[mask_class_1_incorrect]
    # pct_fg_c0_correct = pct_fg[mask_class_0_correct]

    # pct_bg = 1 - pct_fg
    # pct_bg_c1_incorrect = pct_bg[mask_class_1_incorrect]
    # pct_bg_c0_correct = pct_bg[mask_class_0_correct]

    # plt.figure(figsize=(8, 4))
    # plt.hist(pct_bg_c1_incorrect, bins=bins, alpha=0.6, color='green', label='Cancer bien classé')
    # plt.hist(pct_bg_c0_correct, bins=bins, alpha=0.6, color='orange', label='Normal mal classé (FP cancer)')
    # plt.xlabel('Pourcentage de pixels prédits cancer')
    # plt.ylabel('Nombre d’images')
    # plt.legend()
    # plt.title('Distribution du pourcentage de pixels prédits cancer\n(cancer correct vs normal mal classé)')
    # plt.tight_layout()
    # plt.savefig('hist_pct_bg_normal_correct_vs_cancer_misclassified_bloc.png', dpi=300)
    # plt.show()

    # plt.figure(figsize=(8, 4))
    # plt.hist(pct_fg_c1_incorrect, bins=bins, alpha=0.6, color='green', label='Cancer bien classé')
    # plt.hist(pct_fg_c0_correct, bins=bins, alpha=0.6, color='orange', label='Normal mal classé (FP cancer)')
    # plt.xlabel('Pourcentage de pixels prédits cancer')
    # plt.ylabel('Nombre d’images')
    # plt.legend()
    # plt.title('Distribution du pourcentage de pixels prédits cancer\n(cancer correct vs normal mal classé)')
    # plt.tight_layout()
    # plt.savefig('hist_pct_fg_normal_correct_vs_cancer_misclassified_bloc.png', dpi=300)
    # plt.show()





    features_pos = img_ft_1_correct
    features_neg = img_ft_0_incorrect
    # Concaténation
    all_features = np.concatenate([features_pos, features_neg], axis=0)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded_features = tsne.fit_transform(all_features)

    # Séparation
    n_pos = features_pos.shape[0]
    embedded_pos = embedded_features[:n_pos]
    embedded_neg = embedded_features[n_pos:]

    # Visualisation
    plt.figure(figsize=(6, 6))
    plt.scatter(embedded_pos[:, 0], embedded_pos[:, 1], color='green', label='Class 1 - Correct')
    plt.scatter(embedded_neg[:, 0], embedded_neg[:, 1], color='orange', label='Class 0 - Incorrect')

    plt.legend()
    plt.title('t-SNE of pixel features:\nforeground correctly predicted vs background misclassified')
    plt.tight_layout()
    plt.savefig('tsne_correct1_incorrect0_4.png', dpi=300)
    plt.close()

    n_pos = features_pos.shape[0]
    n_neg = features_neg.shape[0]
    true_labels = np.array([1]*n_pos + [0]*n_neg)   

    from sklearn.cluster import KMeans

    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_ids = kmeans.fit_predict(all_features)


    for i in range(n_clusters):
        mask = (cluster_ids == i)
        true_c = true_labels[mask]
        total = len(true_c)
        n_cancer = np.sum(true_c == 1)
        n_normal = np.sum(true_c == 0)
        print(f"Cluster {i}: {total} points → {n_cancer} correct cancer, {n_normal} misclassified normal")


    
    center_cancer_pure=model.get_linear_weights[1].detach().cpu().numpy()

    from sklearn.metrics import pairwise_distances

    anchor_c0 = model.get_linear_weights[0].detach().cpu().numpy().reshape(1, -1)


    for i in range(10):
        cluster_feats = all_features[cluster_ids == i]
        cluster_center = cluster_feats.mean(axis=0, keepdims=True)
        dist = pairwise_distances(cluster_center, anchor_c0).item()
        print(f"Cluster {i}:Distance to anchors = {dist:.3f}")

    anchor_c1 = model.get_linear_weights[1].detach().cpu().numpy().reshape(1, -1)


    for i in range(10):
        cluster_feats = all_features[cluster_ids == i]
        cluster_center = cluster_feats.mean(axis=0, keepdims=True)
        dist = pairwise_distances(cluster_center, anchor_c1).item()
        print(f"Cluster {i}:Distance to anchors = {dist:.3f}")

    # Plot classe 1
    plt.figure(figsize=(8, 4))

    #all_conf_1 = np.concatenate([img_confidence_1_correct, img_confidence_1_incorrect])
    #min_conf = float(np.min(all_conf_1))
    #max_conf = float(np.max(all_conf_1))
    # Ajuste dynamiquement le nombre de bins selon la plage de valeurs
    #n_bins = 15 if max_conf - min_conf < 0.1 else 30
    #bins = np.linspace(min_conf, max_conf + 1e-6, n_bins)

    bins = np.linspace(0, 1, 31)
    counts_correct, bins_correct, _  = plt.hist(img_confidence_1_correct, bins=bins, alpha=0.7, color='blue', label='Class 0 - Correct')
    counts_incorrect, bins_incorrect, _ = plt.hist(img_confidence_1_incorrect, bins=bins, alpha=0.7, color='red', label='Class 0 - Incorrect')

    #counts_correct, bins_correct, _  = plt.hist(img_confidence_1_correct, bins=30, alpha=0.7, color='blue', label='Class 1 - Correct')
    #counts_incorrect, bins_incorrect, _ = plt.hist(img_confidence_1_incorrect, bins=30, alpha=0.7, color='red', label='Class 1 - Incorrect')

    #counts_correct, bins_correct, _ = plt.hist(img_confidence_1_correct, bins=bins, alpha=0.7, color='blue', label='Class 1 - Correct')
    #counts_incorrect, bins_incorrect, _ = plt.hist(img_confidence_1_incorrect, bins=bins, alpha=0.7, color='red', label='Class 1 - Incorrect')


    for count, x in zip(counts_correct, bins_correct[:-1]):
        if count > 0:
            plt.text(x + (bins_correct[1] - bins_correct[0]) / 2, count, str(int(count)), ha='center', va='bottom', fontsize=7)

    for count, x in zip(counts_incorrect, bins_incorrect[:-1]):
        if count > 0:
            plt.text(x + (bins_incorrect[1] - bins_incorrect[0]) / 2, count, str(int(count)), ha='center', va='bottom', fontsize=7)

    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title(f"{title_prefix or ''} Class 0 - Prediction Confidence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hist_class1_proba_{target_dataset}.png"), dpi=300)
    plt.close()

    # Plot classe 0
    plt.figure(figsize=(8, 4))
    counts_correct, bins_correct, _  = plt.hist(img_confidence_0_correct, bins=30, alpha=0.7, color='blue', label='Class 1 - Correct')
    counts_incorrect, bins_incorrect, _ = plt.hist(img_confidence_0_incorrect, bins=30, alpha=0.7, color='red', label='Class 1 - Incorrect')
    for count, x in zip(counts_correct, bins_correct[:-1]):
        if count > 0:
            plt.text(x + (bins_correct[1] - bins_correct[0]) / 2, count, str(int(count)), ha='center', va='bottom', fontsize=7)

    for count, x in zip(counts_incorrect, bins_incorrect[:-1]):
        if count > 0:
            plt.text(x + (bins_incorrect[1] - bins_incorrect[0]) / 2, count, str(int(count)), ha='center', va='bottom', fontsize=7)

    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title(f"{title_prefix or ''} Class 1 - Prediction Confidence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hist_class0_proba_{target_dataset}.png"), dpi=300)
    plt.close()


    # # Plot classe 1
    # plt.figure(figsize=(8, 4))
    # counts_correct, bins_correct, _  = plt.hist(img_energy_class1_correct, bins=30, alpha=0.7, color='blue', label='Class 1 - Correct')
    # counts_incorrect, bins_incorrect, _ = plt.hist(img_energy_class1_incorrect, bins=30, alpha=0.7, color='red', label='Class 1 - Incorrect')
    # for count, x in zip(counts_correct, bins_correct[:-1]):
    #     if count > 0:
    #         plt.text(x + (bins_correct[1] - bins_correct[0]) / 2, count, str(int(count)), ha='center', va='bottom', fontsize=7)

    #     for count, x in zip(counts_incorrect, bins_incorrect[:-1]):
    #         if count > 0:
    #             plt.text(x + (bins_incorrect[1] - bins_incorrect[0]) / 2, count, str(int(count)), ha='center', va='bottom', fontsize=7)

    # plt.xlabel("Energy")
    # plt.ylabel("Count")
    # plt.title(f"{title_prefix or ''} Class 1 - Energy")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, f"hist_class1_energy_{target_dataset}.png"), dpi=300)
    # plt.close()

    # # Plot classe 0
    # plt.figure(figsize=(8, 4))
    # counts_correct, bins_correct, _  = plt.hist(img_confidence_0_correct, bins=30, alpha=0.7, color='blue', label='Class 0 - Correct')
    # counts_incorrect, bins_incorrect, _ = plt.hist(img_confidence_0_incorrect, bins=30, alpha=0.7, color='red', label='Class 0 - Incorrect')
    # for count, x in zip(counts_correct, bins_correct[:-1]):
    #     if count > 0:
    #         plt.text(x + (bins_correct[1] - bins_correct[0]) / 2, count, str(int(count)), ha='center', va='bottom', fontsize=7)

    #     for count, x in zip(counts_incorrect, bins_incorrect[:-1]):
    #         if count > 0:
    #             plt.text(x + (bins_incorrect[1] - bins_incorrect[0]) / 2, count, str(int(count)), ha='center', va='bottom', fontsize=7)

    # plt.xlabel("Confidence")
    # plt.ylabel("Count")
    # plt.title(f"{title_prefix or ''} Class 0 - Prediction Confidence")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, f"hist_class0_proba_{target_dataset}.png"), dpi=300)
    # plt.close()

    # # Plot classe 1
    # plt.figure(figsize=(8, 4))
    # counts_correct, bins_correct, _  = plt.hist(img_energy_class0_correct, bins=30, alpha=0.7, color='blue', label='Class 0 - Correct')
    # counts_incorrect, bins_incorrect, _ = plt.hist(img_energy_class0_incorrect, bins=30, alpha=0.7, color='red', label='Class 0 - Incorrect')
    # for count, x in zip(counts_correct, bins_correct[:-1]):
    #     if count > 0:
    #         plt.text(x + (bins_correct[1] - bins_correct[0]) / 2, count, str(int(count)), ha='center', va='bottom', fontsize=7)

    #     for count, x in zip(counts_incorrect, bins_incorrect[:-1]):
    #         if count > 0:
    #             plt.text(x + (bins_incorrect[1] - bins_incorrect[0]) / 2, count, str(int(count)), ha='center', va='bottom', fontsize=7)

    # plt.xlabel("Energy")
    # plt.ylabel("Count")
    # plt.title(f"{title_prefix or ''} Class 0 - Energy")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, f"hist_class0_energy_{target_dataset}.png"), dpi=300)
    # plt.close()


def plot_energy_based_on_target_image_acc(target_dict, out_dir, save_path=None, title_prefix=None,target_dataset=None):
    os.makedirs(out_dir, exist_ok=True)

    img_classes = target_dict['label_images']
    img_predict = target_dict['pred_images']
    img_confidence = target_dict['probs_images']
    accuracy = (img_classes == img_predict)

    img_energy_correct = target_dict['images'][accuracy == 1]
    img_energy_incorrect = target_dict['images'][accuracy == 0]

    img_confidence_correct = img_confidence[accuracy == 1]
    img_confidence_incorrect = img_confidence[accuracy == 0]



    mask_class_1 = (img_classes == 1)
    mask_class_1_correct = mask_class_1 & (accuracy == 1)
    mask_class_1_incorrect = mask_class_1 & (accuracy == 0)

    mask_class_0 = (img_classes == 0)
    mask_class_0_correct = mask_class_0 & (accuracy == 1)
    mask_class_0_incorrect = mask_class_0 & (accuracy == 0)


    img_energy_class1_correct = target_dict['images'][mask_class_1_correct]
    img_energy_class1_incorrect = target_dict['images'][mask_class_1_incorrect]
    img_confidence_1_correct = img_confidence[mask_class_1_correct]
    img_confidence_1_incorrect = img_confidence[mask_class_1_incorrect]

    img_energy_class0_correct = target_dict['images'][mask_class_0_correct]
    img_energy_class0_incorrect = target_dict['images'][mask_class_0_incorrect]
    img_confidence_0_correct = img_confidence[mask_class_0_correct]
    img_confidence_0_incorrect = img_confidence[mask_class_0_incorrect]


    img_energy = target_dict['images']
    px_energy = target_dict['pixels']


    if target_dataset == constants.CAMELYON512:
        img_pxap_correct = target_dict['cam_performance'][mask_class_1_correct]
        img_pxap_incorrect = target_dict['cam_performance'][mask_class_1_incorrect]
    else:
        img_pxap_correct = target_dict['cam_performance'][accuracy == 1]
        img_pxap_incorrect = target_dict['cam_performance'][accuracy == 0]



    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.hist(img_energy_correct, bins=40, alpha=0.4, label='Correct', color='tab:blue', density=False)
    plt.hist(img_energy_incorrect, bins=40, alpha=0.4, label='Incorrect', color='tab:red', density=False)
    plt.xlabel("Density")
    plt.ylabel("Image Energy")
    plt.title("Image Energy vs Accuracy")
    plt.legend()


    if save_path:
        plt.savefig("PXAP_vs_Energy.png", dpi=300)
        print(f"Saved at {save_path}")
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(8, 6))

    # Image-level
    plt.subplot(1, 2, 1)
    if target_dataset == constants.CAMELYON512:
        plt.scatter(img_energy_class1_correct, img_pxap_correct, alpha=0.4, color='tab:blue', label='Correct')
        plt.scatter(img_energy_class1_incorrect, img_pxap_incorrect, alpha=0.4, color='tab:red', label='Incorrect')
    else:
        plt.scatter(img_energy_correct, img_pxap_correct, alpha=0.4, color='tab:blue', label='Correct')
        plt.scatter(img_energy_incorrect, img_pxap_incorrect, alpha=0.4, color='tab:red', label='Incorrect')
    #plt.scatter(img_energy, img_confidence, alpha=0.5, color='tab:blue', label='Image Energy')
    #plt.scatter(img_energy_correct, img_pxap_correct, alpha=0.4, color='tab:blue', label='Correct')
    #plt.scatter(img_energy_incorrect, img_pxap_incorrect, alpha=0.4, color='tab:red', label='Incorrect')
    plt.xlabel("Image Energy")
    plt.ylabel("PXAP")
    plt.title("Image Energy vs PXAP")
    plt.grid(True)

    if save_path:
        plt.savefig("PXAP_vs_Energy.png", dpi=300)
        print(f"Saved at {save_path}")
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(8, 6))

    # Image-level
    plt.subplot(1, 2, 1)
    #plt.scatter(img_energy, img_confidence, alpha=0.5, color='tab:blue', label='Image Energy')
    plt.scatter(img_energy_correct, img_confidence_correct, alpha=0.4, color='tab:blue', label='Correct')
    plt.scatter(img_energy_incorrect, img_confidence_incorrect, alpha=0.4, color='tab:red', label='Incorrect')
    plt.xlabel("Image Energy")
    plt.ylabel("Model Confidence (max prob)")
    plt.title("Image Energy vs Confidence")
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig("Confidence_vs_Energy.png", dpi=300)
        print(f"Saved at {save_path}")
        plt.close()
    else:
        plt.show()



    plt.figure(figsize=(8, 6))
    # Image-level
    plt.subplot(1, 2, 1)
    #plt.scatter(img_energy, img_confidence, alpha=0.5, color='tab:blue', label='Image Energy')
    plt.scatter(img_energy_class0_correct, img_confidence_0_correct, alpha=0.4, color='tab:blue', label='Correct')
    plt.scatter(img_energy_class0_incorrect, img_confidence_0_incorrect, alpha=0.4, color='tab:red', label='Incorrect')
    plt.xlabel("Image Energy")
    plt.ylabel("Model Confidence (max prob)")
    plt.title("Image Energy vs Confidence")
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig("Confidence_vs_Energy_class_0.png", dpi=300)
        print(f"Saved at {save_path}")
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(8, 6))
    # Image-level
    plt.subplot(1, 2, 1)
    #plt.scatter(img_energy, img_confidence, alpha=0.5, color='tab:blue', label='Image Energy')
    plt.scatter(img_energy_class1_correct, img_confidence_1_correct, alpha=0.4, color='tab:blue', label='Correct')
    plt.scatter(img_energy_class1_incorrect, img_confidence_1_incorrect, alpha=0.4, color='tab:red', label='Incorrect')
    plt.xlabel("Image Energy")
    plt.ylabel("Model Confidence (max prob)")
    plt.title("Image Energy vs Confidence")
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig("Confidence_vs_Energy_class_1.png", dpi=300)
        print(f"Saved at {save_path}")
        plt.close()
    else:
        plt.show()


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

    num_correct_normal = 0
    num_images_normal = 0

    num_correct_cancer = 0
    num_images_cancer = 0

    for i, (images, targets, _, _, _, _, _, _) in enumerate(loader):
        images = images.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            cl_logits = cl_forward(args, model, images)
            pred = cl_logits.argmax(dim=1)

        num_correct += (pred == targets).sum().item()
        num_images += images.size(0)

        # Compute accuracy for each class
        for j in range(len(targets)):
            if targets[j] == 0:
                num_images_normal += 1
                if pred[j] == targets[j]:
                    num_correct_normal += 1
            elif targets[j] == 1:
                num_images_cancer += 1
                if pred[j] == targets[j]:
                    num_correct_cancer += 1
            else:
                raise ValueError("Unknown class label")
            
    # Compute accuracy for each class
    classification_acc_normal = num_correct_normal / float(num_images_normal) * 100 if num_images_normal > 0 else 0
    classification_acc_cancer = num_correct_cancer / float(num_images_cancer) * 100 if num_images_cancer > 0 else 0


    classification_acc = num_correct / float(num_images) * 100
    
    return classification_acc, classification_acc_normal, classification_acc_cancer



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
        #args_dict['pixel_wise_classification'] = False
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
    if parsedargs.external_model == None:
        if "tscam" in encoder_name or "sat" in encoder_name:
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
    else:
        path_eternal_cl = parsedargs.external_model
        encoder_w = torch.load(join(path_eternal_cl, 'encoder.pt'),
                                map_location=get_cpu_device())
        model.encoder.super_load_state_dict(encoder_w, strict=True)

        header_w = torch.load(join(path_eternal_cl, 'classification_head.pt'),
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




    if target_dataset == constants.CUB:
        metadata_root = join(constants.RELATIVE_META_ROOT, 'CUB')
        target_domain_data_paths = config.configure_data_paths(args_dict, 'CUB')

        target_loaders = get_data_loader_natural(
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
            get_splits_eval=['train'],
            #constants.TRAINSET
        )
    else:
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
                loader=target_loaders[split],
                metadata_root=os.path.join(target_metadata_root, split),
                mask_root=args.mask_root,
                iou_threshold_list=args.iou_threshold_list,
                dataset_name=target_dataset,
                split= split,
                cam_curve_interval=args.cam_curve_interval,
                multi_contour_eval=args.multi_contour_eval,
                out_folder=args.outd,
            )



   
    out_dir = "plots_energy_test"
    os.makedirs(out_dir, exist_ok=True)
   
    
    if parsedargs.external_model is not None:
        external_pixel_classifier = True
    else:
        external_pixel_classifier = None


    source_model_name = parsedargs.source_model_name
    target_model_name = parsedargs.target_model_name



    model.eval()
    cl_global, cl_normal, cl_cancer = _compute_accuracy(args, model, target_loaders[split])

    print(f"Classification accuracy on target dataset {target_dataset} is {cl_global:.2f}%")
    print(f"Classification accuracy on target dataset {target_dataset} for normal class is {cl_normal:.2f}%")
    print(f"Classification accuracy on target dataset {target_dataset} for cancer class is {cl_cancer:.2f}%")

    

    if target_dataset == constants.CUB:
        target_energy = compute_energy_distributions_cub(model, target_loaders, target_dataset, energy_fn, split, device, args=args, metadata_root=target_metadata_root, cam_performance = False)
    else:
        target_energy = compute_energy_distributions(model, target_loaders, target_cam_computer, target_dataset, energy_fn, split, device, args=args, metadata_root=target_metadata_root, cam_performance = False)

    
    out_dir = "plots_energy_test"
    os.makedirs(out_dir, exist_ok=True)

    features = target_energy["img_ft"]   # Shape: (N, D)
    labels = target_energy["label_images"]



    image_anchor_weights = model.classification_head.fc.weight[:2] 

    anchor_weights = image_anchor_weights.cpu().detach().numpy()

    X_umap = np.vstack([features, anchor_weights])


    # reducer = umap.UMAP(n_components=2, n_neighbors=100, min_dist=0.05,metric="cosine",random_state=42)
    # embedding = reducer.fit_transform(X_umap) 

    for min_dist in [0.05]:
        for n_neighbors in [1000]:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric="cosine",  # ou "euclidean"
                random_state=42
            )
            embedding = reducer.fit_transform(X_umap)


            embedding_features = embedding[:len(features)]
            embedding_anchors = embedding[len(features):]

            labels = np.array(labels)
            unique_labels = np.unique(labels)

            plt.figure(figsize=(8,6))

            # Plot features par classe
            for label in np.unique(labels):
                idx = labels == label
                plt.scatter(embedding_features[idx, 0], embedding_features[idx, 1],
                            label=f"Classe {label}", alpha=0.7, s=10)

            # Plot anchors (poids du classifieur)
            plt.scatter(embedding_anchors[:,0], embedding_anchors[:,1],
                        c='black', marker='*', s=180, label='Anchors (weights)', edgecolor='white')

            plt.legend()
            plt.title("UMAP features + anchors du classifieur")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.tight_layout()
            filename = f"umap_n{n_neighbors}_mindist{min_dist}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            # plt.savefig("UMAP.png", dpi=300)
            # plt.show()
            #plot_hist_energy_based_on_target_image_acc(target_energy, out_dir, save_path="test", target_dataset=target_dataset, model = model)

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
    parser.add_argument("--external_model", type=str, default=None, help="Path to the external bb+cl.")
    parser.add_argument("--source_model_name", type=str, default=None, help="Name of source model.")
    parser.add_argument("--target_model_name", type=str, default=None, help="Name of target model.")

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
            get_features(exp_path=exp_path, sf_uda_source_folder=parsedargs.path_pre_trained_source,image_ids_to_draw=parsedargs.image_ids_to_draw,image_ids_to_draw_target=parsedargs.image_ids_to_draw_target, checkpoint_type=checkpoint_type, source_dataset=parsedargs.source_dataset,target_dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs)

if __name__ == '__main__':
    fast_eval()