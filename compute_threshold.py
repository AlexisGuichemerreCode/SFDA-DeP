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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import wasserstein_distance, entropy, ks_2samp
from scipy.spatial.distance import cdist
from scipy import linalg
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import confusion_matrix






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

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, precision_score, recall_score
)

from sklearn.metrics import roc_auc_score, average_precision_score





def to_cuda(x):
    return x.cuda()

def to_onehot(label, num_classes):
    identity = to_cuda(torch.eye(num_classes))
    onehot = torch.index_select(identity, 0, label)
    return onehot


def _to_numpy_list(x):
    """Convertit une liste potentiellement composée de tensors/ndarrays -> liste de ndarrays CPU."""
    out = []
    for item in x:
        try:
            import torch
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()
        except Exception:
            pass
        item = np.asarray(item)
        out.append(item)
    return out

def _flatten_batches(arrs):
    """
    arrs: liste de ndarrays éventuellement de forme (B,K), (B,), (K,), ()...
    Retourne:
      - if K>1 -> np.ndarray shape (N,K)
      - if K==1/scalar -> np.ndarray shape (N,)
    """
    flat = []
    for a in arrs:
        a = np.asarray(a)
        if a.ndim == 0:
            flat.append(a.reshape(1))
        elif a.ndim == 1:
            flat.append(a.reshape(-1, *(() if a.shape == () else a.shape[1:])))
        elif a.ndim >= 2:
            flat.append(a.reshape(a.shape[0], -1))  # (B,K'...) -> (B,Kflat)
        else:
            flat.append(a)
    if len(flat) == 0:
        return np.array([])
    cat = np.concatenate(flat, axis=0)
    return cat


def evaluate_from_target_energy(target_energy: dict, out_dir: str, pos_label: int = 1, beta: float = 1.0):
    """
    Lit target_energy['probs_images'], ['label_images'] (et éventuellement ['pred_images']),
    calcule y_true et y_score (proba classe pos_label), trace des courbes et retourne un résumé.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Récup
    probs  = target_energy['probs_images']      # (67, 2)
    labels = target_energy['label_images']      # (67,) ou (67,1)

    # Torch -> NumPy si besoin
    try:
        import torch
        if torch.is_tensor(probs):  probs  = probs.cpu().numpy()
        if torch.is_tensor(labels): labels = labels.cpu().numpy()
    except Exception:
        pass

    labels = labels.reshape(-1).astype(int)
    y_score = probs[:, 1].astype(float)   # proba "cancer"
    y_true = labels.astype(int)

    # Sanity checks
    assert probs.shape[0] == labels.shape[0]
    assert probs.shape[1] == 2
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    # ROC / PR
    fpr, tpr, roc_thr = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    prec, rec, thr = precision_recall_curve(labels, y_score, pos_label=1)
    pr_auc = auc(rec, prec)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR (Recall)")
    plt.title("ROC curve"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=200); plt.close()

    prec, rec, pr_thr = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    pr_auc = auc(rec, prec)

    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall curve"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=200); plt.close()

    # 4) Balayage des seuils
    thr_grid = np.linspace(0.0, 1.0, 101)
    prec_t, rec_t, fbeta_t = [], [], []
    for thr in thr_grid:
        y_hat = (y_score >= thr).astype(int)
        p = precision_score(y_true, y_hat, zero_division=0)
        r = recall_score(y_true, y_hat, zero_division=0)
        if (beta*beta*p + r) > 0:
            f_beta = (1+beta*beta) * p * r / (beta*beta * p + r)
        else:
            f_beta = 0.0
        prec_t.append(p); rec_t.append(r); fbeta_t.append(f_beta)

    prec_t = np.array(prec_t); rec_t = np.array(rec_t); fbeta_t = np.array(fbeta_t)
    best_idx = int(np.argmax(fbeta_t))
    best_thr = float(thr_grid[best_idx]); best_f = float(fbeta_t[best_idx])

    plt.figure(figsize=(6,5))
    plt.plot(thr_grid, fbeta_t); plt.axvline(best_thr, linestyle='--')
    plt.title(f"F{beta:.1f} vs threshold (best={best_f:.4f} @ {best_thr:.3f})")
    plt.xlabel("Threshold"); plt.ylabel(f"F{beta:.1f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"f{beta:.1f}_vs_threshold.png"), dpi=200); plt.close()

    plt.figure(figsize=(6,5))
    plt.plot(thr_grid, prec_t); plt.axvline(best_thr, linestyle='--')
    plt.title("Precision vs threshold"); plt.xlabel("Threshold"); plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "precision_vs_threshold.png"), dpi=200); plt.close()

    plt.figure(figsize=(6,5))
    plt.plot(thr_grid, rec_t); plt.axvline(best_thr, linestyle='--')
    plt.title("Recall vs threshold"); plt.xlabel("Threshold"); plt.ylabel("Recall")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "recall_vs_threshold.png"), dpi=200); plt.close()

    # 5) Youden J & matrice de conf au meilleur F_beta
    J = tpr - fpr
    if len(roc_thr) > 0:
        j_idx = int(np.argmax(J)); j_thr = float(roc_thr[j_idx]); j_val = float(J[j_idx])
    else:
        j_thr, j_val = 0.5, 0.0

    y_hat_best = (y_score >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat_best, labels=[0,1]).ravel()

    conf_txt = os.path.join(out_dir, "confusion_counts.txt")
    with open(conf_txt, "w") as f:
        f.write(f"tn: {int(tn)}\n")
        f.write(f"fp: {int(fp)}\n")
        f.write(f"fn: {int(fn)}\n")
        f.write(f"tp: {int(tp)}\n")


    summary = {
        "N": int(y_true.shape[0]),
        "roc_auc": round(float(roc_auc), 6),
        "pr_auc": round(float(pr_auc), 6),
        "best_fbeta": round(best_f, 6),
        "best_fbeta_threshold": round(best_thr, 6),
        "youden_J": round(j_val, 6),
        "youden_J_threshold": round(j_thr, 6),
        "confusion_at_best_fbeta": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary



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

def plot_histogram_with_counts_and_percentages(correct_data, incorrect_data, title, xlabel, ylabel, out_path, bin):
    plt.figure(figsize=(20, 12))

    counts_correct, bins = np.histogram(correct_data, bins=bin)
    counts_incorrect, _ = np.histogram(incorrect_data, bins=bins)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    total_counts = counts_correct + counts_incorrect

    plt.bar(bin_centers, counts_correct, width=np.diff(bins), alpha=0.7, label='Correctly Predicted', align='center')
    plt.bar(bin_centers, counts_incorrect, width=np.diff(bins), bottom=counts_correct, alpha=0.7, label='Incorrectly Predicted', align='center')

    for total, correct, incorrect, x in zip(total_counts, counts_correct, counts_incorrect, bin_centers):
        if total > 0:
            plt.text(x, total, f"{total}", rotation=90, ha='center', va='bottom', fontsize=8)
            if correct > 0:
                plt.text(x, correct / 2, f"{100 * correct / total:.1f}%", ha='center', va='center', fontsize=7, color='white')
            if incorrect > 0:
                plt.text(x, correct + incorrect / 2, f"{100 * incorrect / total:.1f}%", ha='center', va='center', fontsize=7, color='black')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

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


def compute_energy_distributions_images(
    model,
    loader,
    dataset_name,          # conservé pour compat, pas utilisé ici
    energy_fn,             # ex: lambda logits: -torch.logsumexp(logits, dim=1)
    split,                 # "train" | "valid" | "test"
    device,
    args=None,
    probs_mode: str = "full",   # "full" -> (N,K), "max" -> (N,)
    return_numpy: bool = True,  # True -> np.ndarray en sortie
):
    """
    Version image-only (pas de per-pixel).
    Remplit:
      - images: énergie par image (shape [N])
      - per_class: dict label -> liste d'énergies (float)
      - probs_images: (N,K) si probs_mode="full", sinon (N,) si "max"
      - pred_images: (N,)
      - label_images: (N,)
      - entropy_imgs: (N,)
      - min_logits_img / max_logits_img: (N,)
      - img_ft: (N,D) si model.lin_ft dispo, sinon absent
    """
    model.eval()
    dl = loader[split]
    eps = 1e-8

    energy_data = {
        "images": [],
        "per_class": defaultdict(list),
        "probs_images": [],
        "pred_images": [],
        "label_images": [],
        "entropy_imgs": [],
        "min_logits_img": [],
        "max_logits_img": [],
        "img_ft": [],   # rempli uniquement si dispo
    }

    for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), ncols=80):
        # tolère différents formats de batch
        if len(batch) >= 2:
            images, targets = batch[0], batch[1]
        else:
            raise ValueError("Batch inattendu: attend au moins (images, targets, ...).")

        images  = images.to(device)
        targets = targets.to(device)

        # --- Forward (image-level) ---
        logits = model(images)                # [B,K]
        probs  = torch.softmax(logits, dim=1) # [B,K]
        preds  = probs.argmax(dim=1)          # [B]
        H_img  = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)  # [B]

        # Énergie par image (ex: -logsumexp) -> [B] attendu
        energy_images = energy_fn(logits)     # idéalement [B]
        if energy_images.ndim > 1:
            # fallback si energy_fn renvoie autre chose: on réduit en moyenne
            energy_images = energy_images.view(energy_images.size(0), -1).mean(dim=1)

        max_vals_img, _ = logits.max(dim=1)   # [B]
        min_vals_img, _ = logits.min(dim=1)   # [B]

        # --- Collect ---
        energy_data["images"].append(energy_images.detach().cpu())
        if probs_mode == "full":
            energy_data["probs_images"].append(probs.detach().cpu())
        elif probs_mode == "max":
            energy_data["probs_images"].append(probs.max(dim=1).values.detach().cpu())
        else:
            raise ValueError("probs_mode doit être 'full' ou 'max'.")

        energy_data["pred_images"].append(preds.detach().cpu())
        energy_data["label_images"].append(targets.detach().cpu())
        energy_data["entropy_imgs"].append(H_img.detach().cpu())
        energy_data["max_logits_img"].append(max_vals_img.detach().cpu())
        energy_data["min_logits_img"].append(min_vals_img.detach().cpu())

        # Features image si dispo (utile pour analyses ultérieures)
        if hasattr(model, "lin_ft") and isinstance(model.lin_ft, torch.Tensor):
            img_ft = model.lin_ft.flatten(start_dim=1)   # [B,D]
            energy_data["img_ft"].append(img_ft.detach().cpu())

        # Per-class (pour histogrammes par classe, etc.)
        for e, y in zip(energy_images.detach().cpu(), targets.detach().cpu()):
            energy_data["per_class"][int(y.item())].append(float(e.item()))

    # --- Concat / numpy ---
    def _cat_to_numpy(lst, dim=0):
        t = torch.cat(lst, dim=dim)
        return t.numpy() if return_numpy else t

    energy_data["images"]        = _cat_to_numpy(energy_data["images"]).squeeze()
    # probs_images: concat correctement selon mode
    if len(energy_data["probs_images"]) > 0:
        if probs_mode == "full":
            energy_data["probs_images"] = _cat_to_numpy(energy_data["probs_images"], dim=0)  # (N,K)
        else:
            energy_data["probs_images"] = _cat_to_numpy(energy_data["probs_images"], dim=0)  # (N,)
    energy_data["pred_images"]   = _cat_to_numpy(energy_data["pred_images"])
    energy_data["label_images"]  = _cat_to_numpy(energy_data["label_images"])
    energy_data["entropy_imgs"]  = _cat_to_numpy(energy_data["entropy_imgs"])
    energy_data["max_logits_img"]= _cat_to_numpy(energy_data["max_logits_img"])
    energy_data["min_logits_img"]= _cat_to_numpy(energy_data["min_logits_img"])

    if len(energy_data["img_ft"]) > 0:
        energy_data["img_ft"] = _cat_to_numpy(energy_data["img_ft"], dim=0)
    else:
        # si vide, on le retire pour éviter les surprises en aval
        energy_data.pop("img_ft")

    return energy_data



def compute_energy_distributions(model, loader, cam_computer, dataset_name, energy_fn, split, device, args=None, metadata_root=None):

    cam_perfs = []

    model.eval()
    eps = 1e-8

    for batch_idx, (images, targets, p_glabel, index, raw_imgs, std_cams, _, views) in tqdm(
        enumerate(loader[split]), ncols=constants.NCOLS, total=len(loader[split])
    ):

        images = images.to(device)
        targets = targets.to(device)
        image_size = images.shape[2:]

       
        for image, target, image_id in zip(images, targets, index):

            dataset_source = os.path.basename(args.mask_root)
            if dataset_source == dataset_name:
                mask_root_data = args.mask_root
            else:
                mask_root_data = os.path.join(os.path.dirname(args.mask_root), dataset_name)

            cam_computer = CAMComputer(
                args=deepcopy(args),
                model=model,
                loader=loader[split],
                metadata_root=os.path.join(metadata_root, split),
                mask_root=mask_root_data,
                iou_threshold_list=args.iou_threshold_list,
                dataset_name=dataset_name,
                split=split,
                cam_curve_interval=args.cam_curve_interval,
                multi_contour_eval=args.multi_contour_eval,
                out_folder=args.outd,
            )

            if dataset_name == constants.CAMELYON512 or dataset_name == constants.CAMELYON17_512:
                if target == 1:
                    cam_perf = cam_computer.compute_and_evaluate_cams_one_image(
                        image=image, target=target, image_id=image_id, image_size=image_size
                    )
                    cam_perfs.append(cam_perf)
            else:
                cam_perf = cam_computer.compute_and_evaluate_cams_one_image(
                    image=image, target=target, image_id=image_id, image_size=image_size
                )
                cam_perfs.append(cam_perf)

    return np.array(cam_perfs)


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

    
# def _compute_accuracy(args, model, loader):
#     num_correct = 0
#     num_images = 0

#     num_correct_normal = 0
#     num_images_normal = 0

#     num_correct_cancer = 0
#     num_images_cancer = 0

#     for i, (images, targets, _, _, _, _, _, _) in enumerate(loader):
#         images = images.cuda()
#         targets = targets.cuda()
#         with torch.no_grad():
#             cl_logits = cl_forward(args, model, images)
#             pred = cl_logits.argmax(dim=1)

#         num_correct += (pred == targets).sum().item()
#         num_images += images.size(0)

#         # Compute accuracy for each class
#         for j in range(len(targets)):
#             if targets[j] == 0:
#                 num_images_normal += 1
#                 if pred[j] == targets[j]:
#                     num_correct_normal += 1
#             elif targets[j] == 1:
#                 num_images_cancer += 1
#                 if pred[j] == targets[j]:
#                     num_correct_cancer += 1
#             else:
#                 raise ValueError("Unknown class label")
            
#     # Compute accuracy for each class
#     classification_acc_normal = num_correct_normal / float(num_images_normal) * 100 if num_images_normal > 0 else 0
#     classification_acc_cancer = num_correct_cancer / float(num_images_cancer) * 100 if num_images_cancer > 0 else 0


#     classification_acc = num_correct / float(num_images) * 100
    
#     return classification_acc, classification_acc_normal, classification_acc_cancer


def compute_rates(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    TPR = tp / (tp + fn + 1e-8)
    FPR = fp / (fp + tn + 1e-8)
    TNR = tn / (tn + fp + 1e-8)
    FNR = fn / (fn + tp + 1e-8)
    return {
        "TPR": TPR * 100,
        "FPR": FPR * 100,
        "TNR": TNR * 100,
        "FNR": FNR * 100
    }




def _compute_accuracy(args, model, loader, num_thresholds=100):
    model.eval()

    all_probs = []
    all_targets = []

    # --- Forward pass ---
    for i, (images, targets, _, _, _, _, _, _) in enumerate(loader):
        images = images.cuda()
        targets = targets.cuda()

        cl_logits = cl_forward(args, model, images)
        probs = torch.softmax(cl_logits, dim=1)[:, 1]  # proba classe 1 (cancer)

        all_probs.append(probs.detach().cpu())
        all_targets.append(targets.detach().cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # --- AUC metrics (seuil-indépendant) ---
    auc_roc = roc_auc_score(all_targets, all_probs)
    auc_pr  = average_precision_score(all_targets, all_probs)

    # --- Recherche du meilleur seuil pour équilibrer ---
    thresholds = np.linspace(0, 1, num_thresholds)
    ratios = []

    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        ratio = preds.mean()  # proportion de "cancer"
        ratios.append(ratio)

    best_idx = np.argmin(np.abs(np.array(ratios) - 0.5))
    best_threshold = thresholds[best_idx]
    best_ratio = ratios[best_idx]

    # --- 1) Seuil standard = 0.5 ---
    preds_default = (all_probs >= 0.5).astype(int)
    acc_default   = accuracy_score(all_targets, preds_default) * 100
    prec_default  = precision_score(all_targets, preds_default, zero_division=0) * 100
    rec_default   = recall_score(all_targets, preds_default, zero_division=0) * 100
    f1_default    = f1_score(all_targets, preds_default, zero_division=0) * 100

    # --- 2) Seuil calibré ---
    preds_cal = (all_probs >= best_threshold).astype(int)
    acc_cal   = accuracy_score(all_targets, preds_cal) * 100
    prec_cal  = precision_score(all_targets, preds_cal, zero_division=0) * 100
    rec_cal   = recall_score(all_targets, preds_cal, zero_division=0) * 100
    f1_cal    = f1_score(all_targets, preds_cal, zero_division=0) * 100

    rates_default = compute_rates(all_targets, preds_default)
    rates_cal     = compute_rates(all_targets, preds_cal)

    results = {
        "Default_0.5": {
            "Accuracy": acc_default,
            "Precision": prec_default,
            "Recall": rec_default,
            "F1": f1_default
        },
        f"Calibrated_{best_threshold:.3f}": {
            "Accuracy": acc_cal,
            "Precision": prec_cal,
            "Recall": rec_cal,
            "F1": f1_cal,
            **rates_cal
        },
        "Threshold_independent": {
            "AUC_ROC": auc_roc * 100,
            "AUC_PR": auc_pr * 100
        }
    }
    return results

def _extract_features(args, model, loader):
    model.eval()
    all_feats, all_targets = [], []
    
    for i, (images, targets, _, _, _, _, _, _) in enumerate(loader):
        images = images.cuda()
        targets = targets.cuda()
        cl_logits = model(images)
        
        feats = model.lin_ft  # <-- adapte selon ton modèle
        feats = feats.view(feats.size(0), -1)         # flatten [B, C, H, W] -> [B, D]

        all_feats.append(feats.detach().cpu())
        all_targets.append(targets.detach().cpu())

    all_feats = torch.cat(all_feats, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    return all_feats, all_targets

def gaussian_kernel(x, y, sigma=1.0):
    """
    Gaussian (RBF) kernel entre deux matrices x et y
    x: (N, D), y: (M, D)
    """
    xx = np.sum(x**2, axis=1)[:, np.newaxis]
    yy = np.sum(y**2, axis=1)[np.newaxis, :]
    dist = xx + yy - 2 * np.dot(x, y.T)
    return np.exp(-dist / (2 * sigma**2))

def compute_mmd(features_src, features_tgt, sigma=1.0):
    """
    Calcule MMD^2 entre features_src et features_tgt avec noyau gaussien
    """
    Kxx = gaussian_kernel(features_src, features_src, sigma)
    Kyy = gaussian_kernel(features_tgt, features_tgt, sigma)
    Kxy = gaussian_kernel(features_src, features_tgt, sigma)
    
    m = features_src.shape[0]
    n = features_tgt.shape[0]
    
    mmd2 = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1)) \
         + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1)) \
         - 2 * Kxy.mean()
    return np.sqrt(mmd2)

# ---------- Utils ----------
def _cov(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    denom = max(1, (Xc.shape[0]-1))
    return (Xc.T @ Xc) / denom

def _standardize_pair(X, Y, mode=None, eps=1e-12):
    if mode is None:
        return X, Y
    XY = np.vstack([X, Y])
    if mode == 'zscore':
        mu = XY.mean(0, keepdims=True)
        sd = XY.std(0, keepdims=True) + eps
        return (X - mu)/sd, (Y - mu)/sd
    if mode == 'minmax':
        mn = XY.min(0, keepdims=True)
        mx = XY.max(0, keepdims=True)
        scale = (mx - mn); scale[scale < eps] = 1.0
        return (X - mn)/scale, (Y - mn)/scale
    raise ValueError("standardize must be None | 'zscore' | 'minmax'")

# ---------- JSD (1D, moyenné) ----------
def js_divergence_1d_avg(X, Y, bins=50, eps=1e-10, to_bits=True):
    D = X.shape[1]
    vals = []
    for d in range(D):
        x, y = X[:, d], Y[:, d]
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            continue
        p, _ = np.histogram(x, bins=bins, range=(lo, hi), density=True)
        q, _ = np.histogram(y, bins=bins, range=(lo, hi), density=True)
        p = (p + eps) / (p.sum() + eps*len(p))
        q = (q + eps) / (q.sum() + eps*len(q))
        m = 0.5 * (p + q)
        js = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)  # nats
        if to_bits:
            js /= np.log(2.0)  # borné par 1
        vals.append(js)
    return float(np.mean(vals)) if vals else float('nan')

# ---------- CMD (moments centraux 1..K) ----------
def cmd_distance(X, Y, K=5, eps=1e-12):
    XY = np.vstack([X, Y])
    mn, mx = XY.min(0), XY.max(0)
    scale = (mx - mn); scale[scale < eps] = 1.0
    Xn = 2*(X - mn)/scale - 1
    Yn = 2*(Y - mn)/scale - 1
    mu_x, mu_y = Xn.mean(0), Yn.mean(0)
    s = np.linalg.norm(mu_x - mu_y)  # 1er ordre
    for k in range(2, K+1):
        cx = np.mean((Xn - mu_x)**k, axis=0)
        cy = np.mean((Yn - mu_y)**k, axis=0)
        s += np.linalg.norm(cx - cy)
    return float(s)

# ---------- CORAL (covariance distance) ----------
def coral_distance(X, Y, eps=1e-6, normalize=True):
    Cs = _cov(X) + eps*np.eye(X.shape[1])
    Ct = _cov(Y) + eps*np.eye(Y.shape[1])
    diff = Cs - Ct
    num = np.sqrt((diff * diff).sum())
    if not normalize:
        return float(num)
    den = np.sqrt((Cs*Cs).sum()) + np.sqrt((Ct*Ct).sum()) + 1e-12
    return float(num / den)  # ~[0,1)

# ---------- FID (Fréchet) ----------
def fid_distance(X, Y, eps=1e-6):
    mu_x, mu_y = X.mean(0), Y.mean(0)
    Cx, Cy = _cov(X) + eps*np.eye(X.shape[1]), _cov(Y) + eps*np.eye(Y.shape[1])
    covmean, _ = linalg.sqrtm(Cx.dot(Cy), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu_x - mu_y)**2) + np.trace(Cx + Cy - 2*covmean))

# ---------- Sliced Wasserstein (multi-projections) ----------
def sliced_wasserstein(X, Y, n_proj=128, seed=0):
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    sw = 0.0
    for _ in range(n_proj):
        u = rng.normal(size=D)
        u /= (np.linalg.norm(u) + 1e-12)
        sw += wasserstein_distance(X @ u, Y @ u)
    return float(sw / n_proj)

# ---------- Energy distance ----------
def energy_distance(X, Y, sample_max=3000, seed=0):
    rng = np.random.default_rng(seed)
    if X.shape[0] > sample_max:
        X = X[rng.choice(len(X), sample_max, replace=False)]
    if Y.shape[0] > sample_max:
        Y = Y[rng.choice(len(Y), sample_max, replace=False)]
    d_st = cdist(X, Y, metric="euclidean").mean()
    d_ss = cdist(X, X, metric="euclidean")
    d_tt = cdist(Y, Y, metric="euclidean")
    m_s = d_ss[np.triu_indices_from(d_ss, k=1)].mean()
    m_t = d_tt[np.triu_indices_from(d_tt, k=1)].mean()
    ed2 = 2*d_st - m_s - m_t
    return float(np.sqrt(max(ed2, 0.0)))

# ---------- MK-MMD (multi-kernel RBF) ----------
def _rbf_sum(X, Y, sigmas):
    XX = (X**2).sum(1)[:,None]
    YY = (Y**2).sum(1)[None,:]
    d2 = XX + YY - 2*X.dot(Y.T)
    K = 0.0
    for s in sigmas:
        K += np.exp(-d2 / (2*(s**2)))
    return K

def mk_mmd(X, Y, sigmas=(0.5,1.0,2.0,4.0), sample_max=3000, seed=0):
    rng = np.random.default_rng(seed)
    if X.shape[0] > sample_max:
        X = X[rng.choice(len(X), sample_max, replace=False)]
    if Y.shape[0] > sample_max:
        Y = Y[rng.choice(len(Y), sample_max, replace=False)]
    Kxx = _rbf_sum(X, X, sigmas)
    Kyy = _rbf_sum(Y, Y, sigmas)
    Kxy = _rbf_sum(X, Y, sigmas)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    m, n = len(X), len(Y)
    mmd2 = Kxx.sum()/(m*(m-1)) + Kyy.sum()/(n*(n-1)) - 2*Kxy.mean()
    return float(np.sqrt(max(mmd2, 0.0)))





# def compute_representation_shift(features_src, features_tgt, bins=50, agg="mean", sigma=1.0):
#     """
#     Calcule Wasserstein, KL, KS entre deux ensembles de features.
    
#     features_src : np.array shape [N, D]
#     features_tgt : np.array shape [M, D]
#     bins         : nb de bins pour estimer les distributions (KL)
#     agg          : "mean" ou "median" pour agréger les scores par dimension
#     """
#     wd_list, kl_list, ks_list = [], [], []

#     for d in range(features_src.shape[1]):
#         x, y = features_src[:, d], features_tgt[:, d]

#         # --- Wasserstein ---
#         wd = wasserstein_distance(x, y)
#         wd_list.append(wd)

#         # --- KL divergence ---
#         p_src, _ = np.histogram(x, bins=bins, density=True)
#         p_tgt, _ = np.histogram(y, bins=bins, density=True)

#         p_src = p_src + 1e-10
#         p_tgt = p_tgt + 1e-10
#         p_src = p_src / p_src.sum()
#         p_tgt = p_tgt / p_tgt.sum()

#         kl = entropy(p_src, p_tgt)  # KL(P || Q)
#         kl_list.append(kl)

#         # --- Kolmogorov–Smirnov ---
#         ks_stat, _ = ks_2samp(x, y)
#         ks_list.append(ks_stat)

#     # Agrégation
#     if agg == "mean":
#         wd_final = np.mean(wd_list)
#         kl_final = np.mean(kl_list)
#         ks_final = np.mean(ks_list)
#     elif agg == "median":
#         wd_final = np.median(wd_list)
#         kl_final = np.median(kl_list)
#         ks_final = np.median(ks_list)

#     mmd_final = compute_mmd(features_src, features_tgt, sigma=sigma)

#     return {
#         "Wasserstein": wd_final,
#         "KL": kl_final,
#         "KS": ks_final,
#         "MMD": mmd_final
#     }


# ---------- Fonction principale ----------
def compute_representation_shift(
    features_src, features_tgt,
    bins=50, agg="mean", standardize= None,
    sigma=1.0,                      # conservé pour compat descendante (si tu gardes ton ancien MMD)
    mmd_sigmas=(0.5,1.0,2.0,4.0),   # MK-MMD
    sw_n_proj=128, cmd_K=5, jsd_bits=True,
    sample_max=3000, seed=0,
    return_per_dim=False
):
    """
    Calcule un panel de métriques de shift *post-hoc* sur des features:
      - 1D moyennées: Wasserstein, KL, KS, JSD
      - Multivariées: FID, CORAL, Sliced-Wasserstein, Energy Distance, MK-MMD, CMD

    standardize: None | 'zscore' | 'minmax' (recommandé: 'zscore')
    agg: 'mean' | 'median' pour les métriques 1D par dimension
    """
    X, Y = _standardize_pair(np.asarray(features_src), np.asarray(features_tgt), mode=standardize)

    # --- 1D (par dimension) ---
    wd_list, kl_list, ks_list, js_list = [], [], [], []
    eps = 1e-10
    D = X.shape[1]
    for d in range(D):
        x, y = X[:, d], Y[:, d]
        # Wasserstein 1D
        wd_list.append(wasserstein_distance(x, y))
        # Histogrammes communs pour KL/JSD
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
            p, _ = np.histogram(x, bins=bins, range=(lo, hi), density=True)
            q, _ = np.histogram(y, bins=bins, range=(lo, hi), density=True)
            p = (p + eps) / (p.sum() + eps*len(p))
            q = (q + eps) / (q.sum() + eps*len(q))
            kl_list.append(entropy(p, q))  # KL(P||Q)
            m = 0.5*(p + q)
            js = 0.5*entropy(p, m) + 0.5*entropy(q, m)
            if jsd_bits: js /= np.log(2.0)
            js_list.append(js)
        else:
            kl_list.append(0.0)
            js_list.append(0.0)
        # KS
        ks_stat, _ = ks_2samp(x, y)
        ks_list.append(ks_stat)

    agg_fun = np.mean if agg == "mean" else np.median
    wd_final = agg_fun(wd_list) if wd_list else float('nan')
    kl_final = agg_fun(kl_list) if kl_list else float('nan')
    ks_final = agg_fun(ks_list) if ks_list else float('nan')
    jsd_final = agg_fun(js_list) if js_list else float('nan')

    # --- Multivariées ---
    fid_final   = fid_distance(X, Y)
    coral_final = coral_distance(X, Y, normalize=True)
    swd_final   = sliced_wasserstein(X, Y, n_proj=sw_n_proj, seed=seed)
    ed_final    = energy_distance(X, Y, sample_max=sample_max, seed=seed)
    mmd_final   = mk_mmd(X, Y, sigmas=mmd_sigmas, sample_max=sample_max, seed=seed)
    cmd_final   = cmd_distance(X, Y, K=cmd_K)

    out = {
        "Wasserstein_1D": float(wd_final),
        "KL_1D":          float(kl_final),
        "KS_1D":          float(ks_final),
        "JSD_1D":         float(jsd_final),
        "CMD_K{}".format(cmd_K): float(cmd_final),
        "CORAL_norm":     float(coral_final),
        "FID":            float(fid_final),
        "SWD_proj{}".format(sw_n_proj): float(swd_final),
        "EnergyDist":     float(ed_final),
        "MK-MMD":         float(mmd_final)
    }
    if return_per_dim:
        out["per_dim"] = {
            "Wasserstein_1D": wd_list,
            "KL_1D": kl_list,
            "KS_1D": ks_list,
            "JSD_1D": js_list,
        }
    return out






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


def class_separability_index(features, targets):
    """
    Compute class separability index J = tr(SB) / tr(SW)
    for binary (or multi-class) case, following Duda et al. (2000).
    features: np.ndarray [N, d]
    targets: np.ndarray [N]
    """
    features = np.array(features)
    targets = np.array(targets)
    classes = np.unique(targets)

    # Global mean
    m = features.mean(axis=0)

    # Within-class scatter SW
    SW = np.zeros((features.shape[1], features.shape[1]))
    SB = np.zeros_like(SW)

    for c in classes:
        Xc = features[targets == c]
        mc = Xc.mean(axis=0)
        # scatter intra-classe
        SW += ((Xc - mc).T @ (Xc - mc))
        # scatter inter-classe (pondéré par n_c)
        nc = Xc.shape[0]
        diff = (mc - m).reshape(-1,1)
        SB += nc * (diff @ diff.T)

    # Class separability index
    J = np.trace(SB) / (np.trace(SW) + 1e-8)
    return float(J)

def separability_indices(features, targets):
    res = {}
    if len(np.unique(targets)) > 1 and len(targets) > 10:
        res["J"] = class_separability_index(features, targets)
        res["silhouette"] = float(silhouette_score(features, targets))
        res["davies_bouldin"] = float(davies_bouldin_score(features, targets))
    else:
        res["J"], res["silhouette"], res["davies_bouldin"] = None, None, None
    return res



def get_features(exp_path, sf_uda_source_folder,image_ids_to_draw,image_ids_to_draw_target, checkpoint_type, source_dataset,target_dataset, cudaid, split, tmp_outd='tmp_outd', parsedargs=None, save_results= None):


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
        if parsedargs.source_model_name != constants.METHOD_PIXELCAM:
            args_dict['pixel_wise_classification'] = False
        else:
            args_dict['pixel_wise_classification'] = True
        #args_dict['pixel_wise_classification'] = True
        args_dict['multiple_layer_pixel_classifier'] = False
        args_dict['anchors_ortogonal'] = False
        args_dict['detach_pixel_classifier'] = False
        args_dict['batch_norm_pixel_classifier'] = False
        args_dict['one_layer_pixel_classifier'] = False
        args_dict['cpt_cam_entropy'] = False
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
    assert split == constants.TESTSET or split == constants.CLVALIDSET or split == constants.TRAINSET
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

    if parsedargs.target_dataset == constants.CAMELYON17_512:
        fold_num = parsedargs.fold_cam17
    else:
        fold_num = args.fold
    target_metadata_root = join('./folds/wsol-done-right-splits', target_dataset, f"fold-{fold_num}")
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
            loader=source_loaders[split],
            metadata_root=os.path.join(source_metadata_root, split),
            mask_root=args.mask_root,
            iou_threshold_list=args.iou_threshold_list,
            dataset_name=source_dataset,
            split= split,
            cam_curve_interval=args.cam_curve_interval,
            multi_contour_eval=args.multi_contour_eval,
            out_folder=args.outd,
        )




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
    
        mask_root_data = os.path.join(
        os.path.dirname(args.mask_root),  # -> /export/livia/home/vision/Aguichemerre/datasets
        target_dataset               # -> CAMELYON17
    )
        

        target_cam_computer = CAMComputer(
                args=deepcopy(args),
                model=model,
                loader=target_loaders[split],
                metadata_root=os.path.join(target_metadata_root, split),
                mask_root=mask_root_data,
                iou_threshold_list=args.iou_threshold_list,
                dataset_name=target_dataset,
                split= split,
                cam_curve_interval=args.cam_curve_interval,
                multi_contour_eval=args.multi_contour_eval,
                out_folder=args.outd,
            )



    
    out_dir = "plots_energy_test"
    os.makedirs(out_dir, exist_ok=True)
    #plot_hist_energy_based_on_target_image_acc(source_energy, out_dir, save_path="tmp", target_dataset=source_dataset)




    #plot_logits(source_energy['max_logits_img'], target_energy['max_logits_img'], out_dir, title="Logits Distribution for images", type = "images_max", xlim=(-5, 5), label_src="Source", label_tgt="Target", source_dataset=source_dataset, target_dataset=target_dataset, source_model_name = source_model_name, target_model_name = target_model_name, external_pixel_classifier=external_pixel_classifier)
    
    
    if parsedargs.external_model is not None:
        external_pixel_classifier = True
    else:
        external_pixel_classifier = None


    source_model_name = parsedargs.source_model_name
    target_model_name = parsedargs.target_model_name


    # target_loaders = get_data_loader(
    #     data_roots=target_domain_data_paths,
    #     metadata_root=target_metadata_root,
    #     batch_size=32,#args.batch_size,
    #     workers=args.num_workers,
    #     resize_size=args.resize_size,
    #     crop_size=args.crop_size,
    #     proxy_training_set=args.proxy_training_set,
    #     num_val_sample_per_class=args.num_val_sample_per_class,
    #     std_cams_folder=args.std_cams_folder,
    #     # distributed_eval=False,
    #     get_splits_eval=['test'],
    #     #constants.TRAINSET
    #     eval_batch_size = 32#args.eval_batch_size,
    # )

    model.eval()

    features_src, targets_src = _extract_features(args,model, source_loaders[split])
    features_tgt, targets_tgt = _extract_features(args,model, target_loaders[split])


    Sep_source = separability_indices(features_src, targets_src)
    Sep_target = separability_indices(features_tgt, targets_tgt)

    shifts = compute_representation_shift(features_src, features_tgt, bins=50, sigma=1.0)
    print(shifts)


    results = _compute_accuracy(args, model, target_loaders[split])
    print(results)

    #print(f"Classification accuracy on target dataset {target_dataset} is {cl_global:.2f}%")
    #print(f"Classification accuracy on target dataset {target_dataset} for normal class is {cl_normal:.2f}%")
    #print(f"Classification accuracy on target dataset {target_dataset} for cancer class is {cl_cancer:.2f}%")

    #plot_energy_for_source_images(source_energy['pixels'], out_dir='plots_energy', title="Energy Distribution", source_dataset=source_dataset, target_dataset=target_dataset, source_model_name = source_model_name, target_model_name = target_model_name, external_pixel_classifier=external_pixel_classifier)

    cam_performance = target_cam_computer.compute_and_evaluate_cams()

    if save_results != None:
        # Build dictionnary
        metrics = {
            "shifts": shifts,
            "results": results,
            "cam_performance": cam_performance,
            "separability": {
                "source": {"J": Sep_source},
                "target": {"J": Sep_target}
    }
        }

        # Save in txt
        metrics_file = os.path.join(save_results, "metrics.txt")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"✅ Fichier de métriques sauvegardé : {metrics_file}")

    print(cam_performance)

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
    parser.add_argument("--results_out", type=str, default="results",
                    help="save information")
    parser.add_argument("--fold_cam17", type=int, default=None,
                    help="fold")
    parser.add_argument("--fold_source", type=int, default=None,
                    help="fold")

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

    # # Construire le chemin hiérarchique
    # if parsedargs.target_dataset == constants.CAMELYON17_512:
    #     results_dir = os.path.join(
    #         parsedargs.results_out,
    #         parsedargs.split,        # dossier racine (ex. "results")
    #         parsedargs.source_dataset,     # source dataset
    #         parsedargs.source_model_name,  # nom du modèle source
    #         parsedargs.target_dataset,
    #         str(parsedargs.fold_cam17)
    #     )
    # else:
    #     if parsedargs.source_dataset == constants.CAMELYON17_512:
    #         results_dir = os.path.join(
    #         parsedargs.results_out,        # dossier racine (ex. "results")
    #         parsedargs.split,
    #         parsedargs.source_dataset,
    #         str(parsedargs.fold_source),
    #         parsedargs.source_model_name,  # nom du modèle source
    #         parsedargs.target_dataset
    #         )
    #     else:
    #         results_dir = os.path.join(
    #             parsedargs.results_out,        # dossier racine (ex. "results")
    #             parsedargs.source_dataset,     # source dataset
    #             parsedargs.source_model_name,  # nom du modèle source
    #             parsedargs.target_dataset      # target dataset
    #     )
            
    if parsedargs.source_dataset == constants.CAMELYON17_512:
            if parsedargs.target_dataset == constants.CAMELYON17_512:
                results_dir = os.path.join(
                    parsedargs.results_out,
                    parsedargs.split,        # dossier racine (ex. "results")
                    parsedargs.source_dataset,     # source dataset
                    str(parsedargs.fold_source),
                    parsedargs.source_model_name,  # nom du modèle source
                    parsedargs.target_dataset,
                    str(parsedargs.fold_cam17)
                )
            else:
                results_dir = os.path.join(
                    parsedargs.results_out,        # dossier racine (ex. "results")
                    parsedargs.split,
                    parsedargs.source_dataset,
                    str(parsedargs.fold_source),
                    parsedargs.source_model_name,  # nom du modèle source
                    parsedargs.target_dataset
                    )
    else:
        if parsedargs.target_dataset == constants.CAMELYON17_512:
                results_dir = os.path.join(
                    parsedargs.results_out,
                    parsedargs.split,        # dossier racine (ex. "results")
                    parsedargs.source_dataset,
                    parsedargs.source_model_name,  # nom du modèle source
                    parsedargs.target_dataset,
                    str(parsedargs.fold_cam17)
                )
        else:    
            results_dir = os.path.join(
                parsedargs.results_out,        # dossier racine (ex. "results")
                parsedargs.split,
                parsedargs.source_dataset,     # source dataset
                parsedargs.source_model_name,  # nom du modèle source
                parsedargs.target_dataset      # target dataset
            )



    os.makedirs(results_dir, exist_ok=True)
        
    for checkpoint_type_extended in base_checkpoint_types:
        checkpoint_type = checkpoint_type_extended
        
        split = parsedargs.split
        #assert split == constants.TESTSET or split == constants.CLVALIDSET or split == constants.TRAINSET
        
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
            get_features(exp_path=exp_path, sf_uda_source_folder=parsedargs.path_pre_trained_source,image_ids_to_draw=parsedargs.image_ids_to_draw,image_ids_to_draw_target=parsedargs.image_ids_to_draw_target, checkpoint_type=checkpoint_type, source_dataset=parsedargs.source_dataset,target_dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs, save_results=results_dir)

if __name__ == '__main__':
    fast_eval()