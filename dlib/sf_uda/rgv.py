import torch
import os
import torch.nn.functional as F
from tqdm import tqdm
from math import ceil
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor
from typing import Dict, Iterable, Callable

from dlib.datasets.wsol_loader import get_data_loader
from dlib.configure import constants
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg







__all__ = ['Rgv']

def to_cuda(x):
    return x.cuda()

# ----------------------------------------------------------
#        RGV: Revisiting Source-Free Domain Adaptation
#        Zhu et al., CVPR 2025
# ----------------------------------------------------------


class MemoryBank:
    def __init__(self, feat_dim, logit_dim, max_size=10000, device='cuda'):
        self.device = device
        self.max_size = max_size
        self.feats = torch.zeros((0, feat_dim)).to(device)
        self.logits = torch.zeros((0, logit_dim)).to(device)
        self.names = []

    def update(self, feats, logits, names):
        feats = feats.detach().cuda(self.device)
        logits = logits.detach().cuda(self.device)
        self.feats = torch.cat([self.feats, feats.detach()], dim=0)
        self.logits = torch.cat([self.logits, logits.detach()], dim=0)
        self.names.extend(names)
        if len(self.names) > self.max_size:
            self.feats = self.feats[-self.max_size:]
            self.logits = self.logits[-self.max_size:]
            self.names = self.names[-self.max_size:]

    def get_knn(self, query_feats, k=5, tau=0.07):
        query_norm = F.normalize(query_feats, dim=1)
        memory_norm = F.normalize(self.feats, dim=1)
        sim = torch.mm(query_norm, memory_norm.T)
        topk_sim, idx = torch.topk(sim, k=k, dim=1)
        weights = F.softmax(topk_sim / tau, dim=1)
        neighbor_logits = self.logits[idx]  # (B, k, C)
        refined = torch.bmm(weights.unsqueeze(1), neighbor_logits).squeeze(1)
        return F.softmax(refined, dim=1)
    


class Rgv(object):
    def __init__(self, model_trg, train_loader_trg, num_classes, device= None,
                 sigma=0.2, topk_ratio=0.1):
        self.model = model_trg
        self.train_loader_trg = train_loader_trg
        self.device = device
        self.num_classes = num_classes
        self.sigma = sigma
        self.topk_ratio = topk_ratio
        self.memory = MemoryBank(feat_dim=2048, logit_dim=self.num_classes, device=self.device)


    # ------------------------------------------------------
    # Extract features & logits
    # ------------------------------------------------------
    @torch.no_grad()
    def extract_features(self):
        self.model.eval()
        feats, logits, img_names = [], [], []
        for images, targets, p_glabel, index, raw_imgs, std_cams, masks, views, aug_images in tqdm(self.train_loader_trg, desc="Extracting features"):
            imgs = images.cuda(self.device)
            output = self.model(imgs)
            f = self.model.lin_ft
            l = output
            feats.append(f.cpu())
            logits.append(l.cpu())
            img_names.extend(list(index))
        feats = torch.cat(feats, 0)
        logits = torch.cat(logits, 0)
        probs = F.softmax(logits, dim=1)
        return feats, logits, probs, img_names

    # ------------------------------------------------------
    # C-sampling (classifier-based)
    # ------------------------------------------------------
    def c_sampling(self, probs):
        N, C = probs.shape
        k = int(max(1, self.topk_ratio * N / C))
        selected = []
        for c in range(C):
            idx = torch.topk(probs[:, c], k=k, largest=True).indices
            selected.extend(idx.tolist())
        selected = list(set(selected))
        pseudo = probs[selected].argmax(dim=1)
        return selected, pseudo

    # ------------------------------------------------------
    # T-sampling (target center-based)
    # ------------------------------------------------------
    def t_sampling(self, features, probs):
        N, D = features.size()
        C = probs.size(1)
        mu = torch.zeros((C, D))
        for c in range(C):
            conf = probs[:, c]
            k = int(max(1, self.sigma * N))
            idx = torch.topk(conf, k=k, largest=True).indices
            w = conf[idx].unsqueeze(1)
            mu[c] = (w * features[idx]).sum(0) / (w.sum() + 1e-8)
        mu = F.normalize(mu, dim=1)

        feats_norm = F.normalize(features, dim=1)
        cos_sim = torch.mm(feats_norm, mu.T)

        selected = []
        k_per_c = int(max(1, self.topk_ratio * N / C))
        for c in range(C):
            idx = torch.topk(cos_sim[:, c], k=k_per_c, largest=True).indices
            selected.extend(idx.tolist())
        selected = list(set(selected))
        pseudo = cos_sim[selected].argmax(dim=1)
        return selected, pseudo, mu

    # ------------------------------------------------------
    # I-sampling (intersection)
    # ------------------------------------------------------
    def i_sampling(self, idx_c, idx_t):
        return list(set(idx_c).intersection(set(idx_t)))

    # ------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------
    @torch.no_grad()
    def run(self):
        feats, logits, probs, img_names = self.extract_features()
        self.memory.update(feats, probs, img_names)
        idx_c, pseudo_c = self.c_sampling(probs)
        idx_t, pseudo_t, mu = self.t_sampling(feats, probs)
        idx_i = self.i_sampling(idx_c, idx_t)
        pseudo_i = probs[idx_i].argmax(dim=1)

        print(f"[RGV] Selected samples: C={len(idx_c)}, T={len(idx_t)}, I={len(idx_i)}")

        D_samples = {
            'C': {'names': [img_names[i] for i in idx_c], 'labels': pseudo_c},
            'T': {'names': [img_names[i] for i in idx_t], 'labels': pseudo_t, 'centers': mu},
            'I': {'names': [img_names[i] for i in idx_i], 'labels': pseudo_i}
        }

        self.D_maps = {
            k: {n: l.item() for n, l in zip(v['names'], v['labels'])}
            for k, v in D_samples.items()
        }
        return D_samples
