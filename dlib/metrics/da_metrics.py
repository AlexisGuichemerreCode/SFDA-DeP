import os
import time
from copy import deepcopy
import sys
from os.path import dirname, abspath, join
import threading
from copy import deepcopy
from typing import Optional, Union, Tuple

import cv2
import numpy as np

import torch.utils.data as torchdata
import torch
import torch.nn.functional as F
from scipy.stats import entropy, ks_2samp, wasserstein_distance
from scipy.spatial.distance import cdist
from scipy import linalg



root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


from dlib.datasets.wsol_data_core import get_image_ids
from dlib.datasets.wsol_data_core import get_bounding_boxes
from dlib.datasets.wsol_data_core import get_image_sizes
from dlib.datasets.wsol_data_core import get_mask_paths
from dlib.datasets.wsol_data_core import get_mask
from dlib.datasets.wsol_data_core import RESIZE_LENGTH

from dlib.utils.tools import check_scoremap_validity
from dlib.utils.tools import check_box_convention

#from scipy.stats import wasserstein_distance, ks_2samp


# -----------------------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------------------
def _cov(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    denom = max(1, (Xc.shape[0] - 1))
    return (Xc.T @ Xc) / denom


def _standardize_pair(X, Y, mode="zscore", eps=1e-12):
    if mode is None:
        return X, Y
    XY = np.vstack([X, Y])
    if mode == "zscore":
        mu = XY.mean(0, keepdims=True)
        sd = XY.std(0, keepdims=True) + eps
        return (X - mu) / sd, (Y - mu) / sd
    elif mode == "minmax":
        mn = XY.min(0, keepdims=True)
        mx = XY.max(0, keepdims=True)
        scale = mx - mn
        scale[scale < eps] = 1.0
        return (X - mn) / scale, (Y - mn) / scale
    else:
        raise ValueError("standardize must be None | 'zscore' | 'minmax'")


# -----------------------------------------------------------------------------------------
# Metrics 1D: Wasserstein, KL, KS, JSD
# -----------------------------------------------------------------------------------------
def compute_1d_metrics(X, Y, bins=50):
    eps = 1e-10
    D = X.shape[1]
    wd_list, kl_list, ks_list, js_list = [], [], [], []

    for d in range(D):
        x, y = X[:, d], Y[:, d]

        wd_list.append(wasserstein_distance(x, y))
        ks_list.append(ks_2samp(x, y)[0])

        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        if lo == hi:
            kl_list.append(0.0); js_list.append(0.0)
            continue

        p, _ = np.histogram(x, bins=bins, range=(lo, hi), density=True)
        q, _ = np.histogram(y, bins=bins, range=(lo, hi), density=True)

        p = (p + eps) / (p.sum() + eps * len(p))
        q = (q + eps) / (q.sum() + eps * len(q))
        m = 0.5 * (p + q)

        kl_list.append(entropy(p, q))
        js_list.append((0.5 * entropy(p, m) + 0.5 * entropy(q, m)) / np.log(2))

    return {
        "Wasserstein_1D": float(np.mean(wd_list)),
        "KL_1D": float(np.mean(kl_list)),
        "KS_1D": float(np.mean(ks_list)),
        "JSD_1D": float(np.mean(js_list)),
    }


# -----------------------------------------------------------------------------------------
# Multivariate: CORAL, CMD, FID, Sliced-Wasserstein, Energy distance
# -----------------------------------------------------------------------------------------
def coral_distance(X, Y, eps=1e-6):
    Cs = _cov(X) + eps * np.eye(X.shape[1])
    Ct = _cov(Y) + eps * np.eye(X.shape[1])
    diff = Cs - Ct
    num = np.sqrt((diff * diff).sum())
    den = np.sqrt((Cs * Cs).sum()) + np.sqrt((Ct * Ct).sum()) + 1e-12
    return float(num / den)


def cmd_distance(X, Y, K=5, eps=1e-12):
    XY = np.vstack([X, Y])
    mn, mx = XY.min(0), XY.max(0)
    scale = mx - mn; scale[scale < eps] = 1.0
    Xn = 2*(X - mn)/scale - 1
    Yn = 2*(Y - mn)/scale - 1

    mu_x, mu_y = Xn.mean(0), Yn.mean(0)
    s = np.linalg.norm(mu_x - mu_y)
    for k in range(2, K+1):
        s += np.linalg.norm(np.mean((Xn - mu_x)**k, 0) - np.mean((Yn - mu_y)**k, 0))
    return float(s)


def fid_distance(X, Y, eps=1e-6):
    mu_x, mu_y = X.mean(0), Y.mean(0)
    Cx = _cov(X) + eps * np.eye(X.shape[1])
    Cy = _cov(Y) + eps * np.eye(X.shape[1])
    covmean, _ = linalg.sqrtm(Cx @ Cy, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu_x - mu_y)**2).sum() + np.trace(Cx + Cy - 2 * covmean))


def sliced_wasserstein(X, Y, n_proj=128, seed=0):
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    sw = 0.0
    for _ in range(n_proj):
        u = rng.normal(size=D)
        u /= (np.linalg.norm(u) + 1e-12)
        sw += wasserstein_distance(X @ u, Y @ u)
    return float(sw / n_proj)


def energy_distance(X, Y, sample_max=3000, seed=0):
    rng = np.random.default_rng(seed)
    if X.shape[0] > sample_max:
        X = X[rng.choice(len(X), sample_max, replace=False)]
    if Y.shape[0] > sample_max:
        Y = Y[rng.choice(len(Y), sample_max, replace=False)]
    d_st = cdist(X, Y).mean()
    d_ss = cdist(X, X)
    d_tt = cdist(Y, Y)
    m_s = d_ss[np.triu_indices_from(d_ss, 1)].mean()
    m_t = d_tt[np.triu_indices_from(d_tt, 1)].mean()
    ed2 = 2*d_st - m_s - m_t
    return float(np.sqrt(max(ed2, 0)))


# -----------------------------------------------------------------------------------------
# MK-MMD
# -----------------------------------------------------------------------------------------
def mk_mmd(X, Y, sigmas=(0.5,1.0,2.0,4.0), sample_max=3000, seed=0):
    rng = np.random.default_rng(seed)
    if X.shape[0] > sample_max:
        X = X[rng.choice(len(X), sample_max, replace=False)]
    if Y.shape[0] > sample_max:
        Y = Y[rng.choice(len(Y), sample_max, replace=False)]

    def _rbf(DX, DY):
        AA = (DX**2).sum(1)[:,None]
        BB = (DY**2).sum(1)[None,:]
        return AA + BB - 2*DX@DY.T

    d2_xx = _rbf(X, X)
    d2_yy = _rbf(Y, Y)
    d2_xy = _rbf(X, Y)

    Kxx = sum([np.exp(-d2_xx/(2*s*s)) for s in sigmas])
    Kyy = sum([np.exp(-d2_yy/(2*s*s)) for s in sigmas])
    Kxy = sum([np.exp(-d2_xy/(2*s*s)) for s in sigmas])

    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    m, n = len(X), len(Y)
    mmd2 = Kxx.sum()/(m*(m-1)) + Kyy.sum()/(n*(n-1)) - 2*Kxy.mean()
    return float(np.sqrt(max(mmd2, 0)))


# ==========================================================================================
# -------------------------  ✨  FeatureShiftCalculator (Final)  ✨ --------------------------
# ==========================================================================================

class FeatureShiftCalculator:
    def __init__(self, num_filters=2048):
        self.num_filters = num_filters
        self.feature_source = {k: [] for k in range(num_filters)}
        self.feature_target = {k: [] for k in range(num_filters)}

    def accumulate(self, features, domain):
        for k in range(self.num_filters):
            vals = features[:, k].cpu().tolist()
            if domain == "source":
                self.feature_source[k].extend(vals)
            elif domain == "target":
                self.feature_target[k].extend(vals)
            else:
                raise ValueError("domain must be 'source' or 'target'")

    def domain_shift(self, standardize="zscore", bins=50):
        # Convert to matrices (N × D)
        X = np.vstack([self.feature_source[k] for k in range(self.num_filters)]).T
        Y = np.vstack([self.feature_target[k] for k in range(self.num_filters)]).T

        # Standardization
        X, Y = _standardize_pair(X, Y, mode=standardize)

        # 1D metrics
        one_d = compute_1d_metrics(X, Y, bins=bins)

        # Multivariate metrics
        multi = {
            "CORAL": coral_distance(X, Y),
            "CMD": cmd_distance(X, Y),
            "FID": fid_distance(X, Y),
            "SWD": sliced_wasserstein(X, Y),
            "Energy": energy_distance(X, Y),
            "MK-MMD": mk_mmd(X, Y),
        }

        return {**one_d, **multi}