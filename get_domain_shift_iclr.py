# -*- coding: utf-8 -*-
"""ONE-FILE post-hoc SFDA feature diagnostics for SAT and PixelCAM.

Launch with the same arguments as the previous feature-shift script, plus
optional --output_dir, --skip_tsne, --max_diag_samples.

Calculates source/target shift, class separation, kNN purity, kNN majority-vote
accuracy using true neighbor labels and predicted neighbor pseudo-labels, error-detection
AUROC/AP/precision-at-rho, pseudo-prototype score and image-level target t-SNE.

Scientific protocol:
  * model is in eval mode; no adaptation/training is performed here;
  * target labels are used solely for post-hoc evaluation/visualization;
  * all quantitative distances and kNN are in original, L2-normalized space;
  * t-SNE is qualitative; do NOT compare distances on separate 2D embeddings;
  * neighborhood disagreement and prototype margin are generic proxies, not
    full implementations of NRC, FAUST, RGV, or ensemble uncertainty.

Project-local dlib.* imports remain necessary because they load your model and
wsol datasets. No separate metrics/diagnostic Python script is needed.
"""
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
import json
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


import inspect
import json

from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_samples,
    roc_auc_score,
)


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


# Imports for embedded domain metrics and diagnostic functions.
import inspect
from typing import Dict, Iterable, Mapping, Optional, Sequence
from scipy.stats import entropy, ks_2samp, wasserstein_distance
from scipy.spatial.distance import cdist
from scipy import linalg
from sklearn.metrics import average_precision_score, roc_auc_score


# ======================================================================
# PART 1 — ORIGINAL DOMAIN SHIFT METRICS (source vs target)
# Directly embedded from the user's da_metrics.py functions.
# No local da_metrics.py or separate diagnostics script required.
# ======================================================================

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
            kl_list.append(0.0)
            js_list.append(0.0)
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
    scale = mx - mn
    scale[scale < eps] = 1.0
    Xn = 2 * (X - mn) / scale - 1
    Yn = 2 * (Y - mn) / scale - 1

    mu_x, mu_y = Xn.mean(0), Yn.mean(0)
    s = np.linalg.norm(mu_x - mu_y)
    for k in range(2, K+1):
        s += np.linalg.norm(np.mean((Xn - mu_x)**k, 0) -
                            np.mean((Yn - mu_y)**k, 0))
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
    ed2 = 2 * d_st - m_s - m_t
    return float(np.sqrt(max(ed2, 0)))


def mk_mmd(X, Y, sigmas=(0.5, 1.0, 2.0, 4.0), sample_max=3000, seed=0):
    rng = np.random.default_rng(seed)
    if X.shape[0] > sample_max:
        X = X[rng.choice(len(X), sample_max, replace=False)]
    if Y.shape[0] > sample_max:
        Y = Y[rng.choice(len(Y), sample_max, replace=False)]

    def _rbf(DX, DY):
        AA = (DX ** 2).sum(1)[:, None]
        BB = (DY ** 2).sum(1)[None, :]
        return AA + BB - 2 * DX @ DY.T

    d2_xx = _rbf(X, X)
    d2_yy = _rbf(Y, Y)
    d2_xy = _rbf(X, Y)

    Kxx = sum(np.exp(-d2_xx / (2 * s * s)) for s in sigmas)
    Kyy = sum(np.exp(-d2_yy / (2 * s * s)) for s in sigmas)
    Kxy = sum(np.exp(-d2_xy / (2 * s * s)) for s in sigmas)

    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    m, n = len(X), len(Y)
    mmd2 = Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2 * Kxy.mean()
    return float(np.sqrt(max(mmd2, 0)))


class FeatureShiftCalculator:
    """Original feature-shift calculator; source/target pooled z-scoring."""
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
        X = np.vstack([self.feature_source[k] for k in range(self.num_filters)]).T
        Y = np.vstack([self.feature_target[k] for k in range(self.num_filters)]).T
        if len(X) < 2 or len(Y) < 2:
            raise ValueError("Need at least 2 source and 2 target images for domain metrics")
        X, Y = _standardize_pair(X, Y, mode=standardize)
        one_d = compute_1d_metrics(X, Y, bins=bins)
        multi = {
            "CORAL": coral_distance(X, Y),
            "CMD": cmd_distance(X, Y),
            "FID": fid_distance(X, Y),
            "SWD": sliced_wasserstein(X, Y),
            "Energy": energy_distance(X, Y),
            "MK-MMD": mk_mmd(X, Y),
        }
        return {**one_d, **multi}


# ======================================================================
# PART 2 — TARGET CLASS SEPARATION / ERROR RELIABILITY / t-SNE
# ======================================================================

def _features(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("features must have shape (N, D), N >= 2. Do not pass spatial feature maps.")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite features.")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero/near-zero feature vectors: cosine geometry is undefined.")
    return x / norms


def _inputs(features, labels, probabilities):
    z = _features(features)
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != len(z):
        raise ValueError("labels must have shape (N,).")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must be integer class indices.")
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or len(p) != len(z) or p.shape[1] < 2:
        raise ValueError("probabilities must have shape (N, C), C >= 2 (softmax, not logits).")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Invalid probabilities.")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-4, rtol=0):
        raise ValueError("Probability rows must sum to one.")
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("Label indices must match the order of probability columns (0,...,C-1).")
    return z, y.astype(np.int64, copy=False), p / p.sum(axis=1, keepdims=True)


def _top_fraction(errors, scores, rho):
    """Expected precision under uniform tie-breaking at the selection boundary."""
    n = len(errors)
    m = max(1, int(np.ceil(rho * n)))
    threshold = np.partition(scores, n - m)[n - m]
    above = scores > threshold
    tied = scores == threshold
    hits = float(errors[above].sum())
    hits += (m - int(above.sum())) * float(errors[tied].mean())
    return {
        "n_selected": m,
        "effective_fraction": float(m / n),
        "precision": float(hits / m),
        "recall": float(hits / errors.sum()) if errors.sum() else None,
        "boundary_tie_count": int(tied.sum()),
    }


def _evaluate_scores(errors, scores, rhos):
    valid = len(np.unique(errors)) == 2
    return {
        "n_samples": len(errors),
        "n_errors": int(errors.sum()),
        "error_prevalence": float(errors.mean()),
        "has_correct_and_incorrect_samples": bool(valid),
        "scores": {
            name: {
                "auroc_error": float(roc_auc_score(errors, s)) if valid else None,
                "ap_error": float(average_precision_score(errors, s)) if valid else None,
                "top_fraction": {str(rho): _top_fraction(errors, s, rho) for rho in rhos},
            }
            for name, s in scores.items()
        },
    }


def analyze_target_features(
    features: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    ks: Iterable[int] = (3, 5, 10),
    rhos: Iterable[float] = (0.1, 0.2, 0.4),
    sample_ids: Optional[Sequence[str]] = None,
    group_ids: Optional[Sequence[str]] = None,
    exclude_same_group: bool = False,
    extra_scores: Optional[Mapping[str, np.ndarray]] = None,
    seed: int = 0,
    max_samples: int = 5000,
) -> Dict:
    """Return JSON-serializable geometry and error-detection diagnostics.

    sample_ids: optional unique image IDs, to catch duplicate extraction.
    group_ids: optional actual WSI/patient identifiers; never infer them here.
    exclude_same_group: excludes pairs from the same group from BOTH neighborhood
        and inter/intra distance diagnostics. Run as a sensitivity analysis in
        addition to the default self-exclusion-only protocol.
    extra_scores: optional label-free scores computed elsewhere; larger MUST
        mean more likely erroneous. Can contain view disagreement or a
        pseudo-label/prototype score. True-label-derived scores are invalid here.
    ks: fixed in advance; invalid k is rejected rather than silently reduced.

    Macro purity averages class-wise purity equally. The random-neighbor
    baseline uses each query's eligible class proportions, excluding itself.
    Inter/intra uses cosine distances averaged equally across classes/class pairs.
    No metrics are computed in the t-SNE embedding.
    """
    z, y, p = _inputs(features, labels, probabilities)
    # Silhouette score on original L2-normalized features
    if len(np.unique(y)) >= 2 and len(np.unique(y)) < len(y):
        silhouette_values = silhouette_samples(
            z, y, metric="cosine"
        )

        silhouette = float(silhouette_values.mean())

        silhouette_by_class = {
            str(int(cls)): float(silhouette_values[y == cls].mean())
            for cls in np.unique(y)
        }
    else:
        silhouette = None
        silhouette_by_class = {}

    n, dim = z.shape
    c = p.shape[1]
    if n > max_samples:
        raise ValueError(f"N={n} exceeds max_samples={max_samples}; use a declared subset or a chunked implementation.")
    raw_ks = tuple(ks)
    if not raw_ks or any(not isinstance(k, (int, np.integer)) or k < 1 for k in raw_ks):
        raise ValueError("ks must contain positive integers.")
    ks = tuple(sorted(set(int(k) for k in raw_ks)))
    rhos = tuple(float(r) for r in rhos)
    if not rhos or any(not 0 < r <= 1 for r in rhos):
        raise ValueError("rhos must be in (0, 1].")
    if sample_ids is not None:
        ids = np.asarray(sample_ids, dtype=str)
        if ids.shape != (n,) or len(np.unique(ids)) != n:
            raise ValueError("sample_ids must have length N and be unique.")

    # Unit-vector cosine distances; clip floating-point roundoff.
    distance = np.clip(1.0 - z @ z.T, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    eligible = ~np.eye(n, dtype=bool)
    if exclude_same_group:
        if group_ids is None:
            raise ValueError("Actual group_ids are required to exclude same-group pairs.")
        groups = np.asarray(group_ids, dtype=str)
        if groups.shape != (n,):
            raise ValueError("group_ids must have length N.")
        eligible &= groups[:, None] != groups[None, :]
    if int(eligible.sum(axis=1).min()) < max(ks):
        raise ValueError("Not enough eligible neighbors for the largest requested k.")

    true_classes = np.unique(y)
    warnings = []
    if len(true_classes) < c:
        warnings.append("Some classifier classes are absent from the diagnostic labels; macro results use observed classes only.")
    within = {}
    between = {}
    for a in true_classes:
        ia = np.flatnonzero(y == a)
        block = np.ix_(ia, ia)
        vals = distance[block][eligible[block]]
        within[str(int(a))] = float(vals.mean()) if vals.size else None
        for b in true_classes[true_classes > a]:
            ib = np.flatnonzero(y == b)
            block = np.ix_(ia, ib)
            vals = distance[block][eligible[block]]
            between[f"{int(a)}-{int(b)}"] = float(vals.mean()) if vals.size else None
    intra = (float(np.mean(list(within.values())))
             if within and all(v is not None for v in within.values()) else None)
    inter = (float(np.mean(list(between.values())))
             if between and all(v is not None for v in between.values()) else None)
    ratio = float(inter / intra) if inter is not None and intra is not None and intra > 1e-12 else None

    # Random tie order independent of labels; do not privilege loader class order.
    ranking_distances = np.where(eligible, distance, np.inf)
    perm = np.random.default_rng(seed).permutation(n)
    neighbors = perm[np.argsort(ranking_distances[:, perm], axis=1, kind="stable")[:, :max(ks)]]
    same_true_class = y[:, None] == y[None, :]
    chance = (eligible & same_true_class).sum(axis=1) / eligible.sum(axis=1)
    chance_by_class = {str(int(a)): float(chance[y == a].mean()) for a in true_classes}
    chance_macro = float(np.mean(list(chance_by_class.values())))
    yhat = p.argmax(axis=1)
    errors = (yhat != y).astype(np.int64)
    scores = {"entropy": -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1) / np.log(c)}
    purity = {}
    knn_accuracy = {}

    def _vote_accuracy(neighbor_classes):
        """Majority-vote neighbor labels, breaking ties toward the smallest class ID."""
        classes = np.unique(np.concatenate((y, neighbor_classes.reshape(-1))))
        counts = (neighbor_classes[:, :, None] == classes[None, None, :]).sum(axis=1)
        votes = classes[counts.argmax(axis=1)]
        correct = (votes == y)
        per_class = {
            str(int(a)): float(correct[y == a].mean()) for a in true_classes
        }
        return {
            "micro": float(correct.mean()),
            "macro": float(np.mean(list(per_class.values()))),
            "by_true_class": per_class,
        }

    for k in ks:
        nbr = neighbors[:, :k]
        individual_purity = (y[nbr] == y[:, None]).mean(axis=1)
        by_class = {str(int(a)): float(individual_purity[y == a].mean()) for a in true_classes}
        macro = float(np.mean(list(by_class.values())))
        purity[str(k)] = {
            "micro": float(individual_purity.mean()),
            "macro": macro,
            "by_true_class": by_class,
            "macro_minus_random": macro - chance_macro,
        }
        # Two distinct post-hoc questions:
        # 1. True-label majority vote assesses the feature neighborhood geometry.
        # 2. Predicted-label majority vote assesses whether the source model's
        #    pseudo-labels on those neighbors can recover the query's true label.
        # Both are evaluated against ground truth, but only (1) uses ground
        # truth to construct the neighbor vote (so neither is an SFDA training metric).
        knn_accuracy[str(k)] = {
            "true_neighbor_label_vote": _vote_accuracy(y[nbr]),
            "pseudo_neighbor_label_vote": _vote_accuracy(yhat[nbr]),
        }
        # Crucially: these scores use PREDICTIONS, never ground-truth labels.
        scores[f"knn_hard_disagreement_k{k}"] = (yhat[nbr] != yhat[:, None]).mean(axis=1)
        q = p[nbr].mean(axis=1)
        scores[f"knn_soft_disagreement_k{k}"] = 1.0 - q[np.arange(n), yhat]
    if extra_scores is not None:
        for name, values in extra_scores.items():
            if not isinstance(name, str) or name in scores:
                raise ValueError(f"Invalid/duplicate extra-score name: {name!r}")
            s = np.asarray(values, dtype=np.float64)
            if s.shape != (n,) or not np.isfinite(s).all():
                raise ValueError(f"extra_scores[{name!r}] must be a finite vector of length N.")
            scores[name] = s
    pred_counts = np.bincount(yhat, minlength=c)
    by_prediction = {}
    for a in np.unique(yhat):
        keep = yhat == a
        by_prediction[str(int(a))] = _evaluate_scores(
            errors[keep], {key: val[keep] for key, val in scores.items()}, rhos
        )
    return {
        "protocol": {
            "n_samples": n, "feature_dim": dim, "n_classifier_classes": c,
            "distance": "cosine after row-wise L2 normalization",
            "self_excluded": True, "same_group_excluded": bool(exclude_same_group),
            "ks": list(ks), "rhos": list(rhos), "neighbor_tie_seed": int(seed),
            "score_direction": "larger means more likely erroneous",
            "precision_ties": "expected precision under uniform tie-breaking",
            "knn_vote_ties": "smallest class ID wins majority-vote ties",
            "labels_role": "post-hoc geometry/evaluation only, not detector construction",
            "warnings": warnings,
        },
        "classification": {
            "accuracy": float((yhat == y).mean()),
            "true_class_counts": {str(int(a)): int((y == a).sum()) for a in true_classes},
            "predicted_class_counts": {str(a): int(pred_counts[a]) for a in range(c)},
            "dominant_predicted_classes": [int(a) for a in np.flatnonzero(pred_counts / n > 1.0 / c)],
        },
        "separation": {
            "intra_by_class": within,
            "inter_by_class_pair": between,
            "intra_macro": intra,
            "inter_macro": inter,
            "inter_over_intra": ratio,
            "silhouette": silhouette,
            "silhouette_by_class": silhouette_by_class,
        },
        "knn_purity": purity,
        "knn_accuracy": knn_accuracy,
        "random_neighbor_purity": {
            "micro": float(chance.mean()), "macro": chance_macro, "by_true_class": chance_by_class,
        },
        "error_detection": {
            "all": _evaluate_scores(errors, scores, rhos),
            "by_predicted_class": by_prediction,
        },
    }



def pseudo_prototype_margin(
    features: np.ndarray,
    probabilities: np.ndarray,
) -> Optional[np.ndarray]:
    """Label-free, leave-one-out pseudo-prototype inconsistency score.

    Each target representation is compared with the pseudo-class prototype
    formed from OTHER samples predicted as that class, and with the nearest
    competing pseudo-class prototype. Larger scores = less reliable.

    Returns None if <2 pseudo-classes are present or any predicted class is
    represented by <2 samples. This is NOT the original RGV/prototype score.
    """
    z = _features(features)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != z.shape[0]:
        raise ValueError("probabilities must have shape (N, C).")
    yhat = p.argmax(axis=1)
    classes, counts = np.unique(yhat, return_counts=True)
    if len(classes) < 2 or np.any(counts < 2):
        return None

    sums = {int(c): z[yhat == c].sum(axis=0) for c in classes}
    count_map = {int(c): int(n) for c, n in zip(classes, counts)}
    other_centroids = {}
    for c in classes:
        centroid = sums[int(c)] / count_map[int(c)]
        norm = np.linalg.norm(centroid)
        if norm < 1e-12:
            return None
        other_centroids[int(c)] = centroid / norm

    result = np.empty(len(z), dtype=np.float64)
    for i, c in enumerate(yhat):
        c = int(c)
        own_loo = (sums[c] - z[i]) / (count_map[c] - 1)
        norm = np.linalg.norm(own_loo)
        if norm < 1e-12:
            return None
        own_loo /= norm
        d_own = 1.0 - np.dot(z[i], own_loo)
        d_other = min(1.0 - np.dot(z[i], other_centroids[other])
                      for other in other_centroids if other != c)
        result[i] = d_own - d_other
    return result

def save_target_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_path: str,
    *,
    perplexity: float = 30.0,
    seed: int = 0,
    class_names: Optional[Mapping[int, str]] = None,
    title: str = "Target image representations",
) -> Dict:
    """Save one qualitative PNG plus coordinates; never use these for metrics.

    Run separately for each backbone on exactly the same images, with the same
    settings. Axes and inter-cluster distances are not comparable between plots.
    No learned alignment between representations is attempted.
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    z, y, p = _inputs(features, labels, probabilities)
    if not 0 < perplexity < len(z):
        raise ValueError("t-SNE requires 0 < perplexity < N.")
    if np.max(np.var(z, axis=0)) < 1e-15:
        raise ValueError("All representations are identical; a t-SNE layout would be uninformative.")
    params = dict(n_components=2, metric="cosine", perplexity=float(perplexity),
                  init="random", random_state=seed, learning_rate= 20, n_jobs=-1)
    iteration_parameter = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    params[iteration_parameter] = 300
    embedding = TSNE(**params).fit_transform(z)
    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise ValueError("output_path must end in .png.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for a in np.unique(y):
        mask = y == a
        label = str(int(a)) if class_names is None else class_names.get(int(a), str(int(a)))
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=18, alpha=0.65, label=label)
    error_mask = p.argmax(axis=1) != y
    if error_mask.any():
        ax.scatter(embedding[error_mask, 0], embedding[error_mask, 1],
                   marker="x", s=25, linewidths=0.8, label="Misclassified")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    coordinates_path = path.with_suffix(".coordinates.npz")
    np.savez_compressed(coordinates_path, embedding=embedding, labels=y,
                        predictions=p.argmax(axis=1))
    return {"plot": str(path), "coordinates": str(coordinates_path), "parameters": params}

# ======================================================================
# PART 3 — YOUR EXISTING MODEL/DATA PIPELINE
# ======================================================================

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



def save_target_diagnostics(
    features,
    labels,
    probabilities,
    output_prefix,
    seed=42,
    perplexity=30,
    max_samples=3000,
):
    """
    Target feature diagnostics for SAT / PixelCAM.

    Ground-truth labels are only used for post-hoc evaluation.
    Features: (N, D), from model.lin_ft
    Labels: (N,), 0 = Normal, 1 = Cancer
    Probabilities: (N, C), softmax outputs
    """

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64)

    assert X.ndim == 2
    assert len(X) == len(y) == len(p)
    assert set(np.unique(y)) == {0, 1}
    assert np.isfinite(X).all()
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-4)

    n_total = len(X)
    rng = np.random.default_rng(seed)

    # Fixed random subsample if the dataset is large.
    if n_total > max_samples:
        idx = np.sort(
            rng.choice(n_total, max_samples, replace=False)
        )
        X, y, p = X[idx], y[idx], p[idx]

    # Important: all distance metrics use original features.
    # L2 normalization makes cosine geometry comparable.
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        raise ValueError("Found zero-length feature vectors.")

    X = normalize(X, norm="l2")
    n = len(X)

    if min(np.sum(y == 0), np.sum(y == 1)) < 2:
        raise ValueError("Need at least 2 samples per class.")

    # ====================================================
    # 1. INTRA / INTER CLASS DISTANCES
    # ====================================================

    X_normal = X[y == 0]
    X_cancer = X[y == 1]

    D_normal = cdist(
        X_normal, X_normal, metric="cosine"
    )
    D_cancer = cdist(
        X_cancer, X_cancer, metric="cosine"
    )

    # Exclude self-distances on the diagonal.
    mask_normal = ~np.eye(len(X_normal), dtype=bool)
    mask_cancer = ~np.eye(len(X_cancer), dtype=bool)

    intra_normal = float(
        D_normal[mask_normal].mean()
    )
    intra_cancer = float(
        D_cancer[mask_cancer].mean()
    )

    intra_macro = 0.5 * (
        intra_normal + intra_cancer
    )

    inter = float(
        cdist(
            X_normal, X_cancer, metric="cosine"
        ).mean()
    )

    inter_intra_ratio = (
        inter / intra_macro
        if intra_macro > 1e-12 else None
    )

    # ====================================================
    # 2. SILHOUETTE SCORE
    # ====================================================

    silhouette_values = silhouette_samples(
        X, y, metric="cosine"
    )

    silhouette = float(silhouette_values.mean())

    silhouette_normal = float(
        silhouette_values[y == 0].mean()
    )
    silhouette_cancer = float(
        silhouette_values[y == 1].mean()
    )

    # ====================================================
    # 3. KNN PURITY
    # ====================================================

    ks = [2, 5, 10]
    max_k = max(ks)

    if n <= max_k:
        raise ValueError("Not enough samples for kNN@10.")

    nn = NearestNeighbors(
        n_neighbors=max_k + 1,
        metric="cosine",
        algorithm="brute",
    )

    nn.fit(X)

    neighbor_indices = nn.kneighbors(
        X, return_distance=False
    )

    # Exclude each sample from its own neighborhood.
    neighbor_indices = np.stack([
        row[row != i][:max_k]
        for i, row in enumerate(neighbor_indices)
    ])

    knn_results = {}

    for k in ks:

        neighbors = neighbor_indices[:, :k]

        matches = (
            y[neighbors] == y[:, None]
        ).mean(axis=1)

        purity_normal = float(
            matches[y == 0].mean()
        )
        purity_cancer = float(
            matches[y == 1].mean()
        )

        knn_results[str(k)] = {
            "micro": float(matches.mean()),
            "macro": float(
                0.5 * (purity_normal + purity_cancer)
            ),
            "normal": purity_normal,
            "cancer": purity_cancer,
        }

    # ====================================================
    # 4. ERROR DETECTION
    # ====================================================

    predictions = p.argmax(axis=1)
    errors = (predictions != y).astype(int)

    # Normalized prediction entropy.
    entropy_score = -(
        p * np.log(np.clip(p, 1e-12, 1.0))
    ).sum(axis=1) / np.log(p.shape[1])

    # Neighborhood prediction disagreement.
    knn_disagreement = (
        predictions[neighbor_indices[:, :5]]
        != predictions[:, None]
    ).mean(axis=1)

    detection_results = {}

    for name, score in {
        "entropy": entropy_score,
        "knn_disagreement": knn_disagreement,
    }.items():

        auroc = None

        if len(np.unique(errors)) == 2:
            auroc = float(
                roc_auc_score(errors, score)
            )

        detection_results[name] = {
            "error_auroc": auroc
        }

    # ====================================================
    # 5. TSNE VISUALIZATION
    # ====================================================

    params = dict(
        n_components=2,
        perplexity=min(float(perplexity), n - 1.0),
        metric="cosine",
        init="random",
        learning_rate=200.0,
        random_state=seed,
    )

    iter_key = (
        "max_iter"
        if "max_iter" in inspect.signature(TSNE).parameters
        else "n_iter"
    )

    params[iter_key] = 1500

    embedded = TSNE(**params).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))

    for cls, name, color in [
        (0, "Normal", "#2878B5"),
        (1, "Cancer", "#D95D3E"),
    ]:

        mask = y == cls

        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            c=color,
            s=18,
            alpha=0.65,
            label=f"{name} (n={mask.sum()})",
            rasterized=True,
        )

    # Overlay classification errors.
    if np.any(errors):

        ax.scatter(
            embedded[errors == 1, 0],
            embedded[errors == 1, 1],
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            s=40,
            label="Misclassified",
        )

    ax.set_title("Target feature representation")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False)

    fig.tight_layout()

    plt.savefig(
        output_prefix + "_tsne.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ====================================================
    # 6. SAVE METRICS
    # ====================================================

    results = {
        "n_total": int(n_total),
        "n_analyzed": int(n),
        "class_separability": {
            "intra_normal": intra_normal,
            "intra_cancer": intra_cancer,
            "intra_macro": intra_macro,
            "inter": inter,
            "inter_over_intra": inter_intra_ratio,
            "silhouette": silhouette,
            "silhouette_normal": silhouette_normal,
            "silhouette_cancer": silhouette_cancer,
        },
        "knn_purity": knn_results,
        "error_detection": detection_results,
    }

    with open(
        output_prefix + "_diagnostics.json", "w"
    ) as f:
        json.dump(results, f, indent=4)

    print("\n===== TARGET FEATURE DIAGNOSTICS =====")
    print(f"Silhouette: {silhouette:.4f}")
    print(f"Inter/Intra: {inter_intra_ratio:.4f}")

    for k, value in knn_results.items():
        print(
            f"kNN purity@{k}: "
            f"{value['macro']:.4f}"
        )

    return results



def get_features(exp_path, sf_uda_source_folder,image_ids_to_draw,image_ids_to_draw_target, checkpoint_type, dataset, cudaid, split, tmp_outd='tmp_outd', parsedargs=None, target_method=None):


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


    if 'PixelCAM' in target_method:
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

    elif target_method == 'NEGEV':
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
    
    source_metadata_root = join(constants.RELATIVE_META_ROOT, parsedargs.source_dataset, f"fold-{parsedargs.fold_src_dataset}")
    #read sys var DATASETSH
    args_dict['data_root'] = os.path.join(os.environ['DATASETSH'], 'datasets')
    source_domain_data_paths = config.configure_data_paths(args_dict, parsedargs.source_dataset)

    target_metadata_root = join('./folds/wsol-done-right-splits', parsedargs.target_dataset, f"fold-{parsedargs.fold_trg_dataset}")
    #args_dict['data_root'] = '/export/gauss/vision/Aguichemerre/datasets'
    target_domain_data_paths = config.configure_data_paths(args_dict, parsedargs.target_dataset)

    # loaders = get_data_loader(
    #         data_roots=target_domain_data_paths,
    #         metadata_root=metadata_root,
    #         batch_size=32,#args.batch_size,
    #         workers=args.num_workers,
    #         resize_size=args.resize_size,
    #         crop_size=args.crop_size,
    #         proxy_training_set=args.proxy_training_set,
    #         num_val_sample_per_class=args.num_val_sample_per_class,
    #         std_cams_folder=args.std_cams_folder,
    #         # distributed_eval=False,
    #         get_splits_eval=['test'],
    #         #constants.TRAINSET
    #         eval_batch_size = 32#args.eval_batch_size,
    #     )
    
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
            get_splits_eval=[parsedargs.split],
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
            get_splits_eval=[parsedargs.split],
            #constants.TRAINSET
            eval_batch_size = 32#args.eval_batch_size,
        )
    

    # x = torch.randn((1, 2048, 28, 28), device=device, requires_grad=True)

    # step_size = 1.0
    # noise_scale = 0.01
    # num_steps = 100
    # plot_interval = 1 
    # for step in range(num_steps):
    #     logits, _ = model.pixel_wise_classification_head(x) 
    #     energy = -torch.logsumexp(logits, dim=1)
        
    #     # Calcul du gradient par rapport à x
    #     grad_x = torch.autograd.grad(energy, x, grad_outputs=torch.ones_like(energy), create_graph=False)[0]

    #     # Mise à jour avec SGLD
    #     noise = torch.randn_like(x) * noise_scale
    #     x = x - step_size * grad_x + noise
    #     x = x.detach().clone().requires_grad_(True)
    #     vectors = x.detach().cpu().numpy()
        
    #     # Affichage TSNE tous les 'plot_interval' itérations
    #     if step % plot_interval == 0 or step == num_steps - 1:
    #         output_dir = os.path.join('visualization', 'energy', dataset, 'sgld')
    #         os.makedirs(output_dir, exist_ok=True)

    #         weights = model.pixel_wise_classification_head.conv4.weight.data.cpu().numpy()
    #         weights = weights.reshape(weights.shape[0], -1)

    #         batch_size, vector_dim, height, width = vectors.shape
    #         vectors_reshaped = vectors.reshape(batch_size, vector_dim, height * width).transpose(0, 2, 1)  
    #         vectors_flattened = vectors_reshaped.reshape(height * width, vector_dim)  

    #         combined_data = np.vstack([vectors_flattened, weights])

    #         tsne = TSNE(n_components=2, perplexity=1, random_state=42, init="pca")

    #         reduced_vectors = tsne.fit_transform(weights)

    #         plt.figure(figsize=(8, 6))
    #         plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.6, edgecolors='k')
    #         plt.title(f"t-SNE à l'itération {step}")
    #         plt.xlabel("TSNE Dim 1")
    #         plt.ylabel("TSNE Dim 2")
    #         plt.grid(True)
    #         plt.savefig(f"tsne_step_{step}.png", dpi=300)
    #         plt.close()

    #         reduced_vectors = tsne.fit_transform(combined_data) 
            
    #         generated_points = reduced_vectors[:-weights.shape[0], :]
    #         weight_points = reduced_vectors[-weights.shape[0]:, :]

    #         plt.figure(figsize=(8, 6))
    #         plt.scatter(generated_points[:, 0], generated_points[:, 1], alpha=0.6, edgecolors='k', label="Generated Samples")

    #         # Afficher le premier poids avec un triangle blanc
    #         plt.scatter(weight_points[0, 0], weight_points[0, 1], marker="^", s=200, color="white", edgecolors='black', label="First Weight (White Triangle)")

    #         # Afficher le deuxième poids avec un triangle noir
    #         plt.scatter(weight_points[1, 0], weight_points[1, 1], marker="v", s=200, color="black", edgecolors='black', label="Second Weight (Black Triangle)")

    #         plt.title(f"t-SNE à l'itération {step}")
    #         plt.xlabel("TSNE Dim 1")
    #         plt.ylabel("TSNE Dim 2")
    #         plt.grid(True)
    #         save_path = os.path.join(output_dir, f"tsne_step_{step}.png")
    #         plt.savefig(save_path, dpi=300)
    #         plt.close()
        

    # cam_computer = CAMComputer(
    #         args=deepcopy(args),
    #         model=model,
    #         loader=loaders['test'],
    #         metadata_root=os.path.join(metadata_root, 'test'),
    #         mask_root=args.mask_root,
    #         iou_threshold_list=args.iou_threshold_list,
    #         dataset_name=args.dataset,
    #         split= 'test',
    #         cam_curve_interval=args.cam_curve_interval,
    #         multi_contour_eval=args.multi_contour_eval,
    #         out_folder=args.outd,
    #     )   
    overlay_images = {}
    input_images = {}
    gt_masks = {}

    if "deit" in encoder_name:
        num_filters = 192
    else:
        num_filters = 2048

    # Preserve the original source-vs-target shift calculation.
    domain_shift_measure = FeatureShiftCalculator(num_filters=num_filters)

    # Collect image-level target representations and predictions for a strictly
    # POST-HOC diagnostic. Labels must never be used to train, select or adapt.
    target_feature_batches = []
    target_label_batches = []
    target_probability_batches = []

    for batch_idx, (images, targets, p_glabel, index, raw_imgs, std_cams, _, views, _) in tqdm(
        enumerate(source_loaders[split]), ncols=constants.NCOLS,
        total=len(source_loaders[split])):
        images = images.to(device)
        with torch.no_grad():
            _ = model(images)
            src_img_features = model.lin_ft
            if src_img_features.ndim != 2 or src_img_features.shape[1] != num_filters:
                raise ValueError(f"Source image features expected (B,{num_filters}), "
                                 f"got {tuple(src_img_features.shape)}")
            domain_shift_measure.accumulate(src_img_features.detach(), domain='source')

    for batch_idx, (images, targets, p_glabel, index, raw_imgs, std_cams, _, views, _) in tqdm(
        enumerate(target_loaders[split]), ncols=constants.NCOLS,
        total=len(target_loaders[split])):
        images = images.to(device)
        with torch.no_grad():
            out = model(images)
            trg_img_features = model.lin_ft
            if args.task == constants.STD_CL:
                cl_logits = out
            elif args.task == constants.F_CL:
                cl_logits = out[0]
            else:
                raise NotImplementedError(f"Unsupported task: {args.task}")
            if trg_img_features.ndim != 2 or trg_img_features.shape[1] != num_filters:
                raise ValueError(f"Target image features expected (B,{num_filters}), "
                                 f"got {tuple(trg_img_features.shape)}")
            if cl_logits.ndim != 2 or cl_logits.shape[0] != len(targets):
                raise ValueError(f"Expected classification logits (B,C), got {tuple(cl_logits.shape)}")
            domain_shift_measure.accumulate(trg_img_features.detach(), domain='target')
            target_feature_batches.append(trg_img_features.detach().float().cpu().numpy())
            target_label_batches.append(targets.detach().long().cpu().numpy())
            target_probability_batches.append(
                F.softmax(cl_logits.float(), dim=1).detach().cpu().numpy()
            )


    ############################################################
    # === Build output directory structure ===
    ############################################################
    root_dir = parsedargs.output_dir

    dataset_dir = os.path.join(root_dir, parsedargs.source_dataset, str(parsedargs.fold_src_dataset),parsedargs.wsol_method)
    os.makedirs(dataset_dir, exist_ok=True)



    ############################################################
    # === Build output filename dynamically ===
    ############################################################

    filename = f"{parsedargs.target_dataset}__fold{parsedargs.fold_trg_dataset}__{method_name}__{parsedargs.sfda_method}__{parsedargs.split}.json"
    output_path = os.path.join(dataset_dir, filename)
    
    
    shift_score = domain_shift_measure.domain_shift()

    # Class separability and true error-detection analysis in target train images.
    target_features = np.concatenate(target_feature_batches, axis=0)
    target_labels = np.concatenate(target_label_batches, axis=0)
    target_probabilities = np.concatenate(target_probability_batches, axis=0)

    # Optional unsupervised pseudo-prototype reliability signal. No GT used.
    prototype_score = pseudo_prototype_margin(target_features, target_probabilities)
    extra_scores = ({"pseudo_prototype_margin_loo": prototype_score}
                    if prototype_score is not None else None)

    diagnostics = analyze_target_features(
        target_features,
        target_labels,
        target_probabilities,
        ks=(3, 5, 10),
        rhos=(0.1, 0.2, 0.4),
        extra_scores=extra_scores,
        seed=0,
        max_samples=parsedargs.max_diag_samples,
    )
    diagnostics["protocol"]["feature_source"] = "model.lin_ft, checkpoint specified by --path_pre_trained_source"
    diagnostics["protocol"]["checkpoint_path"] = str(path_cl)
    diagnostics["protocol"]["split"] = str(split)
    diagnostics["protocol"]["notes"] = (
        "Target labels used post-hoc ONLY to evaluate feature geometry and "
        "error detection; model and scores are label-free. Pseudo-prototype "
        "margin is a proxy, NOT an implementation of RGV."
    )

    diag_path = output_path.replace(".json", "__target_diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2, allow_nan=False)

    if not parsedargs.skip_tsne:
        tsne_path = output_path.replace(".json", "__target_tsne.png")
        save_target_tsne(
            target_features, target_labels, target_probabilities, tsne_path,
            class_names={0: "Normal", 1: "Cancer"},
            title=f"{parsedargs.wsol_method} | {parsedargs.target_dataset} fold "
                  f"{parsedargs.fold_trg_dataset} | {split}",
            seed=0,
            perplexity=min(5.0, (len(target_features)-1.0)/3.0),
        )
    np.savez_compressed(
        output_path.replace(".json", "__target_features.npz"),
        features=target_features,
        labels=target_labels,
        probabilities=target_probabilities,
    )
    print(f"Post-hoc target diagnostics saved to: {diag_path}")
    if not parsedargs.skip_tsne:
        print(f"Target t-SNE saved to: {tsne_path}")


    ############################################################
    # === Save individual JSON ===
    ############################################################
    with open(output_path, "w") as f:
        json.dump(shift_score, f, indent=4)


    ############################################################
    # === Append result to global TXT file ===
    ############################################################
    txt_output_path = os.path.join(
        root_dir,
        "domain_shift_results.txt"
    )

    with open(txt_output_path, "a") as f:

        f.write("*" * 80 + "\n")

        f.write(
            f"Source dataset: {parsedargs.source_dataset} "
            f"| Fold: {parsedargs.fold_src_dataset}\n"
        )

        f.write(
            f"Target dataset: {parsedargs.target_dataset} "
            f"| Fold: {parsedargs.fold_trg_dataset}\n"
        )

        f.write(f"WSOL method: {parsedargs.wsol_method}\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"SFDA method: {parsedargs.sfda_method}\n")
        f.write(f"Checkpoint: {parsedargs.checkpoint_type}\n")
        f.write(f"Split: {parsedargs.split}\n")

        f.write("\n")

        f.write("Domain shift metrics:\n")

        if isinstance(shift_score, dict):
            for metric_name, metric_value in shift_score.items():
                f.write(f"{metric_name}: {metric_value}\n")
        else:
            f.write(f"{shift_score}\n")

        f.write("*" * 80 + "\n\n")


    print(f"JSON saved to: {output_path}")
    print(f"TXT results appended to: {txt_output_path}")

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
    parser.add_argument('--image_ids_to_draw', nargs='+', type=str, default=None)
    parser.add_argument('--image_ids_to_draw_target', nargs='+', type=str, default=None)
    parser.add_argument("--tmp_outd", type=str, default='tmp_outd')
    parser.add_argument('--noise_level_for_eval_with_noisy_bbox', nargs='+',
                        type=int, default=[5, 10, 15, 20, 25, 30, 35 ,40, 45, 50])
    parser.add_argument("--fold_src_dataset", type=int, default=0, help="fold.")
    parser.add_argument("--fold_trg_dataset", type=int, default=0, help="fold.")
    parser.add_argument("--target_dataset", type=str, default=None,
                       help="Name of the dataset.")
    parser.add_argument("--source_dataset", type=str, default=None, help="Source dataset")
    #parser.add_argument("--path_pre_trained_source", type=str, default=None, help="Path to the pre-trained source model.")
    parser.add_argument("--path_pre_trained_source", type=str, default=None, help="Path to the pre-trained source model.")
    parser.add_argument("--sfda_method", type=str, default=0, help="fold.")
    parser.add_argument("--wsol_method", type=str, default='wsol', help="fold.")
    

    parser.add_argument(
        "--output_dir", type=str,
        default="/export/livia/home/vision/Aguichemerre/iclr_2027_results/metrics_domain_shift",
        help="Output folder for domain shift and target diagnostics (override on Nibi/FIR).",
    )
    parser.add_argument(
        "--skip_tsne", action="store_true",
        help="Skip t-SNE to speed up repeated runs; all other metrics remain enabled.",
    )
    parser.add_argument(
        "--max_diag_samples", type=int, default=5000,
        help="Safety limit for O(N^2) target diagnostics; for larger N use batches/chunked code.",
    )

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
