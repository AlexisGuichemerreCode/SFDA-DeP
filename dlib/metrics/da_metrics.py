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

from scipy.stats import wasserstein_distance, ks_2samp


class FeatureShiftCalculator:
    def __init__(self, num_filters=2048, bins=50, epsilon=1e-3):

        self.bins = bins
        self.epsilon = epsilon
        self.num_filters = num_filters

        # Stocker les features par filtre
        self.feature_source = {k: [] for k in range(num_filters)}
        self.feature_target = {k: [] for k in range(num_filters)}

    def accumulate(self, features, domain):

        for k in range(self.num_filters):
            if domain == "source":
                self.feature_source[k].extend(features[:, k].cpu().tolist()) 
            elif domain == "target":
                self.feature_target[k].extend(features[:, k].cpu().tolist())  
            else:
                raise ValueError("Mode must be 'source' or 'target'")
            
    def domain_shift(self):
        """
        Compute shift metrics :
        - Wasserstein Distance
        - KL Divergence 
        - Kolmogorov-Smirnov (KS) statistic
        
        Returns:
            dict: Average values for each metric.
        """
        wasserstein_distances = []
        kl_div_per_filter = []
        ks_statistics = []

        for k in range(self.num_filters):
            source_vals = torch.tensor(self.feature_source[k])
            target_vals = torch.tensor(self.feature_target[k])

            if len(source_vals) == 0 or len(target_vals) == 0:
                continue 

            # Wasserstein Distance
            wasserstein_distances.append(wasserstein_distance(source_vals.numpy(), target_vals.numpy()))

            # Kolmogorov-Smirnov (KS) statistic
            ks_stat, _ = ks_2samp(source_vals.numpy(), target_vals.numpy())
            ks_statistics.append(ks_stat)


        # Return average distance
        return {
            "wasserstein": sum(wasserstein_distances) / len(wasserstein_distances) if wasserstein_distances else 0.0,
            "ks": sum(ks_statistics) / len(ks_statistics) if ks_statistics else 0.0
        }