import os
import sys
import time
from os.path import dirname, abspath, join
from typing import Optional, Union, Tuple
from copy import deepcopy
import pickle as pkl
import csv
import math
import datetime as dt
import random as py_random

import numpy as np
import torch
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yaml
import torch.nn.functional as F

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from torch import nn, Tensor
from typing import Dict, Iterable, Callable

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.datasets.wsol_loader import get_data_loader
from dlib.datasets.wsol_loader import get_eval_transforms_global

from dlib.utils.reproducibility import set_seed
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.tools import get_cpu_device
from dlib.utils.tools import get_tag

from dlib.cams.tcam_seeding import TCAMSeeder

from dlib.cams import selflearning
from dlib.learning.inference_wsol import CAMComputer
from dlib.cams import build_std_cam_extractor

from dlib.div_classifiers.parts.has import has as wsol_has
from dlib.div_classifiers.parts.cutmix import cutmix as wsol_cutmix

from dlib.sf_uda import Shot
from dlib.sf_uda import Faust
from dlib.sf_uda import Sdda
from dlib.sf_uda import adadsa
from dlib.sf_uda import Nrc
from dlib.sf_uda import Sfde
from dlib.sf_uda import Cdcl
from dlib.sf_uda import Rgv

from dlib import losses
from dlib.process.instantiators import get_loss
from dlib.process.instantiators import get_optimizer_of_model


__all__ = ['Basic', 'Trainer']


class PerformanceMeter(object):
    def __init__(self, split, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        val = constants.VALIDSET
        # self.value_per_epoch = [] \
        #     if split == val else [-np.inf if higher_is_better else np.inf]
        self.value_per_epoch = []

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        if len(self.value_per_epoch) > 1:
            idx = [i for i, x in enumerate(
                self.value_per_epoch) if x == self.best_value]
            if len(idx) > 0:
                self.best_epoch = idx[-1]
            else:
                self.best_epoch = 0  # issue: nan, inf.
        else:
            self.best_epoch = 0  # issue: nan. inf.


class PerfGistTracker(object):
    def __init__(self, split):
        self.value_per_epoch = dict()
        self.split = split

    def update(self, epoch: int, new_value: dict):
        assert epoch not in self.value_per_epoch
        self.value_per_epoch[epoch] = new_value

    def __getitem__(self, item):
        return self.value_per_epoch[item]


class Basic(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = (constants.TRAINSET, constants.PXVALIDSET, constants.CLVALIDSET,
               constants.TESTSET)
    _EVAL_METRICS = ['loss', constants.CLASSIFICATION_MTR,
                     constants.LOCALIZATION_MTR, constants.F1_MTR,
                     constants.PRECISION_MTR, constants.RECALL_MTR]

    _NUM_CLASSES_MAPPING = {
        constants.CUB: constants.NUMBER_CLASSES[constants.CUB],
        constants.ILSVRC: constants.NUMBER_CLASSES[constants.ILSVRC],
        constants.OpenImages: constants.NUMBER_CLASSES[constants.OpenImages],
        constants.GLAS: constants.NUMBER_CLASSES[constants.GLAS],
        constants.CAMELYON512: constants.NUMBER_CLASSES[constants.CAMELYON512],
        constants.CAMELYON17_512: constants.NUMBER_CLASSES[constants.CAMELYON17_512],
        constants.ICIAR: constants.NUMBER_CLASSES[constants.ICIAR],
        constants.BREAKHIS: constants.NUMBER_CLASSES[constants.BREAKHIS]
    }

    @property
    def _BEST_CRITERION_METRIC(self):
        assert self.inited
        assert self.args is not None

        if self.args.localization_avail:
            return constants.LOCALIZATION_MTR
        else:
            return constants.CLASSIFICATION_MTR

    def __init__(self, args):
        self.args = args

    def _set_perf_gist_tracker(self) -> dict:
        _dict = {
            split: PerfGistTracker(split) for split in self._SPLITS
        }
        return _dict

    def _set_performance_meters(self) -> dict:

        if self.bbox:
            self._EVAL_METRICS += ['localization_IOU_{}'.format(
                threshold) for threshold in self.args.iou_threshold_list]

            self._EVAL_METRICS += ['top1_loc_{}'.format(
                threshold) for threshold in self.args.iou_threshold_list]

            self._EVAL_METRICS += ['top5_loc_{}'.format(
                threshold) for threshold in self.args.iou_threshold_list]

        eval_dict = {
            split: {
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if metric == 'loss' else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict

class KLConsistencyMetrics:
    def __init__(self, device, tau=0.7, window=10, eps=1e-8):
        """
        Args:
            device: 'cuda' or 'cpu'
            tau: confidence threshold on current predictions
            window: sliding window size for normalization
            eps: epsilon to avoid division by zero
        """
        self.device = device
        self.tau = tau
        self.window = window
        self.eps = eps

        # Store raw KL values over time (sliding window)
        self.hist = []
        # Store normalized J_t values for monitoring/early stopping
        self.J_hist = []


        # Store marginal entropy history
        self.hm_hist = []       # raw Hm
        self.htilde_hist = []   # Htilde = log(K) - Hm

        

    @torch.no_grad()
    def compute_batch_kl(self, logits, indices, idx_to_pred):
        """
        Compute KL-consistency between current model predictions and
        source predictions (stored beforehand in idx_to_pred).

        Args:
            logits: [B, K] logits of the current model
            indices: global indices of the current batch samples
            idx_to_pred: dict {index: class predicted by the source model (argmax)}

        Returns:
            Mean KL value for this batch
        """
        p = F.softmax(logits, dim=-1)   # [B, K] current probabilities
        conf = p.max(dim=1).values      # confidence of current predictions

        kl_vals = []
        for i in range(len(indices)):
            idx = int(indices[i])
            if idx in idx_to_pred and conf[i] >= self.tau:
                # Source predicted label (hard argmax)
                y_src = idx_to_pred[idx]
                # Cross-entropy = KL divergence to one-hot source label
                ce = -torch.log(p[i, y_src] + 1e-8)
                kl_vals.append(ce)

        if len(kl_vals) > 0:
            kl = torch.stack(kl_vals).mean()
        else:
            kl = torch.tensor(0.0, device=self.device)

        # Update sliding window history
        self.hist.append(kl.item())
        if len(self.hist) > self.window:
            self.hist = self.hist[-self.window:]

        return kl.item()

    def _normalize(self):
        """
        Normalize the latest KL value within the range of the sliding window.
        Result is between [0,1].
        """
        vals = self.hist
        if len(vals) == 0:
            return 0.0
        v_t = vals[-1]
        vmin, vmax = min(vals), max(vals)
        return (v_t - vmin) / ((vmax - vmin) + self.eps)

    def compute_Jt(self):
        """
        Compute the normalized score J_t for KL (0 = best, 1 = worst).
        This acts like an "unsupervised validation loss".
        """
        J = self._normalize()
        self.J_hist.append(J)
        return J

    def should_stop(self, delta=5e-3, r=3):
        """
        Early stopping criterion based on J_t stability.

        Args:
            delta: minimum decrease required to consider an improvement
            r: number of consecutive epochs to monitor

        Returns:
            True if training should stop, False otherwise
        """
        if len(self.J_hist) < r+1:
            return False
        recent = self.J_hist[-(r+1):]
        # Stop if J_t has not decreased by at least delta in the last r epochs
        return (recent[-1] > (recent[0] - delta))

class Trainer(Basic):

    def __init__(self,
                 args,
                 model,
                 classifier=None,
                 model_src=None
                 ):
        super(Trainer, self).__init__(args=args)

        self.device = torch.device(args.c_cudaid)
        self.args = args

        self.bbox = args.dataset in [constants.CUB, constants.ILSVRC]

        self.performance_meters = self._set_performance_meters()
        self.perf_gist_tracker = self._set_perf_gist_tracker()
        self.model = model
        self.model_src = model_src

        self.loss: losses.MasterLoss = get_loss(args)

        optimizer, lr_scheduler = get_optimizer_of_model(args, model)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.load_tr_masks = args.task == constants.SEG
        self.load_tr_masks &= args.localization_avail
        mask_root = args.mask_root if self.load_tr_masks else ''
        self.mask_root = args.mask_root

        self.chg_staining = args.chg_staining
        self.path_staining = args.path_staining
        self.dist_staining = args.dist_staining

        self.batch_idx = None

        self.silhouette = []
        self.DBI = []
        self.CH = []
        self.J_index = []


        self.store_loss = []
        self.flipped_indices = []
        self.reinforce_indices = []
        self.entropy_all = []
        self.idx_to_pred = {}

        if args.entropy_models:
            self.entropy_models = args.entropy_models
            self.m_entropy_models = args.m_entropy_models

        if args.unlearning_models:
            self.unlearning_models = args.unlearning_models
            self.m_unlearning_models = args.m_unlearning_models
            self.best_unlearning_models = {}

        if args.cl_train_models:
            self.cl_train_models = args.cl_train_models

        
        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            eval_batch_size=self.args.eval_batch_size,
            workers=self.args.num_workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            load_tr_masks=self.load_tr_masks,
            mask_root=mask_root,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            std_cams_folder=self.args.std_cams_folder,
            sfuda_faust=self.args.faust,
            sfuda_n_rnd_views=self._get_faust_n_views(),
            sfda_aug_transform=self.args.sfda_aug_transform
            #chg_staining = self.chg_staining,
            #path_staining = self.path_staining,
            #dist_staining = self.dist_staining
        )

        self.loaders_notransform = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            eval_batch_size=self.args.eval_batch_size,
            workers=self.args.num_workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            load_tr_masks=self.load_tr_masks,
            mask_root=mask_root,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            std_cams_folder=self.args.std_cams_folder,
            sfuda_faust=self.args.faust,
            sfuda_n_rnd_views=self._get_faust_n_views(),
            #chg_staining = self.chg_staining,
            #path_staining = self.path_staining,
            #dist_staining = self.dist_staining
        )
        

        if self.args.target_domain_ds_to_compute_stats != None or self.args.ds_to_compute_acc_trainset_source_target != None:

            self.target_domain_loaders = get_data_loader(
                    data_roots=args.target_domain_data_paths,
                    metadata_root=self.args.target_domain_metadata_root,
                    batch_size=self.args.batch_size,
                    eval_batch_size=self.args.eval_batch_size,
                    workers=self.args.num_workers,
                    resize_size=self.args.resize_size,
                    crop_size=self.args.crop_size,
                    load_tr_masks=self.load_tr_masks,
                    mask_root=mask_root,
                    proxy_training_set=self.args.proxy_training_set,
                    num_val_sample_per_class=self.args.num_val_sample_per_class,
                    std_cams_folder=None,
                    sfuda_faust=self.args.faust,
                    sfuda_n_rnd_views=self._get_faust_n_views(),
                    get_splits_eval=["train", "valcl", "valpx","test"]
                )

            # self.target_domain_loaders = get_data_loader(
            #         data_roots=args.target_domain_data_paths,
            #         metadata_root=self.args.target_domain_metadata_root,
            #         batch_size=self.args.batch_size,
            #         eval_batch_size=self.args.eval_batch_size,
            #         workers=self.args.num_workers,
            #         resize_size=self.args.resize_size,
            #         crop_size=self.args.crop_size,
            #         load_tr_masks=self.load_tr_masks,
            #         mask_root=mask_root,
            #         proxy_training_set=self.args.proxy_training_set,
            #         num_val_sample_per_class=self.args.num_val_sample_per_class,
            #         std_cams_folder=None,
            #         get_splits_eval=[constants.TESTSET]
            #     )
            
            # self.source_domain_loaders = get_data_loader(
            #         data_roots=self.args.data_paths,
            #         metadata_root=self.args.source_domain_metadata_root,
            #         batch_size=self.args.batch_size,
            #         eval_batch_size=self.args.eval_batch_size,
            #         workers=self.args.num_workers,
            #         resize_size=self.args.resize_size,
            #         crop_size=self.args.crop_size,
            #         load_tr_masks=self.load_tr_masks,
            #         mask_root=mask_root,
            #         proxy_training_set=self.args.proxy_training_set,
            #         num_val_sample_per_class=self.args.num_val_sample_per_class,
            #         std_cams_folder=None,
            #         get_splits_eval=[constants.TESTSET]
            #     )
            
            # self.source_train_domain_loaders = get_data_loader(
            #         data_roots=self.args.data_paths,
            #         metadata_root=self.args.source_domain_metadata_root,
            #         batch_size=self.args.batch_size,
            #         eval_batch_size=self.args.eval_batch_size,
            #         workers=self.args.num_workers,
            #         resize_size=self.args.resize_size,
            #         crop_size=self.args.crop_size,
            #         load_tr_masks=self.load_tr_masks,
            #         mask_root=mask_root,
            #         proxy_training_set=self.args.proxy_training_set,
            #         num_val_sample_per_class=self.args.num_val_sample_per_class,
            #         std_cams_folder=None,
            #         get_splits_eval=[constants.TRAINSET]
            #     )
            
            self.target_train_acc_cl = []
            self.target_train_f1 = []
            self.target_train_precision = []
            self.target_train_recall = []
            self.target_train_image_entropy = []
            self.target_train_acc_normal = []
            self.target_train_acc_cancer = []


            self.target_valpx_pxap = []
            self.target_train_dice_bg = []
            self.target_train_dice_fg = []
            self.target_train_miou = []


            # self.target_train_acc_cl = []
            # self.target_test_acc_cl = []
            # self.source_test_acc_cl = []
            # self.source_train_acc_cl = []

            # self.target_test_image_entropy = []
            # self.source_test_image_entropy = []
            # self.source_train_image_entropy = []

            # self.target_test_pixel_entropy = []
            # self.source_test_pixel_entropy = []
            # self.source_train_pixel_entropy = []

            # self.target_valpx_pxap = []
            # self.target_test_pxap = []
            # self.source_test_pxap = []
            # self.source_train_pxap = []

            # self.target_train_dice_bg = []
            # self.target_test_dice_bg = []
            # self.source_test_dice_bg = []
            # self.source_train_dice_bg = []

            # self.target_train_dice_fg = []
            # self.target_test_dice_fg = []
            # self.source_test_dice_fg = []
            # self.source_train_dice_fg = []

            # self.target_train_miou = []
            # self.target_test_miou = []
            # self.source_test_miou = []
            # self.source_train_miou = []

        self.sl_mask_builder = None        
        if args.task in [constants.F_CL, constants.NEGEV] or args.pixel_wise_classification:
            self.sl_mask_builder = self._get_sl(args)

        if args.task == constants.TCAM:
            self.sl_mask_builder: TCAMSeeder = self._get_sl(args)

        self.epoch = 0
        self.counter = 0
        self.seed = int(args.MYSEED)
        self.default_seed = int(args.MYSEED)

        self.best_loc_model = deepcopy(self.model).to(self.cpu_device).eval()
        self.best_cl_model = deepcopy(self.model).to(self.cpu_device).eval()

        self.perf_meters_backup = None
        self.perf_gist_backup = None
        self.inited = True

        self.classifier = classifier
        self.std_cam_extractor = None
        self.tcam_extractor = None


        if args.task in [constants.F_CL, constants.NEGEV, constants.TCAM]:
            assert classifier is not None
            if args.sf_uda == False:
                self.std_cam_extractor = self._build_std_cam_extractor(
                classifier=classifier, args=args)

        self.tcam_extractor = None
        if args.task == constants.TCAM:
            self.tcam_extractor = self._build_tcam_extractor(
                classifier=classifier, args=self.args)


        if args.task in [constants.STD_CL]:
            if args.sf_uda == True and args.esfda == True:
                args_source = deepcopy(args)
                args_source.method = args.sf_uda_source_wsol_method

                self.std_cam_extractor = self._build_std_cam_extractor(
                    classifier=classifier,
                    args=args_source
                )



        self.fcam_argmax = False
        self.fcam_argmax_previous = False

        self.t_init_epoch = dt.datetime.now()
        self.t_end_epoch = dt.datetime.now()

        self.best_valid_tau_loc = None
        self.best_valid_tau_cl = None

        # SFUDA ================================================================
        self.sfuda_master = self.build_sfuda_master(self.args)

        if self.args.sf_uda:
            if self.args.shot or self.args.sfde or self.args.cdcl or self.args.pxsfde or self.args.esfda:
                self.pseudo_labels =[]

            if self.args.correct_pseudo_labels:
                self.corrected_pseudo_labels = self.select_images_to_correct(self.model, self.loaders[constants.TRAINSET], select_imgs_ratio=self.args.correct_pseudo_labels_ratio)

            if self.args.correct_and_incorrect_pseudo_labels:
                self.corrected_pseudo_labels = self.select_images_to_correct_and_incorrect(self.model, self.loaders[constants.TRAINSET], select_imgs_ratio=self.args.correct_pseudo_labels_ratio)


            if args.erl:
                self.index_src_logits = self.init_y_bar(model=self.model,loader=self.loaders[constants.TRAINSET])

        if self.args.sf_uda:

            #self.metrics = KLConsistencyMetrics(device="cuda")

            #self._sf_uda_before_epoch_process()

            if self.args.erl:
                self.update_y_bar_full(model=self.model,loader=self.loaders[constants.TRAINSET])


            # if self.args.esfda:
            #     if self.args.esfda_select_imgs:

            #         self.store_master_loss = []
            #         self.store_ce_flip_loss = []
            #         self.store_ce_not_flip_loss = []
            #         #self.loader.dataset.transform = None

            #         self.pred_distribution, self.overpred_class, self.underpred_class = self.compute_prediction_bias(model=self.model,
            #                     loader_notransform=self.loaders_notransform, top_k=None
            #                 )

            #         # select images to shift label

            #         if self.args.select_distance:
            #             self.flipped_indices, self.reinforce_indices, self.idx_to_pred = self.select_flippable_indices_distances(
            #                 model=self.model,
            #                 loader=self.loaders_notransform,
            #                 select_imgs_ratio=self.args.esfda_select_imgs_ratio,
            #                 random_select_ratio=self.args.random_select_ratio
            #             )

            #         if self.args.select_entropy:

            #             if self.args.entropy_probabilistic:
            #                 self.flipped_indices, self.reinforce_indices, self.idx_to_pred, self.entropy_all, self.all_selected, self.stable_selected, self.stable_labels = self.select_flippable_indices_entropy_probabilistic(
            #                     model=self.model,
            #                     loader=self.loaders_notransform
            #                 )
            #             elif self.args.entropy_gt:
            #                     self.flipped_indices, self.reinforce_indices, self.idx_to_pred, self.entropy_all, self.all_selected, self.stable_selected, self.stable_labels = self.select_flippable_indices_gt(
            #                         model=self.model,
            #                         loader=self.loaders_notransform
            #                     )

            #             elif self.args.entropy_random:
            #                 self.flipped_indices, self.reinforce_indices, self.idx_to_pred, self.entropy_all, self.all_selected, self.stable_selected, self.stable_labels = self.select_flippable_indices_random(
            #                     model=self.model,
            #                     loader=self.loaders_notransform
            #                 )
            #             else:
            #                 self.flipped_indices, self.reinforce_indices, self.idx_to_pred, self.entropy_all, self.all_selected, self.stable_selected, self.stable_labels = self.select_flippable_indices_entropy(
            #                     model=self.model,
            #                     loader=self.loaders_notransform,
            #                     select_imgs_ratio=self.args.esfda_select_imgs_ratio,
            #                     random_select_ratio=self.args.random_select_ratio,
            #                     reverse_imgs=self.args.esfda_reverse_imgs,
            #                     entropy_threshold=self.args.entropy_threshold,
            #                     retain_all_others=self.args.retain_all_others
            #                 )

            #     else:
            #         self.flipped_indices = self.select_flippable_indices(
            #             model=self.model,
            #             loader=self.loaders,
            #             select_imgs_ratio=self.args.esfda_select_imgs_ratio
            #         )

        # ======================================================================

    def _get_faust_n_views(self) -> int:
        cnd = (self.args.sf_uda and self.args.faust)
        cnd &= (self.args.views_ft_consist or self.args.ce_views_soft_pl)

        if cnd:
            assert self.args.faust_n_views > 0, self.args.faust_n_views
            return self.args.faust_n_views

        return 0

    def build_sfuda_master(self, args):

        if not args.sf_uda:
            return None

        if args.shot:
            mask_root = args.mask_root if self.load_tr_masks else ''
            loaders = get_data_loader(
                data_roots=self.args.data_paths,
                metadata_root=self.args.metadata_root,
                batch_size=self.args.batch_size,
                eval_batch_size=self.args.eval_batch_size,
                workers=self.args.num_workers,
                resize_size=self.args.resize_size,
                crop_size=self.args.crop_size,
                load_tr_masks=self.load_tr_masks,
                mask_root=mask_root,
                proxy_training_set=self.args.proxy_training_set,
                num_val_sample_per_class=self.args.num_val_sample_per_class,
                std_cams_folder=None,
                get_splits_eval=[constants.TRAINSET],
                sfuda_faust=False,
                sfuda_n_rnd_views=0
            )
            train_eval_loader = loaders[constants.TRAINSET]
            return Shot(model_trg=self.model,
                        train_loader_trg=train_eval_loader,
                        task=self.args.task,
                        n_cls=self.args.num_classes,
                        shot_freq_epoch=self.args.shot_freq_epoch,
                        shot_dist=self.args.shot_dist_type
                        )
        
        if args.sfde:
            mask_root = args.mask_root if self.load_tr_masks else ''
            loaders = get_data_loader(
                data_roots=self.args.data_paths,
                metadata_root=self.args.metadata_root,
                batch_size=self.args.batch_size,
                eval_batch_size=self.args.eval_batch_size,
                workers=self.args.num_workers,
                resize_size=self.args.resize_size,
                crop_size=self.args.crop_size,
                load_tr_masks=self.load_tr_masks,
                mask_root=mask_root,
                proxy_training_set=self.args.proxy_training_set,
                num_val_sample_per_class=self.args.num_val_sample_per_class,
                std_cams_folder=None,
                get_splits_eval=[constants.TRAINSET]
            )
            train_eval_loader = loaders[constants.TRAINSET]
            return Sfde(model_trg=self.model,
                        train_loader_trg=train_eval_loader,
                        task=self.args.task,
                        n_cls=self.args.num_classes,
                        support_background = self.args.model['support_background'],
                        threshold= self.args.sfde_threshold,
                        variance= self.args.cdd_variance,
                        convergence= self.args.sfde_cvg
                        )

        elif args.faust:
            return Faust(args=self.args, model_trg=self.model)

        elif args.adadsa:
            return adadsa.Adadsa(args=self.args,
                                 model_src=self.model_src,
                                 model_trg=self.model
                                 )

        elif args.sdda:
            return Sdda(args=self.args,
                        model_trg=self.model
                        )

        elif args.nrc:
            mask_root = args.mask_root if self.load_tr_masks else ''
            loaders = get_data_loader(
                data_roots=self.args.data_paths,
                metadata_root=self.args.metadata_root,
                batch_size=self.args.batch_size,
                eval_batch_size=self.args.eval_batch_size,
                workers=self.args.num_workers,
                resize_size=self.args.resize_size,
                crop_size=self.args.crop_size,
                load_tr_masks=self.load_tr_masks,
                mask_root=mask_root,
                proxy_training_set=self.args.proxy_training_set,
                num_val_sample_per_class=self.args.num_val_sample_per_class,
                std_cams_folder=None,
                get_splits_eval=[constants.TRAINSET],
                sfuda_faust=False,
                sfuda_n_rnd_views=0
            )
            return Nrc(model_trg=self.model,
                    train_loader_trg=loaders[constants.TRAINSET],
                    k_neighbors=self.args.nrc_k,
                    k_nearest_neighbors=self.args.nrc_kk,
                    r_nrc=self.args.r_nrc
                    )
        
        elif args.cdcl:
            mask_root = args.mask_root if self.load_tr_masks else ''
            loaders = get_data_loader(
                data_roots=self.args.data_paths,
                metadata_root=self.args.metadata_root,
                batch_size=self.args.batch_size,
                eval_batch_size=self.args.eval_batch_size,
                workers=self.args.num_workers,
                resize_size=self.args.resize_size,
                crop_size=self.args.crop_size,
                load_tr_masks=self.load_tr_masks,
                mask_root=mask_root,
                proxy_training_set=self.args.proxy_training_set,
                num_val_sample_per_class=self.args.num_val_sample_per_class,
                std_cams_folder=None,
                get_splits_eval=[constants.TRAINSET],
                sfuda_faust=False,
                sfuda_n_rnd_views=0
            )
            train_eval_loader = loaders[constants.TRAINSET]
            return Cdcl(model_trg=self.model,
                        train_loader_trg=train_eval_loader,
                        task=self.args.task,
                        n_cls=self.args.num_classes,
                        support_background = self.args.model['support_background'],
                        threshold= self.args.cdcl_threshold,
                        convergence= self.args.cdcl_cvg
                        )
        
        elif args.pxsfde:
            mask_root = args.mask_root if self.load_tr_masks else ''
            loaders = get_data_loader(
                data_roots=self.args.data_paths,
                metadata_root=self.args.metadata_root,
                batch_size=self.args.batch_size,
                eval_batch_size=self.args.eval_batch_size,
                workers=self.args.num_workers,
                resize_size=self.args.resize_size,
                crop_size=self.args.crop_size,
                load_tr_masks=self.load_tr_masks,
                mask_root=mask_root,
                proxy_training_set=self.args.proxy_training_set,
                num_val_sample_per_class=self.args.num_val_sample_per_class,
                std_cams_folder=None,
                get_splits_eval=[constants.TRAINSET],
                sfuda_faust=False,
                sfuda_n_rnd_views=0
            )
            train_eval_loader = loaders[constants.TRAINSET]
            return Shot(model_trg=self.model,
                        train_loader_trg=train_eval_loader,
                        task=self.args.task,
                        n_cls=self.args.num_classes,
                        shot_freq_epoch=self.args.shot_freq_epoch,
                        shot_dist=self.args.shot_dist_type
                        )
        
        elif args.esfda:
            mask_root = args.mask_root if self.load_tr_masks else ''
            loaders = get_data_loader(
                data_roots=self.args.data_paths,
                metadata_root=self.args.metadata_root,
                batch_size=self.args.batch_size,
                eval_batch_size=self.args.eval_batch_size,
                workers=self.args.num_workers,
                resize_size=self.args.resize_size,
                crop_size=self.args.crop_size,
                load_tr_masks=self.load_tr_masks,
                mask_root=mask_root,
                proxy_training_set=self.args.proxy_training_set,
                num_val_sample_per_class=self.args.num_val_sample_per_class,
                std_cams_folder=None,
                get_splits_eval=[constants.TRAINSET],
                sfuda_faust=False,
                sfuda_n_rnd_views=0
            )
            train_eval_loader = loaders[constants.TRAINSET]

        elif args.grsfda:
            mask_root = args.mask_root if self.load_tr_masks else ''
            loaders = get_data_loader(
                data_roots=self.args.data_paths,
                metadata_root=self.args.metadata_root,
                batch_size=self.args.batch_size,
                eval_batch_size=self.args.eval_batch_size,
                workers=self.args.num_workers,
                resize_size=self.args.resize_size,
                crop_size=self.args.crop_size,
                load_tr_masks=self.load_tr_masks,
                mask_root=mask_root,
                proxy_training_set=self.args.proxy_training_set,
                num_val_sample_per_class=self.args.num_val_sample_per_class,
                std_cams_folder=None,
                get_splits_eval=[constants.TRAINSET],
                sfuda_faust=False,
                sfuda_n_rnd_views=0
            )
            train_eval_loader = loaders[constants.TRAINSET]

        elif args.rgv:
            mask_root = args.mask_root if self.load_tr_masks else ''
            loaders = get_data_loader(
                data_roots=self.args.data_paths,
                metadata_root=self.args.metadata_root,
                batch_size=self.args.batch_size,
                eval_batch_size=self.args.eval_batch_size,
                workers=self.args.num_workers,
                resize_size=self.args.resize_size,
                crop_size=self.args.crop_size,
                load_tr_masks=self.load_tr_masks,
                mask_root=mask_root,
                proxy_training_set=self.args.proxy_training_set,
                num_val_sample_per_class=self.args.num_val_sample_per_class,
                std_cams_folder=None,
                get_splits_eval=[constants.TRAINSET],
                sfuda_faust=False,
                sfuda_n_rnd_views=0
            )
            train_eval_loader = loaders[constants.TRAINSET]
            return Rgv(model_trg=self.model,
                    train_loader_trg=train_eval_loader,num_classes=self.args.num_classes, device = self.args.c_cudaid)
        
        else:
            raise NotImplementedError('SFUDA: unspecified method.')

    @staticmethod
    def _build_tcam_extractor(model, args):
        return build_tcam_extractor(model=model, args=args)

    @staticmethod
    def _build_std_cam_extractor(classifier, args):
        classifier.eval()
        return build_std_cam_extractor(classifier=classifier, args=args)

    def _get_sl(self, args):
        if args.task == constants.F_CL:
            return selflearning.MBSeederSLFCAMS(
                    min_=args.sl_min,
                    max_=args.sl_max,
                    ksz=args.sl_ksz,
                    min_p=args.sl_min_p,
                    fg_erode_k=args.sl_fg_erode_k,
                    fg_erode_iter=args.sl_fg_erode_iter,
                    support_background=args.model['support_background'],
                    multi_label_flag=args.multi_label_flag,
                    seg_ignore_idx=args.seg_ignore_idx)

        elif args.task == constants.TCAM:
            return TCAMSeeder(
                seed_tech=args.sl_tc_seed_tech,
                min_=args.sl_tc_min,
                max_=args.sl_tc_max,
                ksz=args.sl_tc_ksz,
                max_p=args.sl_tc_max_p,
                min_p=args.sl_tc_min_p,
                fg_erode_k=args.sl_tc_fg_erode_k,
                fg_erode_iter=args.sl_tc_fg_erode_iter,
                support_background=args.model['support_background'],
                multi_label_flag=args.multi_label_flag,
                seg_ignore_idx=args.seg_ignore_idx,
                cuda_id=args.c_cudaid,
                roi_method=args.sl_tc_roi_method,
                p_min_area_roi=args.sl_tc_roi_min_size,
                use_roi=args.sl_tc_use_roi
            )
            
        
        elif self.args.pixel_wise_classification:

            if args.sl_pc_seeder == constants.SEED_TH:
                return selflearning.MBSeederSLPCAM(
                    min_=args.sl_min,
                    max_=args.sl_max,
                    ksz=args.sl_ksz,
                    min_p=args.sl_min_p,
                    fg_erode_k=args.sl_fg_erode_k,
                    fg_erode_iter=args.sl_fg_erode_iter,
                    support_background=args.model['support_background'],
                    multi_label_flag=args.multi_label_flag,
                    seg_ignore_idx=args.seg_ignore_idx,
                    neg_samples_partial=args.neg_samples_partial)

            elif args.sl_pc_seeder == constants.SEED_PROB:
                return selflearning.MBProbSeederSLPCAM(
                    min_=args.sl_min,
                    max_=args.sl_max,
                    ksz=args.sl_ksz,
                    seg_ignore_idx=args.seg_ignore_idx
                )
            elif args.sl_pc_seeder == constants.SEED_PROB_N_AREA:
                return selflearning.MBProbNegAreaSeederSLPCAM(
                    min_=args.sl_min,
                    max_=args.sl_max,
                    min_p=args.sl_ng_min_p,
                    ksz=args.sl_ksz,
                    seg_ignore_idx=args.seg_ignore_idx,
                    neg_samples_partial=args.neg_samples_partial,
                    equalize=args.sl_pc_equalize
                )
            else:
                raise NotImplementedError
            # return selflearning.MBSeederSLFCAMS(
            #         min_=args.sl_min,
            #         max_=args.sl_max,
            #         ksz=args.sl_ksz,
            #         min_p=args.sl_min_p,
            #         fg_erode_k=args.sl_fg_erode_k,
            #         fg_erode_iter=args.sl_fg_erode_iter,
            #         support_background=args.model['support_background'],
            #         multi_label_flag=args.multi_label_flag,
            #         seg_ignore_idx=args.seg_ignore_idx,
            #         neg_samples_partial=args.neg_samples_partial)

        elif args.task == constants.NEGEV:

            if args.sl_ng_seeder == constants.SEED_TH:
                return selflearning.MBSeederSLNEGEV(
                    min_=args.sl_min,
                    max_=args.sl_max,
                    ksz=args.sl_ksz,
                    min_p=args.sl_min_p,
                    fg_erode_k=args.sl_fg_erode_k,
                    fg_erode_iter=args.sl_fg_erode_iter,
                    support_background=args.model['support_background'],
                    multi_label_flag=args.multi_label_flag,
                    seg_ignore_idx=args.seg_ignore_idx)

            elif args.sl_ng_seeder == constants.SEED_PROB:
                return selflearning.MBProbSeederSLNEGEV(
                    min_=args.sl_min,
                    max_=args.sl_max,
                    ksz=args.sl_ksz,
                    seg_ignore_idx=args.seg_ignore_idx
                )
            elif args.sl_ng_seeder == constants.SEED_PROB_N_AREA:
                return selflearning.MBProbNegAreaSeederSLNEGEV(
                    min_=args.sl_min,
                    max_=args.sl_max,
                    min_p=args.sl_ng_min_p,
                    ksz=args.sl_ksz,
                    seg_ignore_idx=args.seg_ignore_idx
                )
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError


    def prepare_std_cams_disq(self, std_cams: torch.Tensor,
                              image_size: Tuple) -> torch.Tensor:

        assert std_cams.ndim == 4
        cams = std_cams.detach()

        # cams: (bsz, 1, h, w) == image_size.
        assert cams.ndim == 4
        # Quick fix: todo...
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        # todo: unnecessary. cams and images have same size.
        cams = F.interpolate(cams,
                             image_size,
                             mode='bilinear',
                             align_corners=False)  # (bsz, 1, h, w)
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        return cams

    def get_std_cams_minibatch(self, images, targets) -> torch.Tensor:
        # used only for task f_cl/negev
        assert self.args.task in [constants.F_CL, constants.NEGEV]
        assert images.ndim == 4
        image_size = images.shape[2:]

        cams = None
        for idx, (image, target) in enumerate(zip(images, targets)):
            cl_logits = self.classifier(image.unsqueeze(0))
            cam = self.std_cam_extractor(
                class_idx=target.item(), scores=cl_logits, normalized=True)
            # h`, w`
            # todo: set to false (normalize).

            cam = cam.detach().unsqueeze(0).unsqueeze(0)

            if cams is None:
                cams = cam
            else:
                cams = torch.vstack((cams, cam))

        # cams: (bsz, 1, h, w)
        assert cams.ndim == 4
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        cams = F.interpolate(cams,
                             image_size,
                             mode='bilinear',
                             align_corners=False)  # (bsz, 1, h, w)
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)

        return cams

    def get_pseudo_cams_minibatch(self, images, targets) -> torch.Tensor:
        assert images.ndim == 4
        image_size = images.shape[2:]

        cams = None

        for idx, (image, target) in enumerate(zip(images, targets)):
            if self.args.sf_uda_source_wsol_method == constants.METHOD_SAT:
                cl_logits = self.classifier(image.unsqueeze(0), labels = target)
            else:
                cl_logits = self.classifier(image.unsqueeze(0))
                
            cam = self.std_cam_extractor(
                class_idx=target.item(), scores=cl_logits, normalized=True)
            # h`, w`
            # todo: set to false (normalize).

            cam = cam.detach().unsqueeze(0).unsqueeze(0)

            if cams is None:
                cams = cam
            else:
                cams = torch.vstack((cams, cam))

        # cams: (bsz, 1, h, w)
        assert cams.ndim == 4
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        cams = F.interpolate(cams,
                             image_size,
                             mode='bilinear',
                             align_corners=False)  # (bsz, 1, h, w)
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)

        return cams

    def is_seed_required(self, _epoch):
        cmd = (self.args.task in [constants.F_CL, constants.NEGEV])
        cmd &= (('self_learning_fcams' in self.loss.n_holder) or (
            'self_learning_negev' in self.loss.n_holder
        ))
        cmd2 = False
        for _l in self.loss.losses:
            if isinstance(_l, losses.SelfLearningFcams):
                cmd2 = _l.is_on(_epoch=_epoch)

        return cmd and cmd2

    def _do_cutmix(self):
        return (self.args.method == constants.METHOD_CUTMIX and
                self.args.cutmix_prob > np.random.rand(1).item() and
                self.args.cutmix_beta > 0)

    def build_key_arg(self, base_key_arg=None, src_probs=None):

        key_arg = base_key_arg.copy() if base_key_arg is not None else {}

        key_arg["src_probs"] = src_probs

        return key_arg

    def rgv_refine_and_align(self, images, aug_images, index):

        # --- Forward normal ---
        output = self.model(images)
        feats = self.model.lin_ft.clone().detach()
        cl_logits = output.clone()
        probs = F.softmax(cl_logits.detach(), dim=1)

        # --- Pseudo-label refining via MemoryBank (Eq.9) ---
        p_refined = self.sfuda_master.memory.get_knn(feats, k=5, tau=0.07)

        # --- Certainty (Eq.10) ---
        entropy = -(p_refined * p_refined.log()).sum(dim=1)
        max_logp = p_refined.log().max(dim=1)[0]
        entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
        maxlog_norm = (max_logp - max_logp.min()) / (max_logp.max() - max_logp.min() + 1e-8)
        certainty = ((1 - entropy_norm) + maxlog_norm) / 2

        beta = 0.6
        mask_certainty = certainty >= beta

        
        mask_not_DI = torch.tensor(
            [name not in self.sfuda_master.D_maps['I'] for name in index],
            device=self.device, dtype=torch.bool
        )



        if self.model.support_background:
            weights = self.model.classification_head.fc.weight[1:]
        else:
            weights = self.model.classification_head.fc.weight


        aug_imgs = aug_images.to(self.device)
        aug_logits = self.model(aug_imgs)
        aug_feats = self.model.lin_ft

        y_tilde = p_refined.argmax(dim=1)

        self.sfuda_master.memory.update(feats, probs, names=index)

        return certainty, mask_not_DI, y_tilde, aug_feats, weights


    def _one_step_train(self,
                        images,
                        raw_imgs,
                        targets,
                        p_glabel,
                        std_cams,
                        masks,
                        views,
                        index,
                        aug_images):
        args = self.args
        y_global = targets
        y_pl_global = p_glabel

        if self.args.erl:
            y_bar_batch = torch.stack([self.index_src_logits[name] for name in index]).to(self.args.c_cudaid)

            
        z_label = targets

        if args.sf_uda:
            z_label = p_glabel

        if args.method == constants.METHOD_HAS:
            images = wsol_has(image=images,
                              grid_size=args.has_grid_size,
                              drop_rate=args.has_drop_rate)

        cutmix_holder = None
        if args.method == constants.METHOD_CUTMIX:
            if self._do_cutmix():
                images, target_a, target_b, lam = wsol_cutmix(
                    x=images, target=z_label, beta=args.cutmix_beta)
                cutmix_holder = [target_a, target_b, lam]

        if args.sf_uda:

            # if args.pixel_wise_classification and args.ece_adapt:
            #     _, _, h, w = self.model.encoder_last_features.shape
            #     interpolation_mode = 'bilinear'
            #     if std_cams is None:
            #         cams_inter = self.get_std_cams_minibatch(images=images,
            #                                                 targets=z_label)
            #     else:
            #         cams_inter = std_cams

            #     if self.args.low_res:
            #         fcams=self.model.cams
            #     else:
            #         _, _, i, x = cams_inter.shape
            #         fcams= F.interpolate(self.model.cams,
            #                     (i, x),
            #                     mode=interpolation_mode,
            #                     align_corners=False)

            #     with torch.no_grad():
            #         if self.args.low_res:
            #             cams_inter = F.interpolate(cams_inter,
            #                     (h, w),
            #                     mode=interpolation_mode,
            #                     align_corners=False)

            #         seeds = seeds = self.sl_mask_builder(cams_inter, class_idx=p_glabel)

            # else:
            #     seeds = None
            #     fcams = None

            if args.task == constants.STD_CL:
                if args.faust:
                    assert views is not None
                    assert isinstance(views, torch.Tensor), type(views)
                    assert views.ndim == 5, views.ndim  # b, nviews + 1, c, h, w

                    # extra forward so the code does not break
                    with torch.no_grad():
                        output = self.model(images)
                        cl_logits = output

                    faust_out = self.sfuda_master.forward_data(views)

                    logits = cl_logits
                    loss = self.loss(epoch=self.epoch, key_arg=faust_out)

                elif args.nrc:
                    output = self.model(images)
                    cl_logits = output
                    loss = self.loss(epoch=self.epoch,
                                     model=self.model,
                                     cl_logits=cl_logits,
                                     glabel=y_global,
                                     pseudo_glabel=y_pl_global,
                                     cutmix_holder=cutmix_holder,
                                     key_arg=self.nrc_args
                                     )
                    logits = cl_logits
                    

                elif self.args.cdcl:
                    out = self.model(images)
                    features = self.model.lin_ft

                    if args.pixel_wise_classification and args.ece_adapt:
                        _, _, h, w = self.model.encoder_last_features.shape
                        interpolation_mode = 'bilinear'

                        if std_cams is None:
                            cams_inter = self.get_std_cams_minibatch(images=images, targets=z_label)
                        else:
                            cams_inter = std_cams

                        if self.args.low_res:
                            fcams = self.model.cams
                        else:
                            _, _, i, x = cams_inter.shape
                            fcams = F.interpolate(
                                self.model.cams, (i, x),
                                mode=interpolation_mode,
                                align_corners=False
                            )

                        with torch.no_grad():
                            if self.args.low_res:
                                cams_inter = F.interpolate(
                                    cams_inter, (h, w),
                                    mode=interpolation_mode,
                                    align_corners=False
                                )

                            seeds = self.sl_mask_builder(cams_inter, class_idx=p_glabel)
                    else:
                        seeds = None
                        fcams = None
                    
                    with torch.no_grad():
                            output = self.model(images)
                            cl_logits = output

                    cdcl_out = self.sfuda_master.forward_data(features)



                    if self.args.erl:
                        key_args = self.build_key_arg(cdcl_out, src_probs=y_bar_batch)
                    else:
                        key_args = cdcl_out

                    #loss = self.loss(epoch=self.epoch,model=self.model,cl_logits=cl_logits,glabel=y_global,pseudo_glabel=y_pl_global,key_arg=cdcl_out)
                    loss = self.loss(epoch=self.epoch,model=self.model,fcams=fcams, cl_logits=cl_logits,glabel=y_global,pseudo_glabel=y_pl_global,seeds=seeds,key_arg=key_args)
                    
                    logits = cl_logits

                elif self.args.sfde:
                    out = self.model(images)
                    features = self.model.lin_ft

                    if args.pixel_wise_classification and args.ece_adapt:
                        _, _, h, w = self.model.encoder_last_features.shape
                        interpolation_mode = 'bilinear'

                        if std_cams is None:
                            cams_inter = self.get_std_cams_minibatch(images=images, targets=z_label)
                        else:
                            cams_inter = std_cams

                        if self.args.low_res:
                            fcams = self.model.cams
                        else:
                            _, _, i, x = cams_inter.shape
                            fcams = F.interpolate(
                                self.model.cams, (i, x),
                                mode=interpolation_mode,
                                align_corners=False
                            )

                        with torch.no_grad():
                            if self.args.low_res:
                                cams_inter = F.interpolate(
                                    cams_inter, (h, w),
                                    mode=interpolation_mode,
                                    align_corners=False
                                )

                            seeds = self.sl_mask_builder(cams_inter, class_idx=p_glabel)
                    else:
                        seeds = None
                        fcams = None
                    
                    with torch.no_grad():
                            output = self.model(images)
                            cl_logits = output
                    
                    sfde_out = self.sfuda_master.forward_data(features, self.normal_sampler)
                    loss = self.loss(epoch=self.epoch,model=self.model,fcams=fcams, cl_logits=cl_logits,glabel=y_global,pseudo_glabel=y_pl_global,seeds=seeds,key_arg=sfde_out)
                    logits = cl_logits

                elif self.args.pxsfde:
                    out = self.model(images)
                    features = self.model.lin_ft
                    
                    with torch.no_grad():
                            output = self.model(images)
                            cl_logits = output
                    
                    loss = self.loss(epoch=self.epoch,model=self.model,cl_logits=cl_logits,glabel=y_global,pseudo_glabel=y_pl_global)
                    logits = cl_logits

                elif self.args.rgv:

                    out = self.model(images)
                    
                    with torch.no_grad():
                            output = self.model(images)
                            cl_logits = output


                    pseudo_labels = torch.tensor(
                    [self.sfuda_master.D_maps['I'].get(n, -255) for n in index],
                    device=images.device,
                    dtype=torch.long)

                    mask_CE = pseudo_labels != -255


                    certainty_SA, mask_SA, y_tilde_SA, aug_feats_SA, weights = self.rgv_refine_and_align(images, aug_images, index)

                    key_arg = {}

                    # --- Cross-Entropy (D_I) ---
                    key_arg["pseudo_labels_CE"] = pseudo_labels
                    key_arg["mask_CE"] = mask_CE

                    # --- Semantic Alignment (D_U) ---
                    if aug_feats_SA is not None:
                        key_arg["aug_feats_SA"] = aug_feats_SA      
                    if certainty_SA is not None:
                        key_arg["certainty_SA"] = certainty_SA      
                    if mask_SA is not None:
                        key_arg["mask_SA"] = mask_SA   
                    if y_tilde_SA is not None:
                        key_arg["y_tilde_SA"] = y_tilde_SA
                    if weights is not None:
                        key_arg["weights"] = weights

                    

                    #cl_logits = output
                    loss = self.loss(epoch=self.epoch,
                                     model=self.model,
                                     cl_logits=cl_logits,
                                     glabel=y_global,
                                     pseudo_glabel=pseudo_labels,
                                     cutmix_holder=cutmix_holder,
                                     key_arg = key_arg 
                                     )
                    logits = cl_logits


                else:
                    output = self.model(images)
                    cl_logits = output

                    if args.pixel_wise_classification and args.ece_adapt:
                        _, _, h, w = self.model.encoder_last_features.shape
                        interpolation_mode = 'bilinear'
                        if std_cams is None:
                            cams_inter = self.get_std_cams_minibatch(images=images,
                                                                    targets=z_label)
                        else:
                            cams_inter = std_cams

                        if self.args.low_res:
                            fcams=self.model.cams
                        else:
                            _, _, i, x = cams_inter.shape
                            fcams= F.interpolate(self.model.cams,
                                        (i, x),
                                        mode=interpolation_mode,
                                        align_corners=False)

                        with torch.no_grad():
                            if self.args.low_res:
                                cams_inter = F.interpolate(cams_inter,
                                        (h, w),
                                        mode=interpolation_mode,
                                        align_corners=False)

                            seeds = seeds = self.sl_mask_builder(cams_inter, class_idx=p_glabel)

                    else:
                        seeds = None
                        fcams = None


                    loss = self.loss(epoch=self.epoch,
                                     model=self.model,
                                     fcams=fcams,
                                     cl_logits=cl_logits,
                                     glabel=y_global,
                                     pseudo_glabel=y_pl_global,
                                     cutmix_holder=cutmix_holder,
                                     seeds=seeds
                                     )
                    logits = cl_logits

            elif args.task == constants.F_CL:
                raise NotImplementedError

            elif args.task == constants.NEGEV:
                raise NotImplementedError

            elif args.task == constants.SEG:
                raise NotImplementedError


        else:

            output = self.model(images)

            if args.task == constants.STD_CL:
                if args.pixel_wise_classification:
                    _, _, h, w = self.model.encoder_last_features.shape
                    interpolation_mode = 'bilinear'
                    if std_cams is None:
                        cams_inter = self.get_std_cams_minibatch(images=images,
                                                                targets=z_label)
                    else:
                        cams_inter = std_cams

                    if self.args.low_res:
                        fcams=self.model.cams
                    else:
                        _, _, i, x = cams_inter.shape
                        fcams= F.interpolate(self.model.cams,
                                    (i, x),
                                    mode=interpolation_mode,
                                    align_corners=False)

                    with torch.no_grad():
                        if self.args.low_res:
                            cams_inter = F.interpolate(cams_inter,
                                    (h, w),
                                    mode=interpolation_mode,
                                    align_corners=False)

                        seeds = self.sl_mask_builder(cams_inter, class_idx=targets)

                    

                    cl_logits = output
                    loss = self.loss(epoch=self.epoch,
                                    model=self.model,
                                    fcams=fcams,
                                    cl_logits=cl_logits,
                                    glabel=y_global,
                                    pseudo_glabel=y_pl_global,
                                    raw_img=raw_imgs,
                                    cutmix_holder=cutmix_holder,
                                    seeds=seeds
                                    )
                    logits = cl_logits
                else:
                    cl_logits = output
                    loss_params = {'sat_aux_losses': self.model.losses_dict, "sat_area_th": self.args.sat_area_th} if self.args.method == constants.METHOD_SAT else {}
                    loss = self.loss(epoch=self.epoch,
                                    model=self.model,
                                    cl_logits=cl_logits,
                                    glabel=y_global,
                                    pseudo_glabel=y_pl_global,
                                    cutmix_holder=cutmix_holder,
                                    **loss_params
                                    )
                    logits = cl_logits

            elif args.task == constants.F_CL:
                cl_logits, fcams, im_recon = output

                if self.is_seed_required(_epoch=self.epoch):
                    if std_cams is None:
                        cams_inter = self.get_std_cams_minibatch(images=images,
                                                                 targets=z_label)
                    else:
                        cams_inter = std_cams

                    with torch.no_grad():
                        seeds = self.sl_mask_builder(cams_inter)
                else:
                    cams_inter, seeds = None, None

                loss = self.loss(
                    epoch=self.epoch,
                    cams_inter=cams_inter,
                    fcams=fcams,
                    cl_logits=cl_logits,
                    glabel=y_global,
                    pseudo_glabel=y_pl_global,
                    raw_img=raw_imgs,
                    x_in=self.model.x_in,
                    im_recon=im_recon,
                    seeds=seeds
                )
                logits = cl_logits

            elif args.task == constants.NEGEV:
                cl_logits, fcams, im_recon = output

                if self.is_seed_required(_epoch=self.epoch):
                    if std_cams is None:
                        cams_inter = self.get_std_cams_minibatch(images=images,
                                                                 targets=z_label)
                    else:
                        cams_inter = std_cams

                    with torch.no_grad():
                        seeds = self.sl_mask_builder(cams_inter)
                else:
                    cams_inter, seeds = None, None

                loss = self.loss(
                    epoch=self.epoch,
                    cams_inter=cams_inter,
                    fcams=fcams,
                    cl_logits=cl_logits,
                    glabel=y_global,
                    pseudo_glabel=y_pl_global,
                    raw_img=raw_imgs,
                    x_in=self.model.x_in,
                    im_recon=im_recon,
                    seeds=seeds
                )
                logits = cl_logits

            elif args.task == constants.SEG:
                assert masks is not None
                assert isinstance(masks, torch.Tensor)
                assert masks.ndim == 4
                assert masks.shape[1] == 1

                seg_logits = output
                loss = self.loss(seg_logits=seg_logits, masks=masks.squeeze(1))
                logits = None
            else:
                raise NotImplementedError

        return logits, loss

    def _one_step_train_unlearning(self,
                        images,
                        raw_imgs,
                        targets,
                        p_glabel,
                        std_cams,
                        masks,
                        views,
                        cal_mask_forget=None,
                        cal_mask_retain=None,
                        y_pred_batch=None,
                        mask_entropy = None):
        
        args = self.args
        y_global = targets
        y_pl_global = p_glabel

        z_label = targets
        cutmix_holder = None

        if args.sf_uda:
            z_label = p_glabel

        if args.sf_uda:
            if args.task == constants.STD_CL:
                    key_arg = {}
                    # if cal_mask is not None:
                    #     key_arg["cal_mask"] = cal_mask
                    if cal_mask_forget is not None:
                        key_arg["cal_mask_forget"] = cal_mask_forget
                    if cal_mask_retain is not None:
                        key_arg["cal_mask_retain"] = cal_mask_retain
                    if y_pred_batch is not None:
                        key_arg["y_pred_batch"] = y_pred_batch

                    if self.args.esfda_flip_labels_weight:
                        if mask_entropy is not None:
                            key_arg["mask_entropy"] = mask_entropy
                        else:
                            key_arg["mask_entropy"] = None

                    output = self.model(images)
                    cl_logits = output

                    if not self.args.esfda_loc:
                        cams_inter = None
                        attention_map = None

                    if self.args.esfda_loc:
                       interpolation_mode = 'bilinear'
                       fcams = self.model.encoder_last_features.mean(dim=1, keepdim=True)
                       #fattention = torch.sigmoid(attention_map)

                       cams_inter_init = std_cams

                       seeds = None

                       _, _, H, W = fcams.shape

                       cams_inter = F.interpolate(
                            cams_inter_init,      
                            size=(H, W),
                            mode=interpolation_mode,
                            align_corners=False
                        )

                    if args.pixel_wise_classification and args.ece:
                        _, _, h, w = self.model.encoder_last_features.shape
                        interpolation_mode = 'bilinear'
                        if std_cams is None:
                            cams_inter = self.get_pseudo_cams_minibatch(images=images,
                                                                    targets=y_pred_batch)
                        else:
                            cams_inter = std_cams

                        if self.args.low_res:
                            fcams=self.model.cams
                        else:
                            _, _, i, x = cams_inter.shape
                            fcams= F.interpolate(self.model.cams,
                                        (i, x),
                                        mode=interpolation_mode,
                                        align_corners=False)

                        with torch.no_grad():
                            if self.args.low_res:
                                cams_inter = F.interpolate(cams_inter,
                                        (h, w),
                                        mode=interpolation_mode,
                                        align_corners=False)

                            seeds = self.sl_mask_builder(cams_inter, class_idx=y_pred_batch)

                    else:
                        fcams = None
                        seeds = None


                    loss = self.loss(epoch=self.epoch,
                                     model=self.model,
                                     cams_inter=cams_inter, 
                                     fcams=fcams,
                                     cl_logits=cl_logits,
                                     glabel=y_global,
                                     pseudo_glabel=y_pl_global,
                                     raw_img=raw_imgs,
                                     cutmix_holder=cutmix_holder,
                                     seeds=seeds,
                                     key_arg=key_arg
                                    )
                    logits = cl_logits

            elif args.task == constants.F_CL:
                raise NotImplementedError

            elif args.task == constants.NEGEV:
                raise NotImplementedError

            elif args.task == constants.SEG:
                raise NotImplementedError


        else:

            output = self.model(images)

            if args.task == constants.STD_CL:
                if args.pixel_wise_classification:
                    _, _, h, w = self.model.encoder_last_features.shape
                    interpolation_mode = 'bilinear'
                    if std_cams is None:
                        cams_inter = self.get_std_cams_minibatch(images=images,
                                                                targets=z_label)
                    else:
                        cams_inter = std_cams

                    if self.args.low_res:
                        fcams=self.model.cams
                    else:
                        _, _, i, x = cams_inter.shape
                        fcams= F.interpolate(self.model.cams,
                                    (i, x),
                                    mode=interpolation_mode,
                                    align_corners=False)

                    with torch.no_grad():
                        if self.args.low_res:
                            cams_inter = F.interpolate(cams_inter,
                                    (h, w),
                                    mode=interpolation_mode,
                                    align_corners=False)

                        seeds = self.sl_mask_builder(cams_inter, class_idx=targets)

                    

                    cl_logits = output
                    loss = self.loss(epoch=self.epoch,
                                    model=self.model,
                                    fcams=fcams,
                                    cl_logits=cl_logits,
                                    glabel=y_global,
                                    pseudo_glabel=y_pl_global,
                                    raw_img=raw_imgs,
                                    cutmix_holder=cutmix_holder,
                                    seeds=seeds
                                    )
                    logits = cl_logits
                else:
                    cl_logits = output
                    loss_params = {'sat_aux_losses': self.model.losses_dict, "sat_area_th": self.args.sat_area_th} if self.args.method == constants.METHOD_SAT else {}
                    loss = self.loss(epoch=self.epoch,
                                    model=self.model,
                                    cl_logits=cl_logits,
                                    glabel=y_global,
                                    pseudo_glabel=y_pl_global,
                                    cutmix_holder=cutmix_holder,
                                    **loss_params
                                    )
                    logits = cl_logits

            elif args.task == constants.F_CL:
                cl_logits, fcams, im_recon = output

                if self.is_seed_required(_epoch=self.epoch):
                    if std_cams is None:
                        cams_inter = self.get_std_cams_minibatch(images=images,
                                                                 targets=z_label)
                    else:
                        cams_inter = std_cams

                    with torch.no_grad():
                        seeds = self.sl_mask_builder(cams_inter)
                else:
                    cams_inter, seeds = None, None

                loss = self.loss(
                    epoch=self.epoch,
                    cams_inter=cams_inter,
                    fcams=fcams,
                    cl_logits=cl_logits,
                    glabel=y_global,
                    pseudo_glabel=y_pl_global,
                    raw_img=raw_imgs,
                    x_in=self.model.x_in,
                    im_recon=im_recon,
                    seeds=seeds
                )
                logits = cl_logits

            elif args.task == constants.NEGEV:
                cl_logits, fcams, im_recon = output

                if self.is_seed_required(_epoch=self.epoch):
                    if std_cams is None:
                        cams_inter = self.get_std_cams_minibatch(images=images,
                                                                 targets=z_label)
                    else:
                        cams_inter = std_cams

                    with torch.no_grad():
                        seeds = self.sl_mask_builder(cams_inter)
                else:
                    cams_inter, seeds = None, None

                loss = self.loss(
                    epoch=self.epoch,
                    cams_inter=cams_inter,
                    fcams=fcams,
                    cl_logits=cl_logits,
                    glabel=y_global,
                    pseudo_glabel=y_pl_global,
                    raw_img=raw_imgs,
                    x_in=self.model.x_in,
                    im_recon=im_recon,
                    seeds=seeds
                )
                logits = cl_logits

            elif args.task == constants.SEG:
                assert masks is not None
                assert isinstance(masks, torch.Tensor)
                assert masks.ndim == 4
                assert masks.shape[1] == 1

                seg_logits = output
                loss = self.loss(seg_logits=seg_logits, masks=masks.squeeze(1))
                logits = None
            else:
                raise NotImplementedError

        return logits, loss

    #@staticmethod
    def _fill_minibatch(self, _x: torch.Tensor, mbatchsz: int) -> torch.Tensor:
        assert isinstance(_x, torch.Tensor)
        assert isinstance(mbatchsz, int)
        assert mbatchsz > 0

        if _x.shape[0] == mbatchsz or self.args.nrc or self.args.esfda:
            return _x

        s = _x.shape[0]
        t = math.ceil(float(mbatchsz) / s)
        v = torch.cat(t * [_x])
        assert v.shape[1:] == _x.shape[1:]

        out = v[:mbatchsz]
        assert out.shape[0] == mbatchsz
        return out

    #@staticmethod
    def _fill_minibatch_list(self, items, target_size):
        if len(items) == target_size:
            return items
        elif len(items) < target_size:
            repeat_factor = target_size // len(items)
            remainder = target_size % len(items)
            items = items * repeat_factor + items[:remainder]
        else:
            items = items[:target_size]
        return items

    def select_images_to_correct_and_incorrect(self, model, loader, select_imgs_ratio=0.1):

        all_indices = []
        all_true_labels = []

        for _, (images, targets, p_glabel, index, raw_imgs, std_cams, masks, views) in enumerate(loader):
            all_indices.extend(list(index))
            all_true_labels.extend(targets.cpu().tolist())

        total = len(all_indices)
        num_to_select = int(select_imgs_ratio * total)

        py_random.seed(self.seed)
        selected_indices = set(py_random.sample(range(total), num_to_select))

        final_label_dict = {}

        num_classes = len(list(set(all_true_labels)))

        for i in range(total):
            image_id = all_indices[i]
            true_label = all_true_labels[i]

            if i in selected_indices:

                final_label_dict[image_id] = true_label
            else:

                wrong_labels = [c for c in range(num_classes) if c != true_label]
                final_label_dict[image_id] = py_random.choice(wrong_labels)

        return final_label_dict
    
    def select_images_to_correct(self, model, loader, select_imgs_ratio=0.1):

        """
        Select images to correct pseudo-labels.
        :param model: Model to use for selection.
        :param loader: DataLoader to use for selection.
        :param select_imgs_ratio: Ratio of images to select.
        :return: Indices of selected images.
        """
        # Implement the logic to select images based on the model and loader

        all_indices = []
        all_true_labels = []

        for _, (images, targets, p_glabel, index, raw_imgs, std_cams, masks, views) in enumerate(loader):

            all_indices.extend(list(index))
            all_true_labels.extend(targets.cpu().tolist())

        total = len(all_indices)
        num_to_select = int(select_imgs_ratio * total)

        py_random.seed(self.seed)
        selected_indices = py_random.sample(range(total), num_to_select)
        selected_dict = {all_indices[i]: all_true_labels[i] for i in selected_indices}

        return selected_dict
       
    
    def ratio_correct_pseudo_labels(self, pseudo_labels, selected_true_labels_dict):
        """
        Modify pseudo label with true label
        :param true_labels: List of true labels for each epoch.
        :param pseudo_labels: List of pseudo-labels for each epoch.
        :return: new list of pseudo label.
        """

        corrected_pseudo_labels = pseudo_labels.copy()
        for image_name, true_label in selected_true_labels_dict.items():
            if image_name in corrected_pseudo_labels:
                corrected_pseudo_labels[image_name] = true_label  # Only correct existing entries
        return corrected_pseudo_labels


    def _sf_uda_before_epoch_process(self):
        assert self.args.sf_uda


        if self.args.rgv:
            print(f'Running RGV pseudo-label estimation at epoch {self.epoch}')
            if self.epoch % self.args.rgv_round_interval == 0:
                print(f'Updating RGV sample selection at round {self.epoch // self.args.rgv_round_interval}')
                D_samples = self.sfuda_master.run()
                self.D_samples = D_samples  
            else:
                D_samples = getattr(self, "D_samples", None)
                if D_samples is None:
                    print("Initializing RGV samples at epoch 0")
                    D_samples = self.sfuda_master.run()
                    self.D_samples = D_samples

            return D_samples

        
        if self.args.grsfda:
            if self.args.ce_pseudo_lb:
                percent = self.compute_dynamic_percent(
                    base_percent=0.05,
                    epoch=self.epoch,
                    n_epochs_step=2,
                    growth_factor=2.0
                )



                final_selected, pl, acc = self.select_by_entropy_balanced(self.model,self.loaders_notransform[constants.TRAINSET],percent=percent)

                self.loaders[constants.TRAINSET].dataset.set_img_pseudo_labels(
                        pl)

        if self.args.shot:
            if self.args.ce_pseudo_lb:

                print(f'Running img-class pseudo-label estimation SHOT epoch: '
                      f'{self.epoch}')

                pl, acc = self.sfuda_master.update_img_cls_pseudo_lbs()

                if self.args.correct_pseudo_labels:
                    pl = self.ratio_correct_pseudo_labels(
                        pseudo_labels=pl,
                        selected_true_labels_dict=self.corrected_pseudo_labels
                    )

                if self.args.correct_and_incorrect_pseudo_labels:
                    pl = self.corrected_pseudo_labels


                self.loaders[constants.TRAINSET].dataset.set_img_pseudo_labels(
                    pl)
                
                self.pseudo_labels.append(acc)

        elif self.args.adadsa:
            if self.args.ce_pseudo_lb:
                # This is not mandatory by the paper. it is set to measure/track
                # classification accuracy of image-class pseudo-labels.
                # Here, trainset pseudo-labels (PL) estimation is done and the
                # results are set to the dataloader, but they are not used later
                # since the paper uses on the fly PL estimation for each
                # minibatch at each SGD step.

                # ************ THIS CAN BE SAFELY TURNED OFF. ****************

                print(f'Running img-class pseudo-label estimation AdaDSA epoch:'
                      f' {self.epoch}')

                self.sfuda_master.update_lambda_(epoch=max(0, self.epoch - 1),
                                                 mb_idx=0)

                pl = self.sfuda_master.update_img_cls_pseudo_lbs()
                self.loaders[constants.TRAINSET].dataset.set_img_pseudo_labels(
                    pl)

        elif self.args.sdda:
            # WARNING: THIS IS SAFE TO REMOVE. IT IS ADDED TO CHECK QUALITY
            # OF IMAGE-CLASS PSEUDO-LABELS.
            print(f'Running img-class pseudo-label estimation SDDA epoch: '
                  f'{self.epoch}')

            self.sfuda_master.update_lambda_(epoch=max(0, self.epoch - 1),
                                             mb_idx=0)

            pl = self.sfuda_master.update_img_cls_pseudo_lbs()
            self.loaders[constants.TRAINSET].dataset.set_img_pseudo_labels(
                pl)
   

        elif self.args.sfde:
            print(f'running label estimation SFDE epoch {self.epoch}')
            mask_root = self.mask_root if self.load_tr_masks else ''
            sfuda_select_ids_pl, target_hypt,  filtered_classes, self.clustering_acc = self.sfuda_master.solve()

            self.pseudo_labels.append(self.clustering_acc)

            if self.args.correct_pseudo_labels:
                sfuda_select_ids_pl = self.ratio_correct_pseudo_labels(
                    pseudo_labels=sfuda_select_ids_pl,
                    selected_true_labels_dict=self.corrected_pseudo_labels
                )

            if self.args.correct_and_incorrect_pseudo_labels:
                sfuda_select_ids_pl = self.corrected_pseudo_labels

            print('Creation filtered dataloader')
            self.loaders_filtered = get_data_loader(data_roots=self.args.data_paths,
                metadata_root=self.args.metadata_root,
                batch_size=self.args.batch_size,
                eval_batch_size=self.args.eval_batch_size,
                workers=self.args.num_workers,
                resize_size=self.args.resize_size,
                crop_size=self.args.crop_size,
                load_tr_masks=self.load_tr_masks,
                mask_root=mask_root,
                proxy_training_set=self.args.proxy_training_set,
                num_val_sample_per_class=self.args.num_val_sample_per_class,
                std_cams_folder=self.args.std_cams_folder,
                get_splits_eval=[constants.TRAINSET],
                per_split_sfuda_select_ids_pl= {constants.TRAINSET: sfuda_select_ids_pl})
            print('Generating features for surrogate feature sampler')
            self.normal_sampler = self.sfuda_master.construct_surrogate_feature_sampler(filtered_classes, self.loaders_filtered[constants.TRAINSET])
            

        elif self.args.cdcl:
            print(f'running label estimation CDCL epoch {self.epoch}')
            mask_root = self.mask_root if self.load_tr_masks else ''
            sfuda_select_ids_pl, target_hypt,  filtered_classes, self.clustering_acc = self.sfuda_master.solve()

            self.pseudo_labels.append(self.clustering_acc)

            if self.args.correct_pseudo_labels:
                sfuda_select_ids_pl = self.ratio_correct_pseudo_labels(
                    pseudo_labels=sfuda_select_ids_pl,
                    selected_true_labels_dict=self.corrected_pseudo_labels
                )

            if self.args.correct_and_incorrect_pseudo_labels:
                sfuda_select_ids_pl = self.corrected_pseudo_labels
            
            print('Creation filtered dataloader')
            self.loaders_filtered = get_data_loader(data_roots=self.args.data_paths,
                metadata_root=self.args.metadata_root,
                batch_size=self.args.batch_size,
                eval_batch_size=self.args.eval_batch_size,
                workers=self.args.num_workers,
                resize_size=self.args.resize_size,
                crop_size=self.args.crop_size,
                load_tr_masks=self.load_tr_masks,
                mask_root=mask_root,
                proxy_training_set=self.args.proxy_training_set,
                num_val_sample_per_class=self.args.num_val_sample_per_class,
                std_cams_folder=self.args.std_cams_folder,
                #get_splits_eval=[constants.TRAINSET],
                per_split_sfuda_select_ids_pl= {constants.TRAINSET: sfuda_select_ids_pl,constants.PXVALIDSET: None,constants.CLVALIDSET: None,constants.TESTSET: None})
                #per_split_sfuda_select_ids_pl= {constants.TRAINSET: sfuda_select_ids_pl})

        elif self.args.esfda:
            # =====================================================
            # DYNAMIC RE-SAMPLING OF Dforget / Dretain
            # =====================================================
            if (self.epoch == 1):
                # Static mode → one-time precomputation

                self.store_master_loss = []
                self.store_ce_flip_loss = []
                self.store_ce_not_flip_loss = []
                #self.loader.dataset.transform = None

                self.pred_distribution, self.overpred_class, self.underpred_class = self.compute_prediction_bias(model=self.model,
                            loader_notransform=self.loaders_notransform, top_k=None
                        )
                (
                self.flipped_indices, 
                self.reinforce_indices, 
                self.idx_to_pred, 
                self.entropy_all, 
                self.all_selected, 
                self.stable_selected, 
                self.stable_labels,
                self.probs_all,
                self.KL_global,
                self.assigned_labels_map             
                ) = self.select_flippable_indices_entropy(model=self.model,
                                loader=self.loaders_notransform,
                                select_imgs_ratio=self.args.esfda_select_imgs_ratio,
                                random_select_ratio=self.args.random_select_ratio,
                                reverse_imgs=self.args.esfda_reverse_imgs,
                                entropy_threshold=self.args.entropy_threshold,
                                retain_all_others=self.args.retain_all_others
                            )


                print(f"[Static] Sampling done once before training.")

                # ------------------------------------------------------------
                #  BUILD RETAIN SETS : overpred and underpred
                # ------------------------------------------------------------

                # 1) Xretain = all images not in Dforget
                all_seen = list(self.idx_to_pred.keys())

                self.Xretain = [idx for idx in all_seen if idx not in self.flipped_indices]

                # 2) Xretain_overpred = retain images whose predicted class is overpred_class
                over = set([cls for cls, _ in self.overpred_class])      
                self.retain_overpred = [idx for idx in self.Xretain if self.idx_to_pred[idx] in over]

                # 3) Xretain_underpred = retain images whose predicted class is underpred_class
                under = set([cls for cls, _ in self.underpred_class])  
                self.retain_underpred = [idx for idx in self.Xretain if self.idx_to_pred[idx] in under]

            elif (self.args.dynamic_selection and self.epoch % self.args.resample_every == 0):
                # Dynamic mode → resample every N epochs
                self.pred_distribution, self.overpred_class, self.underpred_class = self.compute_prediction_bias(model=self.model,
                            loader_notransform=self.loaders_notransform, top_k=None
                        )
                (
                self.flipped_indices, 
                self.reinforce_indices, 
                self.idx_to_pred, 
                self.entropy_all, 
                self.all_selected, 
                self.stable_selected, 
                self.stable_labels,
                self.probs_all,
                self.KL_global,
                self.assigned_labels_map     
                ) = self.select_flippable_indices_entropy(model=self.model,
                                loader=self.loaders_notransform,
                                select_imgs_ratio=self.args.esfda_select_imgs_ratio,
                                random_select_ratio=self.args.random_select_ratio,
                                reverse_imgs=self.args.esfda_reverse_imgs,
                                entropy_threshold=self.args.entropy_threshold,
                                retain_all_others=self.args.retain_all_others
                            )

                print(f"[Dynamic] Resampled Dforget / Dretain at epoch {self.epoch}.")




    def on_epoch_start(self):
        torch.cuda.empty_cache()

        self.t_init_epoch = dt.datetime.now()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        if self.args.sf_uda:
            self._sf_uda_before_epoch_process()

        #     if self.args.esfda:
        #         if self.args.esfda_select_imgs:
        #             # select images to shift label
                    
        #             self.flipped_indices, self.reinforce_indices, self.idx_to_pred = self.select_flippable_indices_distances(
        #                 model=self.model,
        #                 loader=self.loaders,
        #                 select_imgs_ratio=self.args.esfda_select_imgs_ratio
        #             )

        #         else:
        #             self.flipped_indices = self.select_flippable_indices(
        #                 model=self.model,
        #                 loader=self.loaders,
        #                 select_imgs_ratio=self.args.esfda_select_imgs_ratio
        #             )

        # final
        self.model.train()

        if self.args.sf_uda:
            if self.args.shot or self.args.faust or self.args.sfde or self.args.cdcl or self.args.pxsfde:  # shot/faust/sfde/cdcl method
                self.model.freeze_cl_hypothesis()  # last linear weights +
                # bias of classifier. some wsol methods do not have a last
                # linear classifier: either simple fully conv layers, attention,
                # or no weights (simple max pooling for e.g.)
            elif self.args.esfda or self.args.grsfda:
                if self.args.freeze_classifier_sfda:
                    self.model.freeze_cl_hypothesis()  # last linear weights + bias of
                    # classifier. some wsol methods do not have a last linear
                    # classifier: either simple fully conv layers, attention,
                    # or no weights (simple max pooling for e.g.)
                if self.args.freeze_encoder_sfda:
                    self.model.freeze_encoder()
 

            elif self.args.adadsa:
                self.model = adadsa.adadsa_freeze_all_model_except_bn_a(
                    self.model)
            elif self.args.nrc or self.args.rgv:
                pass
            else:  # todo
                raise NotImplementedError('Add more SFUDA methods.')

    def on_epoch_end(self):
        self.loss.update_t()
        # todo: temp. delete later.
        self.loss.check_losses_status()


        if self.args.ds_to_compute_acc_trainset_source_target == constants.CAMELYON512 and self.epoch % self.args.cmpt_epoch == 0:
            #self.compute_acc_on_source_and_target(self.epoch)
            #self.compute_loc_on_source_and_target(self.epoch)
            if self.args.measure_loc:
                self.compute_loc_on_target(self.epoch)

            self.compute_acc_on_target(self.epoch)
            #self.compute_loc_on_target(self.epoch)


        self.t_end_epoch = dt.datetime.now()
        delta_t = self.t_end_epoch - self.t_init_epoch
        DLLogger.log(fmsg(f'Train epoch runtime: {delta_t}'))

        torch.cuda.empty_cache()

    def random(self):
        self.counter = self.counter + 1
        self.seed = self.seed + self.counter
        set_seed(seed=self.seed, verbose=False)


    @torch.no_grad()
    def select_flippable_indices(self, model, loader, select_imgs_ratio=0.1):
        model.eval()
        all_indices = []
        cancer_pred_indices = []

        loader = loader['train']

        for batch_idx, (images, targets, p_glabel, index,
                        raw_imgs, std_cams, masks, views) in tqdm(
                enumerate(loader), ncols=constants.NCOLS, total=len(loader)):
            images = images.cuda(self.args.c_cudaid)

            logits = model(images)
            preds = logits.argmax(dim=1)

            for i in range(images.size(0)):
                idx = index[i]
                if preds[i].item() == 1:
                    cancer_pred_indices.append(idx)


        py_random.seed(self.seed)
        num = int(len(cancer_pred_indices) * select_imgs_ratio)
        selected_indices = py_random.sample(cancer_pred_indices, num)

        return set(selected_indices) 

    def compute_distance_to_opposite_anchor(self, features, preds, anchors):
        """
        Args:
            features: Tensor [B, D] - feature vectors from the model
            preds: Tensor [B] - predicted class indices (0 or 1)
            anchors: Tensor [2, D] - linear classifier weights for each class
        Returns:
            distances: Tensor [B] - L2 distances to the opposite anchor
        """
        device = features.device
        anchors = torch.from_numpy(anchors).to(device)
        
        
        opp_class_indices = 1 - preds.long()  # [B]
        
        
        opp_anchors = anchors[opp_class_indices]  # [B, D]
        
        # Distance L2
        distances = F.pairwise_distance(features, opp_anchors, p=2)  # [B]
        
        return distances


    @torch.no_grad()
    def init_y_bar(self, model, loader):

        model.eval()
        loader = loader
        #loader_notransform.dataset.transform = get_eval_transforms_global(self.args.crop_size)

        y_bar = {} 

        num_classes = self.args.num_classes

        for batch_idx, (images, targets, p_glabel, index,
                        raw_imgs, std_cams, masks, views, _) in tqdm(
                            enumerate(loader), ncols=constants.NCOLS, total=len(loader)):
            images = images.cuda(self.args.c_cudaid)
            logits = model(images)
            probs = F.softmax(logits, dim=1)  

            for j, name in enumerate(index):
                if isinstance(name, (list, tuple)):
                    name = name[0]
                if torch.is_tensor(name):
                    name = name.item() if name.numel() == 1 else name
                name = str(name)

                y_bar[name] = probs[j].detach().clone()

        return y_bar
    
    @torch.no_grad()
    def update_y_bar_full(self, model, loader):

        model.eval()
        num_classes = self.args.num_classes


        for batch_idx, (images, targets, p_glabel, index,
                    raw_imgs, std_cams, masks, views, _) in tqdm(
                        enumerate(loader), ncols=constants.NCOLS, total=len(loader),
                        desc="Updating y_bar"):

            images = images.cuda(self.args.c_cudaid, non_blocking=True)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            for j, name in enumerate(index):
                if isinstance(name, (list, tuple)):
                    name = name[0]
                if torch.is_tensor(name):
                    name = name.item() if name.numel() == 1 else name
                name = str(name)

                old_y = self.index_src_logits.get(name, torch.zeros(num_classes, device=probs.device))
                new_y = self.args.erl_beta * old_y + (1 - self.args.erl_beta) * probs[j].detach()
                self.index_src_logits[name] = new_y.clone().detach()

    @torch.no_grad()
    def compute_prediction_bias(self, model, loader_notransform, top_k = None):

        model.eval()
        loader_notransform = loader_notransform['train']
        loader_notransform.dataset.transform = get_eval_transforms_global(self.args.crop_size)

        
        
        num_classes = self.args.num_classes
        class_counts = torch.zeros(num_classes, device=self.args.c_cudaid)
        total_samples = 0

        for batch_idx, (images, targets, p_glabel, index,
                        raw_imgs, std_cams, masks, views, _) in tqdm(
                            enumerate(loader_notransform), ncols=constants.NCOLS, total=len(loader_notransform)):

            images = images.cuda(self.args.c_cudaid)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            #num_classes = probs.size(1)

            # if class_counts is None:
            #     class_counts = torch.zeros(num_classes, device=images.device)

            # class_counts += torch.bincount(preds, minlength=num_classes).float()
            # total_samples += preds.numel()

            for c in range(num_classes):
                class_counts[c] += (preds == c).sum()

            total_samples += preds.size(0)

        # --- Distribution ---
        pred_distribution = (class_counts / total_samples).cpu().numpy()

        uniform = 1.0 / self.args.num_classes

        overpred_classes = [(int(i), float(p)) for i, p in enumerate(pred_distribution) if p > uniform]
        underpred_classes = [(int(i), float(p)) for i, p in enumerate(pred_distribution) if p < uniform]

        overpred_classes.sort(key=lambda x: x[1], reverse=True)
        underpred_classes.sort(key=lambda x: x[1])

        return pred_distribution, overpred_classes, underpred_classes

    @torch.no_grad()
    def select_flippable_indices_distances_2(self, model, loader, esfda_distance_normal=4.0, esfda_distance_cancer=6.0, select_imgs_ratio=0.1):
        model.eval()
        flipped_indices = set()
        reinforce_indices = set()
        cancer_pred_indices = []

        loader = loader['train']

        for batch_idx, (images, targets, p_glabel, index,
                        raw_imgs, std_cams, masks, views) in tqdm(
                enumerate(loader), ncols=constants.NCOLS, total=len(loader)):
            
            images = images.cuda(self.args.c_cudaid)
            logits = model(images)
            preds = logits.argmax(dim=1)

            anchors = model.get_linear_weights.detach().cpu().numpy()

            features = model.lin_ft

            distances = self.compute_distance_to_opposite_anchor(features, preds, anchors)

            for i in range(images.size(0)):
                idx = index[i]
                pred = preds[i].item()
                dist = distances[i].item()

                if pred == 1:  
                    if dist < esfda_distance_normal:
                        cancer_pred_indices.append(idx)
                    elif dist > esfda_distance_cancer:
                        reinforce_indices.add(idx)  


        py_random.seed(self.seed)
        num = int(len(cancer_pred_indices) * select_imgs_ratio)
        selected_flippable = py_random.sample(cancer_pred_indices, num)

        return set(selected_flippable), reinforce_indices

    @torch.no_grad()
    def select_flippable_indices_distances(self, model, loader, select_imgs_ratio=0.1,
                                       random_select_ratio=1.0):
        model.eval()
        cancer_pred_distances = []  
        loader = loader['train']

        idx_to_pred = {} 

        for batch_idx, (images, targets, p_glabel, index,
                        raw_imgs, std_cams, masks, views) in tqdm(
                enumerate(loader), ncols=constants.NCOLS, total=len(loader)):
            images = images.cuda(self.args.c_cudaid)
            logits = model(images)
            preds = logits.argmax(dim=1)
            anchors = model.get_linear_weights.detach().cpu().numpy()
            features = model.lin_ft
            distances = self.compute_distance_to_opposite_anchor(features, preds, anchors)
            for i in range(images.size(0)):
                idx = index[i]
                pred = preds[i].item()
                dist = distances[i].item()

                idx_to_pred[idx] = pred


                if pred == 1:
                    cancer_pred_distances.append((idx, dist))

        cancer_pred_distances.sort(key=lambda x: x[1])
        # num = int(len(cancer_pred_distances) * select_imgs_ratio)
        # selected_flippable = set(idx for idx, _ in cancer_pred_distances[:num])
        # reinforce_indices = set() 

        py_random.seed(self.seed)

        preselect_num = int(len(cancer_pred_distances) * select_imgs_ratio)
        preselected_indices = cancer_pred_distances[:preselect_num]

        final_select_num = max(1, int(len(preselected_indices) * random_select_ratio))
        selected_indices = py_random.sample(preselected_indices, final_select_num)

        selected_flippable = set(idx for idx, _ in selected_indices)
        reinforce_indices = set()

        return selected_flippable, reinforce_indices, idx_to_pred

    @torch.no_grad()
    def select_flippable_indices_entropy(
        self,
        model,
        loader,
        select_imgs_ratio=0.1,
        random_select_ratio=1.0,
        reverse_imgs=True,
        stable_per_class=200,
        sub_flippable_ratio=1.0,
        entropy_threshold=None,
        retain_all_others=False,   
    ):
        """
        Select flippable (forget) and stable (retain) image indices using entropy-based
        uncertainty estimation. High-entropy samples are flipped (assigned top-2 label)
        while the rest keep their top-1 prediction.

        Args:
            model: model used to compute entropy and top-k predictions.
            loader: dataloader dictionary containing 'train'.
            select_imgs_ratio: proportion of high-entropy images to consider.
            random_select_ratio: random sub-sampling ratio of selected images.
            reverse_imgs: if True, entropy is sorted descending (highest first).
            stable_per_class: (unused legacy argument).
            sub_flippable_ratio: (unused legacy argument).
            entropy_threshold: if provided, overrides ratio selection with absolute threshold.
            retain_all_others: if True → ALL non-flipped images become Xretain.

        Returns:
            selected_flippable: set of indices selected for flipping.
            reinforce_indices: empty set (legacy placeholder).
            idx_to_pred: mapping idx → predicted class.
            entropy_all: mapping idx → entropy value.
            flippable_subset: same as selected_flippable.
            stable_selected: list of retained indices.
            stable_labels: list of stable predicted labels.
        """

        model.eval()
        loader = loader['train']

        # --- Storage structures ---
        entropy_list = []        # List of (idx, entropy) for the classes selected to flip
        entropy_all = {}         # idx -> entropy
        idx_to_pred = {}         # idx -> predicted class
        idx_to_target = {}       # idx -> GT label (for analysis)
        idx_to_top1 = {}         # idx -> top-1 prediction
        idx_to_top2 = {}         # idx -> top-2 prediction
        by_class_all = {c: [] for c in range(self.args.num_classes)}  # class -> list of (idx, entropy)
        probs_all = {}   # idx -> probability vector
        idx_to_name = {}


        K = self.args.num_classes
        pred_count = torch.zeros(K)        # accumulate class frequencies
        total_samples = 0                  # count total images

        # Classes that are over-predicted and from which we will select Dforget
        freeze_classes = [c for c, _ in self.overpred_class]
        if len(freeze_classes) == 0:
            # Fallback: freeze all classes that appear in predictions
            freeze_classes = list(set(idx_to_pred.values()))

        # ========================================================
        # Pass through the dataset once: compute probabilities,
        # entropy, top-k predictions, and store mappings.
        # ========================================================
        for batch_idx, (
            images, targets, p_glabel, index,
            raw_imgs, std_cams, masks, views, _
        ) in tqdm(
            enumerate(loader),
            ncols=constants.NCOLS,
            total=len(loader)
        ):

            images = images.cuda(self.args.c_cudaid)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            # UPDATE GLOBAL CLASS COUNT
            for c in preds:
                pred_count[c.item()] += 1

            total_samples += preds.size(0)



            C = probs.size(1)
            k = 2 if C >= 2 else 1
            top2_vals, top2_idx = probs.topk(k=k, dim=1, largest=True, sorted=True)

            entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)

            # --- Process each sample in the batch ---
            for i in range(images.size(0)):
                idx = index[i]
                idx_to_name[idx] = idx
                pred = preds[i].item()
                gt = targets[i].item()
                ent = entropy[i].item()

                # Save prediction and entropy
                idx_to_pred[idx] = pred
                idx_to_target[idx] = gt
                entropy_all[idx] = ent
                probs_all[idx] = probs[i].detach().cpu()

                # Save top-1 and top-2
                t1 = int(top2_idx[i, 0].item())
                t2 = int(top2_idx[i, 1].item()) if C >= 2 else t1
                idx_to_top1[idx] = t1
                idx_to_top2[idx] = t2

                # Select candidates only from the over-predicted classes
                if pred in freeze_classes:
                    entropy_list.append((idx, ent))

                # Track per-class distributions
                if pred in by_class_all:
                    by_class_all[pred].append((idx, ent))

        # ========================================================
        # STEP 3 : Compute GLOBAL KL(U || predicted distribution)
        # ========================================================
        p_hat = pred_count / total_samples
        U = torch.ones(K) / K
        KL_global = torch.sum(U * (torch.log(U + 1e-12) - torch.log(p_hat + 1e-12)))
        self.KL_global = KL_global.item()



        # ========================================================
        # Sort entropy in descending order → highest uncertainty first
        # ========================================================
        if reverse_imgs:
            entropy_list.sort(key=lambda x: x[1], reverse=True)
        else:
            entropy_list.sort(key=lambda x: x[1])

        py_random.seed(self.seed)

        # ========================================================
        # Select images to flip: threshold-based or ratio-based
        # ========================================================
        if entropy_threshold is not None:
            preselected_indices = [(idx, e) for (idx, e) in entropy_list if e >= entropy_threshold]
            if len(preselected_indices) == 0:
                print(f"[WARN] No images above entropy threshold {entropy_threshold}.")
        else:
            max_n = int(len(entropy_list) * select_imgs_ratio)
            preselected_indices = entropy_list[:max_n]

        # Random subsampling
        final_n = max(1, int(len(preselected_indices) * random_select_ratio))
        if final_n < len(preselected_indices):
            selected_indices = py_random.sample(preselected_indices, final_n)
        else:
            selected_indices = preselected_indices

        selected_flippable = set(idx for (idx, _) in selected_indices)
        flippable_subset = selected_flippable

        # ========================================================
        # Assign labels:
        #  - flippable -> top-2
        #  - stable    -> top-1
        # ========================================================
        assigned_labels_map = {}
        for idx in idx_to_pred.keys():
            if idx in selected_flippable:
                assigned_labels_map[idx] = idx_to_top2.get(idx, idx_to_top1[idx])
            else:
                assigned_labels_map[idx] = idx_to_top1[idx]

        self.assigned_labels_map = assigned_labels_map
        reinforce_indices = set()   # Placeholder for legacy logic

        # ========================================================
        # Select Xretain (stable) set
        # ========================================================
        stable_selected = []
        stable_labels = []

        # All samples seen during traversal
        all_seen = list(idx_to_pred.keys())

        # ========================================================
        # NEW OPTION: retain absolutely all non-flipped images
        # ========================================================
        if retain_all_others:
            # Xretain = all_seen \ Xforget
            stable_selected = [idx for idx in all_seen if idx not in flippable_subset]
            stable_labels   = [idx_to_pred[idx] for idx in stable_selected]

        else:
            # ====================================================
            # Original behavior:
            # two cases depending on stable_match_strategy
            # ====================================================
            preserve_all = self.args.stable_match_strategy

            if not preserve_all:
                # Simple retain: keep everything not flipped
                nonflipped = [idx for idx in all_seen if idx not in flippable_subset]
                stable_selected = nonflipped
                stable_labels = [idx_to_pred[idx] for idx in stable_selected]

            else:
                # Class-balanced retain among freeze_classes
                total_target = len(flippable_subset)

                if total_target > 0 and len(freeze_classes) > 0:
                    # Compute quotas
                    total_candidates = sum(len(by_class_all[c]) for c in freeze_classes)
                    if total_candidates == 0:
                        targets_per_class = {c: 0 for c in freeze_classes}
                    else:
                        quotas = {
                            c: (len(by_class_all[c]) / total_candidates) * total_target
                            for c in freeze_classes
                        }
                        base = {c: int(quotas[c]) for c in freeze_classes}
                        allocated = sum(base.values())
                        remainder = total_target - allocated

                        # Distribute remainder
                        fracs = sorted(
                            [(c, quotas[c] - base[c]) for c in freeze_classes],
                            key=lambda t: t[1],
                            reverse=True
                        )
                        targets_per_class = base
                        for c, _ in fracs:
                            if remainder <= 0:
                                break
                            targets_per_class[c] += 1
                            remainder -= 1

                    # Sort by entropy ascending (most confident retained first)
                    for c in by_class_all.keys():
                        by_class_all[c].sort(key=lambda x: x[1])

                    # Select stable samples for each class
                    for c in freeze_classes:
                        pool = [(idx, ent) for (idx, ent) in by_class_all[c]
                                if idx not in flippable_subset]

                        take = min(targets_per_class.get(c, 0), len(pool))
                        chosen = pool[:take]
                        stable_selected.extend([idx for (idx, _) in chosen])
                        stable_labels.extend([c] * take)

        # ========================================================
        # Compute monitoring statistics (correctness of flip/retain)
        # ========================================================
        flip_correct = sum(1 for idx in flippable_subset
                        if idx_to_pred.get(idx) == idx_to_target.get(idx))
        flip_total = len(flippable_subset)
        flip_acc = flip_correct / flip_total if flip_total > 0 else 0.0

        stable_correct = sum(1 for idx in stable_selected
                            if idx_to_pred.get(idx) == idx_to_target.get(idx))
        stable_total = len(stable_selected)
        stable_acc = stable_correct / stable_total if stable_total > 0 else 0.0

        results = {
            "retain": flip_acc,
            "forget": stable_acc
        }

        # Save metrics
        out_txt = os.path.join(self.args.outd, "results_unlearning_acc.txt")
        out_pkl = os.path.join(self.args.outd, "results__unlearning_acc.pkl")

        with open(out_txt, "a") as f:
            f.write(f"Epoch {self.epoch}\n")
            f.write(f"Nb retain : {len(stable_selected)}\n")
            f.write(f"retain acc : {flip_acc:.4f}\n")
            f.write(f"Nb forget : {len(flippable_subset)}\n")
            f.write(f"forget acc : {stable_acc:.4f}\n")
            f.write("\n")

        # Load previous results if exists
        if os.path.exists(out_pkl):
            with open(out_pkl, "rb") as f:
                history = pkl.load(f)
        else:
            history = []

        history.append(results)

        with open(out_pkl, "wb") as f:
            pkl.dump(history, f)

        Forget_labels = [assigned_labels_map[idx] for idx in selected_flippable]
        Retain_labels = [assigned_labels_map[idx] for idx in stable_selected]


        # return (
        #     selected_flippable,       
        #     stable_selected,           
        #     assigned_labels_map,       
        #     KL_global 
        #     )

        return (
            selected_flippable,
            reinforce_indices,
            idx_to_pred,
            entropy_all,
            flippable_subset,
            stable_selected,
            stable_labels,
            probs_all,
            KL_global,
            assigned_labels_map
        )


    @torch.no_grad()
    def select_flippable_indices_gt(self,model, loader, n_per_class=2000):
        """
        """
        loader = loader['train']

        idx_to_target = {}
        idx_to_pred   = {}    
        entropy_all   = {}   

        flippable_pool = []   # FP: pred overpred & gt=0
        stable_pool    = []

        class_to_indices = {0: [], 1: []}

        overpred_classes = [c for c, _ in self.overpred_class]

        for batch_idx, (images, targets, p_glabel, index,
                        raw_imgs, std_cams, masks, views) in enumerate(loader):


            images = images.cuda(self.args.c_cudaid) 
            logits = model(images) 
            probs = F.softmax(logits, dim=1) 
            preds = probs.argmax(dim=1)

            for i in range(images.size(0)):
                idx = index[i]
                gt  = targets[i].item()
                pr  = int(preds[i].item())
                idx_to_target[idx] = gt
                idx_to_pred[idx]   = pr    
                entropy_all[idx]   = 0.0  

                if pr in overpred_classes:
                    if gt == 0 and pr != gt:
                        # flippable
                        flippable_pool.append(idx)
                    elif gt == 1 and pr == gt:
                        # stable
                        stable_pool.append(idx)

        k = min(len(flippable_pool), len(stable_pool), n_per_class)

        flippable_selected = set(py_random.sample(flippable_pool, k)) if len(flippable_pool) > k else set(flippable_pool)
        stable_selected    = py_random.sample(stable_pool, k) if len(stable_pool) > k else list(stable_pool)

        # labels stables (pour retour)
        stable_labels = [idx_to_target[idx] for idx in stable_selected]

        # ---- build assigned_labels_map ----
        assigned_labels_map = {}
        for idx, gt in idx_to_target.items():
            if idx in flippable_selected:
                # flip : cancer -> normal (class 0)
                assigned_labels_map[idx] = gt
            else:
                assigned_labels_map[idx] = gt
        self.assigned_labels_map = assigned_labels_map

        print(f"[INFO] Flippables (FP overpred): {len(flippable_selected)} | Stables (TP overpred): {len(stable_selected)} | k={k}")

        reinforce_indices = set()   
        flippable_subset  = set(flippable_selected)

        return flippable_selected, reinforce_indices, idx_to_pred, entropy_all, flippable_subset, stable_selected, stable_labels

    @torch.no_grad()
    def select_flippable_indices_random(
        self,
        model,
        loader,
        flip_ratio=0.1,               # ratio of images to flip
        flip_count=None,              # absolute number of imgs to flip
        mode="pred_cancer",           # "pred_cancer" | "all" | "per_class"
        freeze_classes=(1,),          # for stable
        balance_stables=True        
    ):
        """
        Ablation: random selection of images to flip.
        Returns: selected_flippable, reinforce_indices, idx_to_pred, entropy_all,
                flippable_subset, stable_selected, stable_labels
        Also updates self.assigned_labels_map for the training loop.
        """
        model.eval()
        loader = loader['train']

        import random as py_random
        py_random.seed(getattr(self, "seed", 0))

        idx_to_pred   = {}
        idx_to_target = {}
        entropy_all   = {}


        all_indices = []
        by_pred_class = {}

        # ---- 1) Build pool ----
        for batch_idx, (images, targets, p_glabel, index,
                        raw_imgs, std_cams, masks, views) in enumerate(loader):
            images = images.cuda(self.args.c_cudaid)
            logits = model(images)
            probs  = F.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)

            # entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)

            for i in range(images.size(0)):
                idx = index[i]
                y   = preds[i].item()
                gt  = targets[i].item()

                idx_to_pred[idx]   = y
                idx_to_target[idx] = gt
                entropy_all[idx]   = float(entropy[i].item())

                all_indices.append(idx)
                by_pred_class.setdefault(y, []).append(idx)

        N = len(all_indices)
        if N == 0:
            self.assigned_labels_map = {}
            return set(), set(), idx_to_pred, entropy_all, set(), [], []

        if mode == "pred_cancer":
            pool = by_pred_class.get(1, [])  
        elif mode == "all":
            pool = list(all_indices)
        elif mode == "per_class":
            pool = None  
        else:
            raise ValueError(f"Unknown mode={mode}")

        if flip_count is None:
            base = len(pool) if (mode != "per_class" and pool is not None) else N
            flip_count = max(1, int(base * float(flip_ratio)))

        # ---- random selection to flip ----
        selected_flippable = set()

        if mode in ("pred_cancer", "all"):
            if len(pool) > 0:
                k = min(flip_count, len(pool))
                selected_flippable = set(py_random.sample(pool, k))
        else:
            # "per_class":  
            total_pool = sum(len(v) for v in by_pred_class.values())
            if total_pool > 0:
                quotas_float = {
                    c: (len(by_pred_class[c]) / total_pool) * flip_count
                    for c in by_pred_class
                }
                quotas_int = {c: int(quotas_float[c]) for c in by_pred_class}
                allocated = sum(quotas_int.values())
                remainder = flip_count - allocated
                fracs = sorted(
                    ((c, quotas_float[c] - quotas_int[c]) for c in by_pred_class),
                    key=lambda t: t[1], reverse=True
                )
                for c, _ in fracs:
                    if remainder <= 0:
                        break
                    quotas_int[c] += 1
                    remainder -= 1
                for c, q in quotas_int.items():
                    pool_c = by_pred_class[c]
                    if len(pool_c) > 0 and q > 0:
                        q = min(q, len(pool_c))
                        selected_flippable.update(py_random.sample(pool_c, q))

        # ---- 4) Build assigned_labels_map  ----
        assigned_labels_map = {}
        for idx in idx_to_pred.keys():
            y = idx_to_pred[idx]
            if idx in selected_flippable:
                assigned_labels_map[idx] = 1 - y if y in (0, 1) else (0 if y != 0 else 1)
            else:
                assigned_labels_map[idx] = y

        self.assigned_labels_map = assigned_labels_map

        # ---- 5) Construire stables ----
        all_seen = list(idx_to_pred.keys())
        nonflipped = [idx for idx in all_seen if idx not in selected_flippable]

        if freeze_classes is not None and len(freeze_classes) > 0:
            nonflipped = [idx for idx in nonflipped if idx_to_pred[idx] in set(freeze_classes)]

        if balance_stables:
            # same number of stable to flip
            k_stables = min(len(nonflipped), len(selected_flippable))
            stable_selected = py_random.sample(nonflipped, k_stables) if k_stables > 0 else []
        else:
            # Save all stables
            stable_selected = nonflipped

        stable_labels = [idx_to_pred[idx] for idx in stable_selected]

        reinforce_indices = set()
        flippable_subset  = set(selected_flippable)

        print(f"[RANDOM] mode={mode} | flips={len(selected_flippable)} | "
            f"stables={len(stable_selected)} (balanced={balance_stables})")

        return selected_flippable, reinforce_indices, idx_to_pred, entropy_all, flippable_subset, stable_selected, stable_labels, probs_all     



    @torch.no_grad()
    def select_flippable_indices_entropy_probabilistic(
        self,
        model,
        loader,
        flip_on_pred_classes=(1,),   
        freeze_classes=(0,1),      
        seed_per_epoch: bool = True,
    ):

        model.eval()
        loader = loader['train']
        selected_flippable = set()
        reinforce_indices = set() 
        idx_to_pred = {}
        entropy_all = {}
        idx_to_target = {}

        stable_selected, stable_labels = [], []

        all_idx, all_pred_top1, all_pred_top2, all_H, all_targets = [], [], [], [], []


        K = int(self.args.num_classes)
        logK = math.log(max(2, K))


        set_seed(seed=self.default_seed, verbose=False)

        #flip_set = None if flip_on_pred_classes is None else set(flip_on_pred_classes)
        #freeze_set = set(freeze_classes) if freeze_classes is not None else set()
        flip_set = [c for c, _ in self.overpred_class]
        freeze_classes = [c for c, _ in self.overpred_class]

        for batch_idx, (images, targets, p_glabel, index, raw_imgs, std_cams, masks, views) in tqdm(
            enumerate(loader), ncols=constants.NCOLS, total=len(loader)):

            images = images.cuda(self.args.c_cudaid)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            # top-2 classes & preds
            K = probs.size(1)
            if K >= 2:
                top2_vals, top2_idx = probs.topk(k=2, dim=1)        # [B,2]
                pred_top1 = top2_idx[:, 0]
                pred_top2 = top2_idx[:, 1]
            else:
                pred_top1 = torch.zeros(probs.size(0), dtype=torch.long, device=probs.device)
                pred_top2 = pred_top1.clone()

            ent = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)  # [B]

            
            for b in range(images.size(0)):
                idx  = index[b]
                t1   = int(pred_top1[b].item())
                t2   = int(pred_top2[b].item()) if K >= 2 else int(1 - t1)  
                Hval = float(ent[b].item())

                idx_to_pred[idx] = t1              # prediction top-1 
                entropy_all[idx] = Hval
                gt  = targets[b].item()
                idx_to_target[idx] = gt   # save GT
                
                all_idx.append(idx)
                all_pred_top1.append(t1)
                all_pred_top2.append(t2)
                all_H.append(Hval)
                all_targets.append(int(targets[b].item()))


        # -------- 2) Min–max global sur H --------
        H_t   = torch.tensor(all_H, dtype=torch.float32)
        H_min = float(H_t.min().item())
        H_max = float(H_t.max().item())
        denom = H_max - H_min
        eps   = 1e-8

        # selected_flippable = set()
        # reinforce_indices  = set()
        # stable_selected, stable_labels = [], []

        # Calcul p_i & tirage
        p_t   = ((H_t - H_min) / (denom + eps)).clamp_(0.0, 1.0)    # [N]

        p_np = p_t.detach().cpu().numpy() if hasattr(p_t, "detach") else np.array(p_t)

        plt.figure(figsize=(16,12))
        plt.hist(p_np, bins=40, range=(0,1), alpha=0.7, color="blue", edgecolor="black")
        plt.xlabel("Normalized entropy probability $p_t$")
        plt.ylabel("Frequency")
        plt.title("Histogram of normalized entropy $p_t$")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        out_path = os.path.join(self.args.outd, f"hist_predclass.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

       
        p_by_class = {}  # {pred_label: {"correct": [], "incorrect": []}}

        for i, idx in enumerate(all_idx):
            true_label = all_targets[i]
            pred_label = all_pred_top1[i]
            Hval       = float(H_t[i].item())

            # Normalisation H -> p
            p_val = (Hval - H_min) / (denom + eps)
            #p_val = max(0.0, min(1.0, p_val))  # clamp


            if pred_label not in p_by_class:
                p_by_class[pred_label] = {"correct": [], "incorrect": []}

            if pred_label == true_label:
                
                p_by_class[pred_label]["correct"].append(p_val)
            else:
               
                p_by_class[pred_label]["incorrect"].append(p_val)

        # # Plot
        # h_max = math.log(max(2, self.args.num_classes))
        # bins = 40  # borne sup de l'entropie
        # bin_edges = np.linspace(0.0, h_max, bins)


        # for cls, splits in p_by_class.items():
        #     ent_correct = np.array(splits["correct"])
        #     ent_incorrect = np.array(splits["incorrect"])

        #     # Comptage
        #     counts_correct, _ = np.histogram(ent_correct, bins=bin_edges)
        #     counts_incorrect, _ = np.histogram(ent_incorrect, bins=bin_edges)
        #     total_counts = counts_correct + counts_incorrect

        #     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        #     # Plot
        #     plt.figure(figsize=(16,12))
        #     plt.bar(bin_centers, counts_correct, width=np.diff(bin_edges),
        #             alpha=0.8, label="Correct", align="center", color="#4C9AFF")
        #     plt.bar(bin_centers, counts_incorrect, width=np.diff(bin_edges),
        #             bottom=counts_correct, alpha=0.8, label="Incorrect", align="center", color="#FF9933")

        #     # Ajouter totaux + pourcentages
        #     for total, correct, incorrect, x in zip(total_counts, counts_correct, counts_incorrect, bin_centers):
        #         if total > 0:
        #             plt.text(x, total + 0.5, f"{total}", rotation=90, ha="center", va="bottom", fontsize=7)
        #             if correct > 0:
        #                 plt.text(x, correct / 2, f"{100 * correct / total:.1f}%", 
        #                         ha="center", va="center", fontsize=7, color="white")
        #             if incorrect > 0:
        #                 plt.text(x, correct + incorrect / 2, f"{100 * incorrect / total:.1f}%", 
        #                         ha="center", va="center", fontsize=7, color="black")

        #     plt.title(f"Entropy distribution — Predicted class {cls}")
        #     plt.xlabel("Entropy per image")
        #     plt.ylabel("Nb images")
        #     plt.legend()
        #     plt.grid(True, alpha=0.3)
        #     plt.tight_layout()

        #     out_path = os.path.join(self.args.outd, f"hist_entropy_predclass{cls}.png")
        #     plt.savefig(out_path, dpi=300)
        #     plt.close()
        #     print(f"✅ Histogram saved : {out_path}")

        for cls, splits in p_by_class.items():
            plt.figure(figsize=(6,4))

            if len(splits["correct"]) > 0:
                plt.hist(
                    splits["correct"],
                    bins=30, alpha=0.6, color="blue",
                    edgecolor="black", label="Correct"
                )

            if len(splits["incorrect"]) > 0:
                plt.hist(
                    splits["incorrect"],
                    bins=30, alpha=0.6, color="red",
                    edgecolor="black", label="Incorrect"
                )

            plt.xlabel("Normalized entropy probability p_i")
            plt.ylabel("Frequency")
            plt.title(f"Histogram p_t for predicted class {cls}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()

            # Save
            output_path = os.path.join(self.args.outd, f"hist_p_t_predclass{cls}.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300)
            plt.close()

            print(f"✅ Histogram saved : {output_path}")


        draw_t = torch.bernoulli(p_t).to(torch.bool)                # [N] tirage 0/1

        #flip_set   = None if flip_on_pred_classes is None else set(flip_on_pred_classes)
        #freeze_set = set(freeze_classes) if freeze_classes is not None else set()

        flip_set = [c for c, _ in self.overpred_class]
        freeze_set = [c for c, _ in self.overpred_class]

        selected_flippable = set()
        stable_selected, stable_labels = [], []


        flip_labels_map = {}        # idx -> label (top-2) 
        assigned_labels_map = {}    # idx -> label final (flip: top-2, stable: top-1)
        entropy_flip = []

        for i in range(len(all_idx)):
            idx   = all_idx[i]
            t1    = all_pred_top1[i]
            t2    = all_pred_top2[i]
            draw1 = bool(draw_t[i].item())
            # Flip authorized for this class
            flip_allowed = (flip_set is None) or (t1 in flip_set)

            if draw1 and flip_allowed:
                # FLIP -> vers top-2
                entropy_flip.append(all_H[i])
                selected_flippable.add(idx)
                flip_labels_map[idx] = t2
                assigned_labels_map[idx] = t2
            else:
                # STABLE
                if t1 in freeze_set:
                    stable_selected.append(idx)
                    stable_labels.append(t1)

                assigned_labels_map[idx] = t1

        num_flips = len(selected_flippable)


        #self.assigned_labels_map = assigned_labels_map

        preserve_all = self.args.stable_match_strategy

        if not preserve_all:
            final_stable_selected = stable_selected
            final_stable_labels   = stable_labels

        else:
            total_target = num_flips
            by_class_all = {c: [] for c in freeze_set}
            for idx, label in zip(stable_selected, stable_labels):
                by_class_all[label].append(idx)

            total_candidates = sum(len(by_class_all[c]) for c in freeze_set)
            if total_candidates == 0:
                targets_per_class = {c: 0 for c in freeze_set}
            else:
                quotas = {c: (len(by_class_all[c]) / total_candidates) * total_target
                        for c in freeze_set}
                base = {c: int(quotas[c]) for c in freeze_set}
                allocated = sum(base.values())
                remainder = total_target - allocated

                fracs = sorted(
                    ((c, quotas[c] - base[c]) for c in freeze_set),
                    key=lambda t: t[1], reverse=True
                )
                targets_per_class = base
                for c, _ in fracs:
                    if remainder <= 0:
                        break
                    targets_per_class[c] += 1
                    remainder -= 1

            # Réduire stable_selected selon quotas
            final_stable_selected, final_stable_labels = [], []
            for c in freeze_set:
                pool = by_class_all[c]
                take = min(targets_per_class.get(c, 0), len(pool))
                chosen = pool[:take]
                final_stable_selected.extend(chosen)
                final_stable_labels.extend([c] * take)

        # --- Remplacer ---
        stable_selected = final_stable_selected
        stable_labels   = final_stable_labels

        plt.figure()
        bins = 60  
        plt.hist(entropy_flip, bins=bins, alpha=0.6, label=f"flippable (n={len(entropy_flip)})")
        plt.xlabel("Entropy (nats)")
        plt.ylabel("Count")
        plt.legend()
        plt.title("Entropy distribution: flippable vs stable")
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.outd, "unlearning_entropy_hist.png"), dpi=200)
        plt.close()

        flippable_subset = set(selected_flippable)
        all_selected = set(selected_flippable).union(set(stable_selected))

        flip_correct, flip_total = 0, 0
        for idx in flippable_subset:
            if idx in idx_to_target:
                flip_total += 1
                if idx_to_pred[idx] == idx_to_target[idx]:
                    flip_correct += 1
        flip_acc = flip_correct / flip_total if flip_total > 0 else 0.0

        stable_correct, stable_total = 0, 0
        for idx in stable_selected:
            if idx in idx_to_target:
                stable_total += 1
                if idx_to_pred[idx] == idx_to_target[idx]:
                    stable_correct += 1
        stable_acc = stable_correct / stable_total if stable_total > 0 else 0.0

        results = {
            "retain": flip_acc,
            "forget": stable_acc
        }

        output_path = os.path.join(self.args.outd, "results_unlearning_acc.txt")
        output_pickle = os.path.join(self.args.outd, "results__unlearning_acc.pkl")

        with open(output_path, "w") as f_txt:
            f_txt.write(f"retain : {flip_acc:.4f}\n")
            f_txt.write(f"forget  : {stable_acc:.4f}\n")


        with open(output_pickle, "wb") as f_pkl:
            pkl.dump(results, f_pkl)

        # Enregistre pour usage en aval (p.ex. dans _compute_accuracy_f1)
        self.flip_labels_map = flip_labels_map               # idx -> top-2
        self.assigned_labels_map = assigned_labels_map       # idx -> label final (flip ou stable)
        # 
        self.idx_to_second = {i: s for i, s in zip(all_idx, all_pred_top2)}

        # 
        #flippable_subset = set(selected_flippable)
        reinforce_indices = set()  

        return (
            selected_flippable,
            reinforce_indices,
            idx_to_pred,       # top-1
            entropy_all,
            all_selected,
            stable_selected,
            stable_labels,
        )

    def compute_dynamic_percent(
        self,
        base_percent: float,
        epoch: int,
        n_epochs_step: int = 2,
        growth_factor: float = 2.0,
        max_percent: float = 1.0
        ):
        """
        Ex : base=0.10, step=2 → epochs 1-2: 0.10, 3-4: 0.20, 5-6: 0.40, ...
        """
        step_id = epoch // n_epochs_step
        percent = base_percent * (growth_factor ** step_id)
        percent = min(percent, max_percent)  
        return percent

    
    @torch.no_grad()
    def select_by_entropy_balanced(
        self,
        model,
        loader,
        percent=0.10,
    ):


        model.eval()

        per_class_entropies = {c: [] for c in range(self.args.num_classes)}
        per_class_indices   = {c: [] for c in range(self.args.num_classes)}
        per_class_meta      = {c: [] for c in range(self.args.num_classes)}


        all_image_ids = []
        all_preds = []
        all_gts = []
        all_entropies = []

        for batch_idx, (images, targets, p_glabel, image_ids,
                        raw_imgs, std_cams, masks, views, _) in tqdm(
                            enumerate(loader), ncols=80, total=len(loader)
                        ):

            images = images.cuda(self.args.c_cudaid)
            logits = model(images)
            probs  = F.softmax(logits, dim=1)

            batch_entropy = -(probs * probs.log()).sum(dim=1)   # [B]
            pred = torch.argmax(probs, dim=1)                   # [B]

            for i in range(len(image_ids)):
                cls = int(pred[i].item())
                ent = float(batch_entropy[i].item())
                img_id = image_ids[i]

                per_class_entropies[cls].append(ent)
                per_class_indices[cls].append(img_id)
                per_class_meta[cls].append({
                    "img_id": img_id,
                    "entropy": ent,
                    "pred": cls,
                    "gt": int(targets[i].item())
                })

                all_image_ids.append(img_id)
                all_preds.append(cls)
                all_gts.append(int(targets[i].item()))
                all_entropies.append(ent)


        sorted_per_class = {}
        sorted_meta = {}

        for cls in range(self.args.num_classes):
            ent_list  = torch.tensor(per_class_entropies[cls], dtype=torch.float32)
            idx_list  = per_class_indices[cls]
            meta_list = per_class_meta[cls]

            if len(ent_list) == 0:
                sorted_per_class[cls] = []
                sorted_meta[cls] = []
                continue

            sorted_entropy, sorted_idx = torch.sort(ent_list)
            sorted_img_ids = [idx_list[j] for j in sorted_idx.tolist()]
            sorted_meta_cls = [meta_list[j] for j in sorted_idx.tolist()]

            sorted_per_class[cls] = sorted_img_ids
            sorted_meta[cls] = sorted_meta_cls


        per_class_selected = {}
        per_class_meta_sel = {}
        counts = {}

        for cls in range(self.args.num_classes):
            total_cls = len(sorted_per_class[cls])

            if total_cls == 0:
                per_class_selected[cls] = []
                per_class_meta_sel[cls] = []
                counts[cls] = 0
                continue

            num_cls = max(int(total_cls * percent), 1)

            per_class_selected[cls] = sorted_per_class[cls][:num_cls]
            per_class_meta_sel[cls] = sorted_meta[cls][:num_cls]
            counts[cls] = num_cls

        n_min = min(counts.values()) if len(counts) > 0 else 0

        final_selected_ids = []
        final_selected_meta = []

        for cls in range(self.args.num_classes):
            final_selected_ids.extend(per_class_selected[cls][:n_min])
            final_selected_meta.extend(per_class_meta_sel[cls][:n_min])


        if len(final_selected_meta) > 0:
            acc = sum(m["pred"] == m["gt"] for m in final_selected_meta) / len(final_selected_meta)
            print(f"[Entropy Balanced] Accuracy of selected subset = {acc:.4f}")
        else:
            acc = None


        out_dict = {}

        selected_set = set(final_selected_ids)

        for img_id, pred in zip(all_image_ids, all_preds):
            if img_id in selected_set:
                out_dict[img_id] = pred
            else:
                out_dict[img_id] = -255     

        return final_selected_ids, out_dict, acc


    

    def train(self, split: str, epoch: int) -> dict:
        self.epoch = epoch
        self.random()
        self.on_epoch_start()

        

        assert split == constants.TRAINSET

        if self.args.cdcl or self.args.sfde:
            loader=self.loaders_filtered[split]
        else:
            loader = self.loaders[split]

        total_loss = None
        num_correct = 0
        num_images = 0

        scaler = GradScaler(enabled=self.args.amp)

        mbatchsz = 0
        

        for batch_idx, (images, targets, p_glabel, index,
                        raw_imgs, std_cams, masks, views, aug_images) in tqdm(
                enumerate(loader), ncols=constants.NCOLS, total=len(loader)):

            self.batch_idx = batch_idx
            

            # if self.args.esfda and self.args.esfda_select_imgs:
            #     supervised_labels = torch.full_like(p_glabel, -255)
            #     for i in range(images.size(0)):
            #         img_idx = index[i]
            #         if img_idx in self.flipped_indices:
            #             supervised_labels[i] = 0 
                
                #p_glabel = supervised_labels

            

            if self.args.esfda and self.args.esfda_select_imgs:
                supervised_labels = torch.full_like(p_glabel, -255)   #p_glabel
                entropy_batch = []


                for i in range(images.size(0)):
                    img_idx = index[i]
                    if img_idx in self.all_selected:
                        if self.args.entropy_probabilistic:
                            supervised_labels[i] = int(self.assigned_labels_map[img_idx])
                        else:
                            supervised_labels[i]= int(self.assigned_labels_map[img_idx])

                    # if img_idx in self.flipped_indices:
                    #     supervised_labels[i] = 0  
                    # elif img_idx in self.reinforce_indices:
                    #     supervised_labels[i] = 1 
                    # 
                    if img_idx in self.entropy_all:
                        entropy_batch.append(self.entropy_all[img_idx])
                    else:
                        entropy_batch.append(0.0) 
                #p_glabel = supervised_labels
                # mask_list_forget = [img_name not in self.flipped_indices for img_name in index]
                # #y_pred_batch = torch.tensor([self.idx_to_pred[idx] for idx in index])
                # y_pred_batch = torch.tensor([self.assigned_labels_map[idx] for idx in index])
                # mask_list_retain = [img_name not in self.stable_selected for img_name in index]
                # mask_entropy = torch.tensor(entropy_batch, dtype=torch.float32, device=images.device)
                p_glabel = supervised_labels
                mask_list = [img_name not in self.flipped_indices for img_name in index]
                #y_pred_batch = torch.tensor([self.idx_to_pred[idx] for idx in index])
                y_pred_batch = torch.tensor([self.assigned_labels_map[idx] for idx in index])
                mask_entropy = torch.tensor(entropy_batch, dtype=torch.float32, device=images.device)

                mask_list_retain = [img_name in self.stable_selected for img_name in index]
                mask_list_forget = [img_name in self.flipped_indices for img_name in index]
            
            self.random()
            self.model.train()
            
            if batch_idx == 0:
                mbatchsz = images.shape[0]

            # SFUDA: todo: warning: std_cams must be extracted using the
            #  pseudo-labels not the true labels (offline, online extraction).
            # fill
            images = self._fill_minibatch(images, mbatchsz)
            targets = self._fill_minibatch(targets, mbatchsz)
            p_glabel = self._fill_minibatch(p_glabel, mbatchsz)
            raw_imgs = self._fill_minibatch(raw_imgs, mbatchsz)
            aug_images = self._fill_minibatch(aug_images, mbatchsz)
            index = self._fill_minibatch_list(index, mbatchsz)

            images = images.cuda(self.args.c_cudaid)
            targets = targets.cuda(self.args.c_cudaid)
            p_glabel = p_glabel.cuda(self.args.c_cudaid)
            aug_images = aug_images.cuda(self.args.c_cudaid)
            
            if self.args.esfda and self.args.esfda_select_imgs:
                y_pred_batch = y_pred_batch.cuda(self.args.c_cudaid)

            # SFUDA: estimate img-class pseudo-label on the fly ================
            if self.args.sf_uda:

                if self.args.adadsa:
                    if self.args.ce_pseudo_lb:

                        self.sfuda_master.update_lambda_(
                            epoch=max(0, self.epoch - 1),
                            mb_idx=batch_idx
                        )
                        p_glabel = self.sfuda_master.pseudo_label_imgs(images)

                        self.model = adadsa.adadsa_freeze_all_model_except_bn_a(
                            self.model)

                        # DEBUG // OFF -----------------------------------------
                        # acc = (p_glabel == targets).float().mean() * 100.
                        # msg = f"AdaDSA - MBATCH ACC pseudo-label image-class:" \
                        #       f" {acc} % [{images.shape[0]} samples, " \
                        #       f"lambda: {self.sfuda_master.lambda_}]"
                        # DLLogger.log(fmsg(msg))
                        # ------------------------------------------------------

                elif self.args.nrc:
                    self.nrc_args=self.sfuda_master.update(model=self.model, images=images, index=index)
            # ==================================================================

            if views.ndim == 1:
                views = None

            else:
                views = self._fill_minibatch(views, mbatchsz)
                views = views.cuda(self.args.c_cudaid)

            if masks.ndim == 1:
                masks = None
            else:
                masks = self._fill_minibatch(masks, mbatchsz)
                masks = masks.cuda(self.args.c_cudaid)

            if std_cams.ndim == 1:
                std_cams = None
            else:
                assert std_cams.ndim == 4
                std_cams = self._fill_minibatch(std_cams, mbatchsz)
                std_cams = std_cams.cuda(self.args.c_cudaid)

                with autocast(enabled=self.args.amp):
                    with torch.no_grad():
                        std_cams = self.prepare_std_cams_disq(
                            std_cams=std_cams, image_size=images.shape[2:])

            self.optimizer.zero_grad(set_to_none=True)

            if self.args.esfda and self.args.esfda_entropy_partial:
                with autocast(enabled=self.args.amp):
                    logits, loss = self._one_step_train_unlearning(images,
                                                        raw_imgs,
                                                        targets,
                                                        p_glabel,
                                                        std_cams,
                                                        masks,
                                                        views,
                                                        cal_mask_forget = mask_list_forget,
                                                        cal_mask_retain = mask_list_retain,
                                                        y_pred_batch = y_pred_batch,
                                                        mask_entropy = mask_entropy
                                                        )
                    # logits, loss = self._one_step_train_unlearning(images,
                    #                     raw_imgs,
                    #                     targets,
                    #                     p_glabel,
                    #                     std_cams,
                    #                     masks,
                    #                     views,
                    #                     cal_mask = mask_list,
                    #                     y_pred_batch = y_pred_batch,
                    #                     mask_entropy = mask_entropy
                    #                     )

            else:

                with autocast(enabled=self.args.amp):
                    logits, loss = self._one_step_train(images,
                                                        raw_imgs,
                                                        targets,
                                                        p_glabel,
                                                        std_cams,
                                                        masks,
                                                        views,
                                                        index,
                                                        aug_images
                                                        )

            with torch.no_grad():
                if self.args.task != constants.SEG:
                    pred = logits.argmax(dim=1)
                    num_correct += (pred == targets).sum().detach()

            if total_loss is None:
                total_loss = loss.detach().squeeze() * images.size(0)
            else:
                total_loss += loss.detach().squeeze() * images.size(0)
            num_images += images.size(0)

            self.store_loss.append((loss.detach().squeeze() * images.size(0)).item())

            if loss.requires_grad:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                # loss.backward()
                # self.optimizer.step()


            if self.args.target_domain_ds_to_compute_stats in [constants.CAMELYON512, constants.CAMELYON17_512, constants.OpenImagesTrgt, constants.GLAS] and batch_idx % self.args.cmpt_batch == 0:
                # self.model.eval()
                # with torch.no_grad():
                #     self.compute_acc_on_source_and_target(self.epoch)
                #     self.compute_loc_on_source_and_target(self.epoch)
                if self.args.measure_loc:
                    self.compute_loc_on_target(self.epoch, split = constants.CLVALIDSET)
                # self.model.train()

                self.model.eval()
                with torch.no_grad():
                    self.compute_acc_on_target_came(self.epoch, compute_kl=True, split = constants.CLVALIDSET)
                    self.compute_acc_on_target_came(self.epoch, compute_kl=True, split = constants.TRAINSET)
                    #self.compute_loc_on_target(self.epoch)

                    if self.args.dataset in [constants.CAMELYON512, constants.CAMELYON17_512, constants.OpenImagesTrgt, constants.GLAS] and self.args.cl_train_models:
                        self.update_best_cl_train_model_came(epoch=self.epoch)
                self.model.train()
                

        loss_average = total_loss.item() / float(num_images)

        classification_acc = 0.0
        if self.args.task != constants.SEG:
            classification_acc = num_correct.item() / float(num_images) * 100

        self.performance_meters[split]['classification'].update(
            classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        self.on_epoch_end()

        return dict(classification_acc=classification_acc,
                    loss=loss_average)

    def print_performances(self, checkpoint_type=None):
        # todo: adapt.
        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = ' Argmax: True'
        if checkpoint_type is not None:
            DLLogger.log(fmsg('PERF - CHECKPOINT: {} {}'.format(
                checkpoint_type, tagargmax)))

        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                current_performance = \
                    self.performance_meters[split][metric].current_value
                if current_performance is not None:
                    DLLogger.log(
                        "Split {}, metric {}, current value: {}".format(
                         split, metric, current_performance))
                    if split != constants.TESTSET:
                        DLLogger.log(
                            "Split {}, metric {}, best value: {}".format(
                             split, metric,
                             self.performance_meters[split][metric].best_value))
                        DLLogger.log(
                            "Split {}, metric {}, best epoch: {}".format(
                             split, metric,
                             self.performance_meters[split][metric].best_epoch))

    def serialize_perf_meter(self) -> dict:
        return {
            split: {
                metric: vars(self.performance_meters[split][metric])
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }

    def serialize_perf_gist_tracker(self) -> dict:
        return {
            split: vars(self.perf_gist_tracker[split]) for split in self._SPLITS
        }

    def save_performances(self, epoch: int, checkpoint_type: str):
        assert isinstance(epoch, int)
        assert isinstance(checkpoint_type, str)
        assert checkpoint_type != ''

        assert checkpoint_type in [constants.BEST_LOC, constants.BEST_CL]

        tag = '_{}'.format(checkpoint_type)

        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = '_Argmax_True'

        log_path = join(self.args.outd, 'performance_log{}{}.pickle'.format(
            tag, tagargmax))

        with open(log_path, 'wb') as f:
            pkl.dump(self.serialize_perf_meter(), f,
                     protocol=pkl.HIGHEST_PROTOCOL)
        log_path_g = join(self.args.outd,
                          'performance_gist_tracker_log{}{}.pickle'.format(
                            tag, tagargmax))
        with open(log_path_g, 'wb') as f:
            pkl.dump(self.serialize_perf_gist_tracker(), f,
                     protocol=pkl.HIGHEST_PROTOCOL)

        log_path = join(self.args.outd, 'performance_log{}{}.txt'.format(
            tag, tagargmax))
        with open(log_path, 'w') as f:
            f.write("PERF - CHECKPOINT {}  - EPOCH {}  {} \n".format(
                checkpoint_type, epoch, tagargmax))

            _splits = [constants.TESTSET]
            for split in _splits:
                for metric in self._EVAL_METRICS:
                    if self._skip_print_metric(metric=metric, _split=split,
                                               split=split):
                        continue

                    f.write("REPORT EPOCH/{}: split: {}/metric {}: {} \n"
                            "".format(epoch, split, metric,
                                      self.performance_meters[split][
                                          metric].current_value))
                    f.write(
                        "REPORT EPOCH/{}: split: {}/metric {}: {}_best "
                        "\n".format(epoch, split, metric,
                                    self.performance_meters[split][
                                        metric].best_value))

                if self.args.localization_avail:
                    f.write(f'REPORT EPOCH/{epoch} split: {split}: \n'
                            f'{self.perf_gist_to_str(split, epoch)} \n')

    def perf_gist_to_str(self, split: str, epoch: int) -> str:
        out = [
            f'{k}: '
            f'{self.perf_gist_tracker[split].value_per_epoch[epoch][k]}'
            for k in self.perf_gist_tracker[split].value_per_epoch[epoch]
        ]
        return '\n'.join(out)

    def cl_forward(self, images):
        output = self.model(images)

        if self.args.task == constants.STD_CL:
            cl_logits = output

        elif self.args.task in [constants.F_CL, constants.NEGEV]:
            cl_logits, fcams, im_recon = output

        else:
            raise NotImplementedError

        return cl_logits

    def _compute_accuracy(self, loader):
        torch.cuda.empty_cache()

        num_correct = 0
        num_images = 0

        for i, (images, targets, _, _, _, _, _, _, _) in enumerate(loader):
            images = images.cuda(self.args.c_cudaid)
            targets = targets.cuda(self.args.c_cudaid)
            with torch.no_grad():
                cl_logits = self.cl_forward(images)

                pred = cl_logits.argmax(dim=1)

            num_correct += (pred == targets).sum().detach()
            num_images += images.size(0)

        classification_acc = num_correct / float(num_images) * 100

        torch.cuda.empty_cache()
        return classification_acc.item()

    def _compute_accuracy_binary_metrics(self, loader):
        torch.cuda.empty_cache()

        num_correct = 0
        num_images = 0

        all_preds = []
        all_targets = []
        all_probs = []  

        for i, (images, targets, _, _, _, _, _, _,_) in enumerate(loader):
            images = images.cuda(self.args.c_cudaid)
            targets = targets.cuda(self.args.c_cudaid)

            with torch.no_grad():
                cl_logits = self.cl_forward(images)
                pred = cl_logits.argmax(dim=1)

                probs = torch.softmax(cl_logits, dim=1)[:, 1]


            num_correct += (pred == targets).sum().detach()
            num_images += images.size(0)


            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        # Accuracy
        classification_acc = num_correct / float(num_images) * 100


        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        # Precision / Recall / F1
        precision = precision_score(all_targets, all_preds, average="binary")
        recall = recall_score(all_targets, all_preds, average="binary")
        f1 = f1_score(all_targets, all_preds, average="binary")

        roc_auc = roc_auc_score(all_targets, all_probs)
        pr_auc = average_precision_score(all_targets, all_probs)

        torch.cuda.empty_cache()

        return classification_acc.item(), pr_auc, roc_auc, f1
    

    def _compute_accuracy_and_entropy(self, loader):
        torch.cuda.empty_cache()

        num_correct = 0
        num_images = 0
        total_entropy = 0.0

        for i, (images, targets, *_ ) in enumerate(loader):
            images = images.cuda(self.args.c_cudaid)
            targets = targets.cuda(self.args.c_cudaid)

            with torch.no_grad():
                cl_logits = self.cl_forward(images)

                # Accuracy
                pred = cl_logits.argmax(dim=1)
                num_correct += (pred == targets).sum().detach()
                num_images += images.size(0)

                # Entropy
                probs = torch.softmax(cl_logits, dim=1)
                log_probs = torch.log_softmax(cl_logits, dim=1)
                entropy = -torch.sum(probs * log_probs, dim=1)  # [B]
                total_entropy += entropy.sum().item()

        classification_acc = num_correct / float(num_images) * 100
        mean_entropy = total_entropy / float(num_images)

        torch.cuda.empty_cache()
        return classification_acc.item(), mean_entropy
    
    
    
    def _compute_accuracy_entropy(self, loader):
        torch.cuda.empty_cache()

        num_correct = 0
        num_images = 0

        cam_size = 0
        images_total_entropy = 0
        pixel_total_entropy = 0

        for i, (images, targets, _, _, _, _, _, _) in enumerate(loader):
            images = images.cuda(self.args.c_cudaid)
            targets = targets.cuda(self.args.c_cudaid)

            _,_,x,y = images.size()

            with torch.no_grad():
                cl_logits = self.cl_forward(images)
                pred = cl_logits.argmax(dim=1)

                images_probs = torch.softmax(cl_logits, dim=1)
                images_entropy = self.compute_entropy(images_probs)
                # images_dist = torch.distributions.Categorical(images_probs)
                # images_entropies = images_dist.entropy()
                images_total_entropy += images_entropy.sum().item()

                pixel_logits = self.model.cams
                pixel_probs = torch.softmax(pixel_logits, dim=1)
                pixel_entropy = self.compute_entropy(pixel_probs)
                # pixel_dist = torch.distributions.Categorical(pixel_probs)
                # pixel_entropies = pixel_dist.entropy()
                pixel_total_entropy += pixel_entropy.sum().item()
                
            num_correct += (pred == targets).sum().detach()
            num_images += images.size(0)

        classification_acc = num_correct / float(num_images) * 100

        images_entropy = images_total_entropy / num_images

        nb_pixels = x * y * num_images
        pixel_entropy = pixel_total_entropy / nb_pixels

        torch.cuda.empty_cache()
        return classification_acc.item(), images_entropy, pixel_entropy

    # def _plot_entropy_hist_per_class(self, ent_correct, ent_incorrect, class_name, out_path, num_classes):
    #     h_max = math.log(max(2, num_classes))
    #     bins = np.linspace(0.0, h_max, 30)

    #     plt.figure(figsize=(7, 4.5))
    #     plt.hist(ent_correct, bins=bins, alpha=0.7, label="Correctly classified", density=True)
    #     plt.hist(ent_incorrect, bins=bins, alpha=0.7, label="Misclassified", density=True)
    #     plt.title(f"Histogram — {class_name}")
    #     plt.xlabel("Entropy per image")
    #     plt.ylabel("Density")
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     os.makedirs(os.path.dirname(out_path), exist_ok=True)
    #     plt.savefig(out_path, dpi=200)
    #     plt.close()

    def _plot_entropy_hist_per_class(self, ent_correct, ent_incorrect, class_name, out_path, num_classes, bins=60, figsize=(16, 10)):

        eps = 1e-4


        h_max = math.log(max(2, num_classes))
        bin_edges = np.linspace(0.0, h_max + eps, bins)

        plt.figure(figsize=figsize)


        counts_correct, _ = np.histogram(ent_correct, bins=bin_edges)
        counts_incorrect, _ = np.histogram(ent_incorrect, bins=bin_edges)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        total_counts = counts_correct + counts_incorrect


        plt.bar(bin_centers, counts_correct, width=np.diff(bin_edges), 
                alpha=0.8, label="Correctly Predicted", align="center", color="#4C9AFF")
        plt.bar(bin_centers, counts_incorrect, width=np.diff(bin_edges), 
                bottom=counts_correct, alpha=0.8, label="Incorrectly Predicted", align="center", color="#FF9933")


        for total, correct, incorrect, x in zip(total_counts, counts_correct, counts_incorrect, bin_centers):
            if total > 0:
                # Total 
                plt.text(x, total + 0.5, f"{total}", rotation=90, ha="center", va="bottom", fontsize=8)


                if correct > 0:
                    plt.text(x, correct / 2, f"{100 * correct / total:.1f}%", 
                            ha="center", va="center", fontsize=8, color="white")

                if incorrect > 0:
                    plt.text(x, correct + incorrect / 2, f"{100 * incorrect / total:.1f}%", 
                            ha="center", va="center", fontsize=8, color="black")

        plt.title(f"Entropy distribution — {class_name}")
        plt.xlabel("Entropy per image")
        plt.ylabel("Nb images")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()

    def compute_J_index(self, features, labels):
        features = np.array(features)
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        mu_global = features.mean(axis=0)

        inter = 0.0
        intra = 0.0
        for c in unique_labels:
            mask = (labels == c)
            feats_c = features[mask]
            mu_c = feats_c.mean(axis=0)
            inter += len(feats_c) * np.sum((mu_c - mu_global) ** 2)
            intra += np.sum((feats_c - mu_c) ** 2)
        return inter / intra if intra > 0 else 0.0
    
    def compute_kl_uniform(self, preds, num_classes):
        hist = torch.bincount(preds, minlength=num_classes).float()
        p = hist / hist.sum()          
        u = torch.ones_like(p) / num_classes  
        kl = (p * (p / u).log()).sum()
        return float(kl.item())
    
    def _compute_accuracy_f1(self, loader, split = constants.CLVALIDSET, compute_kl = True):
        torch.cuda.empty_cache()

        num_correct = 0
        num_images = 0

        num_correct_normal = 0
        num_images_normal = 0

        num_correct_cancer = 0
        num_images_cancer = 0

        cam_size = 0
        images_total_entropy = 0
        pixel_total_entropy = 0

        # --- Entropy buckets for plotting (per class & correctness)
        ent_correct_by_class = {c: [] for c in range(self.args.num_classes)}
        ent_incorrect_by_class = {c: [] for c in range(self.args.num_classes)}

        y_pred = []
        y_true = []

        features_list = []
        labels_list = []

        # --- KL metrics ---
        kl_vals = []

        # --- Marginal entropy accumulators ---
        sum_probs = torch.zeros(self.args.num_classes, device=self.args.c_cudaid)
        num_samples = 0

        


        master_loss = 0
        ce_flip_loss = 0
        ce_not_flip_loss = 0

        correct_flip, total_flip = 0, 0
        correct_stable, total_stable = 0, 0


        all_probs = []
        all_logits = []
        all_labels = []


        ece_vals = []
        nll_vals = []
        brier_vals = []

        for batch_idx, (images, targets, p_glabel, index, raw_imgs, std_cams, masks, views, _) in enumerate(loader):
            images = images.cuda(self.args.c_cudaid)
            targets = targets.cuda(self.args.c_cudaid)

            _,_,x,y = images.size()

            supervised_labels = torch.full_like(p_glabel, -255)
            entropy_batch = []

            for i in range(images.size(0)):
                img_idx = index[i]
                if img_idx in self.flipped_indices:
                    supervised_labels[i] = self.assigned_labels_map[img_idx] 
                # elif img_idx in self.reinforce_indices:
                #     supervised_labels[i] = 1 
                # 
                if img_idx in self.entropy_all:
                    entropy_batch.append(self.entropy_all[img_idx])
                else:
                    entropy_batch.append(0.0) 
                    
            # p_glabel = supervised_labels
            # mask_list_forget = [img_name not in self.flipped_indices for img_name in index]
            # mask_list_retain = [img_name not in self.stable_selected for img_name in index]
            # y_pred_batch = torch.tensor([self.idx_to_pred[idx] for idx in index])
            # mask_entropy = torch.tensor(entropy_batch, dtype=torch.float32, device=images.device)
            # y_pred_batch = y_pred_batch.cuda(self.args.c_cudaid)

            # p_glabel = supervised_labels
            # mask_list  = [img_name not in self.flipped_indices for img_name in index]
            # # mask_list_retain = [img_name not in self.stable_selected for img_name in index]
            # #y_pred_batch = torch.tensor([self.idx_to_pred[idx] for idx in index])
            # y_pred_batch = torch.tensor([self.assigned_labels_map[idx] for idx in index])
            # mask_entropy = torch.tensor(entropy_batch, dtype=torch.float32, device=images.device)
            # y_pred_batch = y_pred_batch.cuda(self.args.c_cudaid)


            with torch.no_grad():
                cl_logits = self.cl_forward(images)
                feats = self.model.lin_ft.detach().cpu().numpy()
                features_list.append(feats)
                labels_list.append(targets.cpu().numpy())


                pred = cl_logits.argmax(dim=1)

                images_probs = torch.softmax(cl_logits, dim=1)
                images_entropy = self.compute_entropy(images_probs)
                
                images_total_entropy += images_entropy.sum().item()

                ent_per_img = -(images_probs.clamp(min=1e-8) * images_probs.clamp(min=1e-8).log()).sum(dim=1)

                all_probs.append(images_probs.detach().cpu())
                all_logits.append(cl_logits.detach().cpu())
                all_labels.append(targets.detach().cpu())

                with torch.no_grad():
                    for b in range(images.size(0)):
                        true_cls = int(targets[b].item())
                        pred_cls = int(pred[b].item())
                        #corr = int(pred[b].item() == true_cls)
                        h = float(ent_per_img[b].item())

                        if pred_cls == true_cls:
                            ent_correct_by_class[pred_cls].append(h)
                        else:
                            ent_incorrect_by_class[pred_cls].append(h)


                        # if true_cls in (0, 1):  
                        #     if corr:
                        #         ent_correct_by_class[pred_cls].append(h)
                        #     else:
                        #         ent_incorrect_by_class[pred_cls].append(h)


                # accumulate for marginal entropy
                sum_probs += images_probs.sum(dim=0)
                num_samples += images_probs.size(0)

                # --- KL-consistency if enabled ---
                if compute_kl:
                    #conf = images_probs.max(dim=1).values
                    for b in range(images.size(0)):
                        #idx = int(indices[b])
                        idx = index[b]  # Convert to Python int
                        if idx in self.idx_to_pred:
                            y_src = self.idx_to_pred[idx]
                            ce = -torch.log(images_probs[b, y_src] + 1e-8)
                            kl_vals.append(ce.item())

                # pixel_logits = self.model.cams
                # pixel_probs = torch.softmax(pixel_logits, dim=1)
                # pixel_entropy = self.compute_entropy(pixel_probs)
                # # pixel_dist = torch.distributions.Categorical(pixel_probs)
                # # pixel_entropies = pixel_dist.entropy()
                # pixel_total_entropy += pixel_entropy.sum().item()

            for b in range(images.size(0)):
                idx_b = index[b]
                pred_b = pred[b].item()

                # Forget
                if idx_b in self.flipped_indices:
                    init_cls = self.assigned_labels_map[idx_b]
                    #tgt_flip = 0
                    correct_flip += int(pred_b == init_cls)
                    total_flip += 1

                # Retain
                if idx_b in self.stable_selected:
                    #pos = self.stable_selected.index(idx_b)
                    tgt_stable = int(self.assigned_labels_map[idx_b])
                    correct_stable += int(pred_b == tgt_stable)
                    total_stable += 1
                

            num_correct += (pred == targets).sum().detach()
            num_images += images.size(0)


            y_pred.extend(pred.cpu().tolist())
            y_true.extend(targets.cpu().tolist())

            if self.args.num_classes == 2:
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

        features_np = np.concatenate(features_list, axis=0)
        labels_np = np.concatenate(labels_list, axis=0)


        silhouette = silhouette_score(features_np, labels_np)
        dbi = davies_bouldin_score(features_np, labels_np)
        CH = calinski_harabasz_score(features_np, labels_np)
        J_index = self.compute_J_index(features_np, labels_np)


        all_probs = torch.cat(all_probs, dim=0)     
        all_logits = torch.cat(all_logits, dim=0)  
        all_labels = torch.cat(all_labels, dim=0)   

        ECE_global = self.compute_ece(all_probs, all_labels, n_bins=15)
        NLL_global = self.compute_nll(all_logits, all_labels)
        Brier_global = self.compute_brier(all_probs, all_labels, num_classes=self.args.num_classes)


        self.silhouette.append(silhouette)
        self.DBI.append(dbi)
        self.CH.append(CH)
        self.J_index.append(J_index)


        fig_outd = os.path.join(self.args.outd, "entropy_hist", split)
        os.makedirs(fig_outd, exist_ok=True)

        #fig_outd = os.path.join(self.args.outd, "entropy_hist")



        # After loop — plotting
        for c in range(self.args.num_classes):

            if self.args.target_domain_ds_to_compute_stats == constants.GLAS:
                CLASS_NAME = f"entropy_hist_class{c}_ep{self.epoch:03d}.png"
            else:
                if self.batch_idx is None:
                    self.batch_idx = 0
                CLASS_NAME = f"entropy_hist_class{c}_ep{self.epoch:03d}_b{self.batch_idx:05d}.png"

            out_path = os.path.join(fig_outd, CLASS_NAME)

            self._plot_entropy_hist_per_class(
                ent_correct_by_class[c],
                ent_incorrect_by_class[c],
                class_name=f"Class {c}",
                out_path=out_path,
                num_classes=self.args.num_classes
            )


        # if self.args.target_domain_ds_to_compute_stats == constants.GLAS:
        #     CLASS_NORMAL_NAME = f"entropy_hist_normal_ep{self.epoch:03d}.png"
        #     CLASS_CANCER_NAME = f"entropy_hist_cancer_ep{self.epoch:03d}.png"
        #     out_normal = os.path.join(fig_outd, CLASS_NORMAL_NAME)
        #     out_cancer = os.path.join(fig_outd, CLASS_CANCER_NAME)
        # else:
        #     if self.batch_idx == None:
        #         self.batch_idx = 0
        #     CLASS_NORMAL_NAME = f"entropy_hist_normal_ep{self.epoch:03d}_b{self.batch_idx:05d}.png"
        #     CLASS_CANCER_NAME = f"entropy_hist_cancer_ep{self.epoch:03d}_b{self.batch_idx:05d}.png"
        #     out_normal = os.path.join(fig_outd, CLASS_NORMAL_NAME)
        #     out_cancer = os.path.join(fig_outd, CLASS_CANCER_NAME)



        # # Classe 0 : Normal
        # self._plot_entropy_hist_per_class(
        #     ent_correct_by_class[0],
        #     ent_incorrect_by_class[0],
        #     class_name="Normal (label=0)",
        #     out_path=out_normal,
        #     num_classes=self.args.num_classes
        # )

        # # Classe 1 : Cancer
        # self._plot_entropy_hist_per_class(
        #     ent_correct_by_class[1],
        #     ent_incorrect_by_class[1],
        #     class_name="Cancer (label=1)",
        #     out_path=out_cancer,
        #     num_classes=self.args.num_classes
        # )


        self.store_master_loss.append(master_loss) 
        self.store_ce_flip_loss.append(ce_flip_loss)
        self.store_ce_not_flip_loss.append(ce_not_flip_loss)

        classification_acc = num_correct / float(num_images) * 100
        classification_acc_normal = num_correct_normal / float(num_images_normal) * 100 if num_images_normal > 0 else 0
        classification_acc_cancer = num_correct_cancer / float(num_images_cancer) * 100 if num_images_cancer > 0 else 0

        images_entropy = images_total_entropy / num_images

        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

        labels = np.unique(np.concatenate([y_true_np, y_pred_np]))

        cm = confusion_matrix(y_true_np, y_pred_np, labels=labels)

        if self.args.num_classes == 2:
            try:
                tn, fp, fn, tp = cm.ravel()
            except ValueError:
                tn = fp = fn = tp = np.nan

            f1 = f1_score(y_true_np, y_pred_np, average="binary")
            precision = precision_score(y_true_np, y_pred_np, average="binary")
            recall = recall_score(y_true_np, y_pred_np, average="binary")

        else:
            tn = fp = fn = tp = np.nan   

            f1 = f1_score(y_true_np, y_pred_np, average="macro")
            precision = precision_score(y_true_np, y_pred_np, average="macro")
            recall = recall_score(y_true_np, y_pred_np, average="macro")

        # === Build row for CSV ===
        row = {
            "epoch": self.epoch,
            "silhouette": silhouette,
            "DBI": dbi,
            "CH_J": CH,
            "J_index": J_index, 
            "accuracy": classification_acc.item(),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "ECE": ECE_global,
            "NLL": NLL_global,
            "Brier": Brier_global,
        }

        cm_file = os.path.join(self.args.outd, "confusion_evolution.csv")

        # === Append or create ===
        file_exists = os.path.isfile(cm_file)

        with open(cm_file, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        preds_all = pred
        #KL consistency (mean over all images)
        kl_uniform = self.compute_kl_uniform(preds_all, num_classes= self.args.num_classes)
        #self.metrics.hist.append(kl_uniform)

        #Marginal entropy
        pbar = sum_probs / num_samples
        Hm = -(pbar * (pbar + 1e-8).log()).sum().item()
        Htilde = max(0.0, math.log(self.args.num_classes) - Hm)

        #self.metrics.hm_hist.append(Hm)
        #self.metrics.htilde_hist.append(Htilde)

        #Store accuracy unlearned and not unlearned ---
        acc_flip = correct_flip / total_flip if total_flip > 0 else 0.0
        acc_stable = correct_stable / total_stable if total_stable > 0 else 0.0

        total_correct = correct_flip + correct_stable
        total_samples = total_flip + total_stable

        acc_global = total_correct / total_samples if total_samples > 0 else 0.0


        if not hasattr(self, "history_acc_flip"):
            self.history_acc_flip = []
        if not hasattr(self, "history_acc_stable"):
            self.history_acc_stable = []

        self.history_acc_flip.append(acc_flip)
        self.history_acc_stable.append(acc_stable)

        results = {
            "classification_acc": classification_acc.item(),
            "classification_acc_normal": classification_acc_normal,
            "classification_acc_cancer": classification_acc_cancer,
            "images_entropy": images_entropy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "silhouette": silhouette,
            "DBI": dbi,
            "CH": CH,
            "J_index": J_index,
            "kl_uniform": kl_uniform,
            "Hm": Hm,
            "Htilde": Htilde,
            "acc_flip": acc_flip,
            "acc_stable": acc_stable,
            "acc_global": acc_global,
            "ECE": ECE_global,
            "NLL": NLL_global,
            "Brier": Brier_global,
        }

        torch.cuda.empty_cache()
        return results
        #return classification_acc.item(), classification_acc_normal, classification_acc_cancer, images_entropy, f1, precision, recall
    

    def compute_entropy(self, probs):
        """ Computes the entropy of a probability distribution.

        Handles both classification (B, C) and localization maps (B, C, H, W).

        Args:
            pixel_probs (torch.Tensor): A tensor of probabilities. 
                                        Shape can be:
                                        - (B, C) for global classification.
                                        - (B, C, H, W) for localization maps.

        Returns:
            torch.Tensor: The entropy. 
                            - Shape (B) for classification.
                            - Shape (B, H, W) for localization maps.
        """
        # Avoid numerical issues with log(0)
        epsilon = 1e-9
        probs = torch.clamp(probs, min=epsilon)

        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs), dim=1)

        return entropy 
    
    def compute_acc_on_source_and_target(self, epoch, split=constants.TESTSET):
        self.model.eval()
        with torch.no_grad():
            target_test_acc, target_test_image_entropy, target_test_pixel_entropy = self._compute_accuracy_entropy(self.target_domain_loaders[constants.TESTSET])
            self.target_test_acc_cl.append(target_test_acc)
            self.target_test_image_entropy.append(target_test_image_entropy)
            self.target_test_pixel_entropy.append(target_test_pixel_entropy)

            source_test_acc, source_test_image_entropy, source_test_pixel_entropy = self._compute_accuracy_entropy(self.source_domain_loaders[constants.TESTSET])
            self.source_test_acc_cl.append(source_test_acc)
            self.source_test_image_entropy.append(source_test_image_entropy)
            self.source_test_pixel_entropy.append(source_test_pixel_entropy)

            source_train_acc, source_train_image_entropy, source_train_pixel_entropy = self._compute_accuracy_entropy(self.source_train_domain_loaders[constants.TRAINSET])
            self.source_train_acc_cl.append(source_train_acc)
            self.source_train_image_entropy.append(source_train_image_entropy)
            self.source_train_pixel_entropy.append(source_train_pixel_entropy)


    def compute_acc_on_target(self, epoch, split=constants.TRAINSET):
        self.model.eval()
        with torch.no_grad():
            target_train_acc, target_train_acc_normal, target_train_acc_cancer,  images_entropy, f1, precision, recall = self._compute_accuracy_f1(self.target_domain_loaders[constants.CLVALIDSET], compute_kl = True)
            
            self.target_train_acc_cl.append(target_train_acc)
            self.target_train_f1.append(f1)
            self.target_train_precision.append(precision)
            self.target_train_recall.append(recall)
            self.target_train_image_entropy.append(images_entropy)
            self.target_train_acc_normal.append(target_train_acc_normal)
            self.target_train_acc_cancer.append(target_train_acc_cancer)

            #self.target_train_image_entropy.append(target_train_image_entropy)
            #self.target_train_pixel_entropy.append(target_train_pixel_entropy)

            with torch.no_grad():
                accuracy = 0.0
                # if self.args.task != constants.SEG:
                #     accuracy = self._compute_accuracy(loader=self.loaders[split])

                torch.cuda.empty_cache()

                model_cl = deepcopy(self.model).to(self.cpu_device).eval()
                #model_state = model_entropy.state_dict()

                if not hasattr(self, "best_accuracy"):
                    self.cl_train_model = None
                    self.best_accuracy = -1.0
                    self.best_epoch = -1

                if target_train_acc > self.best_accuracy:
                    self.best_accuracy = target_train_acc
                    self.cl_train_model = deepcopy(model_cl)
                    self.best_epoch = epoch
                    print(f"[Accuracy Update] New best model at epoch {epoch} with accuracy {target_train_acc:.2f}%")
                else:
                    print(f"[Accuracy Skip] Model at epoch {epoch} with accuracy {target_train_acc:.2f}% was not better than best ({self.best_accuracy:.2f}%)")

    def save_unlearn_models(self, epoch, max_epochs, milestones=[0.10, 0.25, 0.50, 0.75, 1.00]):
        """
        Return True if the current epoch matches one of the milestone percentages.
        """
        # Convert milestones into exact epoch numbers
        milestone_epochs = {max(1, round(max_epochs * m)) for m in milestones}

        return epoch in milestone_epochs



    def compute_ece(self, probs, labels, n_bins=15):
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


    def compute_nll(self, logits, labels):
        return float(F.cross_entropy(logits, labels).item())


    def compute_brier(self, probs, labels, num_classes):
        one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        return float(((probs - one_hot)**2).mean().item())

    def compute_acc_on_target_came(self, epoch, split, compute_kl= False):

        if not hasattr(self, "metrics"):
            self.metrics = {
                "train": {
                    "acc_cl": [],
                    "acc_normal": [],
                    "acc_cancer": [],
                    "f1": [],
                    "precision": [],
                    "recall": [],
                    "entropy": [],
                    "silhouette": [],
                    "DBI": [],
                    "CH": [],
                    "J_index": [],
                    "kl_uniform": [],
                    "Hm": [],
                    "Htilde": [],
                    "acc_flip": [],
                    "acc_stable": [],
                    "acc_global": [],
                    "ece": [],
                    "nll": [],
                    "brier": [],
                },
                "valcl": {
                    "acc_cl": [],
                    "acc_normal": [],
                    "acc_cancer": [],
                    "f1": [],
                    "precision": [],
                    "recall": [],
                    "entropy": [],
                    "silhouette": [],
                    "DBI": [],
                    "CH": [],
                    "J_index": [],
                    "kl_uniform": [],
                    "Hm": [],
                    "Htilde": [],
                    "acc_flip": [],
                    "acc_stable": [],
                    "acc_global": [],
                    "ece": [],
                    "nll": [],
                    "brier": [],
            }
        }


        self.model.eval()
        with torch.no_grad():
            #target_train_acc, target_train_acc_normal, target_train_acc_cancer,  images_entropy, f1, precision, recall = self._compute_accuracy_f1(self.target_domain_loaders[split], compute_kl = True)
            stats = self._compute_accuracy_f1(self.target_domain_loaders[split], split = split, compute_kl = True)
            
            acc_cl         = stats["classification_acc"]
            acc_n          = stats["classification_acc_normal"]
            acc_c          = stats["classification_acc_cancer"]
            f1             = stats["f1"]
            precision      = stats["precision"]
            recall         = stats["recall"]
            images_entropy = stats["images_entropy"]
            silhouette     = stats["silhouette"]
            DBI            = stats["DBI"]
            CH             = stats["CH"]
            J_index        = stats["J_index"]
            kl_uniform     = stats["kl_uniform"]
            Hm             = stats["Hm"]
            Htilde         = stats["Htilde"]
            acc_flip       = stats["acc_flip"]
            acc_stable     = stats["acc_stable"]
            acc_global     = stats["acc_global"]
            ece            = stats["ECE"]
            nll            = stats["NLL"]
            brier          = stats["Brier"]


            self.metrics[split]["acc_cl"].append(acc_cl)
            self.metrics[split]["acc_normal"].append(acc_n)
            self.metrics[split]["acc_cancer"].append(acc_c)

            self.metrics[split]["f1"].append(f1)
            self.metrics[split]["precision"].append(precision)
            self.metrics[split]["recall"].append(recall)

            self.metrics[split]["entropy"].append(images_entropy)

            self.metrics[split]["silhouette"].append(silhouette)
            self.metrics[split]["DBI"].append(DBI)
            self.metrics[split]["CH"].append(CH)
            self.metrics[split]["J_index"].append(J_index)

            self.metrics[split]["kl_uniform"].append(kl_uniform)
            self.metrics[split]["Hm"].append(Hm)
            self.metrics[split]["Htilde"].append(Htilde)

            self.metrics[split]["acc_flip"].append(acc_flip)
            self.metrics[split]["acc_stable"].append(acc_stable)
            self.metrics[split]["acc_global"].append(acc_global)

            self.metrics[split]["ece"].append(ece)
            self.metrics[split]["nll"].append(nll)
            self.metrics[split]["brier"].append(brier)


            if self.args.save_multiple_unlearn_models:
                if self.save_unlearn_models(self.epoch, self.args.max_epochs):
                    current_unlearn_model = deepcopy(self.model).to(self.cpu_device).eval()

                    unlearning_log_path = os.path.join(self.args.outd, "best_models_unlearning.txt")
            
                    checkpoint_type = f"B-Unlearning-epoch{self.epoch}"
                    tag = get_tag(self.args, checkpoint_type=checkpoint_type)
                    path = os.path.join(self.args.outd, tag)

                    if not os.path.isdir(path):
                        os.makedirs(path)
                    if self.args.task == constants.STD_CL:
                        if self.args.method in [constants.METHOD_ACOL,
                                                constants.METHOD_ADL,
                                                constants.METHOD_SPG,
                                                constants.METHOD_TSCAM,
                                                constants.METHOD_SAT]:
                            torch.save(current_unlearn_model.state_dict(),
                                    join(path, 'model.pt'))

                        elif self.args.method == constants.METHOD_MAXMIN:
                            torch.save(current_unlearn_model.encoder.state_dict(),
                                    join(path, 'encoder.pt'))
                            torch.save(current_unlearn_model.classification_head1.state_dict(),
                                    join(path, 'classification_head1.pt'))
                            torch.save(current_unlearn_model.classification_head2.state_dict(),
                                    join(path, 'classification_head2.pt'))
                            if current_unlearn_model.mask_head is not None:
                                torch.save(current_unlearn_model.mask_head.state_dict(),
                                        join(path, 'mask_head.pt'))

                        elif self.args.method == constants.METHOD_PIXELCAM:
                            if "deit" in self.args.model['encoder_name']:
                                torch.save(current_unlearn_model.state_dict(),
                                    join(path, 'model.pt'))
                            else:
                                torch.save(current_unlearn_model.encoder.state_dict(),
                                        join(path, 'encoder.pt'))
                                torch.save(current_unlearn_model.classification_head.state_dict(),
                                        join(path, 'classification_head.pt')),
                                torch.save(current_unlearn_model.pixel_wise_classification_head.state_dict(),
                                            join(path, 'pixel_wise_classification_head.pt'))
                        else:
                            torch.save(current_unlearn_model.encoder.state_dict(),
                                    join(path, 'encoder.pt'))
                            torch.save(current_unlearn_model.classification_head.state_dict(),
                                    join(path, 'classification_head.pt'))
                        
                    self._save_args(path=join(path, 'config_model.yaml'))
                    DLLogger.log(message="Stored Model [CP: {} \t EPOCH: {} \t TAG: {}]:"
                                        " {}".format(checkpoint_type, epoch, tag, path))


            # self.target_train_acc_cl.append(target_train_acc)
            # self.target_train_f1.append(f1)
            # self.target_train_precision.append(precision)
            # self.target_train_recall.append(recall)
            # self.target_train_image_entropy.append(images_entropy)
            # self.target_train_acc_normal.append(target_train_acc_normal)
            # self.target_train_acc_cancer.append(target_train_acc_cancer)
            # self.current_acc_cl = target_train_acc

        # if split == constants.CLVALIDSET:
        #     with torch.no_grad():
        #         accuracy = 0.0
        #         # if self.args.task != constants.SEG:
        #         #     accuracy = self._compute_accuracy(loader=self.loaders[split])

        #         torch.cuda.empty_cache()

        #         model_cl = deepcopy(self.model).to(self.cpu_device).eval()
        #         #model_state = model_entropy.state_dict()

        #         if not hasattr(self, "best_accuracy"):
        #             self.cl_train_model = None
        #             self.best_accuracy = -1.0
        #             self.best_epoch = -1

        #         if acc_cl > self.best_accuracy:
        #             self.best_accuracy = acc_cl
        #             self.cl_train_model = deepcopy(model_cl)
        #             self.best_epoch = epoch
        #             print(f"[Accuracy Update] New best model at epoch {epoch} with accuracy {acc_cl:.2f}%")
        #         else:
        #             print(f"[Accuracy Skip] Model at epoch {epoch} with accuracy {acc_cl:.2f}% was not better than best ({self.best_accuracy:.2f}%)")


    def compute_loc_on_target(self, epoch, split=constants.TRAINSET):
        self.model.eval()

        cam_computer_target_train = CAMComputer(
            args=deepcopy(self.args),
            model=self.model,
            loader=self.target_domain_loaders[split],
            metadata_root=os.path.join(self.args.target_domain_metadata_root, split),
            mask_root=self.args.mask_root_target,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.target_domain_ds_to_compute_stats,
            split=split,
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            out_folder=self.args.outd,
            fcam_argmax=self.fcam_argmax,
            best_valid_tau= None
        )


        cam_performance_target_train = cam_computer_target_train.compute_and_evaluate_cams()

        self.target_valpx_pxap.append(cam_computer_target_train.evaluator.perf_gist[constants.MTR_PXAP])
        #self.source_test_pxap.append(cam_computer_source_test.evaluator.perf_gist[constants.MTR_PXAP])
        #self.source_train_pxap.append(cam_computer_source_train.evaluator.perf_gist[constants.MTR_PXAP])

        self.target_train_dice_bg.append(cam_computer_target_train.evaluator.perf_gist[constants.MTR_DICEBG_05])
        #self.source_test_dice_bg.append(cam_computer_source_test.evaluator.perf_gist[constants.MTR_DICEBG_05])
        #self.source_train_dice_bg.append(cam_computer_source_train.evaluator.perf_gist[constants.MTR_DICEBG_05])

        self.target_train_dice_fg.append(cam_computer_target_train.evaluator.perf_gist[constants.MTR_DICEFG_05])
        #self.source_test_dice_fg.append(cam_computer_source_test.evaluator.perf_gist[constants.MTR_DICEFG_05])
        #self.source_train_dice_fg.append(cam_computer_source_train.evaluator.perf_gist[constants.MTR_DICEFG_05])

        self.target_train_miou.append(cam_computer_target_train.evaluator.perf_gist[constants.MTR_MIOU_05])
        #self.source_test_miou.append(cam_computer_source_test.evaluator.perf_gist[constants.MTR_MIOU_05])
        #self.source_train_miou.append(cam_computer_source_train.evaluator.perf_gist[constants.MTR_MIOU_05])

    def compute_loc_on_source_and_target(self, epoch, split=constants.TESTSET):
        self.model.eval()

        cam_computer_source_test = CAMComputer(
            args=deepcopy(self.args),
            model=self.model,
            loader=self.source_domain_loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset,
            split=split,
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            out_folder=self.args.outd,
            fcam_argmax=self.fcam_argmax,
            best_valid_tau= None
        )

        cam_computer_source_train = CAMComputer(
            args=deepcopy(self.args),
            model=self.model,
            loader=self.source_train_domain_loaders['train'],
            metadata_root=os.path.join(self.args.metadata_root, 'train'),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset,
            split='train',
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            out_folder=self.args.outd,
            fcam_argmax=self.fcam_argmax,
            best_valid_tau= None
        )

        cam_computer_target_test = CAMComputer(
            args=deepcopy(self.args),
            model=self.model,
            loader=self.target_domain_loaders[split],
            metadata_root=os.path.join(self.args.target_domain_metadata_root, split),
            mask_root=self.args.mask_root_target,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.target_domain_ds_to_compute_stats,
            split=split,
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            out_folder=self.args.outd,
            fcam_argmax=self.fcam_argmax,
            best_valid_tau= None
        )

        
        cam_performance_source_test = cam_computer_source_test.compute_and_evaluate_cams()
        cam_performance_source_train = cam_computer_source_train.compute_and_evaluate_cams()

        cam_performance_target_test = cam_computer_target_test.compute_and_evaluate_cams()

        self.target_test_pxap.append(cam_computer_target_test.evaluator.perf_gist[constants.MTR_PXAP])
        self.source_test_pxap.append(cam_computer_source_test.evaluator.perf_gist[constants.MTR_PXAP])
        self.source_train_pxap.append(cam_computer_source_train.evaluator.perf_gist[constants.MTR_PXAP])

        self.target_test_dice_bg.append(cam_computer_target_test.evaluator.perf_gist[constants.MTR_DICEBG_05])
        self.source_test_dice_bg.append(cam_computer_source_test.evaluator.perf_gist[constants.MTR_DICEBG_05])
        self.source_train_dice_bg.append(cam_computer_source_train.evaluator.perf_gist[constants.MTR_DICEBG_05])

        self.target_test_dice_fg.append(cam_computer_target_test.evaluator.perf_gist[constants.MTR_DICEFG_05])
        self.source_test_dice_fg.append(cam_computer_source_test.evaluator.perf_gist[constants.MTR_DICEFG_05])
        self.source_train_dice_fg.append(cam_computer_source_train.evaluator.perf_gist[constants.MTR_DICEFG_05])

        self.target_test_miou.append(cam_computer_target_test.evaluator.perf_gist[constants.MTR_MIOU_05])
        self.source_test_miou.append(cam_computer_source_test.evaluator.perf_gist[constants.MTR_MIOU_05])
        self.source_train_miou.append(cam_computer_source_train.evaluator.perf_gist[constants.MTR_MIOU_05])

    def compute_loc_on_target(self, epoch, split=constants.TESTSET):
        self.model.eval()

        cam_computer_source_train = CAMComputer(
            args=deepcopy(self.args),
            model=self.model,
            loader=self.target_domain_loaders['valpx'],
            metadata_root=os.path.join(self.args.metadata_root, 'valpx'),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset,
            split='valpx',
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            out_folder=self.args.outd,
            fcam_argmax=self.fcam_argmax,
            best_valid_tau= None
        )
  
        cam_performance_source_train = cam_computer_source_train.compute_and_evaluate_cams()

        self.target_valpx_pxap.append(cam_computer_source_train.evaluator.perf_gist[constants.MTR_PXAP])

        self.target_train_dice_bg.append(cam_computer_source_train.evaluator.perf_gist[constants.MTR_DICEBG_05])

        self.target_train_dice_fg.append(cam_computer_source_train.evaluator.perf_gist[constants.MTR_DICEFG_05])

        self.target_train_miou.append(cam_computer_source_train.evaluator.perf_gist[constants.MTR_MIOU_05])


    def save_curves(self, task, cmpt_epoch):
         #Store data in a pickle
        curves_data = {
            'target_train_acc_cl': self.target_train_acc_cl,
            'target_train_f1': self.target_train_f1,
            'target_train_precision': self.target_train_precision,
            'target_train_recall': self.target_train_recall,
            'target_train_image_entropy': self.target_train_image_entropy,
            'target_train_acc_normal': self.target_train_acc_normal,
            'target_train_acc_cancer': self.target_train_acc_cancer,

            'target_valpx_pxap': self.target_valpx_pxap,
            'target_train_dice_bg': self.target_train_dice_bg,
            'target_train_dice_fg': self.target_train_dice_fg,
            'target_train_miou': self.target_train_miou,

        }
        
        #Store data in a pickle
        # curves_data = {
        #     'target_test_acc_cl': self.target_test_acc_cl,
        #     'source_test_acc_cl': self.source_test_acc_cl,
        #     'source_train_acc_cl': self.source_train_acc_cl,

        #     'target_test_image_entropy': self.target_test_image_entropy,
        #     'source_test_image_entropy': self.source_test_image_entropy,
        #     'source_train_image_entropy': self.source_train_image_entropy,

        #     'target_test_pixel_entropy': self.target_test_pixel_entropy,
        #     'source_test_pixel_entropy': self.source_test_pixel_entropy,
        #     'source_train_pixel_entropy': self.source_train_pixel_entropy,

        #     'target_test_pxap': self.target_test_pxap,
        #     'source_test_pxap': self.source_test_pxap,
        #     'source_train_pxap': self.source_train_pxap,

        #     'target_test_dice_bg': self.target_test_dice_bg,
        #     'source_test_dice_bg': self.source_test_dice_bg,
        #     'source_train_dice_bg': self.source_train_dice_bg,

        #     'target_test_dice_fg': self.target_test_dice_fg,
        #     'source_test_dice_fg': self.source_test_dice_fg,
        #     'source_train_dice_fg': self.source_train_dice_fg,

        #     'target_test_miou': self.target_test_miou,
        #     'source_test_miou': self.source_test_miou,
        #     'source_train_miou': self.source_train_miou
        # }

        pickle_path = os.path.join(self.args.outd, 'results_source_target_data.pickle')
        with open(pickle_path, 'wb') as f:
            pkl.dump(curves_data, f)

    def save_loss_esfda(self):
         #Store data in a pickle
        curves_data = {
            'master_loss': self.store_master_loss,
            'ce_flip_loss': self.store_ce_flip_loss,
            'ce_not_flip_loss': self.store_ce_not_flip_loss,
        }
    

        pickle_path = os.path.join(self.args.outd, 'results_loss_esfda_data.pickle')
        with open(pickle_path, 'wb') as f:
            pkl.dump(curves_data, f)

    def save_unlearning_acc(self, filename="unlearning_acc_history.pickle"):
        """
        Sauvegarde l'historique des accuracies flip et stable dans un pickle.
        """
        data_to_save = {
            "history_acc_flip": getattr(self, "history_acc_flip", []),
            "history_acc_stable": getattr(self, "history_acc_stable", [])
        }

        pickle_path = os.path.join(self.args.outd, filename)

        with open(pickle_path, "wb") as f:
            pkl.dump(data_to_save, f)

        print(f"Unlearning accuracies saved to {pickle_path}")

    def save_metrics(self, filename="metrics_history.pickle"):
        """
        Save metrics histories (KL, Hm, Htilde) to a pickle file.
        
        Args:
            metrics: your metrics object (with .hist, .hm, .h_tilde attributes)
            filename: output pickle file name
        """
        #data_to_save = {
        #    "kl": self.metrics.hist,
        #    "hm": self.metrics.hm_hist,
        #    "h_tilde": self.metrics.htilde_hist
        #}

        pickle_path = os.path.join(self.args.outd,filename)

        with open(pickle_path, "wb") as f:
            pkl.dump(data_to_save, f)

        print(f"Metrics saved to {filename}")


    def save_loss(self, filename="loss_history.pickle"):
        """
        Save loss history to a pickle file.
        
        Args:
            loss: your metrics object (with .hist, .hm, .h_tilde attributes)
            filename: output pickle file name
        """
        data_to_save = {
            "loss": self.store_loss
        }

        pickle_path = os.path.join(self.args.outd,filename)

        with open(pickle_path, "wb") as f:
            pkl.dump(data_to_save, f)

        print(f"Loss saved to {filename}")
    

    def plot_source_target_acc_curves(self, task, cmpt_epoch):
        if task == "cl":
            source_data = self.source_test_acc_cl
            target_data = self.target_test_acc_cl
            ylabel = 'Classification'
        else:
            source_data = self.source_test_acc_loc
            target_data = self.target_test_acc_loc
            ylabel = 'Localization'

        plt.figure(figsize=(12, 3))
        plt.plot(np.arange(cmpt_epoch, len(source_data) * cmpt_epoch + cmpt_epoch, cmpt_epoch), source_data, label='Source')
        plt.plot(np.arange(cmpt_epoch, len(target_data) * cmpt_epoch + cmpt_epoch, cmpt_epoch), target_data, label='Target')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        #set y axis labels only in integer with max value to len of source and target acc.
        #epochs = np.arange(0, len(source_data) * cmpt_epoch + 1, cmpt_epoch)
        plt.xticks(np.arange(cmpt_epoch, len(source_data) * cmpt_epoch + cmpt_epoch, cmpt_epoch))
        plt.tight_layout()
        #plt.savefig(os.path.join(self.args.outd, 'Classification accuracy curve on test set between source and target dataset.png'))
        file_prefix = f"{ylabel}_Accuracy"
        output_path = os.path.join(self.args.outd, f'{file_prefix}_curve_on_test_set_between_source_and_target_dataset.png')
        plt.savefig(output_path)
        plt.close()

        #Store data in a pickle
        curves_data = {
            'source_acc_cl': source_data,
            'target_acc_cl': target_data
        }

        pickle_path = os.path.join(self.args.outd, f'{file_prefix}_results_source_target_data.pickle')
        with open(pickle_path, 'wb') as f:
            pkl.dump(curves_data, f)

    # def plot_target_acc_curves(self, task, cmpt_epoch):
    #     if task == "cl":
    #         target_data = self.target_train_acc_cl
    #         ylabel = 'Classification'

    #     elif task == "silhouette":
    #         target_data = self.silhouette
    #         ylabel = 'Silhouette'
        
    #     elif task == "DBI":
    #         target_data = self.DBI
    #         ylabel = 'DBI'
        
    #     elif task == "CH":
    #         target_data = self.CH
    #         ylabel = 'CH'
        
    #     elif task == "J_index":
    #         target_data = self.J_index
    #         ylabel = 'J_index'

    #     elif task == "f1":
    #         target_data = self.target_train_f1
    #         ylabel = 'F1 Score'

    #     elif task == "precision":
    #         target_data = self.target_train_precision
    #         ylabel = 'Precision'

    #     elif task == "recall":
    #         target_data = self.target_train_recall
    #         ylabel = 'Recall'

    #     elif task == "image_entropy":
    #         target_data = self.target_train_image_entropy
    #         ylabel = 'Image Entropy'

    #     elif task == "acc_normal":
    #         target_data = self.target_train_acc_normal
    #         ylabel = 'Normal Classification'
            
    #     elif task == "acc_cancer":
    #         target_data = self.target_train_acc_cancer
    #         ylabel = 'Cancer Classification'
            
    #     else:
    #         target_data = self.target_valpx_pxap
    #         ylabel = 'Localization'

    #     plt.figure(figsize=(12, 3))
    #     #plt.plot(np.arange(cmpt_epoch, len(source_data) * cmpt_epoch + cmpt_epoch, cmpt_epoch), source_data, label='Source')
    #     plt.plot(np.arange(cmpt_epoch, len(target_data) * cmpt_epoch + cmpt_epoch, cmpt_epoch), target_data, label='Target')
    #     plt.xlabel('Epoch')
    #     plt.ylabel(ylabel)
    #     plt.legend()
    #     #set y axis labels only in integer with max value to len of source and target acc
    #     #epochs = np.arange(0, len(source_data) * cmpt_epoch + 1, cmpt_epoch)
    #     plt.xticks(np.arange(cmpt_epoch, len(target_data) * cmpt_epoch + cmpt_epoch, cmpt_epoch))
    #     plt.tight_layout()
    #     #plt.savefig(os.path.join(self.args.outd, 'Classification accuracy curve on test set between source and target dataset.png'))
    #     file_prefix = f"{ylabel}_Accuracy"
    #     output_path = os.path.join(self.args.outd, f'{file_prefix}_curve_on_test_set_on_target_dataset.png')
    #     plt.savefig(output_path)
    #     plt.close()

    #     #Store data in a pickle
    #     curves_data = {
    #         'target_acc_cl': target_data
    #     }

    #     pickle_path = os.path.join(self.args.outd, f'{file_prefix}_results_target_data.pickle')
    #     with open(pickle_path, 'wb') as f:
    #         pkl.dump(curves_data, f)

    def fmt(self,x):
        """Format number/string to be filename-safe."""
        s = str(x)
        s = s.replace('.', 'p')
        s = s.replace('-', 'm')
        return s

    def plot_target_acc_curves(self, task, cmpt_epoch, split):
        """
        task : str in {
            'cl', 'silhouette', 'DBI', 'CH', 'J_index',
            'f1', 'precision', 'recall',
            'image_entropy', 'acc_normal', 'acc_cancer',
            'acc_flip', 'acc_stable', 'kl_uniform', 'acc_global'
        }
        split : constants.TRAINSET or constants.CLVALIDSET
        """

        lr = self.fmt(self.args.optimizer.get("opt__lr", "NA"))
        step_size = self.fmt(self.args.optimizer.get("opt__step_size", "NA"))
        Resample = self.fmt(getattr(self.args, "resample_every", "NA"))
        imgs_ratio = self.fmt(getattr(self.args, "esfda_select_imgs_ratio", "NA"))
        Retain_lambda = self.fmt(getattr(self.args, "CERetain_lambda", "NA"))
        Forget_lambda = self.fmt(getattr(self.args, "CEForget_lambda", "NA"))

        hp_suffix = f"LR{lr}_STEP{step_size}_K{Resample}_P{imgs_ratio}_LRETAIN{Retain_lambda}_LFORGET{Forget_lambda}"


        task_map = {
            "cl":           ("acc_cl",           "Classification",          "Classification"),
            "silhouette":   ("silhouette",    "Silhouette",              "Silhouette"),
            "DBI":          ("DBI",           "DBI",                     "DBI"),
            "CH":           ("CH",            "CH",                      "CH"),
            "J_index":      ("J_index",       "J_index",                 "J_index"),
            "f1":           ("f1",            "F1 Score",                "F1"),
            "precision":    ("precision",     "Precision",               "Precision"),
            "recall":       ("recall",        "Recall",                  "Recall"),
            "image_entropy":("entropy",       "Image Entropy",           "Image_Entropy"),
            "acc_normal":   ("acc_normal",    "Normal Classification",   "Acc_Normal"),
            "acc_cancer":   ("acc_cancer",    "Cancer Classification",   "Acc_Cancer"),
            "acc_flip":     ("acc_flip",      "Flip Accuracy",           "Acc_Flip"),
            "acc_stable":   ("acc_stable",    "Stable Accuracy",         "Acc_Stable"),
            "kl_uniform":   ("kl_uniform",    "KL Divergence (Uniform)", "KL_Uniform"),
            "acc_global":   ("acc_global",    "Global Accuracy",         "Acc_Global"),
            "ECE":          ("ece",    "ECE",         "ECE"),
            "NLL":          ("nll",    "NLL",         "NLL"),
            "Brier":        ("brier",    "Brier",         "Brier"),
        }

        if task not in task_map:
            raise ValueError(f"Unknown task '{task}'. Available: {list(task_map.keys())}")

        metric_key, ylabel, file_prefix = task_map[task]

        if not hasattr(self, "metrics"):
            raise RuntimeError("self.metrics is not initialized.")

        if split not in self.metrics:
            raise RuntimeError(f"Split '{split}' not found in self.metrics. Available: {list(self.metrics.keys())}")

        if metric_key not in self.metrics[split]:
            raise RuntimeError(f"Metric '{metric_key}' not found in self.metrics['{split}'].")

        target_data = self.metrics[split][metric_key]

        if len(target_data) == 0:
            print(f"[plot_target_acc_curves] No data for metric '{metric_key}' on split '{split}'.")
            return

        base_dir = os.path.join(self.args.outd, "unlearning_metrics", str(split))
        os.makedirs(base_dir, exist_ok=True)

        epochs = np.arange(cmpt_epoch, len(target_data) * cmpt_epoch + cmpt_epoch, cmpt_epoch)

        # --- Plot
        plt.figure(figsize=(12, 3))
        plt.plot(epochs, target_data, label=f'{split}')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.xticks(epochs)
        plt.tight_layout()

        png_name = f"{file_prefix}_curve_{split}_{hp_suffix}.png"
        output_path = os.path.join(base_dir, png_name)
        plt.savefig(output_path)
        plt.close()

        curves_data = {
            "epochs": epochs.tolist(),
            "values": target_data,
            "metric_key": metric_key,
            "ylabel": ylabel,
            "split": split,
        }
        pickle_name = f"{file_prefix}_results_{split}_{hp_suffix}.pickle"
        pickle_path = os.path.join(base_dir, pickle_name)
        with open(pickle_path, "wb") as f:
            pkl.dump(curves_data, f)

        all_metrics_path = os.path.join(base_dir, f"all_metrics_{split}_{hp_suffix}.pickle")
        with open(all_metrics_path, "wb") as f:
            pkl.dump(self.metrics[split], f)

    def plot_unlearning_acc(self):
        plt.figure(figsize=(8, 5))

        
        batches = range(1, len(self.history_acc_flip) + 1)

        plt.plot(batches, self.history_acc_flip, label="Accuracy Flip", linewidth=2, color="red")
        plt.plot(batches, self.history_acc_stable, label="Accuracy Stable", linewidth=2, color="blue")

        plt.title("Unlearning Accuracies per Batch")
        plt.xlabel("Batch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1) 
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(self.args.outd, 'unlearning_acc_curves.png')
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Figure saved at {output_path}")
        plt.close()

    def plot_losses(self):
        plt.figure(figsize=(8,5))


        epochs = range(1, len(self.store_master_loss) + 1)

        plt.plot(epochs, self.store_master_loss, label="Master Loss", linewidth=2)
        plt.plot(epochs, self.store_ce_flip_loss, label="CE Flip Loss", linewidth=2)
        plt.plot(epochs, self.store_ce_not_flip_loss, label="CE Not Flip Loss", linewidth=2)

        plt.title("Loss curves during training")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(self.args.outd, 'curve_on_train_esfda.png')
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Figure saved at {output_path}")
        plt.close()

    def evaluate_entropy(self, epoch, split, checkpoint_type=None, fcam_argmax=False):
        torch.cuda.empty_cache()
        assert split in [constants.TESTSET, constants.VALIDSET, constants.TRAINSET]

        if split == constants.TESTSET:
            splitpx = split
            splitcl = split
        elif split == constants.VALIDSET:
            splitpx = constants.PXVALIDSET
            splitcl = constants.CLVALIDSET
        elif split == constants.TRAINSET:
            splitpx = split
            splitcl = split
        else:
            raise NotImplementedError

        if fcam_argmax:
            assert self.args.task in [constants.F_CL, constants.NEGEV,
                                    constants.SEG]

        self.fcam_argmax_previous = self.fcam_argmax
        self.fcam_argmax = fcam_argmax
        tagargmax = ''
        if self.args.task in [constants.F_CL, constants.NEGEV]:
            tagargmax = 'Argmax {}'.format(fcam_argmax)

        DLLogger.log(fmsg("Evaluate: Epoch {} Split {} {}".format(
            epoch, split, tagargmax)))

        outd = None
        if split == constants.TESTSET:
            assert checkpoint_type is not None
            if fcam_argmax:
                outd = join(self.args.outd, checkpoint_type, 'argmax-true',
                            split)
            else:
                outd = join(self.args.outd, checkpoint_type, split)
        elif split == constants.VALIDSET:
            _chpt = 'training' if checkpoint_type is None else checkpoint_type
            if fcam_argmax:
                outd = join(self.args.outd, _chpt, 'argmax-true', split)
            else:
                outd = join(self.args.outd, _chpt, split)

        elif split == constants.TRAINSET:
            _chpt = 'training' if checkpoint_type is None else checkpoint_type
            if fcam_argmax:
                outd = join(self.args.outd, _chpt, 'argmax-true', split)
            else:
                outd = join(self.args.outd, _chpt, split)
        else:
            raise NotImplementedError

        os.makedirs(outd, exist_ok=True)

        set_seed(seed=self.default_seed, verbose=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model.eval()

        # cl.
        accuracy = 0.0
        if self.args.task != constants.SEG:
            accuracy, entropy = self._compute_accuracy_and_entropy(loader=self.loaders[splitcl])

        self.performance_meters[
            splitcl][constants.CLASSIFICATION_MTR].update(accuracy)

        torch.cuda.empty_cache()

    def evaluate(self, epoch, split, checkpoint_type=None, fcam_argmax=False):
        torch.cuda.empty_cache()
        assert split in [constants.TESTSET, constants.VALIDSET]

        if split == constants.TESTSET:
            splitpx = split
            splitcl = split
        elif split == constants.VALIDSET:
            splitpx = constants.PXVALIDSET
            splitcl = constants.CLVALIDSET
        else:
            raise NotImplementedError

        if fcam_argmax:
            assert self.args.task in [constants.F_CL, constants.NEGEV,
                                      constants.SEG]

        self.fcam_argmax_previous = self.fcam_argmax
        self.fcam_argmax = fcam_argmax
        tagargmax = ''
        if self.args.task in [constants.F_CL, constants.NEGEV]:
            tagargmax = 'Argmax {}'.format(fcam_argmax)

        DLLogger.log(fmsg("Evaluate: Epoch {} Split {} {}".format(
            epoch, split, tagargmax)))

        outd = None
        if split == constants.TESTSET:
            assert checkpoint_type is not None
            if fcam_argmax:
                outd = join(self.args.outd, checkpoint_type, 'argmax-true',
                            split)
            else:
                outd = join(self.args.outd, checkpoint_type, split)
        elif split == constants.VALIDSET:
            _chpt = 'training' if checkpoint_type is None else checkpoint_type
            if fcam_argmax:
                outd = join(self.args.outd, _chpt, 'argmax-true', split)
            else:
                outd = join(self.args.outd, _chpt, split)
        else:
            raise NotImplementedError

        os.makedirs(outd, exist_ok=True)

        set_seed(seed=self.default_seed, verbose=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model.eval()

        # cl.
        accuracy = 0.0
        precision, recall, f1 = 0.0, 0.0, 0.0
        if self.args.task != constants.SEG:
            accuracy = self._compute_accuracy(loader=self.loaders[splitcl])
            #accuracy,precision, recall, f1 = self._compute_accuracy_binary_metrics(loader=self.loaders[splitcl])

        self.performance_meters[
            splitcl][constants.CLASSIFICATION_MTR].update(accuracy)

        self.performance_meters[
            splitcl][constants.PRECISION_MTR].update(precision)
        self.performance_meters[
            splitcl][constants.RECALL_MTR].update(recall)
        self.performance_meters[
            splitcl][constants.F1_MTR].update(f1)

        # loc.
        if not self.args.localization_avail:
            return None

        cam_curve_interval = self.args.cam_curve_interval
        cmdx = (split == constants.VALIDSET)
        cmdx &= self.args.dataset in [constants.CUB, constants.ILSVRC]
        if cmdx:
            cam_curve_interval = constants.VALID_FAST_CAM_CURVE_INTERVAL

        if split == constants.VALIDSET:
            best_valid_tau = None
        elif split == constants.TESTSET:
            if checkpoint_type == constants.BEST_LOC:
                best_valid_tau = self.best_valid_tau_loc
            elif checkpoint_type == constants.BEST_CL:
                best_valid_tau = self.best_valid_tau_cl
            else:
                raise NotImplementedError
            assert best_valid_tau is not None
        else:
            raise NotImplementedError
        # todo: remove code above. to measure segmentation performance,
        #  we estimate the bes tthreshold over the current set. e.g. in the
        #  case of testset, we estimate the best segmentation metrics with
        #  the best threshold obtained with miou.
        best_valid_tau = None

        cam_computer = CAMComputer(
            args=deepcopy(self.args),
            model=self.model,
            loader=self.loaders[splitpx],
            metadata_root=os.path.join(self.args.metadata_root, splitpx),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset,
            split=splitpx,
            cam_curve_interval=cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            out_folder=outd,
            fcam_argmax=fcam_argmax,
            best_valid_tau=best_valid_tau
        )
        t0 = dt.datetime.now()

        cam_performance = cam_computer.compute_and_evaluate_cams()

        DLLogger.log(fmsg("CAM EVALUATE TIME of {} split: {}".format(
            split, dt.datetime.now() - t0)))

        #if split == constants.TESTSET:
            #cam_computer.draw_some_best_pred()

        avg = self.args.multi_iou_eval
        avg |= self.args.dataset in [constants.OpenImages,constants.OpenImagesSrc, constants.OpenImagesTrgt, constants.GLAS,
                                     constants.CAMELYON512, constants.CAMELYON17_512]
        if avg:
            loc_score = np.average(cam_performance)
        else:
            loc_score = cam_performance[self.args.iou_threshold_list.index(50)]

        self.performance_meters[splitpx][constants.LOCALIZATION_MTR].update(
            loc_score)

        self.perf_gist_tracker[splitpx].update(
            epoch=epoch, new_value=cam_computer.evaluator.perf_gist)

        if split in [constants.TESTSET, constants.VALIDSET]:

            curves = cam_computer.evaluator.curve_s
            if split == constants.TESTSET:
                with open(join(outd, 'curves.pkl'), 'wb') as fc:
                    pkl.dump(curves, fc, protocol=pkl.HIGHEST_PROTOCOL)
            elif split == constants.VALIDSET:
                if checkpoint_type is None:
                    _outd = None
                    if self._is_best_model_cl(epoch):
                        _outd = join(outd, constants.BEST_CL)
                        os.makedirs(_outd, exist_ok=True)
                        with open(join(_outd, 'curves.pkl'), 'wb') as fc:
                            pkl.dump(curves, fc, protocol=pkl.HIGHEST_PROTOCOL)
                    if self._is_best_model_loc(epoch):
                        _outd = join(outd, constants.BEST_LOC)
                        os.makedirs(_outd, exist_ok=True)
                        with open(join(_outd, 'curves.pkl'), 'wb') as fc:
                            pkl.dump(curves, fc, protocol=pkl.HIGHEST_PROTOCOL)

                else:
                    _outd = outd
                    os.makedirs(_outd, exist_ok=True)
                    with open(join(_outd, 'curves.pkl'), 'wb') as fc:
                        pkl.dump(curves, fc, protocol=pkl.HIGHEST_PROTOCOL)

            if split == constants.TESTSET:
                title = get_tag(self.args, checkpoint_type=checkpoint_type)

                if fcam_argmax:
                    title += '_argmax_true.'
                else:
                    title += '_argmax_false.'
                title += r' Best $\tau$: {}'.format(
                    curves[constants.MTR_BESTTAU][0])
                self.plot_loc_perf_curves(
                    curves=curves, fdout=outd, title=title,
                    checkpoint_type=checkpoint_type)

        torch.cuda.empty_cache()

    def plot_loc_perf_curves(self, curves: dict, fdout: str, title: str,
                             checkpoint_type: str):

        ncols = 4
        ks = ['y', constants.MTR_MIOU, constants.MTR_TP, constants.MTR_TN,
              constants.MTR_FP, constants.MTR_FN, constants.MTR_DICEFG,
              constants.MTR_DICEBG]

        best_tau = curves[constants.MTR_BESTTAU][0]
        x_tau = curves['threshold_list_right_edge'][:-1]
        x_pxap = curves['x']
        nbr_th = len(x_tau)
        idx = curves['idx']
        hand = checkpoint_type
        assert idx < len(x_tau)
        assert idx < x_pxap.size

        if len(ks) > ncols:
            nrows = math.ceil(len(ks) / float(ncols))
        else:
            nrows = 1
            ncols = len(ks)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False)

        t = 0
        for i in range(nrows):
            for j in range(ncols):
                if t >= len(ks):
                    axes[i, j].set_visible(False)
                    t += 1
                    continue

                xx_t = None
                if t == 0:
                    val = curves['y']
                    x = x_pxap
                    subtitle = 'Recall/precision'
                    x_label = 'Recall'
                    y_label = 'Precision'
                    axes[i, j].plot(x, val, color='tab:orange')
                else:
                    val = curves[ks[t]][:nbr_th]
                    x = x_tau
                    n = len(x)
                    subtitle = ks[t]
                    x_label = r'$\tau$'
                    y_label = None
                    xx_t = list(range(n))
                    axes[i, j].plot(xx_t, val, color='tab:orange')

                axes[i, j].set_title(subtitle, fontsize=4)
                axes[i, j].xaxis.set_tick_params(labelsize=4)
                axes[i, j].yaxis.set_tick_params(labelsize=4)
                axes[i, j].set_xlabel(x_label, fontsize=4)
                if y_label is not None:
                    axes[i, j].set_ylabel(y_label, fontsize=4)

                axes[i, j].grid(True)

                if ks[t] != 'y':
                    # todo: not perfect but ok.
                    a = axes[i, j].get_xticks().tolist()
                    aa = [z / 1000. for z in a]
                    axes[i, j].set_xticklabels(aa)

                if t == 0:
                    axes[i, j].plot([x[idx]], [val[idx]],
                                    marker='o',
                                    markersize=5,
                                    color=constants.COLOUR_BEST_CP[hand],
                                    label=hand
                                    )
                    axes[i, j].legend(loc='best', prop={'size': 5})
                else:
                    axes[i, j].plot([xx_t[idx]], [val[idx]],
                                    marker='o',
                                    markersize=5,
                                    color=constants.COLOUR_BEST_CP[hand]
                                    )

                t += 1

        fig.suptitle(title, fontsize=8)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()
        fig.savefig(join(fdout, 'curves_loc_perf.png'), bbox_inches='tight',
                    dpi=300)

    def capture_perf_meters(self):
        self.perf_meters_backup = deepcopy(self.performance_meters)
        self.perf_gist_backup = deepcopy(self.perf_gist_tracker)

    def switch_perf_meter_to_captured(self):
        self.performance_meters = deepcopy(self.perf_meters_backup)
        self.perf_gist_tracker = deepcopy(self.perf_gist_backup)
        self.fcam_argmax = self.fcam_argmax_previous

    def save_args(self):
        self._save_args(path=join(self.args.outd, 'config_obj_final.yaml'))

    def _save_args(self, path):
        _path = path
        with open(_path, 'w') as f:
            self.args.tend = dt.datetime.now()
            yaml.dump(vars(self.args), f)

    @property
    def cpu_device(self):
        return get_cpu_device()

    def save_pseudo_labels(self, filename="pseudo_label_distribution.png"):

        os.makedirs(self.args.outd, exist_ok=True)

        pickle_path = os.path.join(self.args.outd, "pseudo_labels.pkl")
        with open(pickle_path, "wb") as f:
            pkl.dump(self.pseudo_labels, f)
        print(f"[Save] Pseudo-labels saved to {pickle_path}")


        accuracies = self.pseudo_labels
        epochs = list(range(len(accuracies)))
        
        plt.figure()
        plt.plot(epochs, accuracies, marker='o')

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy of pseudo labels")
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.args.outd, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Pseudo-labels saved : {save_path}")




    def save_unlearning_models(self):

        sorted_models = sorted(
            self.best_unlearning_models.items(), 
            key=lambda x: x[1][0], 
            reverse=True            
        )
            
        unlearning_log_path = os.path.join(self.args.outd, "best_models_unlearning.txt")

        # with open(unlearning_log_path, "w") as f:
        #     f.write("Rank\tEpoch\tcl\n")
        #     for rank, (epoch, (entropy, _model)) in enumerate(sorted_models, start=1):
        #         f.write(f"{rank}\t{epoch}\t{entropy:.6f}\n")

        
        for rank, (epoch, (entropy, _model)) in enumerate(sorted_models, start=1):
            checkpoint_type = f"B-UNLEARNING{rank}"
            tag = get_tag(self.args, checkpoint_type=checkpoint_type)
            path = os.path.join(self.args.outd, tag)

            if not os.path.isdir(path):
                os.makedirs(path)
            if self.args.task == constants.STD_CL:
                if self.args.method in [constants.METHOD_ACOL,
                                        constants.METHOD_ADL,
                                        constants.METHOD_SPG,
                                        constants.METHOD_TSCAM,
                                        constants.METHOD_SAT]:
                    torch.save(_model.state_dict(),
                            join(path, 'model.pt'))

                elif self.args.method == constants.METHOD_MAXMIN:
                    torch.save(_model.encoder.state_dict(),
                            join(path, 'encoder.pt'))
                    torch.save(_model.classification_head1.state_dict(),
                            join(path, 'classification_head1.pt'))
                    torch.save(_model.classification_head2.state_dict(),
                            join(path, 'classification_head2.pt'))
                    if _model.mask_head is not None:
                        torch.save(_model.mask_head.state_dict(),
                                join(path, 'mask_head.pt'))

                elif self.args.method == constants.METHOD_PIXELCAM:
                    if "deit" in self.args.model['encoder_name']:
                        torch.save(_model.state_dict(),
                            join(path, 'model.pt'))
                    else:
                        torch.save(_model.encoder.state_dict(),
                                join(path, 'encoder.pt'))
                        torch.save(_model.classification_head.state_dict(),
                                join(path, 'classification_head.pt'))
                        torch.save(_model.pixel_wise_classification_head.state_dict(),
                                    join(path, 'pixel_wise_classification_head.pt'))
                else:
                    torch.save(_model.encoder.state_dict(),
                            join(path, 'encoder.pt'))
                    torch.save(_model.classification_head.state_dict(),
                            join(path, 'classification_head.pt'))
                
            self._save_args(path=join(path, 'config_model.yaml'))
            DLLogger.log(message="Stored Model [CP: {} \t EPOCH: {} \t TAG: {}]:"
                                " {}".format(checkpoint_type, epoch, tag, path))



    

    def save_best_entropy_models(self):

        sorted_models = sorted(
            self.best_entropy_models.items(), 
            key=lambda x: x[1][0], 
            reverse=True            
        )
            
        entropy_log_path = os.path.join(self.args.outd, "best_models_entropy.txt")

        with open(entropy_log_path, "w") as f:
            f.write("Rank\tEpoch\tEntropy\n")
            for rank, (epoch, (entropy, _model)) in enumerate(sorted_models, start=1):
                f.write(f"{rank}\t{epoch}\t{entropy:.6f}\n")

        #for epoch, (entropy, _model) in self.best_entropy_models.items():
        for rank, (epoch, (entropy, _model)) in enumerate(sorted_models, start=1):
            checkpoint_type = f"B-EPOCH{rank}"
            tag = get_tag(self.args, checkpoint_type=checkpoint_type)
            path = os.path.join(self.args.outd, tag)

            if not os.path.isdir(path):
                os.makedirs(path)
            if self.args.task == constants.STD_CL:
                if self.args.method in [constants.METHOD_ACOL,
                                        constants.METHOD_ADL,
                                        constants.METHOD_SPG,
                                        constants.METHOD_TSCAM,
                                        constants.METHOD_SAT]:
                    torch.save(_model.state_dict(),
                            join(path, 'model.pt'))

                elif self.args.method == constants.METHOD_MAXMIN:
                    torch.save(_model.encoder.state_dict(),
                            join(path, 'encoder.pt'))
                    torch.save(_model.classification_head1.state_dict(),
                            join(path, 'classification_head1.pt'))
                    torch.save(_model.classification_head2.state_dict(),
                            join(path, 'classification_head2.pt'))
                    if _model.mask_head is not None:
                        torch.save(_model.mask_head.state_dict(),
                                join(path, 'mask_head.pt'))

                elif self.args.method == constants.METHOD_PIXELCAM:
                    if "deit" in self.args.model['encoder_name']:
                        torch.save(_model.state_dict(),
                            join(path, 'model.pt'))
                    else:
                        torch.save(_model.encoder.state_dict(),
                                join(path, 'encoder.pt'))
                        torch.save(_model.classification_head.state_dict(),
                                join(path, 'classification_head.pt')),
                        torch.save(_model.pixel_wise_classification_head.state_dict(),
                                    join(path, 'pixel_wise_classification_head.pt'))
                else:
                    torch.save(_model.encoder.state_dict(),
                            join(path, 'encoder.pt'))
                    torch.save(_model.classification_head.state_dict(),
                            join(path, 'classification_head.pt'))
                
            self._save_args(path=join(path, 'config_model.yaml'))
            DLLogger.log(message="Stored Model [CP: {} \t EPOCH: {} \t TAG: {}]:"
                                " {}".format(checkpoint_type, epoch, tag, path))
            

    # def save_best_cl_train_models(self):

    #     model = self.cl_train_model  
    #     epoch = self.best_epoch      
    #     acc = self.best_accuracy    

    #     save_dir = os.path.join(self.args.outd, "best_model_cl")
    #     os.makedirs(save_dir, exist_ok=True)

    #     # Save in  .txt
    #     with open(os.path.join(self.args.outd, "best_model_info.txt"), "w") as f:
    #         f.write(f"Best classification model at epoch {epoch} with accuracy {acc:.2f}%\n")

    #     # Sauvegarde du modèle selon la méthode utilisée
    #     if self.args.task == constants.STD_CL:
    #         method = self.args.method

    #         if method in [constants.METHOD_ACOL,
    #                     constants.METHOD_ADL,
    #                     constants.METHOD_SPG,
    #                     constants.METHOD_TSCAM,
    #                     constants.METHOD_SAT]:
    #             torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

    #         elif method == constants.METHOD_MAXMIN:
    #             torch.save(model.encoder.state_dict(), os.path.join(save_dir, 'encoder.pt'))
    #             torch.save(model.classification_head1.state_dict(), os.path.join(save_dir, 'classification_head1.pt'))
    #             torch.save(model.classification_head2.state_dict(), os.path.join(save_dir, 'classification_head2.pt'))
    #             if model.mask_head is not None:
    #                 torch.save(model.mask_head.state_dict(), os.path.join(save_dir, 'mask_head.pt'))

    #         elif method == constants.METHOD_PIXELCAM:
    #             if "deit" in self.args.model['encoder_name']:
    #                 torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    #             else:
    #                 torch.save(model.encoder.state_dict(), os.path.join(save_dir, 'encoder.pt'))
    #                 torch.save(model.classification_head.state_dict(), os.path.join(save_dir, 'classification_head.pt'))
    #                 torch.save(model.pixel_wise_classification_head.state_dict(), os.path.join(save_dir, 'pixel_wise_classification_head.pt'))

    #         else:  
    #             torch.save(model.encoder.state_dict(), os.path.join(save_dir, 'encoder.pt'))
    #             torch.save(model.classification_head.state_dict(), os.path.join(save_dir, 'classification_head.pt'))

        
    #     self._save_args(path=os.path.join(save_dir, 'config_model.yaml'))

    #     DLLogger.log(message=f"[SAVE] Best classification model (Epoch {epoch}, Acc {acc:.2f}%) saved to: {save_dir}")

    def save_best_cl_train_models(self, criterion=constants.CLVALIDSET):
        """
        criterion: "valcl" or "valpx"
        """

        # --------------------------------------------------
        # 1. Select best model & metrics depending on criterion
        # --------------------------------------------------
        if criterion == constants.CLVALIDSET:
            model = self.cl_train_model
            epoch = self.best_epoch
            acc = self.best_accuracy
            metric_name = "Accuracy"
            metric_value = acc
            save_suffix = "valcl"

        elif criterion == constants.PXVALIDSET:
            model = self.pxap_train_model
            epoch = self.best_epoch_pxap
            acc = self.best_accuracy_pxap
            metric_name = "PxAP"
            metric_value = acc
            save_suffix = "valpx"

        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        # --------------------------------------------------
        # 2. Create save directory
        # --------------------------------------------------
        #save_dir = os.path.join(self.args.outd, f"best_model_{save_suffix}")
        #os.makedirs(save_dir, exist_ok=True)

        checkpoint_type = f"B-UNLEARNING_{save_suffix}"
        tag = get_tag(self.args, checkpoint_type=checkpoint_type)
        save_dir = os.path.join(self.args.outd, tag)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # --------------------------------------------------
        # 3. Save info file
        # --------------------------------------------------
        with open(os.path.join(save_dir, "best_model_info.txt"), "w") as f:
            f.write(
                f"Best model selected by {metric_name}\n"
                f"Epoch: {epoch}\n"
                f"{metric_name}: {metric_value:.4f}\n"
            )

        # --------------------------------------------------
        # 4. Save model weights (method-aware)
        # --------------------------------------------------
        if self.args.task == constants.STD_CL:
            method = self.args.method

            if method in [
                constants.METHOD_ACOL,
                constants.METHOD_ADL,
                constants.METHOD_SPG,
                constants.METHOD_TSCAM,
                constants.METHOD_SAT,
            ]:
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

            elif method == constants.METHOD_MAXMIN:
                torch.save(model.encoder.state_dict(), os.path.join(save_dir, "encoder.pt"))
                torch.save(model.classification_head1.state_dict(), os.path.join(save_dir, "classification_head1.pt"))
                torch.save(model.classification_head2.state_dict(), os.path.join(save_dir, "classification_head2.pt"))
                if model.mask_head is not None:
                    torch.save(model.mask_head.state_dict(), os.path.join(save_dir, "mask_head.pt"))

            elif method == constants.METHOD_PIXELCAM:
                if "deit" in self.args.model["encoder_name"]:
                    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
                else:
                    torch.save(model.encoder.state_dict(), os.path.join(save_dir, "encoder.pt"))
                    torch.save(model.classification_head.state_dict(), os.path.join(save_dir, "classification_head.pt"))
                    torch.save(
                        model.pixel_wise_classification_head.state_dict(),
                        os.path.join(save_dir, "pixel_wise_classification_head.pt"),
                    )

            else:
                torch.save(model.encoder.state_dict(), os.path.join(save_dir, "encoder.pt"))
                torch.save(model.classification_head.state_dict(), os.path.join(save_dir, "classification_head.pt"))

        # --------------------------------------------------
        # 5. Save config
        # --------------------------------------------------
        self._save_args(path=os.path.join(save_dir, "config_model.yaml"))

        DLLogger.log(
            message=(
                f"[SAVE] Best model ({criterion}) | "
                f"Epoch {epoch} | {metric_name}: {metric_value:.4f} | "
                f"Saved to {save_dir}"
            )
        )

    def save_best_epoch(self):
        if self.args.localization_avail:
            split = constants.PXVALIDSET

            best_loc_epoch = self.performance_meters[split][
                constants.LOCALIZATION_MTR].best_epoch
            self.args.best_loc_epoch = best_loc_epoch

        if self.args.task != constants.SEG:
            split = constants.CLVALIDSET

            best_cl_epoch = self.performance_meters[split][
                constants.CLASSIFICATION_MTR].best_epoch
            self.args.best_cl_epoch = best_cl_epoch

    def save_checkpoints(self):
        if self.args.localization_avail:
            split = constants.PXVALIDSET

            best_epoch = self.performance_meters[split][
                constants.LOCALIZATION_MTR].best_epoch

            self._save_model(checkpoint_type=constants.BEST_LOC,
                             epoch=best_epoch)

        if self.args.task != constants.SEG:
            split = constants.CLVALIDSET

            best_epoch = self.performance_meters[split][
                constants.CLASSIFICATION_MTR].best_epoch
            self._save_model(checkpoint_type=constants.BEST_CL,
                             epoch=best_epoch)

    def _save_model(self, checkpoint_type, epoch):
        assert checkpoint_type in [constants.BEST_LOC, constants.BEST_CL]

        if checkpoint_type == constants.BEST_CL:
            _model = deepcopy(self.best_cl_model).to(self.cpu_device).eval()
        elif checkpoint_type == constants.BEST_LOC:
            _model = deepcopy(self.best_loc_model).to(self.cpu_device).eval()
        else:
            raise NotImplementedError

        tag = get_tag(self.args, checkpoint_type=checkpoint_type)
        path = join(self.args.outd, tag)
        if not os.path.isdir(path):
            os.makedirs(path)

        if self.args.task == constants.STD_CL:
            if self.args.method in [constants.METHOD_ACOL,
                                    constants.METHOD_ADL,
                                    constants.METHOD_SPG,
                                    constants.METHOD_TSCAM,
                                    constants.METHOD_SAT]:
                torch.save(_model.state_dict(),
                           join(path, 'model.pt'))

            elif self.args.method == constants.METHOD_MAXMIN:
                torch.save(_model.encoder.state_dict(),
                           join(path, 'encoder.pt'))
                torch.save(_model.classification_head1.state_dict(),
                           join(path, 'classification_head1.pt'))
                torch.save(_model.classification_head2.state_dict(),
                           join(path, 'classification_head2.pt'))
                if _model.mask_head is not None:
                    torch.save(_model.mask_head.state_dict(),
                               join(path, 'mask_head.pt'))

            elif self.args.method == constants.METHOD_PIXELCAM:
                if "deit" in self.args.model['encoder_name']:
                    torch.save(_model.state_dict(),
                           join(path, 'model.pt'))
                else:
                    torch.save(_model.encoder.state_dict(),
                            join(path, 'encoder.pt'))
                    torch.save(_model.classification_head.state_dict(),
                            join(path, 'classification_head.pt')),
                    torch.save(_model.pixel_wise_classification_head.state_dict(),
                                join(path, 'pixel_wise_classification_head.pt'))
            else:
                torch.save(_model.encoder.state_dict(),
                           join(path, 'encoder.pt'))
                torch.save(_model.classification_head.state_dict(),
                           join(path, 'classification_head.pt'))

        elif self.args.task in [constants.F_CL, constants.NEGEV]:

            torch.save(_model.encoder.state_dict(), join(path, 'encoder.pt'))
            torch.save(_model.decoder.state_dict(), join(path, 'decoder.pt'))
            torch.save(_model.segmentation_head.state_dict(),
                       join(path, 'segmentation_head.pt'))
            torch.save(_model.classification_head.state_dict(),
                       join(path, 'classification_head.pt'))

            if _model.reconstruction_head is not None:
                torch.save(_model.reconstruction_head.state_dict(),
                           join(path, 'reconstruction_head.pt'))

        elif self.args.task == constants.SEG:
            torch.save(_model.encoder.state_dict(), join(path, 'encoder.pt'))
            torch.save(_model.decoder.state_dict(), join(path, 'decoder.pt'))
            torch.save(_model.segmentation_head.state_dict(),
                       join(path, 'segmentation_head.pt'))
        else:
            raise NotImplementedError

        self._save_args(path=join(path, 'config_model.yaml'))
        DLLogger.log(message="Stored Model [CP: {} \t EPOCH: {} \t TAG: {}]:"
                             " {}".format(checkpoint_type, epoch, tag, path))
        

    def update_best_unlearning_model(self, epoch, m_unlearning_models):
        torch.cuda.empty_cache()
        self.model.eval()
        #torch.cuda.empty_cache()

        self.model.flush()
        model_unlearning = deepcopy(self.model).to(self.cpu_device).eval()
        #model_state = model_entropy.state_dict()

        if not hasattr(self, "best_unlearning_models"):
            self.best_entropy_models = {}  # {epoch: (entropy, state_dict)}


        if epoch <= m_unlearning_models:
            self.best_unlearning_models[epoch] = (epoch, model_unlearning)
            print(f"[Unlearning Sep: ] Model at epoch {epoch} is saved.")

    def update_best_entropy_model(self, epoch, split):
        torch.cuda.empty_cache()
        self.model.eval()
        accuracy = 0.0
        if self.args.task != constants.SEG:
            accuracy, entropy = self._compute_accuracy_and_entropy(loader=self.loaders[split])

        #torch.cuda.empty_cache()

        if self.args.method != constants.METHOD_MAXMIN:
            self.model.flush()

        self.model.flush()
        model_entropy = deepcopy(self.model).to(self.cpu_device).eval()
        #model_state = model_entropy.state_dict()

        if not hasattr(self, "best_entropy_models"):
            self.best_entropy_models = {}  # {epoch: (entropy, state_dict)}

        if len(self.best_entropy_models) < self.m_entropy_models:
            self.best_entropy_models[epoch] = (entropy, model_entropy)

        else:
            min_epoch, (min_entropy, _) = min(self.best_entropy_models.items(), key=lambda x: x[1][0])

            if entropy > min_entropy:
                del self.best_entropy_models[min_epoch]
                self.best_entropy_models[epoch] = (entropy, model_entropy)
                print(f"[Entropy Replace] Replaced model from epoch {min_epoch} (entropy {min_entropy:.4f}) with epoch {epoch} (entropy {entropy:.4f})")
            else:
                print(f"[Entropy Skip] Model at epoch {epoch} with entropy {entropy:.4f} was not selected.")


    def update_best_cl_train_model(self, epoch, split):
        torch.cuda.empty_cache()
        self.model.eval()
        accuracy = 0.0
        # if self.args.task != constants.SEG:
        #     accuracy = self._compute_accuracy(loader=self.loaders[split])

        torch.cuda.empty_cache()

        model_cl = deepcopy(self.model).to(self.cpu_device).eval()
        #model_state = model_entropy.state_dict()

        if not hasattr(self, "best_accuracy"):
            self.cl_train_model = None
            self.best_accuracy = -1.0
            self.best_epoch = -1

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.cl_train_model = deepcopy(model_cl)
            self.best_epoch = epoch
            print(f"[Accuracy Update] New best model at epoch {epoch} with accuracy {accuracy:.2f}%")
        else:
            print(f"[Accuracy Skip] Model at epoch {epoch} with accuracy {accuracy:.2f}% was not better than best ({self.best_accuracy:.2f}%)")

    def update_best_cl_train_model_came(self, epoch):
        torch.cuda.empty_cache()
        self.model.eval()
        # if self.args.task != constants.SEG:
        #     accuracy = self._compute_accuracy(loader=self.loaders[split])

        accuracy = 0.0
        # if self.args.task != constants.SEG:
        #     accuracy = self._compute_accuracy(loader=self.loaders[split])

        torch.cuda.empty_cache()

        model_cl = deepcopy(self.model).to(self.cpu_device).eval()
        #model_state = model_entropy.state_dict()

        acc_cl = self.metrics[constants.CLVALIDSET]['acc_cl'][-1]

        if not hasattr(self, "best_accuracy"):
            self.cl_train_model = None
            self.best_accuracy = -1.0
            self.best_epoch = -1

        if acc_cl > self.best_accuracy:
            self.best_accuracy = acc_cl
            self.cl_train_model = deepcopy(model_cl)
            self.best_epoch = epoch
            print(f"[Accuracy Update] New best model at epoch {epoch} with accuracy {acc_cl:.2f}%")
        else:
            print(f"[Accuracy Skip] Model at epoch {epoch} with accuracy {acc_cl:.2f}% was not better than best ({self.best_accuracy:.2f}%)")

        if self.args.measure_loc:
            acc_pxap = self.target_valpx_pxap[-1]

            if not hasattr(self, "best_accuracy_pxap"):
                self.pxap_train_model = None
                self.best_accuracy_pxap = -1.0
                self.best_epoch_pxap = -1
            if acc_pxap > self.best_accuracy_pxap:
                self.best_accuracy_pxap = acc_pxap
                self.pxap_train_model = deepcopy(model_cl)
                self.best_epoch_pxap = epoch
                print(f"[PXAP Update] New best PXAP model at epoch {epoch} with PXAP {acc_pxap:.2f}%")
            else:
                print(f"[PXAP Skip] PXAP model at epoch {epoch} with PXAP {acc_pxap:.2f}% was not better than best ({self.best_accuracy_pxap:.2f}%)")







    def _is_best_model_loc(self, epoch: int) -> bool:
        cnd = self.args.localization_avail
        cnd &= (self.performance_meters[constants.PXVALIDSET][
                    constants.LOCALIZATION_MTR].best_epoch) == epoch

        return cnd

    def _is_best_model_cl(self, epoch: int) -> bool:
        cnd = self.args.task != constants.SEG
        cnd &= (self.performance_meters[constants.CLVALIDSET][
                    constants.CLASSIFICATION_MTR].best_epoch) == epoch

        return cnd

    def model_selection(self, epoch):

        if self.args.method != constants.METHOD_MAXMIN:
            self.model.flush()

        if self._is_best_model_loc(epoch):
            self.best_loc_model = deepcopy(
                self.model).to(self.cpu_device).eval()
            self.best_valid_tau_loc = self.perf_gist_tracker[
                constants.PXVALIDSET][epoch][constants.MTR_BESTTAU][0]
            self.args.best_valid_tau_loc = self.best_valid_tau_loc

        if self._is_best_model_cl(epoch):
            self.best_cl_model = deepcopy(self.model).to(
                self.cpu_device).eval()

            if self.args.localization_avail:
                self.best_valid_tau_cl = self.perf_gist_tracker[
                    constants.PXVALIDSET][epoch][constants.MTR_BESTTAU][0]
                self.args.best_valid_tau_cl = self.best_valid_tau_cl

    def load_checkpoint(self, checkpoint_type):
        assert checkpoint_type in [constants.BEST_LOC, constants.BEST_CL]
        tag = get_tag(self.args, checkpoint_type=checkpoint_type)
        path = join(self.args.outd, tag)

        if self.args.task == constants.STD_CL:
            if self.args.method in [constants.METHOD_ACOL,
                                    constants.METHOD_ADL,
                                    constants.METHOD_SPG,
                                    constants.METHOD_TSCAM,
                                    constants.METHOD_SAT]:
                weights = torch.load(join(path, 'model.pt'),
                                     map_location=self.device)
                self.model.load_state_dict(weights, strict=True)

            elif self.args.method == constants.METHOD_MAXMIN:
                weights = torch.load(join(path, 'encoder.pt'),
                                     map_location=self.device)
                self.model.encoder.super_load_state_dict(
                    weights, strict=True)

                weights = torch.load(join(path, 'classification_head1.pt'),
                                     map_location=self.device)
                self.model.classification_head1.load_state_dict(
                    weights, strict=True)

                weights = torch.load(join(path, 'classification_head2.pt'),
                                     map_location=self.device)
                self.model.classification_head2.load_state_dict(
                    weights, strict=True)

                if self.model.mask_head is not None:
                    weights = torch.load(join(path, 'mask_head.pt'),
                                         map_location=self.device)
                    self.model.mask_head.load_state_dict(
                        weights, strict=True)

            else:
                if self.args.method == constants.METHOD_PIXELCAM and "deit" in self.args.model['encoder_name']:
                    weights = torch.load(join(path, 'model.pt'),
                                     map_location=self.device)
                    self.model.load_state_dict(weights, strict=True)
                else:
                    weights = torch.load(join(path, 'encoder.pt'),
                                        map_location=self.device)
                    self.model.encoder.super_load_state_dict(
                        weights, strict=True)

                    weights = torch.load(join(path, 'classification_head.pt'),
                                        map_location=self.device)
                    self.model.classification_head.load_state_dict(
                        weights, strict=True)
                    
                    if self.args.method == constants.METHOD_PIXELCAM:
                        weights = torch.load(join(path, 'pixel_wise_classification_head.pt'),
                                        map_location=self.device)
                        self.model.pixel_wise_classification_head.load_state_dict(
                            weights, strict=True)   

        elif self.args.task in [constants.F_CL, constants.NEGEV]:

            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=self.device)
            self.model.encoder.super_load_state_dict(weights,
                                                              strict=True)

            weights = torch.load(join(path, 'decoder.pt'),
                                 map_location=self.device)
            self.model.decoder.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'segmentation_head.pt'),
                                 map_location=self.device)
            self.model.segmentation_head.load_state_dict(weights,
                                                                  strict=True)

            weights = torch.load(join(path, 'classification_head.pt'),
                                 map_location=self.device)
            self.model.classification_head.load_state_dict(
                weights, strict=True)

            if self.model.reconstruction_head is not None:
                weights = torch.load(join(path, 'reconstruction_head.pt'),
                                     map_location=self.device)
                self.model.reconstruction_head.load_state_dict(
                    weights, strict=True)

        elif self.args.task == constants.SEG:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=self.device)
            self.model.encoder.super_load_state_dict(weights,
                                                              strict=True)

            weights = torch.load(join(path, 'decoder.pt'),
                                 map_location=self.device)
            self.model.decoder.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'segmentation_head.pt'),
                                 map_location=self.device)
            self.model.segmentation_head.load_state_dict(weights, strict=True)
        else:
            raise NotImplementedError

        DLLogger.log("Checkpoint {} loaded.".format(path))

    def report_train(self, train_performance: dict, epoch: int):
        split = constants.TRAINSET

        if self.args.task != constants.SEG:
            DLLogger.log('REPORT EPOCH/{}: {}/classification: {}'.format(
                epoch, split, train_performance['classification_acc']))

        DLLogger.log('REPORT EPOCH/{}: {}/loss: {}'.format(
            epoch, split, train_performance['loss']))

    def report(self, epoch, split, checkpoint_type=None):
        # todo: adapt.
        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = ' Argmax: True'
        if checkpoint_type is not None:
            DLLogger.log(fmsg('PERF - CHECKPOINT: {} {}'.format(
                checkpoint_type, tagargmax)))

        _splits = []
        if split in [constants.TRAINSET, constants.TESTSET]:
            _splits = [split]
        elif split == constants.VALIDSET:
            _splits = []
            if self.args.task != constants.SEG:
                _splits = [constants.CLVALIDSET]
            if self.args.localization_avail:
                _splits += [constants.PXVALIDSET]
            assert _splits != []
        else:
            raise NotImplementedError

        for _split in _splits:
            for metric in self._EVAL_METRICS:
                if self._skip_print_metric(metric=metric, _split=_split,
                                           split=split):
                    continue

                DLLogger.log(
                    "REPORT EPOCH/{}: split: {}/metric {}: {} ".format(
                        epoch, _split, metric,
                        self.performance_meters[_split][metric].current_value))
                DLLogger.log("REPORT EPOCH/{}: split: {}/metric {}: "
                             "{}_best ".format(
                              epoch, _split, metric,
                              self.performance_meters[_split][
                                  metric].best_value))

        if self.args.localization_avail and split in [
                constants.TESTSET, constants.VALIDSET]:
            _split = constants.TESTSET if split == constants.TESTSET else \
                constants.PXVALIDSET

            if _split == constants.TESTSET:
                _epoch = epoch
            else:
                _epoch = self.performance_meters[_split][
                    constants.LOCALIZATION_MTR].best_epoch
            DLLogger.log(f'REPORT EPOCH/{_epoch} split: {_split}: [BEST]\n'
                         f'{self.perf_gist_to_str(_split, _epoch)} \n')

    def _skip_print_metric(self, metric, _split, split):
        if (metric == constants.CLASSIFICATION_MTR) and (
                self.args.task == constants.SEG):
            return True
        if (metric == constants.LOCALIZATION_MTR) and (
                not self.args.localization_avail):
            return True

        if (metric == 'loss') and split in [
                    constants.TESTSET, constants.VALIDSET]:
            return True

        if (metric == constants.LOCALIZATION_MTR) and (
                _split == constants.CLVALIDSET):
            return True

        if (metric == constants.CLASSIFICATION_MTR) and (
                _split == constants.PXVALIDSET):
            return True

        return False

    def adjust_learning_rate(self):
        self.lr_scheduler.step()

    def plot_meter(self, metrics: dict, filename: str, title: str = '',
                   xlabel: str = '', best_iter_cl: int = None,
                   best_iter_loc: int = None):

        ncols = 4
        ks = list(metrics.keys())
        if len(ks) > ncols:
            nrows = math.ceil(len(ks) / float(ncols))
        else:
            nrows = 1
            ncols = len(ks)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False)
        t = 0
        for i in range(nrows):
            for j in range(ncols):
                if t >= len(ks):
                    axes[i, j].set_visible(False)
                    t += 1
                    continue

                val = metrics[ks[t]]['value_per_epoch']
                x = list(range(len(val)))
                axes[i, j].plot(x, val, color='tab:orange')
                subtitle = ks[t]
                if subtitle == constants.LOCALIZATION_MTR:
                    subtitle = 'PXAP localization'
                if subtitle == constants.CLASSIFICATION_MTR:
                    subtitle = 'Classification accuracy'

                axes[i, j].set_title(subtitle, fontsize=4)
                axes[i, j].xaxis.set_tick_params(labelsize=4)
                axes[i, j].yaxis.set_tick_params(labelsize=4)
                axes[i, j].set_xlabel('#{}'.format(xlabel), fontsize=4)
                axes[i, j].grid(True)
                axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

                for hand, best_iter in zip([constants.BEST_CL,
                                            constants.BEST_LOC],
                                           [best_iter_cl, best_iter_loc]):

                    if best_iter is not None:
                        _best_iter = best_iter if best_iter < len(x) else \
                            len(x) - 1
                        if i == j == 0:
                            axes[i, j].plot([x[_best_iter]], [val[_best_iter]],
                                            marker='o',
                                            markersize=5,
                                            color=constants.COLOUR_BEST_CP[
                                                hand],
                                            label=hand
                                            )
                            axes[i, j].legend(loc='best', prop={'size': 5})
                        else:
                            axes[i, j].plot([x[_best_iter]], [val[_best_iter]],
                                            marker='o',
                                            markersize=5,
                                            color=constants.COLOUR_BEST_CP[
                                                hand]
                                            )

                t += 1

        fig.suptitle(title, fontsize=4)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        fig.savefig(join(self.args.outd, '{}.png'.format(filename)),
                    bbox_inches='tight', dpi=300)

    def clean_metrics(self, metric: dict) -> dict:
        _metric = deepcopy(metric)
        l = []
        for k in _metric.keys():
            cd = (_metric[k]['value_per_epoch'] == [])
            cd |= (_metric[k]['value_per_epoch'] == [np.inf])
            cd |= (_metric[k]['value_per_epoch'] == [-np.inf])

            if cd:
                l.append(k)

        for k in l:
            _metric.pop(k, None)

        return _metric

    def plot_gist_tracker(self, gist_tracker: PerfGistTracker,
                          filename: str, title: str = '', xlabel: str = '',
                          best_iter_cl: int = None, best_iter_loc: int = None):

        ncols = 4
        ks = list(gist_tracker.value_per_epoch[0].keys())
        if len(ks) > ncols:
            nrows = math.ceil(len(ks) / float(ncols))
        else:
            nrows = 1
            ncols = len(ks)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False)
        t = 0
        for i in range(nrows):
            for j in range(ncols):
                if t >= len(ks):
                    axes[i, j].set_visible(False)
                    t += 1
                    continue

                val = [gist_tracker.value_per_epoch[elem][ks[t]] for elem in
                       gist_tracker.value_per_epoch]
                x = list(range(len(val)))
                axes[i, j].plot(x, val, color='tab:orange')
                subtitle = ks[t]
                axes[i, j].set_title(subtitle, fontsize=4)
                axes[i, j].xaxis.set_tick_params(labelsize=4)
                axes[i, j].yaxis.set_tick_params(labelsize=4)
                axes[i, j].set_xlabel('#{}'.format(xlabel), fontsize=4)
                axes[i, j].grid(True)
                axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

                for hand, best_iter in zip([constants.BEST_CL,
                                            constants.BEST_LOC],
                                           [best_iter_cl, best_iter_loc]):

                    if best_iter is not None:
                        _best_iter = best_iter if best_iter < len(x) else \
                            len(x) - 1
                        if i == j == 0:
                            axes[i, j].plot([x[_best_iter]], [val[_best_iter]],
                                            marker='o',
                                            markersize=5,
                                            color=constants.COLOUR_BEST_CP[
                                                hand],
                                            label=hand
                                            )
                            axes[i, j].legend(loc='best', prop={'size': 5})
                        else:
                            axes[i, j].plot([x[_best_iter]], [val[_best_iter]],
                                            marker='o',
                                            markersize=5,
                                            color=constants.COLOUR_BEST_CP[
                                                hand]
                                            )

                t += 1

        fig.suptitle(title, fontsize=4)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        fig.savefig(join(self.args.outd, '{}.png'.format(filename)),
                    bbox_inches='tight', dpi=300)

    def plot_perfs_meter(self):
        meters = self.serialize_perf_meter()
        xlabel = 'epochs'

        if self.args.localization_avail:
            best_loc_epoch = self.performance_meters[constants.PXVALIDSET][
                constants.LOCALIZATION_MTR].best_epoch
            if self.args.task != constants.SEG:
                best_cl_epoch = self.performance_meters[constants.CLVALIDSET][
                    constants.CLASSIFICATION_MTR].best_epoch
            else:
                best_cl_epoch = None
            splits = [constants.TRAINSET, constants.PXVALIDSET,
                      constants.CLVALIDSET]
        else:
            best_loc_epoch = None
            best_cl_epoch = self.performance_meters[constants.CLVALIDSET][
                constants.CLASSIFICATION_MTR].best_epoch
            splits = [constants.TRAINSET,  constants.CLVALIDSET]

        for split in splits:
            title = f'DS: {self.args.dataset}, Split: {split}. ' \
                    f'Best iter. CL:{best_cl_epoch} ' \
                    f'Best iter. LOC: {best_loc_epoch} ({xlabel})'

            filename = '{}-{}'.format(self.args.dataset, split)
            self.plot_meter(
                self.clean_metrics(meters[split]),
                filename=filename,
                title=title,
                xlabel=xlabel,
                best_iter_cl=best_cl_epoch,
                best_iter_loc=best_loc_epoch
            )

            if split == constants.PXVALIDSET:
                filename = '{}-{}-gist'.format(self.args.dataset, split)
                self.plot_gist_tracker(
                    self.perf_gist_tracker[split],
                    filename=filename,
                    title=title,
                    xlabel=xlabel,
                    best_iter_cl=best_cl_epoch,
                    best_iter_loc=best_loc_epoch
                )


class FeatureExtractor_for_source_code(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

class FeatureExtractor_for_source_code(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features
