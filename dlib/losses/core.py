import sys
from os.path import dirname, abspath
from typing import Optional, Union, List, Tuple
from itertools import cycle

import re
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.losses.elb import ELB
from dlib.losses.entropy import Entropy
from dlib.losses.energy_marginal import Energy_Marginal
from dlib import crf

from dlib.div_classifiers.parts.spg import get_loss as get_spg_loss
from dlib.div_classifiers.parts.acol import get_loss as get_acol_loss

from dlib.configure import constants
from dlib.losses.element import ElementaryLoss
from dlib.losses.ortho_px import OrthogonalProjectionLoss

__all__ = [
    'MasterLoss',
    'ClLoss',
    'SpgLoss',
    'AcolLoss',
    'CutMixLoss',
    'MaxMinLoss',
    'SegLoss',
    'ImgReconstruction',
    'SelfLearningFcams',
    'ConRanFieldFcams',
    'EntropyFcams',
    'MaxSizePositiveFcams',
    'SelfUnLearningFattention',
    'SelfLearningNegev',
    'ConRanFieldNegev',
    'JointConRanFieldNegev',
    'MaxSizePositiveNegev',
    'NegativeSamplesNegev',
    'EnergyCEloss',
    'ForgetEntropyLoss',
    'CalLoss',
    'CalPxLoss',
    'PartialEntropy',
    'PxOrtognalityloss',
    'Energy_Marginal',
    'CENotFlipLoss',
    'CEFlipLoss',
    'EntropyFcamsLoss',
]


class CalLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(CalLoss, self).__init__(**kwargs)

        self.cal_lambda: float = 0.0

        self.loss = Entropy().to(self._device)

        self.already_set = False

    # def entropy_loss(self, logits):
    #     probs = torch.softmax(logits, dim=1)
    #     log_probs = torch.log_softmax(logits, dim=1)
    #     entropy = -torch.sum(probs * log_probs, dim=1) 
    #     return entropy.mean()

    def set_it(self, cal_lambda: float):
        assert isinstance(cal_lambda, float), type(cal_lambda)
        assert 0 <= cal_lambda <= 1., cal_lambda

        self.cal_lambda = cal_lambda

        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(CalLoss, self).forward(epoch=epoch)

        assert self.already_set

        assert cl_logits is not None, "cl_logits must be provided for entropy loss"

        assert cl_logits.ndim == 2, cl_logits.ndim

        probs = torch.softmax(cl_logits, dim=1)

        loss = - self.loss(probs).mean()

        return loss * self.cal_lambda


        #loss = self.entropy_loss(cl_logits)

        #return -loss * self.cal_lambda


class PartialEntropy(ElementaryLoss):
    def __init__(self, **kwargs):
        super(PartialEntropy, self).__init__(**kwargs)

        self.cal_lambda: float = 0.0

        self.loss = Entropy().to(self._device)

        self.already_set = False

    # def entropy_loss(self, logits):
    #     probs = torch.softmax(logits, dim=1)
    #     log_probs = torch.log_softmax(logits, dim=1)
    #     entropy = -torch.sum(probs * log_probs, dim=1) 
    #     return entropy.mean()

    def set_it(self, esfda_entropy_partial_lambda: float):
        assert isinstance(esfda_entropy_partial_lambda, float), type(esfda_entropy_partial_lambda)
        assert 0 <= esfda_entropy_partial_lambda <= 1., esfda_entropy_partial_lambda

        self.esfda_entropy_partial_lambda = esfda_entropy_partial_lambda

        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(PartialEntropy, self).forward(epoch=epoch)

        assert self.already_set

        assert cl_logits is not None, "cl_logits must be provided for entropy loss"

        assert cl_logits.ndim == 2, cl_logits.ndim

        probs = torch.softmax(cl_logits, dim=1)


        # Get mask
        if key_arg is not None and "cal_mask" in key_arg:
            cal_mask = key_arg["cal_mask"]
            cal_mask = torch.tensor(cal_mask, dtype=torch.bool, device=probs.device)
            assert cal_mask.shape[0] == probs.shape[0], "Mismatch cal_mask / batch size"
            probs = probs[cal_mask]

        if probs.shape[0] == 0:
            return torch.tensor(0.0, device=cl_logits.device, requires_grad=False)

        loss = self.loss(probs).mean()
        return loss * self.esfda_entropy_partial_lambda
    

class CENotFlipLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(CENotFlipLoss, self).__init__(**kwargs)
        self.ce_label_smoothing: float = 0.0
        self.lambda_: float = 1.0
        self.loss = nn.CrossEntropyLoss(reduction="mean", label_smoothing=self.ce_label_smoothing).to(self._device)
        self.already_set = False
        

    def _weight_for_entropy(self, entropy: torch.Tensor):

        if self.entropy_mode is None:
            return entropy

        if self.entropy_mode == "linear":
            return 1.0 - entropy
        elif self.entropy_mode == "exp":
            return 1.0 / torch.exp(entropy)
        else:
            raise ValueError(f"Unknown entropy_mode: {self.entropy_mode}")

    def set_it(self, lambda_: float, esfda_flip_labels_weight = False, ce_label_smoothing: float = 0.0, esfda_weight_entropy: str | None = None,):
        assert isinstance(ce_label_smoothing, float)
        assert 0 <= ce_label_smoothing <= 1.
        self.ce_label_smoothing = ce_label_smoothing

        assert isinstance(lambda_, float)
        assert 0 <= lambda_ <= 1.
        self.lambda_ = lambda_

        # self.loss = nn.CrossEntropyLoss(reduction="mean", label_smoothing=self.ce_label_smoothing).to(self._device)
        # self.already_set = True

        reduction = "none" if esfda_flip_labels_weight else "mean"

        self.loss = nn.CrossEntropyLoss(reduction=reduction,label_smoothing=self.ce_label_smoothing).to(self._device)

        self.esfda_flip_labels_weight = esfda_flip_labels_weight

        self.entropy_mode = esfda_weight_entropy

        self.already_set = True



    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(CENotFlipLoss, self).forward(epoch=epoch)
        assert self.already_set

        if not self.is_on():
            return self._zero

        assert cl_logits is not None 
        assert cl_logits.shape[0] == glabel.shape[0], "Mismatch between logits and labels"

        y_pred_batch = key_arg["y_pred_batch"]
        assert y_pred_batch.shape[0] == cl_logits.shape[0], "Mismatch between logits and pseudo-labels"

        # if key_arg is not None and "cal_mask_retain" in key_arg:
        #     cal_mask = torch.tensor(key_arg["cal_mask_retain"], dtype=torch.bool, device=cl_logits.device)
        #     assert cal_mask.shape[0] == cl_logits.shape[0]

        #     retain_mask = ~cal_mask
        #     keep_mask = retain_mask
        #     cl_logits = cl_logits[retain_mask]
        #     y_pred_batch_not_flip = y_pred_batch[retain_mask]
        if key_arg is not None and "cal_mask_retain" in key_arg:
            cal_mask = torch.tensor(key_arg["cal_mask_retain"], dtype=torch.bool, device=cl_logits.device)
            assert cal_mask.shape[0] == cl_logits.shape[0], "Mask size mismatch"

            keep_mask = cal_mask
            cl_logits = cl_logits[keep_mask]
            y_pred_batch_not_flip = y_pred_batch[keep_mask]
            
        if self.esfda_flip_labels_weight:
            per_sample_loss = self.loss(cl_logits, y_pred_batch_not_flip)  # (batch_size,)


            mask_entropy = key_arg["mask_entropy"].to(keep_mask.device)  


            filtered_entropy = mask_entropy[keep_mask]

            # Call entropy weighting function
            weights = self._weight_for_entropy(filtered_entropy)

            if weights is not None:
                weights = weights.to(cl_logits.device)
                loss = torch.mean(weights * per_sample_loss) * self.lambda_
            else:
                loss = per_sample_loss.mean() * self.lambda_
        else:
            loss = self.loss(cl_logits, y_pred_batch_not_flip) * self.lambda_



        if cl_logits.shape[0] == 0:
            return torch.tensor(0.0, device=cl_logits.device, requires_grad=False)

        return loss
        #return self.loss(input=cl_logits, target=y_pred_batch_not_flip) * self.lambda_


class CEFlipLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(CEFlipLoss, self).__init__(**kwargs)
        self.ce_label_smoothing: float = 0.0
        self.lambda_: float = 1.0
        self.loss = nn.CrossEntropyLoss(reduction="mean", label_smoothing=self.ce_label_smoothing).to(self._device)
        self.already_set = False

    def set_it(self, lambda_: float, ce_label_smoothing: float = 0.0):
        assert isinstance(ce_label_smoothing, float)
        assert 0 <= ce_label_smoothing <= 1.
        self.ce_notflip_label_smoothing = ce_label_smoothing

        assert isinstance(lambda_, float)
        assert 0 <= lambda_ <= 1.
        self.lambda_ = lambda_

        self.loss = nn.CrossEntropyLoss(reduction="mean", label_smoothing=self.ce_label_smoothing).to(self._device)
        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(CEFlipLoss, self).forward(epoch=epoch)
        assert self.already_set
        if not self.is_on():
            return self._zero


        assert cl_logits is not None 
        assert cl_logits.shape[0] == glabel.shape[0], "Mismatch between logits and labels"

        y_pred_batch = key_arg["y_pred_batch"]
        assert y_pred_batch.shape[0] == cl_logits.shape[0], "Mismatch between logits and pseudo-labels"

        # if key_arg is not None and "cal_mask_forget" in key_arg:
        #     cal_mask = torch.tensor(key_arg["cal_mask_forget"], dtype=torch.bool, device=cl_logits.device)
        #     assert cal_mask.shape[0] == cl_logits.shape[0], "Mask size mismatch"

        #     flip_mask = ~cal_mask
        #     cl_logits = cl_logits[flip_mask]
        #     y_pred_batch_to_flip = y_pred_batch[flip_mask]

        if key_arg is not None and "cal_mask_forget" in key_arg:
            # cal_mask = torch.tensor(key_arg["cal_mask"], dtype=torch.bool, device=cl_logits.device)
            # assert cal_mask.shape[0] == cl_logits.shape[0], "Mask size mismatch"

            # flip_mask = ~cal_mask
            flip_mask = torch.tensor(key_arg["cal_mask_forget"], dtype=torch.bool, device=cl_logits.device)

            cl_logits = cl_logits[flip_mask]
            y_pred_batch_to_flip = y_pred_batch[flip_mask]
            
            

        if cl_logits.shape[0] == 0:
            return torch.tensor(0.0, device=cl_logits.device, requires_grad=False)

        #flipped_label = 1 - y_pred_batch_to_flip
        flipped_label = y_pred_batch_to_flip
        return self.loss(input=cl_logits, target=flipped_label) * self.lambda_
    





class CalPxLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(CalPxLoss, self).__init__(**kwargs)

        self.cal_px_lambda: float = 0.0

        self.already_set = False

    def pixel_entropy_loss(self, fcams):
        # fcams: (B, C, H, W)
        probs = torch.softmax(fcams, dim=1)          
        log_probs = torch.log_softmax(fcams, dim=1)

        entropy_map = -torch.sum(probs * log_probs, dim=1)
        return entropy_map.mean()

    def set_it(self, cal_px_lambda: float):
        assert isinstance(cal_px_lambda, float), type(cal_px_lambda)
        assert 0 <= cal_px_lambda <= 1., cal_px_lambda

        self.cal_px_lambda = cal_px_lambda

        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(CalPxLoss, self).forward(epoch=epoch)

        assert self.already_set

        assert fcams is not None
        loss = self.pixel_entropy_loss(fcams)

        return -loss * self.cal_px_lambda
 

class ClLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(ClLoss, self).__init__(**kwargs)

        self.ce_label_smoothing: float = 0.0

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=self.ce_label_smoothing
        ).to(self._device)

        self.already_set = False

    def set_it(self, ce_label_smoothing: float):
        assert isinstance(ce_label_smoothing, float), type(ce_label_smoothing)
        assert 0 <= ce_label_smoothing <= 1., ce_label_smoothing

        self.ce_label_smoothing = ce_label_smoothing

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=self.ce_label_smoothing
        ).to(self._device)

        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(ClLoss, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        return self.loss(input=cl_logits, target=glabel) * self.lambda_


class SpgLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(SpgLoss, self).__init__(**kwargs)

        self.spg_threshold_1h = None
        self.spg_threshold_1l = None
        self.spg_threshold_2h = None
        self.spg_threshold_2l = None
        self.spg_threshold_3h = None
        self.spg_threshold_3l = None

        self.hyper_p_set = False

        self.ce_label_smoothing: float = 0.0
        self.already_set = False


    @property
    def spg_thresholds(self):
        assert self.hyper_p_set

        h1 = self.spg_threshold_1h
        l1 = self.spg_threshold_1l

        h2 = self.spg_threshold_2h
        l2 = self.spg_threshold_2l

        h3 = self.spg_threshold_3h
        l3 = self.spg_threshold_3l

        return (h1, l1), (h2, l2), (h3, l3)

    def set_it(self, ce_label_smoothing: float):
        assert isinstance(ce_label_smoothing, float), type(ce_label_smoothing)
        assert 0 <= ce_label_smoothing <= 1., ce_label_smoothing

        self.ce_label_smoothing = ce_label_smoothing

        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(SpgLoss, self).forward(epoch=epoch)

        assert self.hyper_p_set
        assert self.already_set

        if not self.is_on():
            return self._zero

        return get_spg_loss(output_dict=model.logits_dict,
                            target=glabel,
                            spg_thresholds=self.spg_thresholds,
                            ce_label_smoothing=self.ce_label_smoothing
                            )


class AcolLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(AcolLoss, self).__init__(**kwargs)

        self.ce_label_smoothing: float = 0.0
        self.already_set = False

    def set_it(self, ce_label_smoothing: float):
        assert isinstance(ce_label_smoothing, float), type(ce_label_smoothing)
        assert 0 <= ce_label_smoothing <= 1., ce_label_smoothing

        self.ce_label_smoothing = ce_label_smoothing

        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(AcolLoss, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        return get_acol_loss(output_dict=model.logits_dict,
                             gt_labels=glabel,
                             ce_label_smoothing=self.ce_label_smoothing
                             )


class CutMixLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(CutMixLoss, self).__init__(**kwargs)

        self.ce_label_smoothing: float = 0.0

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=self.ce_label_smoothing
        ).to(self._device)

        self.already_set = False

    def set_it(self, ce_label_smoothing: float):
        assert isinstance(ce_label_smoothing, float), type(ce_label_smoothing)
        assert 0 <= ce_label_smoothing <= 1., ce_label_smoothing

        self.ce_label_smoothing = ce_label_smoothing

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=self.ce_label_smoothing
        ).to(self._device)

        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(CutMixLoss, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        if cutmix_holder is None:
            return self.loss(input=cl_logits, target=glabel) * self.lambda_

        assert isinstance(cutmix_holder, list)
        assert len(cutmix_holder) == 3
        target_a, target_b, lam = cutmix_holder
        loss = (self.loss(cl_logits, target_a) * lam + self.loss(
            cl_logits, target_b) * (1. - lam))

        return loss


class MaxMinLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(MaxMinLoss, self).__init__(**kwargs)

        self.dataset_name: str = ''
        assert isinstance(self.elb, ELB)
        self.lambda_size = 0.
        self.lambda_neg = 0.

        self._lambda_size_set = False
        self._lambda_neg_set = False

        self.ce_label_smoothing: float = 0.0

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=self.ce_label_smoothing
        ).to(self._device)

        self.BCE = nn.BCEWithLogitsLoss(reduction="mean").to(self._device)

        self.softmax = nn.Softmax(dim=1)

    def set_ce_label_smoothing(self, ce_label_smoothing: float = 0.0):
        assert isinstance(ce_label_smoothing, float), type(ce_label_smoothing)
        assert 0 <= ce_label_smoothing <= 1., ce_label_smoothing

        self.ce_label_smoothing = ce_label_smoothing

    def set_lambda_neg(self, lambda_neg: float):
        assert isinstance(lambda_neg, float)
        assert lambda_neg >= 0.
        self.lambda_neg = lambda_neg

        self._lambda_neg_set = True

    def set_lambda_size(self, lambda_size: float):
        assert isinstance(lambda_size, float)
        assert lambda_size >= 0.
        self.lambda_size = lambda_size

        self._lambda_size_set = True

    def set_dataset_name(self, dataset_name: str):
        self._assert_dataset_name(dataset_name=dataset_name)
        self.dataset_name = dataset_name

    def _is_ready(self):
        assert self._lambda_size_set
        assert self._lambda_neg_set
        self._assert_dataset_name(dataset_name=self.dataset_name)

    def _assert_dataset_name(self, dataset_name: str):
        assert isinstance(dataset_name, str)
        assert dataset_name in [constants.GLAS, constants.CAMELYON512, constants.CAMELYON17_512]

    def kl_uniform_loss(self, logits):
        assert logits.ndim == 2
        logsoftmax = torch.log2(self.softmax(logits))
        return (-logsoftmax).mean(dim=1).mean()

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(MaxMinLoss, self).forward(epoch=epoch)

        self._is_ready()

        if not self.is_on():
            return self._zero

        logits = model.logits_dict['logits']
        logits_pos = model.logits_dict['logits_pos']
        logits_neg = model.logits_dict['logits_neg']

        cam = model.logits_dict['cam']
        cam_logits = model.logits_dict['cam_logits']

        assert cam.ndim == 4
        assert cam.shape[1] == 1
        assert cam.shape == cam_logits.shape
        bs, _, _, _ = cam.shape

        cl_losss = self.loss(input=logits, target=glabel)
        total_l = cl_losss
        size = cam.contiguous().view(bs, -1).sum(dim=-1).view(-1, )

        if self.dataset_name == constants.GLAS:
            size_loss = self.elb(-size) + self.elb(-1. + size)
            total_l = total_l + self.lambda_size * size_loss * 0.0

            total_l = total_l + self.loss(input=logits_pos, target=glabel) * 0.
            total_l = total_l + self.lambda_neg * self.kl_uniform_loss(
                logits=logits_neg) * 0.0

        if self.dataset_name in [constants.CAMELYON512, constants.CAMELYON17_512]:
            # pos
            ind_metas = (glabel == 1).nonzero().view(-1)
            if ind_metas.numel() > 0:
                tmps = size[ind_metas]
                size_loss = self.elb(-tmps) + self.elb(-1. + tmps)
                total_l = total_l + self.lambda_size * size_loss

            # neg
            ind_normal = (glabel == 0).nonzero().view(-1)
            if ind_normal.numel() > 0:
                trg_cams = torch.zeros(
                    (ind_normal.numel(), 1, cam.shape[2], cam.shape[3]),
                    dtype=torch.float, device=cam.device)

                total_l = total_l + self.BCE(input=cam_logits[ind_normal],
                                             target=trg_cams)

        return total_l


class SegLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(SegLoss, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.seg_ignore_idx).to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(SegLoss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag

        return self.loss(input=seg_logits, target=masks) * self.lambda_


class ImgReconstruction(ElementaryLoss):
    def __init__(self, **kwargs):
        super(ImgReconstruction, self).__init__(**kwargs)

        self.loss = nn.MSELoss(reduction="none").to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(ImgReconstruction, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        n = x_in.shape[0]
        loss = self.elb(self.loss(x_in, im_recon).view(n, -1).mean(
            dim=1).view(-1, ))
        return self.lambda_ * loss.mean()


class SelfLearningFcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(SelfLearningFcams, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.seg_ignore_idx).to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(SelfLearningFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag

        return self.loss(input=fcams, target=seeds) * self.lambda_


class SelfUnLearningFattention(ElementaryLoss):
    def __init__(self, mode = "mse", lambda_=1.0, use_std=False, **kwargs):
        super(SelfUnLearningFattention, self).__init__(**kwargs)

        self.lambda_ = lambda_
        self.mode = mode
        self.use_std = use_std  # A_STD_COSINE
        eps = 1e-8
        #self.loss_mse = nn.MSELoss(reduction="mean").to(self._device)


        if mode == constants.A_MSE:
            self.loss = nn.MSELoss(reduction="mean").to(self._device)
        elif mode == constants.A_COSINE:
            self.loss = nn.CosineSimilarity(dim=1, eps=eps).to(self._device)
        else:
            raise ValueError(f"[SelfUnLearningFattention] Unknown mode '{self.mode}'. Use 'mse' or 'cosine'.")


    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,    # attention from source model
                fcams=None,         # attention from unlearned model
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None):

        super(SelfUnLearningFattention, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert cams_inter is not None and fcams is not None, \
            "[SelfLearningFcams] Both cams_inter (from source) and fcams (from unlearned) must be provided."
        

        if cams_inter.shape[-2:] != fcams.shape[-2:]:
            fcams = F.interpolate(fcams, size=cams_inter.shape[-2:], mode="bilinear", align_corners=False)

        # --- 1) MSE standard ---
        if self.mode == "mse":
            loss_val = self.loss(fcams, cams_inter)


        # --- 2) Cosine Similarity ---
        elif self.mode == "cosine":
            b = fcams.shape[0]

            # Flatten attention [B, 1, H, W] -> [B, H*W]
            fcams_flat = fcams.contiguous().view(b, -1)
            cams_inter_flat = cams_inter.contiguous().view(b, -1)

            # Standardise mao (A_STD_COSINE)
            if self.use_std:
                mean = fcams_flat.mean(dim=-1, keepdim=True)
                std = torch.std(fcams_flat, dim=-1, keepdim=True, correction=1) + 1e-4
                fcams_flat = (fcams_flat - mean) / std

            # Cosine Similarity
            cos_sim = self.loss(fcams_flat, cams_inter_flat)  # [B]


            loss_val = (1. - cos_sim).mean()

        else:
            raise ValueError(f"[SelfUnLearningFattention] Invalid mode '{self.mode}'.")

        return loss_val * self.lambda_

        #return self.loss_mse(fcams, cams_inter) * self.lambda_



class ConRanFieldFcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(ConRanFieldFcams, self).__init__(**kwargs)

        self.loss = crf.DenseCRFLoss(
            weight=self.lambda_, sigma_rgb=self.sigma_rgb,
            sigma_xy=self.sigma_xy, scale_factor=self.scale_factor
        ).to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(ConRanFieldFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero
        
        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        return self.loss(images=raw_img, segmentations=fcams_n)


class EntropyFcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(EntropyFcams, self).__init__(**kwargs)

        self.loss = Entropy().to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(EntropyFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert fcams.ndim == 4
        bsz, c, h, w = fcams.shape

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        probs = fcams_n.permute(0, 2, 3, 1).contiguous().view(bsz * h * w, c)

        return self.lambda_ * self.loss(probs).mean()


class MaxSizePositiveFcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(MaxSizePositiveFcams, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(MaxSizePositiveFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag  # todo

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        n = fcams.shape[0]
        loss = None
        for c in [0, 1]:
            bl = fcams_n[:, c].view(n, -1).sum(dim=-1).view(-1, )
            if loss is None:
                loss = self.elb(-bl)
            else:
                loss = loss + self.elb(-bl)

        return self.lambda_ * loss * (1./2.)


class SelfLearningNegev(SelfLearningFcams):
    def __init__(self, **kwargs):
        super(SelfLearningNegev, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.seg_ignore_idx).to(self._device)

        self.apply_negative_samples: bool = False
        self.negative_c: int = 0

        self._is_already_set = False

    def set_it(self, apply_negative_samples: bool, negative_c: int):
        assert isinstance(apply_negative_samples, bool)
        assert isinstance(negative_c, int)
        assert negative_c >= 0

        self.negative_c = negative_c
        self.apply_negative_samples = apply_negative_samples

        self._is_already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(SelfLearningFcams, self).forward(epoch=epoch)

        assert self._is_already_set

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag

        if self.apply_negative_samples:
            return self.loss(input=fcams, target=seeds) * self.lambda_

        ind_non_neg = (glabel != self.negative_c).nonzero().view(-1)

        nbr = ind_non_neg.numel()

        if nbr == 0:
            return self._zero

        fcams_n_neg = fcams[ind_non_neg]
        seeds_n_neg = seeds[ind_non_neg]
        return self.loss(input=fcams_n_neg, target=seeds_n_neg) * self.lambda_


class ConRanFieldNegev(ConRanFieldFcams):
    pass


class MaxSizePositiveNegev(MaxSizePositiveFcams):
    def __init__(self, **kwargs):
        super(MaxSizePositiveNegev, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)

        self.apply_negative_samples: bool = False
        self.negative_c: int = 0

        self._is_already_set = False

    def set_it(self, apply_negative_samples: bool, negative_c: int):
        assert isinstance(apply_negative_samples, bool)
        assert isinstance(negative_c, int)
        assert negative_c >= 0

        self.negative_c = negative_c
        self.apply_negative_samples = apply_negative_samples

        self._is_already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(MaxSizePositiveFcams, self).forward(epoch=epoch)

        assert self._is_already_set

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag  # todo

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        fcams_n_input = fcams_n

        if not self.apply_negative_samples:
            ind_non_neg = (glabel != self.negative_c).nonzero().view(-1)

            nbr = ind_non_neg.numel()

            if nbr == 0:
                return self._zero

            fcams_n_input = fcams_n[ind_non_neg]

        n = fcams_n_input.shape[0]
        loss = None

        for c in [0, 1]:
            bl = fcams_n_input[:, c].view(n, -1).sum(dim=-1).view(-1, )
            if loss is None:
                loss = self.elb(-bl)
            else:
                loss = loss + self.elb(-bl)

        return self.lambda_ * loss * (1./2.)


class JointConRanFieldNegev(ElementaryLoss):
    def __init__(self, **kwargs):
        super(JointConRanFieldNegev, self).__init__(**kwargs)

        self.loss = crf.ColorDenseCRFLoss(
            weight=self.lambda_, sigma_rgb=self.sigma_rgb,
            scale_factor=self.scale_factor).to(self._device)

        self.pair_mode: str = ''
        self.n: int = 0

        self.dataset_name: str = ''

        self._already_set = False

    def _assert_dataset_name(self, dataset_name: str):
        assert dataset_name in [constants.GLAS, constants.CAMELYON512, constants.CAMELYON17_512]

    def _assert_pair_mode(self, pair_mode: str):
        assert pair_mode in [constants.PAIR_SAME_C, constants.PAIR_MIXED_C,
                             constants.PAIR_DIFF_C]

    def _assert_n(self, n: int):
        assert isinstance(n, int)
        assert n > 0

    def set_it(self, pair_mode: str, n: int, dataset_name: str):
        self._assert_pair_mode(pair_mode)
        self._assert_n(n)
        self._assert_dataset_name(dataset_name)

        self.pair_mode = pair_mode
        self.n = n
        self.dataset_name = dataset_name
        self._already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(JointConRanFieldNegev, self).forward(epoch=epoch)
        assert self._already_set

        if not self.is_on():
            return self._zero

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        raw_img_grey = rgb_to_grayscale(img=raw_img, num_output_channels=1)

        p_imgs, p_cams = self.pair_samples(imgs=raw_img_grey, glabel=glabel,
                                           prob_cams=fcams_n)

        return self.loss(images=p_imgs, segmentations=p_cams)

    def pair_samples(self,
                     imgs: torch.Tensor,
                     glabel: torch.Tensor,
                     prob_cams: torch.Tensor) -> Tuple[torch.Tensor,
                                                       torch.Tensor]:

        assert imgs.ndim == 4
        assert prob_cams.ndim == 4

        b, c, h, w = imgs.shape
        out_img = None
        out_prob_cams = None
        all_idx = torch.arange(b)
        for i in range(b):
            _c = glabel[i]

            if self.pair_mode == constants.PAIR_SAME_C:
                idx = torch.nonzero(glabel == _c, as_tuple=False).squeeze()
            elif self.pair_mode == constants.PAIR_DIFF_C:
                idx = torch.nonzero(glabel != _c, as_tuple=False).squeeze()
            elif self.pair_mode == constants.PAIR_MIXED_C:
                idx = all_idx
            else:
                raise NotImplementedError

            idx = idx[idx != i]

            nbr = idx.numel()
            if (nbr == 0) or (nbr == 1):
                continue

            tmp_img = imgs[i].unsqueeze(0)
            tmp_prob_cams = prob_cams[i].unsqueeze(0)

            didx = torch.randperm(nbr)
            n_max = min(nbr, self.n)
            pool = cycle(list(range(n_max)))

            for _ in range(self.n):
                z = next(pool)
                # cat width.
                tmp_img = torch.cat(
                    (tmp_img, imgs[idx[didx[z]]].unsqueeze(0)), dim=3)
                tmp_prob_cams = torch.cat(
                    (tmp_prob_cams, prob_cams[idx[didx[z]]].unsqueeze(0)),
                    dim=3)

            if out_img is None:
                out_img = tmp_img
                out_prob_cams = tmp_prob_cams
            else:
                out_img = torch.cat((out_img, tmp_img), dim=0)
                out_prob_cams = torch.cat((out_prob_cams, tmp_prob_cams), dim=0)

        return out_img, out_prob_cams


class NegativeSamplesNegev(ElementaryLoss):
    def __init__(self, **kwargs):
        super(NegativeSamplesNegev, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(reduction="mean").to(self._device)

        self.negative_c: int = 0
        self._is_already_set = False

    def set_it(self, negative_c: int):
        assert isinstance(negative_c, int)
        assert negative_c >= 0

        self.negative_c = negative_c
        self._is_already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(NegativeSamplesNegev, self).forward(epoch=epoch)
        assert self._is_already_set

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag

        ind_neg = (glabel == self.negative_c).nonzero().view(-1)
        nbr = ind_neg.numel()

        if nbr == 0:
            return self._zero

        b, c, h, w = fcams.shape

        trg = torch.zeros(
            (ind_neg.numel(), h, w), dtype=torch.long, device=fcams.device)
        logits = fcams[ind_neg]

        return self.loss(input=logits, target=trg) * self.lambda_


class MasterLoss(nn.Module):
    def __init__(self, cuda_id: int, name=None):
        super().__init__()
        self._name = name

        self.losses = []
        self.l_holder = []
        self.n_holder = [self.__name__]
        self._device = torch.device(cuda_id)

    def add(self, loss_: ElementaryLoss):
        self.losses.append(loss_)
        self.n_holder.append(loss_.__name__)

    def update_t(self):
        for loss in self.losses:
            loss.update_t()

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name

    def forward(self, **kwargs):
        assert self.losses != []

        self.l_holder = []
        for loss in self.losses:

            self.l_holder.append(loss(**kwargs).to(self._device))

        loss = sum(self.l_holder)
        self.l_holder = [loss] + self.l_holder
        return loss

    def to_device(self):
        for loss in self.losses:
            loss.to(self._device)

    def check_losses_status(self):
        print('-' * 60)
        print('Losses status:')

        for i, loss in enumerate(self.losses):
            if hasattr(loss, 'is_on'):
                print(self.n_holder[i+1], ': ... ',
                      loss.is_on(),
                      "({}, {})".format(loss.start_epoch, loss.end_epoch))
        print('-' * 60)

    def __str__(self):
        return "{}():".format(
            self.__class__.__name__, ", ".join(self.n_holder))


if __name__ == "__main__":
    from dlib.utils.reproducibility import set_seed
    set_seed(seed=0)
    b, c = 10, 4
    cudaid = 0
    torch.cuda.set_device(cudaid)

    loss = MasterLoss(cuda_id=cudaid)
    print(loss.__name__, loss, loss.l_holder, loss.n_holder)
    loss.add(SelfLearningFcams(cuda_id=cudaid))
    loss.add(SelfLearningNegev(cuda_id=cudaid))

    for l in loss.losses:
        print(l, isinstance(l, SelfLearningNegev))

    for e in loss.n_holder:
        print(e)

class SatLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(SatLoss, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(reduction="mean").to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                sat_aux_losses = None,
                sat_area_th = None,
                pseudo_glabel = None,
                cutmix_holder = None,
                ):
        super(SatLoss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        loss_cls = self.loss(input=cl_logits, target=glabel) * self.lambda_
        
        ba_loss, norm_loss = sat_aux_losses['ba_loss'].mean(0), sat_aux_losses['norm_loss'].mean(0)  
            
        loss = loss_cls +  torch.abs(ba_loss - sat_area_th) + norm_loss 
        
        # loss =  loss_cls +  torch.abs(sat_aux_losses['ba_loss'] - sat_area_th).mean(0) + sat_aux_losses['norm_loss']
        return loss

# class EnergyCEloss(ElementaryLoss):
#     def __init__(self, **kwargs):
#         super(EnergyCEloss, self).__init__(**kwargs)

#         self.ece_lambda = 0.0
#         self.negative_c: int = 0
#         self._is_already_set = False
#         self.dataset = kwargs['dataset']
#         self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=-255).to(self._device)

#     def set_it(self, ece_lambda):

#         self.ece_lambda = ece_lambda
#         self._is_already_set = True

#     def forward(self,
#                 epoch=0,
#                 model=None,
#                 cams_inter=None,
#                 fcams=None,
#                 cl_logits=None,
#                 glabel=None,
#                 raw_img=None,
#                 x_in=None,
#                 im_recon=None,
#                 seeds=None,
#                 sat_aux_losses = None,
#                 sat_area_th = None,
#                 pseudo_glabel = None,
#                 cutmix_holder = None,
#                 ):
#         super(EnergyCEloss, self).forward(epoch=epoch)

#         if not self.is_on():
#             return self._zero
        
#         if self.dataset in constants.CAMELYON512:
#             ind_neg = (glabel == self.negative_c).nonzero().view(-1)
#             nbr = ind_neg.numel()

#             if nbr == 0:
#                 return self._zero

#             _, h, w = seeds.shape

#             zero_matrix = torch.full((h, w), -255, dtype=seeds.dtype, device=seeds.device)

#             for idx in ind_neg:
#                 seeds[idx] = zero_matrix

#         loss = self.loss(model.cams,seeds)  * self.ece_lambda


#         return loss

class EnergyCEloss(SelfLearningFcams):
    def __init__(self, **kwargs):
        super(EnergyCEloss, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.seg_ignore_idx).to(self._device)

        self.ece_lambda = 0.0
        self.apply_negative_samples: bool = False
        self.negative_c: int = 0

        self._is_already_set = False

    def set_it(self,ece_lambda, apply_negative_samples: bool, negative_c: int):
        assert isinstance(apply_negative_samples, bool)

        if apply_negative_samples:
            assert negative_c is not None
            assert isinstance(negative_c, int)
            assert negative_c >= 0

        self.ece_lambda = ece_lambda
        self.negative_c = negative_c
        self.apply_negative_samples = apply_negative_samples

        self._is_already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(SelfLearningFcams, self).forward(epoch=epoch)

        # assert self._is_already_set

        # if not self.is_on():
        #     return self._zero

        # assert not self.multi_label_flag

        # if not self.apply_negative_samples:
        #     return self.loss(input=fcams, target=seeds) * self.ece_lambda

        # ind_non_neg = (glabel != self.negative_c).nonzero().view(-1)

        # nbr = ind_non_neg.numel()

        # if nbr == 0:
        #     return self._zero

        # fcams_n_neg = fcams[ind_non_neg]
        # seeds_n_neg = seeds[ind_non_neg]
        # return self.loss(input=fcams_n_neg, target=seeds_n_neg) * self.ece_lambda


        assert self._is_already_set
        if not self.is_on():
            return self._zero
        assert not self.multi_label_flag

        cal_mask_forget = None
        cal_mask_retain = None

        if key_arg is not None:
            if "cal_mask_forget" in key_arg:
                cal_mask_forget = key_arg["cal_mask_forget"]
            if "cal_mask_retain" in key_arg:
                cal_mask_retain = key_arg["cal_mask_retain"]


        # --- Combine both masks (logical OR)
        if cal_mask_forget is not None or cal_mask_retain is not None:
            # convert both to tensors (if needed)
            if cal_mask_forget is None:
                cal_mask_forget = torch.zeros_like(torch.tensor(cal_mask_retain, device=fcams.device, dtype=torch.bool))
            if cal_mask_retain is None:
                cal_mask_retain = torch.zeros_like(torch.tensor(cal_mask_forget, device=fcams.device, dtype=torch.bool))

            if not torch.is_tensor(cal_mask_forget):
                cal_mask_forget = torch.tensor(cal_mask_forget, device=fcams.device, dtype=torch.bool)
            if not torch.is_tensor(cal_mask_retain):
                cal_mask_retain = torch.tensor(cal_mask_retain, device=fcams.device, dtype=torch.bool)

            # Logical OR: use if either mask is True
            cal_mask_final = cal_mask_forget | cal_mask_retain

            ind_selected = cal_mask_final.nonzero(as_tuple=True)[0]
            if ind_selected.numel() == 0:
                return self._zero

            fcams = fcams[ind_selected]
            seeds = seeds[ind_selected]
            if glabel is not None:
                glabel = glabel[ind_selected]

        if not self.apply_negative_samples:
            return self.loss(input=fcams, target=seeds) * self.ece_lambda

        ind_non_neg = (glabel != self.negative_c).nonzero(as_tuple=True)[0]
        if ind_non_neg.numel() == 0:
            return self._zero

        fcams = fcams[ind_non_neg]
        seeds = seeds[ind_non_neg]
        return self.loss(input=fcams, target=seeds) * self.ece_lambda
    

class EntropyFcamsLoss(SelfLearningFcams):
    def __init__(self, **kwargs):
        super(EntropyFcamsLoss, self).__init__(**kwargs)

        self.entropy_lambda = 0.0
        self._is_already_set = False

    def set_it(self, entropy_lambda: float):
        self.entropy_lambda = entropy_lambda
        self._is_already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(SelfLearningFcams, self).forward(epoch=epoch)

        assert self._is_already_set
        if not self.is_on():
            return self._zero
        assert not self.multi_label_flag

        # -------------------------------------------------
        # Handle masks exactly like in EnergyCEloss
        # -------------------------------------------------
        cal_mask_forget = None
        cal_mask_retain = None

        key_arg = None

        if key_arg is not None:
            cal_mask_forget = key_arg.get("cal_mask_forget", None)
            cal_mask_retain = key_arg.get("cal_mask_retain", None)

        if cal_mask_forget is not None or cal_mask_retain is not None:
            if cal_mask_forget is None:
                cal_mask_forget = torch.zeros_like(
                    torch.tensor(cal_mask_retain, device=fcams.device, dtype=torch.bool)
                )
            if cal_mask_retain is None:
                cal_mask_retain = torch.zeros_like(
                    torch.tensor(cal_mask_forget, device=fcams.device, dtype=torch.bool)
                )

            if not torch.is_tensor(cal_mask_forget):
                cal_mask_forget = torch.tensor(cal_mask_forget, device=fcams.device, dtype=torch.bool)
            if not torch.is_tensor(cal_mask_retain):
                cal_mask_retain = torch.tensor(cal_mask_retain, device=fcams.device, dtype=torch.bool)

            cal_mask_final = cal_mask_forget | cal_mask_retain
            ind_selected = cal_mask_final.nonzero(as_tuple=True)[0]

            if ind_selected.numel() == 0:
                return self._zero

            fcams = fcams[ind_selected]

        # -------------------------------------------------
        # Entropy computation
        # -------------------------------------------------
        # fcams: [N, C, H, W] or [N, C]
        p = torch.softmax(fcams, dim=1)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=1)

        # mean over pixels
        loss = entropy.mean()

        return loss * self.entropy_lambda

class ForgetEntropyLoss(SelfLearningFcams):
    def __init__(self, **kwargs):
        super(ForgetEntropyLoss, self).__init__(**kwargs)
        self.forget_lambda = 0.0
        self._is_already_set = False

    def set_it(self, forget_lambda: float):
        self.forget_lambda = forget_lambda
        self._is_already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None):

        super(SelfLearningFcams, self).forward(epoch=epoch)

        assert self._is_already_set
        if not self.is_on():
            return self._zero

        if seeds is None:
            return self._zero
        

        cal_mask_forget = key_arg.get("cal_mask_forget", None)

        bg_seed_mask = (seeds == 0)  # [B, H, W]

        if cal_mask_forget is None:
            return self._zero

        if not torch.is_tensor(cal_mask_forget):
            cal_mask_forget = torch.tensor(
                cal_mask_forget, device=fcams.device, dtype=torch.bool)

        ind_forget = cal_mask_forget.nonzero(as_tuple=True)[0]
        bg_seed_mask = bg_seed_mask[ind_forget]  # [Bf, H, W]

        if ind_forget.numel() == 0:
            return self._zero

        logits = fcams[ind_forget]

        p_fg = logits[:, 1]            # [Bf, H, W]

        valid_mask = (~bg_seed_mask)  

       
        p_fg_valid = p_fg[valid_mask]   # [N_valid]
        p_fg_valid = p_fg_valid.float()

        if p_fg_valid.numel() == 0:
            return self._zero

        eps = 1e-4
        p_fg_valid = torch.clamp(p_fg_valid, eps, 1 - eps)

        entropy = -(p_fg_valid * torch.log(p_fg_valid)
                    + (1 - p_fg_valid) * torch.log(1 - p_fg_valid))

        return self.forget_lambda * entropy.mean()


class EnergyCEAdaptloss(SelfLearningFcams):
    def __init__(self, **kwargs):
        super(EnergyCEAdaptloss, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.seg_ignore_idx).to(self._device)

        self.ece_adapt_lambda = 0.0
        self.apply_negative_samples: bool = False
        self.negative_c: int = 0

        self._is_already_set = False

    def set_it(self,ece_adapt_lambda, apply_negative_samples: bool, negative_c: int):
        assert isinstance(apply_negative_samples, bool)
        assert isinstance(negative_c, int)
        assert negative_c >= 0

        self.ece_lambda = ece_adapt_lambda
        self.negative_c = negative_c
        self.apply_negative_samples = apply_negative_samples

        self._is_already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(SelfLearningFcams, self).forward(epoch=epoch)

        assert self._is_already_set

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag

        if not self.apply_negative_samples:
            return self.loss(input=fcams, target=seeds) * self.ece_lambda

        #ind_non_neg = (pseudo_glabel != -255) & (pseudo_glabel != self.negative_c)
        #ind_non_neg = (pseudo_glabel != self.negative_c).nonzero().view(-1)

        ind_non_neg = ((pseudo_glabel != -255) & (pseudo_glabel != 0)).nonzero().view(-1)


        nbr = ind_non_neg.numel()

        if nbr == 0:
            return self._zero

        fcams_n_neg = fcams[ind_non_neg]
        seeds_n_neg = seeds[ind_non_neg]
        return self.loss(input=fcams_n_neg, target=seeds_n_neg) * self.ece_lambda




    
class PxOrtognalityloss(SelfLearningFcams):
    def __init__(self, **kwargs):
        super(PxOrtognalityloss, self).__init__(**kwargs)

        self.loss = OrthogonalProjectionLoss()

        self.pxortho_lambda = 0.0
        self.apply_negative_samples: bool = False
        self.negative_c: int = 0

        self._is_already_set = False

    def set_it(self,pxortho_lambda):

        self.pxortho_lambda = pxortho_lambda
        self._is_already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(SelfLearningFcams, self).forward(epoch=epoch)

        assert self._is_already_set

        if not self.is_on():
            return self._zero

        encoder_last_features_upsampled = F.interpolate(
            model.encoder_last_features, size=seeds.shape[1:], mode='bilinear', align_corners=False)

        return self.loss(features=encoder_last_features_upsampled, labels=seeds) * self.pxortho_lambda
    

class ConRanFieldPxcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(ConRanFieldPxcams, self).__init__(**kwargs)

        self.loss = crf.DenseCRFLoss(
            weight=self.lambda_, sigma_rgb=self.sigma_rgb,
            sigma_xy=self.sigma_xy, scale_factor=self.scale_factor
        ).to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(ConRanFieldPxcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero
        
        fcams = model.cams
        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        return self.loss(images=raw_img, segmentations=fcams_n)
    

    
class EnergyMGloss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(EnergyMGloss, self).__init__(**kwargs)

        self.eng_lambda = 0.0
        self._is_already_set = False
        self.loss = Energy_Marginal()

    def set_it(self, eng_lambda,sgld_lr =1.0, sgld_std=0.01):

        self.eng_lambda = eng_lambda
        self.sgld_lr = sgld_lr
        self.sgld_std = sgld_std
        self._is_already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                sat_aux_losses = None,
                sat_area_th = None,
                pseudo_glabel = None,
                cutmix_holder = None,
                ):
        super(EnergyMGloss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero
        
        #Position of foreground and background from pre-trained CAM
        indices_1 = (seeds == 1).nonzero(as_tuple=False)
        indices_0 = (seeds == 1).nonzero(as_tuple=False)

        retrieved_values_1 = []

        for idx in indices_1:
            batch_idx, x, y = idx
            values = model.cams[batch_idx, :, x, y]
            retrieved_values_1.append(values)
        
        retrieved_values_1 = torch.stack(retrieved_values_1)

        retrieved_values_0 = []

        for idx in indices_0:
            batch_idx, x, y = idx
            values = model.cams[batch_idx, :, x, y]
            retrieved_values_0.append(values)
        
        retrieved_values_0 = torch.stack(retrieved_values_0)


        combined_values = torch.cat((retrieved_values_1, retrieved_values_0), dim=0).to(self._device)

        first_dimension = combined_values.shape[0]

        model.eval()
        n_steps = 20 

        y = None
        uniform_tensor = torch.FloatTensor(32, 2048, 1, 1).uniform_(-1, 1).half()

        x_k = torch.autograd.Variable(uniform_tensor, requires_grad=True).to(self._device)

        for k in range(n_steps):
            f_prime = torch.autograd.grad(model.pixel_wise_classification_head(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += self.sgld_lr * f_prime + self.sgld_std * torch.randn_like(x_k)

        #loss = self.loss(combined_values, combined_labels) * self.ece_lambda

        model.train()
        final_samples = x_k.detach()

        batch_size = 32
        channels = combined_values.size(1)
        #height = combined_values.size(2)
        #width = combined_values.size(3)
        
        random_indices_h = torch.randint(0, 224, (batch_size,))
        random_indices_w = torch.randint(0, 224, (batch_size,))

        output_tensor = model.cams[torch.arange(batch_size), :, random_indices_h, random_indices_w]

        output_tensor = output_tensor.squeeze(-1).squeeze(-1)
        energy_target = output_tensor.sum(dim=1)

        energy_sample = model.pixel_wise_classification_head(final_samples).sum(dim=1).squeeze()

        loss = self.loss(energy_target, energy_sample) * self.eng_lambda
        # loss =  loss_cls +  torch.abs(sat_aux_losses['ba_loss'] - sat_area_th).mean(0) + sat_aux_losses['norm_loss']
        return loss