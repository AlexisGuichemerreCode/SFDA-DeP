"""
Xu et al., "Revisiting Source-Free Domain Adaptation: a New Perspective via
Uncertainty Control" (UCon-SFDA), ICLR 2025.
https://proceedings.iclr.cc/paper_files/paper/2025/file/e85454a113e8b41e017c81875ae68d47-Paper-Conference.pdf
Official code: https://github.com/xugezheng/UCon_SFDA

Port of `train_target_UCon_SFDA.py` + `utils_partial_label.py` to this code
base.

Per training step (official loop):
  - weak view x -> (f, p) with the adapted model; strong view x' -> p'.
  - partial-label bank update with top-k predictions (k fixed, or k* 'cal').
  - uncertain-sample selection bank: p_max / p_2nd < tau (rolling, cumulative).
  - feature / score bank update and K-NN retrieval (cosine).
  - CL-POS: neighbors consistency  sum_k KL(p || p_nk)  (AaD-style).
  - CL-NEG: alpha * mean_i sum_{j!=i} p_i . p_j, alpha decayed with
    (1 + 10 * it / max_it)^(-beta) * alpha0.
  - DC (dispersion control): CE(smooth(softmax(z_weak).detach()), p_strong).
  - PL (partial label): partial-label loss on the selected uncertain samples.

Loss terms are computed in `dlib/losses/sf_uda.py` (UCon* classes) from the
dict returned by `UCon.step()`.

Adaptation to this code base:
  - features = `model.lin_ft` (pooled features before the last linear layer).
  - weak view = the regular training view (`images`), strong view is provided
    by the loader in `views` (see `get_sfuda_strong_transform`).
  - banks are initialized with a pass over the target trainset with the eval
    transform (official: weak train transform, in eval mode).
  - image ids (str) are mapped to integer dataset indices.
"""
import sys
from os.path import dirname, abspath

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg


__all__ = ['UCon']


# ==============================================================================
# Official helpers (utils_partial_label.py)
# ==============================================================================
def partial_label_bank_update(partial_label_bank, tar_idx, outputs, k_values):
    if isinstance(k_values, int):
        _, batch_label = torch.topk(outputs.clone().detach(), k=k_values,
                                    dim=-1, largest=True)
        tar_idx_reshaped = tar_idx.unsqueeze(1).expand_as(batch_label)
        partial_label_bank[tar_idx_reshaped, batch_label] = 1
    else:
        num_classes = outputs.size(1)
        device = outputs.device
        sorted_indices = outputs.clone().detach().argsort(dim=-1,
                                                          descending=True)
        col_indices = torch.arange(num_classes, device=device).unsqueeze(0)
        mask = (col_indices < k_values.view(-1, 1)).to(device)
        tar_idx = tar_idx.to(device)
        selected_rows = tar_idx.unsqueeze(1).expand_as(mask)[mask]
        selected_cols = sorted_indices[mask]
        partial_label_bank[selected_rows, selected_cols] = 1

    return partial_label_bank


def calculate_k_values(outputs, k_max=2):
    batch_size, num_classes = outputs.size()
    eps1 = 0.1
    output_sorted, _ = torch.topk(outputs.detach().clone(), k=num_classes,
                                  dim=1)
    sorted_accumu_1 = torch.cat(
        [(torch.sum(output_sorted[:, :i + 1], dim=1) - eps1) / (i + 1)
         for i in range(num_classes)]
    ).reshape(num_classes, batch_size).transpose(0, 1)
    _, k_star = torch.max(sorted_accumu_1, dim=1)
    k_star += 1
    k_values = torch.where(k_star < k_max, k_star, k_max)
    return k_star, k_values


def selection_mask_bank_update(sample_selection_mask_bank, tar_idx, outputs,
                               tau_type: str, ratio: float):
    logits, _ = torch.topk(outputs, k=2, dim=-1, largest=True, sorted=True)
    logits_ratio = logits[:, 0] / logits[:, 1]

    if tau_type in ['fixed', 'stat']:
        selected_indx = torch.where(logits_ratio < ratio)[0].to(tar_idx.device)

    elif tau_type == 'cal':
        logits_max = logits[:, 0]
        logits_second = logits[:, 1]
        n_cls = outputs.size(1)
        eps2 = 1.0 / n_cls
        level_1 = 1.0 / n_cls + eps2
        level_2 = logits_second + eps2
        selected_indx = torch.where(torch.logical_or(
            logits_max < level_1, logits_max < level_2))[0].to(tar_idx.device)
    else:
        raise NotImplementedError(tau_type)

    sample_selection_mask_bank[tar_idx[selected_indx]] = 1
    return sample_selection_mask_bank, logits_ratio[selected_indx]


def logits_ratio_calculation(outputs):
    logits, _ = torch.topk(outputs, k=2, dim=-1, largest=True, sorted=True)
    return (logits[:, 0] / logits[:, 1]).detach().cpu().numpy()


# ==============================================================================
class UCon(object):
    def __init__(self,
                 args,
                 model_trg: nn.Module,
                 train_eval_loader,
                 n_cls: int,
                 n_iters_per_epoch: int,
                 max_epochs: int
                 ):
        self.args = args
        self.device = torch.device(f'cuda:{args.c_cudaid}')

        self.model = model_trg
        self.train_eval_loader = train_eval_loader
        self.n_cls = n_cls

        self.K = args.ucon_k
        self.alpha0 = args.ucon_alpha
        self.beta = args.ucon_beta
        self.alpha_decay = args.ucon_alpha_decay
        self.partial_k_type = args.ucon_partial_k_type
        self.partial_k = args.ucon_partial_k
        self.tau_type = args.ucon_tau_type

        assert self.partial_k_type in ['fixed', 'cal'], self.partial_k_type
        assert self.tau_type in ['fixed', 'stat', 'cal'], self.tau_type
        assert 1 <= self.partial_k <= n_cls, self.partial_k

        self.iter_num = 0
        self.max_iter = max(1, int(max_epochs * n_iters_per_epoch))

        image_ids = train_eval_loader.dataset.image_ids
        self.id_to_idx = {k: i for i, k in enumerate(image_ids)}
        num_sample = len(image_ids)

        self.uncertain_ratio_list = []

        # banks are built with a full pass (official init).
        self._init_banks(num_sample)

    # --------------------------------------------------------------------------
    def ids_to_idx(self, image_ids) -> torch.Tensor:
        return torch.tensor([self.id_to_idx[k] for k in image_ids],
                            dtype=torch.long, device=self.device)

    def _forward(self, x: torch.Tensor):
        logits = self.model(x)
        feats = self.model.lin_ft
        assert feats is not None
        return torch.flatten(feats, 1), logits

    @torch.no_grad()
    def _init_banks(self, num_sample: int):
        was_training = self.model.training
        self.model.eval()

        fea, scores, gts, idxs_all = [], [], [], []
        for batch in tqdm(self.train_eval_loader, ncols=constants.NCOLS,
                          desc='UCon banks'):
            images, targets, image_ids = batch[0], batch[1], batch[3]
            images = images.cuda(self.args.c_cudaid)
            feats, logits = self._forward(images)
            fea.append(F.normalize(feats.float(), dim=1).cpu())
            scores.append(F.softmax(logits.float(), dim=1))
            gts.append(targets.to(self.device))
            idxs_all.append(self.ids_to_idx(image_ids))

        fea = torch.cat(fea)
        scores = torch.cat(scores)
        gts = torch.cat(gts)
        idxs_all = torch.cat(idxs_all)

        feat_dim = fea.shape[1]
        self.fea_bank = torch.randn(num_sample, feat_dim)  # cpu (official)
        self.score_bank = torch.randn(num_sample, self.n_cls,
                                      device=self.device)
        self.partial_label_bank = torch.zeros(num_sample, self.n_cls,
                                              dtype=torch.long,
                                              device=self.device)
        self.sample_selection_mask_bank = torch.zeros(num_sample,
                                                      dtype=torch.long,
                                                      device=self.device)

        self.fea_bank[idxs_all.cpu()] = fea
        self.score_bank[idxs_all] = scores

        self.sample_selection_R = float(self.args.ucon_tau)
        for s in range(0, len(idxs_all), 256):
            ix = idxs_all[s: s + 256]
            sc = scores[s: s + 256]
            self._update_partial_labels(ix, sc)
            self.sample_selection_mask_bank, _ = selection_mask_bank_update(
                self.sample_selection_mask_bank, ix, sc, self.tau_type,
                self.sample_selection_R)

        # official `obtain_sample_R_ratio`.
        if self.tau_type == 'stat':
            ratio = logits_ratio_calculation(self.score_bank)
            self.sample_selection_R = float(np.percentile(ratio, [10])[0])

        acc = (scores.argmax(1) == gts).float().mean().item() * 100.
        DLLogger.log(fmsg(
            f'UCon-SFDA: init banks. N={num_sample}, feat_dim={feat_dim}, '
            f'K={self.K}, tau={self.sample_selection_R:.4f}, '
            f'selected={int(self.sample_selection_mask_bank.sum())}, '
            f'acc(trainset, source model)={acc:.2f}%'))

        self.model.train(was_training)

    def _update_partial_labels(self, idx: torch.Tensor, probs: torch.Tensor):
        if self.partial_k_type == 'fixed':
            self.partial_label_bank = partial_label_bank_update(
                self.partial_label_bank, idx, probs, k_values=self.partial_k)
        else:
            _, k_values = calculate_k_values(probs)
            self.partial_label_bank = partial_label_bank_update(
                self.partial_label_bank, idx, probs, k_values=k_values)

    def on_epoch_start(self):
        """Official: tau re-estimated at every evaluation interval (here: each
        epoch) for tau_type in ['stat', 'cal']."""
        if self.tau_type == 'stat':
            ratio = logits_ratio_calculation(self.score_bank)
            self.sample_selection_R = float(np.percentile(ratio, [10])[0])
        elif self.tau_type == 'cal' and len(self.uncertain_ratio_list) > 0:
            cat = torch.cat(self.uncertain_ratio_list)
            if cat.numel() > 0:
                self.sample_selection_R = float(cat.max().item())

        self.uncertain_ratio_list = []
        DLLogger.log(fmsg(
            f'UCon-SFDA: tau={self.sample_selection_R:.4f}, selected '
            f'samples={int(self.sample_selection_mask_bank.sum())}'))

    # --------------------------------------------------------------------------
    def step(self, images_w: torch.Tensor, views: torch.Tensor,
             image_ids) -> dict:
        """
        :param images_w: weak view. b, c, h, w.
        :param views: strong views. b, >=1, c, h, w.
        :param image_ids: list of str.
        :return: dict of tensors used by the UCon losses.
        """
        assert views is not None and views.ndim == 5, \
            'UCon-SFDA needs 1 strong view (`ucon_n_strong_views`).'
        self.iter_num += 1

        tar_idx = self.ids_to_idx(image_ids)

        # alpha decay (official).
        if self.alpha_decay:
            alpha = (1 + 10 * self.iter_num / self.max_iter) ** (
                -self.beta) * self.alpha0
        else:
            alpha = self.alpha0
        alpha = max(alpha, 5e-6)

        # strong view (dispersion control). forward first so that model
        # attributes (cams, lin_ft, ...) refer to the weak view afterwards.
        _, outputs_strong = self._forward(views[:, 0])

        # weak (main) view.
        features_test, outputs_test = self._forward(images_w)
        softmax_out = F.softmax(outputs_test, dim=1)

        with torch.no_grad():
            sm = softmax_out.detach().float()
            self._update_partial_labels(tar_idx, sm)

            self.sample_selection_mask_bank, uncertain_ratio = \
                selection_mask_bank_update(self.sample_selection_mask_bank,
                                           tar_idx, sm, self.tau_type,
                                           self.sample_selection_R)
            self.uncertain_ratio_list.append(uncertain_ratio.detach().clone())
            batch_selection_mask = self.sample_selection_mask_bank[tar_idx]

            output_f_ = F.normalize(features_test.float(), dim=1).cpu()
            self.score_bank[tar_idx] = sm.clone()
            self.fea_bank[tar_idx.cpu()] = output_f_.clone()

            distance = output_f_ @ self.fea_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True,
                                     k=self.K + 1)
            idx_near = idx_near[:, 1:].to(self.device)  # b x K
            score_near = self.score_bank[idx_near]  # b x K x C

            partial_Y = self.partial_label_bank[tar_idx]

        return {
            'softmax_out': softmax_out,
            'outputs_test': outputs_test,
            'outputs_strong': outputs_strong,
            'score_near': score_near,
            'alpha': alpha,
            'partial_Y': partial_Y,
            'selection_mask': batch_selection_mask,
        }
