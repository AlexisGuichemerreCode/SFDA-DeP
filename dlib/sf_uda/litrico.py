"""
Litrico et al., "Guiding Pseudo-labels with Uncertainty Estimation for
Source-free Unsupervised Domain Adaptation", CVPR 2023.
https://arxiv.org/abs/2303.03770
Official code:
https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation

Port of `train_target.py` + `moco.py` (AdaMoCo) to this code base.

Per training step (official `train()`):
  1. weak view x_w -> (feats_w, logits_w) with the adapted model.
  2. pseudo-label refinement: soft k-NN voting over a (features, probs) bank
     -> pseudo_labels_w, refined probs_w.
  3. strong view x_s1 -> query (adapted model); strong view x_s2 -> key
     (momentum model, EMA m=0.999). MoCo logits vs. a queue of keys.
  4. contrastive loss where negatives sharing a temporal pseudo-label with
     the query are masked out (temporal queue of length T).
  5. update queue + temporal labels memory.
  6. uncertainty weight w = exp(-H(probs_w) / log(C)).
  7. cls loss: w * NL(logits_q, pseudo_labels_w) (negative learning,
     default in official code) or w * CE(...).
  8. diversity loss: div(logits_w) + div(logits_q).
  9. bank FIFO update with (feats_w, softmax(logits_w)).
After every epoch, the bank is rebuilt from a full pass over the target train
set with the eval transform (official `eval_and_label_dataset`).

Loss terms are computed in `dlib/losses/sf_uda.py` (Litrico* classes) from
the dict returned by `Litrico.step()`.

Adaptation to this code base:
  - features = `model.lin_ft` (pooled features before the last linear layer).
  - weak view = the regular training view (`images`), strong views are
    provided by the loader in `views` (see `get_sfuda_strong_transform`).
  - image ids (str) are mapped to integer dataset indices.
"""
import sys
from os.path import dirname, abspath
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg


__all__ = ['Litrico']


def _get_distances(x: torch.Tensor, y: torch.Tensor,
                   dist_type: str = 'cosine') -> torch.Tensor:
    if dist_type == 'euclidean':
        return torch.cdist(x, y)
    elif dist_type == 'cosine':
        return 1. - torch.matmul(F.normalize(x, dim=1),
                                 F.normalize(y, dim=1).T)
    raise NotImplementedError(dist_type)


class Litrico(object):
    def __init__(self,
                 args,
                 model_trg: nn.Module,
                 train_eval_loader,
                 n_cls: int,
                 num_neighbors: int = 10,
                 temporal_length: int = 5,
                 queue_size: int = 16384,
                 moco_m: float = 0.999,
                 moco_t: float = 0.07,
                 dist_type: str = 'cosine'
                 ):
        self.args = args
        self.device = torch.device(f'cuda:{args.c_cudaid}')

        self.model = model_trg
        self.train_eval_loader = train_eval_loader
        self.n_cls = n_cls

        self.num_neighbors = num_neighbors
        self.temporal_length = temporal_length
        self.moco_m = moco_m
        self.moco_t = moco_t
        self.dist_type = dist_type

        # image_id (str) -> int index.
        image_ids = train_eval_loader.dataset.image_ids
        self.id_to_idx = {k: i for i, k in enumerate(image_ids)}
        self.dataset_length = len(image_ids)

        # MoCo key encoder: EMA of the adapted model.
        self.momentum_model = deepcopy(self.model)
        self.momentum_model.requires_grad_(False)

        self.K = min(queue_size, self.dataset_length)
        self.queue_ptr = 0
        self.mem_ptr = 0

        # bank for pseudo-label refinement.
        self.banks = None
        self._skip_next_refresh = False

        # initial bank (official: eval_and_label_dataset(0, ...)).
        # also gives the feature dimension.
        feat_dim = self.refresh_banks(log=True)
        # first `on_epoch_start` must not rebuild the bank again.
        self._skip_next_refresh = True

        # MoCo queue + temporal pseudo-label memory.
        self.features = F.normalize(
            torch.randn(feat_dim, self.K, device=self.device), dim=0)
        self.labels = torch.randint(0, n_cls, (self.K,), device=self.device)
        self.idxs = torch.randint(0, self.dataset_length, (self.K,),
                                  device=self.device)
        self.mem_labels = torch.randint(
            0, n_cls, (self.dataset_length, self.temporal_length),
            device=self.device)

        DLLogger.log(fmsg(f'Litrico: init. N={self.dataset_length}, '
                          f'feat_dim={feat_dim}, K={self.K}, '
                          f'num_neighbors={num_neighbors}, '
                          f'temporal_length={temporal_length}'))

    # --------------------------------------------------------------------------
    def ids_to_idx(self, image_ids) -> torch.Tensor:
        return torch.tensor([self.id_to_idx[k] for k in image_ids],
                            dtype=torch.long, device=self.device)

    def _forward(self, model: nn.Module, x: torch.Tensor):
        logits = model(x)
        feats = model.lin_ft
        assert feats is not None
        feats = torch.flatten(feats, 1)
        return feats, logits

    # --------------------------------------------------------------------------
    @torch.no_grad()
    def refresh_banks(self, log: bool = False) -> int:
        """Official `eval_and_label_dataset`: full pass over target trainset
        (eval transform), random subset of size <= 16384 as bank."""
        was_training = self.model.training
        self.model.eval()

        features, probs, gt = [], [], []
        for batch in tqdm(self.train_eval_loader, ncols=constants.NCOLS,
                          desc='Litrico bank'):
            images, targets = batch[0], batch[1]
            images = images.cuda(self.args.c_cudaid)
            feats, logits = self._forward(self.model, images)
            features.append(feats.float())
            probs.append(F.softmax(logits.float(), dim=1))
            gt.append(targets.to(self.device))

        features = torch.cat(features)
        probs = torch.cat(probs)
        gt = torch.cat(gt)

        rand_idxs = torch.randperm(len(features), device=self.device)
        self.banks = {
            'features': features[rand_idxs][: 16384],
            'probs': probs[rand_idxs][: 16384],
            'ptr': 0
        }

        if log:
            pred_labels, _ = self.refine_predictions(features)
            acc = (pred_labels == gt).float().mean().item() * 100.
            DLLogger.log(fmsg(f'Litrico: refined pseudo-labels accuracy on '
                              f'target trainset: {acc:.2f}%'))

        self.model.train(was_training)
        return features.shape[1]

    def on_epoch_start(self):
        if self._skip_next_refresh:
            self._skip_next_refresh = False
            return
        self.refresh_banks(log=True)

    # --------------------------------------------------------------------------
    @torch.no_grad()
    def refine_predictions(self, features: torch.Tensor):
        """Official `soft_k_nearest_neighbors`."""
        features_bank = self.banks['features']
        probs_bank = self.banks['probs']

        pred_probs = []
        for feats in features.split(64):
            distances = _get_distances(feats, features_bank, self.dist_type)
            _, idxs = distances.sort()
            idxs = idxs[:, : self.num_neighbors]
            pred_probs.append(probs_bank[idxs, :].mean(1))

        pred_probs = torch.cat(pred_probs)
        _, pred_labels = pred_probs.max(dim=1)
        return pred_labels, pred_probs

    @torch.no_grad()
    def _update_banks(self, features: torch.Tensor, logits: torch.Tensor):
        """Official `update_labels`: FIFO update of the bank."""
        probs = F.softmax(logits, dim=1)
        n_bank = len(self.banks['features'])
        start = self.banks['ptr']
        end = start + len(features)
        idxs_replace = torch.arange(start, end, device=self.device) % n_bank
        self.banks['features'][idxs_replace, :] = features
        self.banks['probs'][idxs_replace, :] = probs
        self.banks['ptr'] = end % n_bank

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.model.parameters(),
                                    self.momentum_model.parameters()):
            param_k.data = param_k.data * self.moco_m + \
                           param_q.data * (1. - self.moco_m)

    @torch.no_grad()
    def _update_memory(self, epoch: int, idxs: torch.Tensor,
                       keys: torch.Tensor, pseudo_labels: torch.Tensor):
        """Official `AdaMoCo.update_memory`."""
        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end, device=self.device) % self.K
        self.features[:, idxs_replace] = keys.T
        self.labels[idxs_replace] = pseudo_labels
        self.idxs[idxs_replace] = idxs
        self.queue_ptr = end % self.K

        self.mem_labels[idxs, self.mem_ptr] = pseudo_labels
        self.mem_ptr = epoch % self.temporal_length

    # --------------------------------------------------------------------------
    def step(self, images_w: torch.Tensor, views: torch.Tensor, image_ids,
             epoch: int) -> dict:
        """
        :param images_w: weak view. b, c, h, w.
        :param views: strong views. b, 2, c, h, w.
        :param image_ids: list of str.
        :param epoch: int.
        :return: dict of tensors used by the Litrico losses.
        """
        assert views is not None and views.ndim == 5, \
            'Litrico needs 2 strong views (`litrico_n_strong_views`).'
        assert views.shape[1] >= 2, views.shape

        idxs = self.ids_to_idx(image_ids)
        strong_x = views[:, 0]
        strong_x2 = views[:, 1]

        # 1. weak view (with grad: used by the diversity loss).
        feats_w, logits_w = self._forward(self.model, images_w)

        # 2. pseudo-label refinement.
        with torch.no_grad():
            if self.args.litrico_label_refinement:
                pseudo_labels_w, probs_w = self.refine_predictions(
                    feats_w.float())
            else:
                probs_w = F.softmax(logits_w.float(), dim=1)
                pseudo_labels_w = probs_w.max(1)[1]

        # 3. MoCo: query (adapted model) / key (momentum model).
        feats_q, logits_q = self._forward(self.model, strong_x)
        q = F.normalize(feats_q.float(), dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            self.momentum_model.train(self.model.training)
            k, _ = self._forward(self.momentum_model, strong_x2)
            k = F.normalize(k.float(), dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.features.clone().detach()])
        logits_ins = torch.cat([l_pos, l_neg], dim=1) / self.moco_t

        # temporal labels used to mask negatives (before memory update).
        mem_labels_query = self.mem_labels[idxs].clone()
        mem_labels_queue = self.mem_labels[self.idxs].clone()

        # 5. update queue and temporal memory.
        self._update_memory(epoch, idxs, k, pseudo_labels_w)

        # 6. uncertainty weights.
        with torch.no_grad():
            max_entropy = torch.log2(torch.tensor(float(self.n_cls),
                                                  device=self.device))
            w = -torch.sum(probs_w * torch.log2(probs_w + 1e-5), dim=1)
            w = torch.exp(-(w / max_entropy))

        # 9. bank update.
        self._update_banks(feats_w.detach().float(), logits_w.detach().float())

        return {
            'logits_w': logits_w,
            'logits_q': logits_q,
            'logits_ins': logits_ins,
            'pseudo_labels_w': pseudo_labels_w,
            'weights': w,
            'mem_labels_query': mem_labels_query,
            'mem_labels_queue': mem_labels_queue,
        }
