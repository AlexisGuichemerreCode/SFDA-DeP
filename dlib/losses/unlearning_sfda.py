import torch
import torch.nn as nn

class ForgetClassSuppressionLoss(nn.Module):
    """
    Suppresses confidence for a given class.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, forget_class):
        """
        logits       : [B, K]
        forget_class : [B] (int)
        """
        probs = torch.softmax(logits, dim=1)
        p_c = probs.gather(1, forget_class.unsqueeze(1)).squeeze(1)
        return -torch.log(1.0 - p_c + self.eps)

FORGET_REDISTRIBUTION_MODES = ('none', 'uniform', 'hybrid',
                               'uniform_all', 'hybrid_all')


def redistribution_forget_target(logits: torch.Tensor,
                                 forget_class: torch.Tensor,
                                 mode: str,
                                 alpha: float = 0.5,
                                 forget_set=None) -> torch.Tensor:
    """
    Soft target q for selective forgetting with probability redistribution.

    logits       : [B, K] classifier logits.
    forget_class : [B] long, class d to forget per sample.
    mode         : 'uniform' -> q_j = 1/(K-1) for j != d.
                   'hybrid'  -> q_j = (1-alpha) p_j/(1-p_d) + alpha/(K-1),
                                j != d.
    q_d = 0 in both cases; each row of q sums to 1.
    p_j/(1-p_d) is computed as a softmax over the logits with class d
    masked out: exact, and stable even when p_d rounds to 1.
    'uniform_all' / 'hybrid_all': same, but every class in forget_set (all
    classes forgotten over the forget set: list/1-D tensor of class ids) is
    removed,
    not only d: K-1 -> |K \\ F| and p_j/(1-p_d) -> p_j/sum_{k not in F} p_k.
    Rows where F covers all classes fall back to removing only d.
    Returned without gradient (stop-gradient target).
    """
    with torch.no_grad():
        logits = logits.detach().float()
        assert logits.ndim == 2, logits.ndim
        k = logits.shape[1]
        assert k >= 2, k
        assert mode in FORGET_REDISTRIBUTION_MODES[1:], mode
        assert 0.0 <= alpha <= 1.0, alpha

        d = forget_class.long().unsqueeze(1)                   # [B, 1]
        keep = torch.ones_like(logits, dtype=torch.bool)
        keep.scatter_(1, d, False)                             # False at d

        if mode.endswith('_all'):
            assert forget_set is not None, mode
            fset = torch.as_tensor(forget_set, dtype=torch.long,
                                   device=logits.device).view(-1)
            keep_all = keep.clone()
            keep_all[:, fset] = False                          # False on F
            ok = keep_all.any(dim=1, keepdim=True)             # F != all K
            keep = torch.where(ok, keep_all, keep)
            mode = mode[:-len('_all')]

        uniform = keep.float() / keep.float().sum(dim=1, keepdim=True)
        if mode == 'uniform':
            return uniform

        # p_j / (1 - p_d) for j != d, 0 at d.
        renorm = torch.softmax(logits.masked_fill(~keep, float('-inf')),
                               dim=1)
        return (1.0 - alpha) * renorm + alpha * uniform.masked_fill(~keep, 0.)


def redistribution_forget_loss(logits: torch.Tensor,
                               forget_class: torch.Tensor,
                               mode: str,
                               alpha: float = 0.5,
                               forget_set=None) -> torch.Tensor:
    """
    Per-sample soft-target CE: L = -sum_j q_j log p_j, with q from
    redistribution_forget_target (q is a stop-gradient target).
    Returns [B].
    """
    log_p = torch.log_softmax(logits.float(), dim=1)
    q = redistribution_forget_target(logits, forget_class, mode, alpha,
                                     forget_set)
    return -(q * log_p).sum(dim=1)
