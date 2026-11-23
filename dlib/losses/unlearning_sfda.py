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

FORGET_REDISTRIBUTION_MODES = ('none', 'uniform', 'hybrid')


def redistribution_forget_target(logits: torch.Tensor,
                                 forget_class: torch.Tensor,
                                 mode: str,
                                 alpha: float = 0.5) -> torch.Tensor:
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
    Returned without gradient (stop-gradient target).
    """
    with torch.no_grad():
        logits = logits.detach().float()
        assert logits.ndim == 2, logits.ndim
        k = logits.shape[1]
        assert k >= 2, k
        assert mode in ('uniform', 'hybrid'), mode
        assert 0.0 <= alpha <= 1.0, alpha

        d = forget_class.long().unsqueeze(1)                   # [B, 1]
        keep = torch.ones_like(logits, dtype=torch.bool)
        keep.scatter_(1, d, False)                             # False at d

        uniform = keep.float() / float(k - 1)                  # [B, K]
        if mode == 'uniform':
            return uniform

        # p_j / (1 - p_d) for j != d, 0 at d.
        renorm = torch.softmax(logits.masked_fill(~keep, float('-inf')),
                               dim=1)
        return (1.0 - alpha) * renorm + alpha * uniform.masked_fill(~keep, 0.)


def redistribution_forget_loss(logits: torch.Tensor,
                               forget_class: torch.Tensor,
                               mode: str,
                               alpha: float = 0.5) -> torch.Tensor:
    """
    Per-sample soft-target CE: L = -sum_j q_j log p_j, with q from
    redistribution_forget_target (q is a stop-gradient target).
    Returns [B].
    """
    log_p = torch.log_softmax(logits.float(), dim=1)
    q = redistribution_forget_target(logits, forget_class, mode, alpha)
    return -(q * log_p).sum(dim=1)
