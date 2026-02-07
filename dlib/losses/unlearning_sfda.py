import torch
import torch.nn as nn

class ForgetClassSuppressionLoss(nn.Module):
    """
    Suppresses confidence for a given class without enforcing another class.
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