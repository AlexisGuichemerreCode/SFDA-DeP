from torch import nn
#from utils.utils import to_cuda
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class NLL(torch.nn.Module):
    def __init__(self, mu: torch.Tensor, var: torch.Tensor, pi: torch.Tensor, **kwargs):

        self.mu = mu
        self.var = var
        self.pi = pi

    def forward(self, pixel_features):

        _, K, D = self.mu.shape

        N, D, H, W = pixel_features.shape

        device = pixel_features.device

        pixel_features = pixel_features.permute(0, 2, 3, 1).reshape(N * H * W, D)  # (N_pixels, D)


        diff = pixel_features.unsqueeze(1) - self.mu.to(device).unsqueeze(0) # (N, K, D)
        sigma_inv = 1.0 / self.var  # (1, K, D)

        log_det_sigma = torch.sum(torch.log(self.var), dim=-1).to(device)  # (1, K)


        #diff = pixel_features.unsqueeze(1) - self.mu.to(device).unsqueeze(0) # (N, K, D)
        diff = diff.squeeze(0)
        mahalanobis = torch.sum(diff**2 * sigma_inv.to(device), dim=-1)  # (N, K)


        #mahalanobis = torch.einsum('nkd,kde,nke->nk', diff.squeeze(0), sigma_inv, diff.squeeze(0))  # (N, K)

        log_norm = -0.5 * (D * torch.log(torch.tensor(2 * torch.pi)) + log_det_sigma)  # (1, K)
        log_prob = log_norm - 0.5 * mahalanobis  # (N, K)

        weighted_log_prob = log_prob + torch.log(self.pi.to(device).squeeze())  # (N, K)

        loss = torch.logsumexp(weighted_log_prob, dim=1)  # (N,)

        return loss