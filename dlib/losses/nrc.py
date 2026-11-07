from torch import nn
#from utils.utils import to_cuda
import torch
import torch.nn.functional as F

# class NA(torch.nn.Module):
#     def forward(self, output_re, score_near_kk, weight_kk):
#         const = torch.mean((F.kl_div(output_re, score_near_kk, reduction='none').sum(-1)*weight_kk.cuda()).sum(1)) # kl_div here equals to dot product since we do not use log for score_near_kk
#         neig_aff = torch.mean(const)
#         return neig_aff
    
# class ENA(torch.nn.Module):
#     def forward(self, softmax_out_un, score_near,weight):
#         ext_neig_aff = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)*weight.cuda()).sum(1))
#         return ext_neig_aff

class NA(nn.Module):
    def forward(self, output_re, score_near_kk, weight_kk):

        affinity = -(output_re * score_near_kk).sum(dim=-1)

        return (affinity * weight_kk.to(affinity.device)).sum(dim=1).mean()


class ENA(nn.Module):
    def forward(self, softmax_out_un, score_near, weight):

        affinity = -(softmax_out_un * score_near).sum(dim=-1)

        return (affinity * weight.to(affinity.device)).sum(dim=1).mean()

# class KL(torch.nn.Module):
#     def __init__(self, epsilon):
#         self.epsilon = epsilon

#     def forward(self, msoftmax,weight_kk):
#         gentropy = torch.sum(msoftmax*torch.log(msoftmax + self.epsilon))
#         return gentropy

class KL(nn.Module):
    def __init__(self, epsilon=0.001):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, msoftmax, weight_kk=None):

        p = msoftmax.float()

        if not torch.isfinite(p).all():
            raise RuntimeError(
                f"NRC KL: invalid msoftmax = {p.detach()}"
            )

        p = p.clamp(min=self.epsilon, max=1.0)

        gentropy = torch.sum(p * torch.log(p))

        if not torch.isfinite(gentropy):
            raise RuntimeError("NRC KL produced NaN/Inf")

        return gentropy
