import torch
from torch import nn
from einops import reduce


class NeRFLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.lambda_opa = 1e-3

    def forward(self, results, rgbs):
        d = {}
        d['rgb'] = reduce((results['rgb']-rgbs)**2, 'n c -> n', 'mean')

        o = results['opacity']+1e-8
        # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = self.lambda_opa*(-o*torch.log(o))

        return d
