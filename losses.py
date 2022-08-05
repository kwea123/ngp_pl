from torch import nn


class NeRFLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, results, target, **kwargs):
        d = {}
        d['rgb'] = (results['rgb']-target['rgb'])**2
        d['opacity'] = (results['opacity']-target['alpha'])**2

        return d
