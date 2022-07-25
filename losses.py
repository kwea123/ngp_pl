import torch
from torch import nn


def shiftscale_inv_depthloss(disp_pred, disp_gt):
    """
    Computes the shift- scale- invariant depth loss as proposed in
    https://arxiv.org/pdf/1907.01341.pdf.
    Inputs:
        disp_pred: (N) disp predicted by the network
        disp_gt: (N) disparity produced by image-based method.
    Outputs:
        loss: (N)
    """
    t_pred = torch.median(disp_pred)
    s_pred = torch.mean(torch.abs(disp_pred-t_pred))
    t_gt = torch.median(disp_gt)
    s_gt = torch.mean(torch.abs(disp_gt-t_gt))

    disp_pred_n = (disp_pred-t_pred)/s_pred
    disp_gt_n = (disp_gt-t_gt)/s_gt
    loss = (disp_pred_n-disp_gt_n)**2
    return loss


class NeRFLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.lambda_opa = 1e-3

    def forward(self, results, target, **kwargs):
        d = {}
        d['rgb'] = (results['rgb']-target['rgb'])**2

        o = results['opacity']+1e-10
        # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = self.lambda_opa*(-o*torch.log(o))

        return d
