import torch
from torch import nn
import vren
from torch.cuda.amp import custom_fwd, custom_bwd


class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in mipnerf360

    Inputs:
        ws: (N)
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_prefix_sum, wts_prefix_sum = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_prefix_sum, wts_prefix_sum, ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dloss):
        ws_prefix_sum, wts_prefix_sum, ws, deltas, ts, rays_a = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_prefix_sum, wts_prefix_sum,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


class NeRFLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.lambda_dist = 1e-2
        self.lambda_opa = 1e-3

    def forward(self, results, target, **kwargs):
        d = {}
        d['rgb'] = (results['rgb']-target['rgb'])**2

        o = results['opacity']+1e-10
        # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = self.lambda_opa*(-o*torch.log(o))

        # # TODO: use normalized ts and deltas
        # d['distortion'] = self.lambda_dist * \
        #     DistortionLoss.apply(results['ws'], results['deltas'],
        #                          results['ts'], results['rays_a'])

        return d
