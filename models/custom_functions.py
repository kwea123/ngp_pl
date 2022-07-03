import torch
import vren
from torch.cuda.amp import custom_fwd, custom_bwd


class RayAABBIntersector(torch.autograd.Function):
    """
    Computes the intersections of rays and axis-aligned voxels.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) normalized ray directions
        centers: (N_voxels, 3) voxel centers
        half_sizes: (N_voxels, 3) voxel half sizes
        max_hits: maximum number of intersected voxels to keep for one ray
                  (for a cubic scene, this is at most 3*N_voxels^(1/3)-2)

    Outputs:
        hits_cnt: (N_rays) number of hits for each ray
        (followings are from near to far)
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit)
        hits_voxel_idx: (N_rays, max_hits) hit voxel indices (-1 if no hit)
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, center, half_size, max_hits):
        return vren.ray_aabb_intersect(rays_o, rays_d, center, half_size, max_hits)


class RayMarcher(torch.autograd.Function):
    """
    March the rays to get sample point positions and directions.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) normalized ray directions
        hits_t: (N_rays, 2) near and far bounds from aabb intersection
        density_bitfield: (C*G**3//8)
        scale: float
        exp_step_factor: the exponential factor to scale the steps
        perturb: whether to perturb the first sample's depth
        grid_size: int
        max_samples: int
        mean_samples: int, mean total samples per batch

    Outputs:
        rays_a: (N_rays) ray_idx, start_idx, N_samples
        (the followings contain only valid samples, with some padding)
        xyzs: (N, 3) sample positions
        dirs: (N, 3) sample view directions
        deltas: (N) dt for integration
        ts: (N) sample ts
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, hits_t,
                density_bitfield, scale, exp_step_factor,
                perturb, grid_size, max_samples):

        rays_a, xyzs, dirs, deltas, ts, counter = \
            vren.raymarching_train(
                rays_o, rays_d, hits_t,
                density_bitfield, scale,
                exp_step_factor, perturb, grid_size, max_samples)

        total_samples = counter[0] # total samples for all rays
        # remove redundant output
        xyzs = xyzs[:total_samples]
        dirs = dirs[:total_samples]
        deltas = deltas[:total_samples]
        ts = ts[:total_samples]

        return rays_a, xyzs, dirs, deltas, ts, total_samples


class VolumeRenderer(torch.autograd.Function):
    """
    Volume rendering with different number of samples per ray

    Inputs:
        sigmas: (N)
        rgbs: (N, 3)
        deltas: (N)
        ts: (N)
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]
        rgb_bg: (3) background color
        T_threshold: float, stop the ray if the transmittance is below it

    Outputs:
        opacity: (N_rays)
        depth: (N_rays)
        rgb: (N_rays, 3)
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, deltas, ts, rays_a, rgb_bg, T_threshold):
        opacity, depth, rgb = \
            vren.composite_train_fw(sigmas, rgbs, deltas, ts,
                                    rays_a, rgb_bg, T_threshold)
        ctx.save_for_backward(sigmas, rgbs, deltas, ts, rays_a,
                              opacity, depth, rgb, rgb_bg)
        ctx.T_threshold = T_threshold
        return opacity, depth, rgb

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dopacity, dL_ddepth, dL_drgb):
        sigmas, rgbs, deltas, ts, rays_a, \
        opacity, depth, rgb, rgb_bg = ctx.saved_tensors
        dL_dsigmas, dL_drgbs, dL_drgb_bg = \
            vren.composite_train_bw(dL_dopacity, dL_ddepth,
                                    dL_drgb, sigmas, rgbs, deltas, ts,
                                    rays_a,
                                    opacity, depth, rgb, rgb_bg,
                                    ctx.T_threshold)
        return dL_dsigmas, dL_drgbs, None, None, None, dL_drgb_bg, None


class TruncExp(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))
