import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange

import wandb

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch
import torch
from torch import nn
from torch.nn import functional as F



from utils import slim_ckpt, load_ckpt

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.warmup_steps = 256
        self.update_interval = 16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act)
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

    def forward(self, batch, split):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']

        return render(self.model, rays_o, rays_d, **kwargs)


def save_ckpt(model, save_path, epoch, step, best_val_loss):
    ckpt = {'model': model.state_dict(),
            'epoch': epoch,
            'step': step,
            'best_val_loss': best_val_loss}
    torch.save(ckpt, save_path)

def configure_optimizers(hparams, model):
    # define additional parameters
    if hparams.optimize_ext:
        N = len(train_dataset.poses)
        model.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=hparams.device)))
        model.register_parameter('dT',
            nn.Parameter(torch.zeros(N, 3, device=hparams.device)))

    load_ckpt(model, hparams.weight_path)

    net_params = []
    for n, p in model.named_parameters():
        if n not in ['dR', 'dT']: net_params += [p]

    opts = []
    net_opt = FusedAdam(net_params, hparams.lr, eps=1e-15)
    opts += [net_opt]
    if hparams.optimize_ext:
        opts += [FusedAdam([model.dR, model.dT], 1e-6)] # learning rate is hard-coded
    net_sch = CosineAnnealingLR(net_opt,
                                hparams.num_epochs,
                                hparams.lr/30)

    return opts, [net_sch]


if __name__ == '__main__':
    # wandb.init(project="ngp-pl", entity="praveen998")
    hparams = get_opts()
    # breakpoint()
    wandb.config = hparams
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams).to(hparams.device)

    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir,
                'downsample': hparams.downsample}
    train_dataset = dataset(split=hparams.split, **kwargs)
    train_dataset.batch_size = hparams.batch_size
    train_dataset.ray_sampling_strategy = hparams.ray_sampling_strategy

    test_dataset = dataset(split='test', **kwargs)

    directions = system.directions = train_dataset.directions.to(hparams.device)
    poses = system.poses = train_dataset.poses.to(hparams.device)

    optimizers = configure_optimizers(hparams, system)

    train_dataloader = DataLoader(train_dataset,
                                  num_workers=16,
                                  persistent_workers=True,
                                  batch_size=None,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset,
                                num_workers=8,
                                batch_size=None,
                                pin_memory=True)
    # breakpoint()
    system.model.mark_invisible_cells(train_dataset.K.to(hparams.device),
                                        poses,
                                        train_dataset.img_wh)
    # start train
    system.train()
    for epoch in range(hparams.num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(hparams.device) for k, v in batch.items()}
            batch['directions'] = directions[batch['pix_idxs']]
            batch['poses'] = poses[batch['img_idxs']]
            global_step = epoch * len(train_dataloader) + batch_idx
            if global_step % system.update_interval == 0:
                system.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=global_step<system.warmup_steps,
                                           erode=hparams.dataset_name=='colmap')
            # Floating point exception (core dumped) on loss backward

            for op in optimizers[0]:
                op.zero_grad()
            results = system(batch, 'train')
            loss_d = system.loss(results, batch)
            if hparams.use_exposure:
                zero_radiance = torch.zeros(1, 3, device=hparams.device)
                unit_exposure_rgb = system.model.log_radiance_to_rgb(zero_radiance,
                                        **{'exposure': torch.ones(1, 1, device=hparams.device)})
                loss_d['unit_exposure'] = \
                    0.5*(unit_exposure_rgb-train_dataset.unit_exposure_rgb)**2
            loss = sum(lo.mean() for lo in loss_d.values())
            print(loss_d)
            # Floating point exception (core dumped) on loss backward
            loss.backward()

            with torch.no_grad():
                psnr = system.train_psnr(results['rgb'], batch['rgb'])

            for op in optimizers[0]:
                op.step()
            print(loss_d)
            # if hparams.optimize_ext:
            #     save_ckpt(system.model, hparams.ckpt_path, epoch, global_step, 0)
        for scheduler in optimizers[1]:
            scheduler.step()
        # save_ckpt(system.model, hparams.ckpt_path, epoch, global_step, 0)

    # start test
    torch.cuda.empty_cache()
    if not hparams.no_save_test:
        system.val_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}'
        os.makedirs(system.val_dir, exist_ok=True)

    if hparams.val_only:
        system.eval()
        for batch in test_dataloader:
            batch = {k: v.to(hparams.device) for k, v in batch.items()}
            rgb_gt = batch['rgb']
            results = system(batch, 'test')
            log = {}
            system.val_psnr(results['rgb'], rgb_gt)
            log['psnr'] = system.val_psnr.compute()
            system.val_psnr.reset()

            w, h = train_dataset.img_wh
            rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
            rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
            system.val_ssim(rgb_pred, rgb_gt)
            log['ssim'] = system.val_ssim.compute()
            system.val_ssim.reset()
            if hparams.eval_lpips:
                system.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                            torch.clip(rgb_gt*2-1, -1, 1))
                log['lpips'] = system.val_lpips.compute()
                system.val_lpips.reset()
            
            if not hparams.no_save_test: # save test image to disk
                idx = batch['img_idxs']
                rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
                rgb_pred = (rgb_pred*255).astype(np.uint8)
                depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
                imageio.imsave(os.path.join(system.val_dir, f'{idx:03d}.png'), rgb_pred)
                imageio.imsave(os.path.join(system.val_dir, f'{idx:03d}_d.png'), depth)
                
            
        # system.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # system.log('val_psnr', system.val_psnr, on_step=True, on_epoch=True, prog_bar=True)
        # system.log('val_ssim', system.val_ssim, on_step=True, on_epoch=True, prog_bar=True)
        # if hparams.eval_lpips:
        #     system.log('val_lpips', system.val_lpips, on_step=True, on_epoch=True, prog_bar=True)
        # wandb.log(system.logger.experiment[-1])
        # system.logger.experiment = []


    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

    if (not hparams.no_save_test) and \
       hparams.dataset_name=='nsvf' and \
       'Synthetic' in hparams.root_dir: # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)
