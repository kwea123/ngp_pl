import torch
from opt import get_opts
import os
import imageio
import numpy as np
import cv2

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from metrics import psnr

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import slim_ckpt

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.loss = NeRFLoss()

        self.model = NGP(scale=hparams.scale)
        # save grid coordinates for training
        G = self.model.grid_size
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

        self.S = 16 # the interval to update density grid

    def forward(self, rays, split):
        kwargs = {'test_time': split!='train'}

        return render(self.model, rays, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[hparams.dataset_name]
        kwargs = {'root_dir': hparams.root_dir,
                  'downsample': hparams.downsample}
        self.train_dataset = dataset(split=hparams.split, **kwargs)
        self.test_dataset = dataset(split='test', **kwargs)

    def configure_optimizers(self):
        self.opt = FusedAdam(self.model.parameters(), hparams.lr)
        self.sch = CosineAnnealingLR(self.opt,
                                     hparams.num_epochs,
                                     hparams.lr/30)

        return [self.opt], [self.sch]

    def train_dataloader(self):
        # hard sampling (remove converged rays in training)
        if hparams.hard_sampling:
            if self.current_epoch == 0:
                self.register_buffer('weights',
                        torch.ones(len(self.train_dataset.rays), device=self.device))
            else:
                non_converged_mask = self.weights>1e-5
                self.train_dataset.rays = self.train_dataset.rays[non_converged_mask]
                self.weights = self.weights[non_converged_mask]

        self.train_dataset.batch_size = hparams.batch_size
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          num_workers=16,
                          batch_size=None,
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        if self.global_step%self.S == 0:
            # gradually increase the threshold to remove floater
            a_thr = min(self.current_epoch+1, 25)/50 # alpha threshold, at most 0.5
            self.model.update_density_grid(a_thr*MAX_SAMPLES/(2*3**0.5),
                                           warmup=self.global_step<256)

        rays, rgb = batch['rays'], batch['rgb']
        results = self(rays, split='train')
        loss_d = self.loss(results, rgb)
        if hparams.hard_sampling:
            self.weights[batch['idx']] = loss_d['rgb'].detach()
        loss = sum(lo.mean() for lo in loss_d.values())

        self.log('lr', self.opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/psnr', psnr(results['rgb'], rgb), prog_bar=True)

        return loss

    def on_validation_start(self):
        if not hparams.no_save_test:
            self.val_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        rays, rgb_gt = batch['rays'], batch['rgb']
        results = self(rays, split='test')
        log = {'psnr': psnr(results['rgb'], rgb_gt)}

        if not hparams.no_save_test: # save test image to disk
            w, h = self.train_dataset.img_wh
            rgb_pred = results['rgb'].reshape(h, w, 3).cpu().numpy()
            log['rgb'] = rgb_pred = (rgb_pred*255).astype(np.uint8)
            log['depth'] = depth = \
                depth2img(results['depth'].reshape(h, w).cpu().numpy())
            imageio.imsave(os.path.join(self.val_dir, f'{batch_nb:03d}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{batch_nb:03d}_d.png'), depth)

        return log

    def validation_epoch_end(self, outputs):
        mean_psnr = torch.stack([x['psnr'] for x in outputs]).mean()
        self.log('test/psnr', mean_psnr, prog_bar=True)

        if not hparams.no_save_test: # save video
            imageio.mimsave(os.path.join(self.val_dir, 'rgb.mp4'),
                            [x['rgb'] for x in outputs],
                            fps=30, macro_block_size=1)
            imageio.mimsave(os.path.join(self.val_dir, 'depth.mp4'),
                            [x['depth'] for x in outputs],
                            fps=30, macro_block_size=1)


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              every_n_epochs=hparams.ckpt_freq,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      reload_dataloaders_every_n_epochs=1 if hparams.hard_sampling else 0,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='auto',
                      devices=1, # tinycudann doesn't support multigpu...
                      num_sanity_val_steps=0,
                      precision=16)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    # save slimmed ckpt for the last epoch
    ckpt_ = slim_ckpt(f'ckpts/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt')
    torch.save(ckpt_, f'ckpts/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')
