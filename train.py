import torch
from torch import nn
from opt import get_opts
import numpy as np
import cv2

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays, get_ray_directions

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import PeakSignalNoiseRatio

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

        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss()
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)

        self.model = NGP(scale=self.hparams.scale, use_a=self.hparams.use_a)
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

    def forward(self, batch, split):
        poses = self.poses[batch['cam']].clone() # (3, 4)

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['cam']].unsqueeze(0))[0]
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['cam']]

        directions = \
            get_ray_directions(self.train_dataset.Hs[batch['cam']],
                               self.train_dataset.Ws[batch['cam']],
                               self.train_dataset.Ks[batch['cam']],
                               device=self.device)[batch['pix_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'cam': torch.LongTensor([batch['cam']]).to(self.device)}

        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample,
                  'scene': self.hparams.scene,
                  'take': self.hparams.take}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size

    def configure_optimizers(self):
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(92, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(92, 3, device=self.device)))

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']:
                net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)]

        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def training_step(self, batch, batch_nb, *args):
        if self.global_step%self.update_interval == 0:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=self.global_step<self.warmup_steps)

        results = self(batch, split='train')
        loss_d = self.loss(results, batch)
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/s_per_ray', results['total_samples']/len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)

        return loss

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.scene}/{hparams.take}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}/{hparams.scene}",
                               name=hparams.take,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=1,
                      num_sanity_val_steps=0,
                      precision=16)

    trainer.fit(system)

    ckpt_ = \
        slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.scene}/{hparams.take}/epoch={hparams.num_epochs-1}.ckpt',
                  save_poses=hparams.optimize_ext)
    torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.scene}/{hparams.take}/epoch={hparams.num_epochs-1}_slim.ckpt')
