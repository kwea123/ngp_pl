import torch
import glob
import json
#### Under construction. Don't use now

import numpy as np
import os
import imageio
import cv2
from einops import rearrange
from tqdm import tqdm

from .ray_utils import get_ray_directions

from .base import BaseDataset


def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)


class RTMVDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        with open(os.path.join(self.root_dir, '00000.json'), 'r') as f:
            meta = json.load(f)['camera_data']
        self.shift = np.array(meta['scene_center_3d_box'])
        self.scale = (np.array(meta['scene_max_3d_box'])-
                      np.array(meta['scene_min_3d_box'])).max()/2 * 1.05 # enlarge a little

        fx = meta['intrinsics']['fx'] * downsample
        fy = meta['intrinsics']['fy'] * downsample
        cx = meta['intrinsics']['cx'] * downsample
        cy = meta['intrinsics']['cy'] * downsample
        w = int(meta['width']*downsample)
        h = int(meta['height']*downsample)
        K = np.float32([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]])
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

        self.read_meta(split)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'train': start_idx, end_idx = 0, 100
        elif split == 'trainval': start_idx, end_idx = 0, 105
        elif split == 'test': start_idx, end_idx = 105, 150
        else: raise ValueError(f'{split} split not recognized!')
        imgs = sorted(glob.glob(os.path.join(self.root_dir, '*[0-9].exr')))[start_idx:end_idx]
        poses = sorted(glob.glob(os.path.join(self.root_dir, '*.json')))[start_idx:end_idx]

        print(f'Loading {len(imgs)} {split} images ...')
        for img, pose in tqdm(zip(imgs, poses)):
            with open(pose, 'r') as f:
                m = json.load(f)['camera_data']
            c2w = np.zeros((3, 4), dtype=np.float32)
            c2w[:, :3] = -np.array(m['cam2world'])[:3, :3].T
            c2w[:, 3] = np.array(m['location_world'])-self.shift
            c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
            self.poses += [c2w]

            img = imageio.imread(img)[..., :3]
            img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_LANCZOS4)
            # img = np.clip(linear_to_srgb(img), 0, 1)
            img = rearrange(torch.FloatTensor(img), 'h w c -> (h w) c')

            self.rays += [img]

        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
