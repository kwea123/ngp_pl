import torch
import glob
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset


class RTMVDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, '00000.json'), 'r') as f:
            meta = json.load(f)['camera_data']

        self.shift = np.array(meta['scene_center_3d_box'])
        self.scale = (np.array(meta['scene_max_3d_box'])-
                      np.array(meta['scene_min_3d_box'])).max()/2 * 1.05 # enlarge a little

        fx = meta['intrinsics']['fx'] * self.downsample
        fy = meta['intrinsics']['fy'] * self.downsample
        cx = meta['intrinsics']['cx'] * self.downsample
        cy = meta['intrinsics']['cy'] * self.downsample
        w = int(meta['width']*self.downsample)
        h = int(meta['height']*self.downsample)
        K = np.float32([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]])
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'train': start_idx, end_idx = 0, 100
        elif split == 'trainval': start_idx, end_idx = 0, 105
        elif split == 'test': start_idx, end_idx = 105, 150
        else: start_idx, end_idx = 0, 150
        img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))[start_idx:end_idx]
        poses = sorted(glob.glob(os.path.join(self.root_dir, '*.json')))[start_idx:end_idx]

        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path, pose in tqdm(zip(img_paths, poses)):
            with open(pose, 'r') as f:
                p = json.load(f)['camera_data']
            c2w = np.array(p['cam2world']).T[:3]
            c2w[:, 1:3] *= -1
            if 'bricks' in self.root_dir:
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # bound in [-0.5, 0.5]
            self.poses += [c2w]

            img = read_image(img_path, self.img_wh)
            self.rays += [img]

        self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
