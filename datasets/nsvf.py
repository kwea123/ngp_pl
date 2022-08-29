import torch
import glob
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset


class NSVFDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            xyz_min, xyz_max = \
                np.loadtxt(os.path.join(root_dir, 'bbox.txt'))[:6].reshape(2, 3)
            self.shift = (xyz_max+xyz_min)/2
            self.scale = (xyz_max-xyz_min).max()/2 * 1.05 # enlarge a little

            # hard-code fix the bound error for some scenes...
            if 'Mic' in self.root_dir: self.scale *= 1.2
            elif 'Lego' in self.root_dir: self.scale *= 1.1

            self.read_meta(split)

    def read_intrinsics(self):
        if 'Synthetic' in self.root_dir or 'Ignatius' in self.root_dir:
            with open(os.path.join(self.root_dir, 'intrinsics.txt')) as f:
                fx = fy = float(f.readline().split()[0]) * self.downsample
            if 'Synthetic' in self.root_dir:
                w = h = int(800*self.downsample)
            else:
                w, h = int(1920*self.downsample), int(1080*self.downsample)

            K = np.float32([[fx, 0, w/2],
                            [0, fy, h/2],
                            [0,  0,   1]])
        else:
            K = np.loadtxt(os.path.join(self.root_dir, 'intrinsics.txt'),
                           dtype=np.float32)[:3, :3]
            if 'BlendedMVS' in self.root_dir:
                w, h = int(768*self.downsample), int(576*self.downsample)
            elif 'Tanks' in self.root_dir:
                w, h = int(1920*self.downsample), int(1080*self.downsample)
            K[:2] *= self.downsample

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'test_traj': # BlendedMVS and TanksAndTemple
            if 'Ignatius' in self.root_dir:
                poses_path = \
                    sorted(glob.glob(os.path.join(self.root_dir, 'test_pose/*.txt')))
                poses = [np.loadtxt(p) for p in poses_path]
            else:
                poses = np.loadtxt(os.path.join(self.root_dir, 'test_traj.txt'))
                poses = poses.reshape(-1, 4, 4)
            for pose in poses:
                c2w = pose[:3]
                c2w[:, 0] *= -1 # [left down front] to [right down front]
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                self.poses += [c2w]
        else:
            if split == 'train': prefix = '0_'
            elif split == 'trainval': prefix = '[0-1]_'
            elif split == 'trainvaltest': prefix = '[0-2]_'
            elif split == 'val': prefix = '1_'
            elif 'Synthetic' in self.root_dir: prefix = '2_' # test set for synthetic scenes
            elif split == 'test': prefix = '1_' # test set for real scenes
            else: raise ValueError(f'{split} split not recognized!')
            img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'rgb', prefix+'*.png')))
            poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', prefix+'*.txt')))

            print(f'Loading {len(img_paths)} {split} images ...')
            for img_path, pose in tqdm(zip(img_paths, poses)):
                c2w = np.loadtxt(pose)[:3]
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                self.poses += [c2w]

                img = read_image(img_path, self.img_wh)
                if 'Jade' in self.root_dir or 'Fountain' in self.root_dir:
                    # these scenes have black background, changing to white
                    img[torch.all(img<=0.1, dim=-1)] = 1.0

                self.rays += [img]

            self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
