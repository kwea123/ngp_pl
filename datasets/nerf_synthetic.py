import torch
import json
import numpy as np
import os
from PIL import Image
from einops import rearrange

from .ray_utils import *

from .base import BaseDataset


class NeRFSyntheticDataset(BaseDataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800), **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()
        if split.startswith('train'):
            rays_train = self.read_meta('train')
            if split == 'trainval':
                rays_val = self.read_meta('val')
                self.rays = torch.cat(list(rays_train.values())+
                                      list(rays_val.values()))
            else:
                self.rays = torch.cat(list(rays_train.values()))
        else: # val, test
            self.rays = self.read_meta(split)

    def read_meta(self, split):
        with open(os.path.join(self.root_dir,
                        f"transforms_{split}.json"), 'r') as f:
            meta = json.load(f)

        w, h = self.img_wh
        fx = fy = 0.5*800/np.tan(0.5*meta['camera_angle_x'])
        fx *= w/800; fy *= h/800

        # intrinsics, common for all images
        K = np.float32([[fx, 0, w/2],
                        [0, fy, h/2],
                        [0,  0,   1]])

        directions = get_ray_directions(h, w, K)

        rays = {} # {frame_idx: ray tensor}
        for idx in range(len(meta['frames'])):
            frame = meta['frames'][idx]
            c2w = torch.Tensor(frame['transform_matrix'])[:3, :4]
            c2w[:, 3] *= 0.6 # to bound the scene inside [-1, 1]
            rays_o, rays_d = get_rays(directions, c2w)

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, h, w)
            img = rearrange(img, 'c h w -> (h w) c', c=4) # RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            rays[idx] = torch.cat([rays_o, rays_d, img], 1) # (h*w, 9)

        return rays

    
