import torch
import numpy as np
import os
import cv2
from PIL import Image
from einops import rearrange
from tqdm import tqdm

from .ray_utils import *
from .depth_utils import read_pfm
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

from .base import BaseDataset


class ColmapDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        if split.startswith('train'):
            rays_train = self.read_meta('train')
            if split == 'trainval':
                rays_val = self.read_meta('test')
                self.rays = torch.cat(list(rays_train.values())+
                                      list(rays_val.values()))
            else:
                self.rays = torch.cat(list(rays_train.values()))
        else: # val, test
            self.rays = self.read_meta(split)

    def read_meta(self, split):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height*self.downsample)
        w = int(camdata[1].width*self.downsample)
        self.img_wh = (w, h)
        fx = fy = camdata[1].params[0]*self.downsample
        self.K = np.float32([[fx, 0, w/2],
                             [0, fy, h/2],
                             [0,  0,   1]])
        directions = get_ray_directions(h, w, self.K)

        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)
        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, 'images', name)
                     for name in sorted(img_names)]
        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices

        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d]) # (N, 3)

        self.poses, self.pts3d = center_poses(poses, pts3d)

        scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        self.poses[..., 3] /= scale
        self.pts3d /= scale

        # use precomputed test poses
        rays = {} # {frame_idx: ray tensor}
        if split == 'test_traj':
            self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean())
            for idx, pose in enumerate(self.poses):
                rays_o, rays_d = get_rays(directions, torch.cuda.FloatTensor(pose))

                rays[idx] = torch.cat([rays_o, rays_d], 1).cpu() # (h*w, 6)
            return rays

        # use every 8th image as test set
        if split=='train':
            img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i%8!=0])
        elif split=='test':
            img_paths = [x for i, x in enumerate(img_paths) if i%8==0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i%8==0])

        print(f'Loading {len(img_paths)} {split} images ...')
        for idx, pose in enumerate(tqdm(self.poses)):
            rays_o, rays_d = \
                get_rays(directions, torch.cuda.FloatTensor(pose))
            ray = [rays_o, rays_d]

            img_path = img_paths[idx]
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img).cuda() # (c, h, w)
            img = rearrange(img, 'c h w -> (h w) c')
            ray += [img]

            try: # try to read depth files if there are
                disp_path = img_path.replace('images', 'disps').split('.')[-2]+'.pfm'
                disp = read_pfm(disp_path)[0]
                disp = cv2.resize(disp, self.img_wh, interpolation=cv2.INTER_NEAREST)
                disp = torch.cuda.FloatTensor(disp).reshape(-1, 1)
                ray += [disp]
            except: pass

            rays[idx] = torch.cat(ray, 1).cpu()

        return rays