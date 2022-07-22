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


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses, pts3d):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of 3d point cloud.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = pts3d.mean(0)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses, pts3d):
    """
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3) reconstructed point cloud

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pts3d_centered: (N, 3) centered point cloud
    """

    pose_avg = average_poses(poses, pts3d) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_inv = np.linalg.inv(pose_avg_homo)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = pose_avg_inv @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    pts3d_centered = pts3d @ pose_avg_inv[:, :3].T + pose_avg_inv[:, 3:].T

    return poses_centered, pts3d_centered


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

        # use every 8th image as test set
        if split=='train':
            img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i%8!=0])
        elif split=='test':
            img_paths = [x for i, x in enumerate(img_paths) if i%8==0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i%8==0])

        print(f'Loading {len(img_paths)} {split} images ...')
        rays = {} # {frame_idx: ray tensor}
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