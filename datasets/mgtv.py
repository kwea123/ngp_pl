import cv2
import glob
import torch
import numpy as np
import imageio
import os
from einops import rearrange
from tqdm import tqdm

from .base import BaseDataset


class MGTVDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.scene = kwargs['scene'] # "F1_06"
        self.take = kwargs['take'] # "000000"
        self.read_meta(split)

    def read_meta(self, split):
        self.rays = {}
        self.poses = []

        self.Hs, self.Ws = [], []
        self.Ds, self.Ks = [], []

        for cam in range(92): # read all cameras
            xml_path = os.path.join(self.root_dir,
                "camera_parameters", self.scene, str(cam+1), "intrinsic.xml")
            fs = cv2.FileStorage(xml_path, cv2.FileStorage_READ)
            K = fs.getNode('M').mat()
            if K[0, 0] < 4000: # hack to get image height and width
                self.Hs += [int(2048*self.downsample)]
                self.Ws += [int(2592*self.downsample)]
            else:
                self.Hs += [int(3072*self.downsample)]
                self.Ws += [int(4096*self.downsample)]
            K[:2] *= self.downsample
            self.Ks += [K]
            self.Ds += [fs.getNode('D').mat()]

            xml_path = os.path.join(self.root_dir, 
                "camera_parameters", self.scene, str(cam+1), "extrinsics.xml")
            fs = cv2.FileStorage(xml_path, cv2.FileStorage_READ)
            R = fs.getNode('R').mat()
            T = fs.getNode('T').mat() # in meters

            w2c = np.eye(4)
            w2c[:3] = np.concatenate([R, T], 1) # (3, 4)
            c2w = np.linalg.inv(w2c)[:3]
            if self.scene=='M3_02':
                # special case, large scene
                c2w[:, 3] /= 3.3
            else:
                c2w[:, 3] /= 2
            c2w[2, 3] += 0.45
            self.poses += [c2w]
        self.Ks = torch.FloatTensor(np.array(self.Ks))
        self.poses = torch.FloatTensor(np.array(self.poses)) # (92, 3, 4)

        img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                            split, self.scene, self.take, '*.png')))

        if not img_paths: return
        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path in tqdm(img_paths):
            filename = img_path.split('/')[-1]
            cam = int(filename[9:11])-1

            img = imageio.imread(img_path).astype(np.float32)/255.0
            img[..., :3] = img[..., :3]*img[..., -1:]

            img = cv2.resize(img, (self.Ws[cam], self.Hs[cam]))
            img = rearrange(img, 'h w c -> (h w) c')
            img = torch.FloatTensor(img)

            self.rays[cam] = img