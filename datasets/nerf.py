import cv2
import torch
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset


class MGTVDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.scene = kwargs['scene'] # "F1_06"
        self.take = kwargs['take'] # "000000"
        self.read_meta(split)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        for cam in range(92):
            xml_path = os.path.join(self.root_dir, 
                "camera_parameters", self.scene, str(cam+1), "intrinsic.xml")
            fs = cv2.FileStorage(xml_path, cv2.FileStorage_READ)
            K = fs.getNode('M').mat()
            K[:2] *= self.downsample
            D = fs.getNode('D').mat()

            xml_path = os.path.join(self.root_dir, 
                "camera_parameters", self.scene, str(cam+1), "extrinsic.xml")
            fs = cv2.FileStorage(xml_path, cv2.FileStorage_READ)
            R = fs.getNode('R').mat()
            T = fs.getNode('T').mat()

        self.Ks = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split):

        with open(os.path.join(self.root_dir, f"transforms_{split}.json"), 'r') as f:
            meta = json.load(f)

        print(f'Loading {len(meta["frames"])} {split} images ...')
        for frame in tqdm(meta['frames']):
            c2w = np.array(frame['transform_matrix'])[:3, :4]
            c2w[:, 1:3] *= -1
            c2w[:, 3] /= np.linalg.norm(c2w[:, 3])/1.5
            self.poses += [c2w]

            img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            img = read_image(img_path, self.img_wh)
            self.rays += [img]

        self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
