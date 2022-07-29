from torchvision import transforms as T
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train', downsample=1.0):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.define_transforms()

    def read_intrinsics(self):
        raise NotImplementedError

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.poses)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            # randomly select images
            img_idxs = np.random.choice(len(self.poses), self.batch_size)
            # randomly select pixels
            pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            sample = {'rgb': self.rays[img_idxs, pix_idxs],
                      'img_idxs': img_idxs,
                      'pix_idxs': pix_idxs}
        else:
            sample = {'pose': self.poses[idx],
                      'img_idxs': idx}
            if len(self.rays)>0: # if ground truth rgb available
                sample['rgb'] = self.rays[idx]

        return sample