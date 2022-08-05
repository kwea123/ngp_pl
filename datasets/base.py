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

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        # uniform from all pixels
        pix_idxs = np.random.choice(len(self.rays), self.batch_size)
        rays = self.rays[pix_idxs]

        # bg_idxs = np.random.choice(len(self.rays_bg), int(self.batch_size*0.1))
        # fg_idxs = np.random.choice(len(self.rays_fg), int(self.batch_size*0.9))
        # rays = torch.cat([self.rays_bg[bg_idxs], self.rays_fg[fg_idxs]])

        sample = {'rays_o': rays[:, :3],
                    'rays_d': rays[:, 3:6],
                    'rgb': rays[:, 6:9],
                    'alpha': rays[:, 9]}
        return sample