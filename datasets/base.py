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
        cam = np.random.choice(list(self.rays.keys()), 1)[0]
        pix_idxs = np.random.choice(len(self.rays[cam]), self.batch_size)
        rays = self.rays[cam][pix_idxs]

        sample = {'cam': cam,
                  'pix_idxs': pix_idxs,
                  'rgb': rays[:, :3],
                  'alpha': rays[:, 3]}
        return sample