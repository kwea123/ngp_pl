from torchvision import transforms as T
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self):
        pass

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.rays)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            idxs = np.random.choice(len(self.rays), self.batch_size)
            sample = {'rays': self.rays[idxs, :6],
                      'rgb': self.rays[idxs, 6:9],
                      'idxs': idxs}
        else:
            sample = {'rays': self.rays[idx][:, :6],
                      'rgb': self.rays[idx][:, 6:9]}

        return sample