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
            return 1000 * self.batch_size
        return len(self.rays)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            idx = np.random.randint(len(self.rays))
            sample = {'rays': self.rays[idx, :6],
                      'rgb': self.rays[idx, 6:9],
                      'idx': idx}
        else:
            sample = {'rays': self.rays[idx][:, :6],
                      'rgb': self.rays[idx][:, 6:9]}

        return sample