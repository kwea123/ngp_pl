from .nsvf import NSVFDataset
from .colmap import ColmapDataset


dataset_dict = {'nsvf': NSVFDataset,
                'colmap': ColmapDataset}