from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .rtmv import RTMVDataset
from .nerfpp import NeRFPPDataset


dataset_dict = {'nsvf': NSVFDataset,
                'colmap': ColmapDataset,
                'rtmv': RTMVDataset,
                'nerfpp': NeRFPPDataset}