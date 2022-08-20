import os
from tqdm import tqdm
import torch
import os
import numpy as np
from models.networks import NGP
from models.rendering import render
from datasets import dataset_dict
from datasets.ray_utils import get_ray_directions, get_rays
from utils import load_ckpt
import cc3d
import imageio
from kornia.utils.grid import create_meshgrid3d
import vren


if __name__ == '__main__':
    root_dir = '/home/ubuntu/hdd/data/mgtv'
    rect_text_path = '/home/ubuntu/hdd/data/mgtv/evaluation_code/test_b_rect.txt'

    split_dir = f'{root_dir}/test_b_'
    scenes = sorted(os.listdir(split_dir))
    for scene in tqdm(scenes):
        takes = sorted(os.listdir(f'{split_dir}/{scene}'))
        for take in tqdm(takes):
            os.makedirs(f'results/mgtv_test_b/{scene}/{take}', exist_ok=True)
            with open(rect_text_path, 'r') as f:
                lines = f.readlines()
            crop_dict = {}
            for line in lines:
                filename, x, y, w, h = line.split()
                w = int(w)
                h = int(h)
                scene_, take_, filename_ = filename.split('/')
                if scene_==scene and take_==take[:6]:
                    cam = int(filename_[9:11])-1
                    crop_dict[cam] = (int(x), int(y), w, h)

            dataset = dataset_dict['mgtv'](
                root_dir, scene=scene, take=take,
                split='test', downsample=1.0)

            model = NGP(scale=0.5, use_a=scene[0]=='F').cuda()
            load_ckpt(model, f'ckpts/mgtv/{scene}/{take}/epoch=19_slim.ckpt')

            G = model.grid_size
            xyz = create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3)
            indices = vren.morton3D(xyz.cuda()).long().cpu().numpy()

            _density_bitfield = model.density_bitfield
            density_bitfield = torch.zeros(model.cascades*G**3//8, 8, dtype=torch.bool)
            for i in range(8):
                density_bitfield[:, i] = _density_bitfield & torch.tensor([2**i], device='cuda')
            density_bitfield = density_bitfield.reshape(model.cascades, G**3).cpu().numpy()

            largest_connected_components = \
                cc3d.largest_k(density_bitfield[0, indices].reshape(G, G, G),
                            k=2 if scene=='M3_02' else 1, connectivity=6)
            new_density_grid = torch.zeros(model.cascades, G**3, device='cuda')
            new_density_grid[0, indices] = \
                torch.cuda.FloatTensor(largest_connected_components.reshape(model.cascades, -1).astype(np.float32))

            # update
            vren.packbits(new_density_grid, 0.5, model.density_bitfield)

            _density_bitfield = model.density_bitfield
            density_bitfield = torch.zeros(model.cascades*G**3//8, 8, dtype=torch.bool)
            for i in range(8):
                density_bitfield[:, i] = _density_bitfield & torch.tensor([2**i], device='cuda')
            density_bitfield = density_bitfield.reshape(model.cascades, G**3).cpu()

            for cam in tqdm(range(92)):
                directions = \
                    get_ray_directions(dataset.Hs[cam],
                                       dataset.Ws[cam],
                                       dataset.Ks[cam],
                                       flatten=False)
                if cam in crop_dict:
                    x, y, w, h = crop_dict[cam]
                    directions = directions[y:y+h, x:x+w]
                else: continue
                directions = directions.reshape(-1, 3)
                rays_o, rays_d = get_rays(directions.cuda(), dataset.poses[cam].cuda())
                
                results = render(model, rays_o, rays_d, 
                                **{'test_time': True,
                                   'cam': torch.cuda.LongTensor([0])})

                pred = results['rgb'].reshape(h, w, 3).cpu().numpy()
                pred = (pred*255).astype(np.uint8)
                imageio.imwrite(f'results/mgtv_test_b/{scene}/{take}/image.cam{cam+1:02d}_{take}.jpg', pred)
                
                torch.cuda.empty_cache()