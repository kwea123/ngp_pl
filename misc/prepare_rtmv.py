import imageio
import glob
import sys
from tqdm import tqdm
import os
import numpy as np
sys.path.append('datasets')
from color_utils import linear_to_srgb

import warnings; warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # convert hdr images to ldr by applying linear_to_srgb and clamping tone-mapping
    # and save into images/ folder to accelerate reading
    root_dir = sys.argv[1]
    envs = sorted(os.listdir(root_dir))
    print('Generating ldr images from hdr images ...')
    for env in tqdm(envs):
        for scene in tqdm(sorted(os.listdir(os.path.join(root_dir, env)))):
            os.makedirs(os.path.join(root_dir, env, scene, 'images'), exist_ok=True)
            for i, img_p in enumerate(tqdm(sorted(glob.glob(os.path.join(root_dir, env, scene, '*[0-9].exr'))))):
                img = imageio.imread(img_p) # hdr
                img[..., :3] = linear_to_srgb(img[..., :3])
                img = (255*img).astype(np.uint8)
                imageio.imsave(os.path.join(root_dir, env, scene, f'images/{i:05d}.png'), img)