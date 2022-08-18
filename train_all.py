import os
import glob


if __name__ == '__main__':
    split = 'test_b_'
    root_dir = f'/home/ubuntu/hdd/data/mgtv/{split}'

    scenes = ['F2_07']#sorted(os.listdir(root_dir))
    for scene in scenes:
        takes = sorted(os.listdir(f'{root_dir}/{scene}'))
        for take in takes:
            print(f'training {scene} {take}')
            cmd = 'python train.py --root_dir /home/ubuntu/hdd/data/mgtv'
            cmd += f' --split {split} --scene {scene} --take {take}'
            if scene[0] == 'F':
                cmd += ' --use_a --optimize_ext'
            os.system(cmd)
