import os


if __name__ == '__main__':
    root_dir = '/home/ubuntu/hdd/data/mgtv'

    split_dir = f'{root_dir}/test_b_'
    scenes = sorted(os.listdir(split_dir))
    for scene in scenes:
        takes = sorted(os.listdir(f'{split_dir}/{scene}'))
        for take in takes:
            print(f'training {scene} {take}')
            cmd = f'python train.py --root_dir {root_dir}'
            cmd += f' --split test_b_ --scene {scene} --take {take}'
            if scene[0] == 'F':
                cmd += ' --use_a --optimize_ext'
            os.system(cmd)
