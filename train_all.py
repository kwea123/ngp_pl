import os
import glob


if __name__ == '__main__':
    split = 'test_a_'
    root_dir = f'/home/ubuntu/hdd/data/mgtv/{split}'

    scenes = sorted(os.listdir(root_dir))
    for scene in scenes:
        takes = sorted(os.listdir(f'{root_dir}/{scene}'))
        for take in takes:
            print(f'training {scene} {take}')
            cmd = 'python train.py --root_dir /home/ubuntu/hdd/data/mgtv'
            cmd += f' --split {split} --scene {scene} --take {take}'
            if scene[0] == 'F':
                cmd += ' --use_a'
            if scene == 'F1_06':
                cmd += ' --optimize_ext'
            os.system(cmd)

    # # copy GT files
    # root_dir = '/home/ubuntu/hdd/data/mgtv/val_crop_gt'

    # scenes = sorted(os.listdir(root_dir))
    # for scene in scenes:
    #     os.makedirs(f'/home/ubuntu/hdd/data/mgtv/gan/{scene}/gt', exist_ok=True)
    #     takes = sorted(os.listdir(f'{root_dir}/{scene}'))
    #     for take in takes:
    #         imgs = sorted(glob.glob(f'{root_dir}/{scene}/{take}/*.jpg'))
    #         for img in imgs:
    #             os.system(f'cp {img} /home/ubuntu/hdd/data/mgtv/gan/{scene}/gt/{img.split("/")[-1]}')