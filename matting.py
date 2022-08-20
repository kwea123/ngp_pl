import os


if __name__ == '__main__':
    root_dir = '/home/ubuntu/hdd/data/mgtv/test_b'

    scenes = sorted(os.listdir(root_dir))
    for scene in scenes:
        takes = sorted(os.listdir(f'{root_dir}/{scene}'))
        for take in takes:
            print(f'processsing {scene} {take}')
            os.system(f'''python tools/predict.py \
                --config configs/modnet/modnet-hrnet_w18.yml \
                --model_path ../modnet-hrnet_w18.pdparams \
                --image_path {root_dir}/{scene}/{take} \
                --save_dir {root_dir}_/{scene}/{take} \
                --fg_estimate True''')