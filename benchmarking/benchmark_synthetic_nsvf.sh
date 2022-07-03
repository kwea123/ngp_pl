#!/bin/bash

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/Synthetic_NSVF/Wineholder \
    --exp_name Wineholder

# python train.py \
#     --root_dir /home/ubuntu/data/nerf_data/Synthetic_NSVF/Steamtrain \
#     --exp_name Steamtrain

# python train.py \
#     --root_dir /home/ubuntu/data/nerf_data/Synthetic_NSVF/Toad \
#     --exp_name Toad

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/Synthetic_NSVF/Robot \
    --exp_name Robot

# python train.py \
#     --root_dir /home/ubuntu/data/nerf_data/Synthetic_NSVF/Bike \
#     --exp_name Bike

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/Synthetic_NSVF/Palace \
    --exp_name Palace

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/Synthetic_NSVF/Spaceship \
    --exp_name Spaceship

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/Synthetic_NSVF/Lifestyle \
    --exp_name Lifestyle