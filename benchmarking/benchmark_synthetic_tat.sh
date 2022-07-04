#!/bin/bash

export DOWNSAMPLE=0.5 # to avoid OOM

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/TanksAndTemple/Ignatius \
    --exp_name Ignatius --downsample $DOWNSAMPLE

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/TanksAndTemple/Truck \
    --exp_name Truck --downsample $DOWNSAMPLE

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/TanksAndTemple/Barn \
    --exp_name Barn --downsample $DOWNSAMPLE

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/TanksAndTemple/Caterpillar \
    --exp_name Caterpillar --downsample $DOWNSAMPLE

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/TanksAndTemple/Family \
    --exp_name Family --downsample $DOWNSAMPLE
