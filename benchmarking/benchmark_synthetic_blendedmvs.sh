#!/bin/bash

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/BlendedMVS/Jade \
    --exp_name Jade

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/BlendedMVS/Fountain \
    --exp_name Fountain

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/BlendedMVS/Character \
    --exp_name Character

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/BlendedMVS/Statues \
    --exp_name Statues
