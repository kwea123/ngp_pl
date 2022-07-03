#!/bin/bash

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/Synthetic_NeRF/Chair \
    --exp_name Chair

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/Synthetic_NeRF/Drums \
    --exp_name Drums

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/Synthetic_NeRF/Ficus \
    --exp_name Ficus

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/Synthetic_NeRF/Hotdog \
    --exp_name Hotdog

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/Synthetic_NeRF/Lego \
    --exp_name Lego

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/Synthetic_NeRF/Materials \
    --exp_name Materials

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/Synthetic_NeRF/Mic \
    --exp_name Mic

python train.py \
    --root_dir /home/ubuntu/data/nerf_data/Synthetic_NeRF/Ship \
    --exp_name Ship