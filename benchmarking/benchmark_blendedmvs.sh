#!/bin/bash

export ROOT_DIR=/home/ubuntu/data/nerf_data/BlendedMVS

python train.py \
    --root_dir $ROOT_DIR/Jade \
    --exp_name Jade --no_save_test

python train.py \
    --root_dir $ROOT_DIR/Fountain \
    --exp_name Fountain --no_save_test

python train.py \
    --root_dir $ROOT_DIR/Character \
    --exp_name Character --no_save_test

python train.py \
    --root_dir $ROOT_DIR/Statues \
    --exp_name Statues --no_save_test
