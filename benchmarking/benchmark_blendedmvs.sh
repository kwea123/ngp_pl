#!/bin/bash

export ROOT_DIR=/home/ubuntu/data/nerf_data/BlendedMVS

python train.py \
    --root_dir $ROOT_DIR/Jade \
    --exp_name Jade --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Fountain \
    --exp_name Fountain --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Character \
    --exp_name Character --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Statues \
    --exp_name Statues --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips
