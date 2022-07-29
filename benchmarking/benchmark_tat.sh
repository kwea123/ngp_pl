#!/bin/bash

export ROOT_DIR=/home/ubuntu/data/nerf_data/TanksAndTemple
export DOWNSAMPLE=0.5 # to avoid OOM

python train.py \
    --root_dir $ROOT_DIR/Ignatius \
    --exp_name Ignatius --downsample $DOWNSAMPLE --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Truck \
    --exp_name Truck --downsample $DOWNSAMPLE --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Barn \
    --exp_name Barn --downsample $DOWNSAMPLE --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Caterpillar \
    --exp_name Caterpillar --downsample $DOWNSAMPLE --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Family \
    --exp_name Family --downsample $DOWNSAMPLE --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips
