#!/bin/bash

export ROOT_DIR=/home/ubuntu/data/nerf_data/Synthetic_NSVF

python train.py \
    --root_dir $ROOT_DIR/Wineholder \
    --exp_name Wineholder --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Steamtrain \
    --exp_name Steamtrain --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Toad \
    --exp_name Toad --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Robot \
    --exp_name Robot --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Bike \
    --exp_name Bike --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 1e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Palace \
    --exp_name Palace --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Spaceship \
    --exp_name Spaceship --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Lifestyle \
    --exp_name Lifestyle --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips