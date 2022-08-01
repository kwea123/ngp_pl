#!/bin/bash

export ROOT_DIR=/home/ubuntu/data/nerf_data/360_v2
export DOWNSAMPLE=0.25 # to avoid OOM

python train.py \
    --root_dir $ROOT_DIR/bicycle --dataset_name colmap \
    --exp_name bicycle --downsample $DOWNSAMPLE --no_save_test \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/bonsai --dataset_name colmap \
    --exp_name bonsai --downsample $DOWNSAMPLE --no_save_test \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/counter --dataset_name colmap \
    --exp_name counter --downsample $DOWNSAMPLE --no_save_test \
    --num_epochs 20 --scale 16.0 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/garden --dataset_name colmap \
    --exp_name garden --downsample $DOWNSAMPLE --no_save_test \
    --num_epochs 20 --scale 16.0 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/kitchen --dataset_name colmap \
    --exp_name kitchen --downsample $DOWNSAMPLE --no_save_test \
    --num_epochs 20 --scale 4.0 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/room --dataset_name colmap \
    --exp_name room --downsample $DOWNSAMPLE --no_save_test \
    --num_epochs 20 --scale 4.0 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/stump --dataset_name colmap \
    --exp_name stump --downsample $DOWNSAMPLE --no_save_test \
    --num_epochs 20 --batch_size 4096 --scale 64.0 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/flowers --dataset_name colmap \
    --exp_name bicycle --downsample $DOWNSAMPLE --no_save_test \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/treehill --dataset_name colmap \
    --exp_name bicycle --downsample $DOWNSAMPLE --no_save_test \
    --num_epochs 20 --batch_size 4096 --scale 64.0 --eval_lpips