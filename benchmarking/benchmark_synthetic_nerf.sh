#!/bin/bash

export ROOT_DIR=/home/ubuntu/data/nerf_data/Synthetic_NeRF

python train.py \
    --root_dir $ROOT_DIR/Chair \
    --exp_name Chair --no_save_test

python train.py \
    --root_dir $ROOT_DIR/Drums \
    --exp_name Drums --no_save_test

python train.py \
    --root_dir $ROOT_DIR/Ficus \
    --exp_name Ficus --no_save_test

python train.py \
    --root_dir $ROOT_DIR/Hotdog \
    --exp_name Hotdog --no_save_test

python train.py \
    --root_dir $ROOT_DIR/Lego \
    --exp_name Lego --no_save_test

python train.py \
    --root_dir $ROOT_DIR/Materials \
    --exp_name Materials --no_save_test

python train.py \
    --root_dir $ROOT_DIR \
    --exp_name Mic --no_save_test

python train.py \
    --root_dir $ROOT_DIR \
    --exp_name Ship --no_save_test