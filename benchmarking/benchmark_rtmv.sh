#!/bin/bash

# for other environments, change the paths accordingly
# for amazon_berkely, set scale=1.0
export ROOT_DIR=/home/ubuntu/hdd/data/RTMV/bricks

python train.py \
    --root_dir $ROOT_DIR/4_Privet_Drive \
    --exp_name 4_Privet_Drive --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Action_Comics_#1_Superman \
    --exp_name Action_Comics_#1_Superman --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Buried_Treasure! \
    --exp_name Buried_Treasure! --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Fire_temple \
    --exp_name Fire_temple --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/First_Order_Star_Destroyer \
    --exp_name First_Order_Star_Destroyer --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Four_Weapons_Blacksmith \
    --exp_name Four_Weapons_Blacksmith --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/NASA_apollo_lunar_excursion_module \
    --exp_name NASA_apollo_lunar_excursion_module --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Night_Fury_Dragon_-_Lego_Elves_Style \
    --exp_name Night_Fury_Dragon_-_Lego_Elves_Style --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/Oak_Tree \
    --exp_name Oak_Tree --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

python train.py \
    --root_dir $ROOT_DIR/V8 \
    --exp_name V8 --no_save_test \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips