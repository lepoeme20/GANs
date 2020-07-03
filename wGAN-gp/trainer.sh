#!/bin/bash

# To training AttackGAN

#=== Set parameters ===
epochs=200
batch_size=512
lr=0.0002
dataset='mnist' # cifar10 | cifar100 | mnist
data_path='/media/lepoeme20/Data/basics'
image_size=32
image_channels=1
n_gpus=1

# Training AttackGAN
echo "Training on ($dataset & $classifier)"
python main.py --gpus $n_gpus --distributed_backend ddp --batch_size $batch_size\
    --dataset $dataset --image_size $image_size --data-root-path $data_path\
    --channels $image_channels --lr $lr --epochs $epochs