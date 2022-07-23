#!/bin/bash

python main.py --algorithm ours --dataset cifar10 --model lenet5 --ks 5 --in_ch 3 --nclass 2 --nsample_pc 20 --noniid --shard --load_data  --num_users 13 --rounds 25 --frac 0.3 --local_bs 10 --local_ep 5 --lr 0.01 --momentum 0.5 --pruning_percent 10 --pruning_target 30 --dist_thresh 0.0001 --acc_thresh 50 --is_print --print_freq 100