#!/bin/bash

#SBATCH --job-name=yuto
#SBATCH --partition=math-alderaan
#SBATCH --time=10:00:00            # Max wall-clock time 1 hour
#SBATCH --ntasks=1                # number of cores 

python main.py   --dataset cifar10 --model lenet5 --ks 5 --in_ch 3 --nclass 2 --nsample_pc 20 --noniid --shard --load_data    --lr 0.01   --momentum 0.5  --pruning_target 70 --dist_thresh 0.0001 --acc_thresh 50 --is_print --local_bs 32 --local_ep 5 --pruning_percent 20 --num_users 100 --rounds 1000 --frac 0.1 --algorithm ours --print_freq 50 --alpha 0.1 --delta_r 20 --load_initial --layer_wise_fill_weights True 