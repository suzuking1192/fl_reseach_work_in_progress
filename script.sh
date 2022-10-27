#!/bin/bash

#SBATCH --job-name=yuto
#SBATCH --partition=math-alderaan
#SBATCH --time=90:00:00            # Max wall-clock time 90 hour
#SBATCH --ntasks=1                # number of cores 

python main.py  --n_layer 5  --dataset cifar10 --model lenet5 --ks 5 --in_ch 3 --nclass 2 --nsample_pc 20 --noniid --shard  --load_data   --lr 0.01  --momentum 0.5  --pruning_target 70 --dist_thresh 0.0001 --acc_thresh 50  --local_bs 32 --local_ep 10 --pruning_percent 20 --num_users 3 --rounds 100 --frac 1 --algorithm create_data_to_learn_positive_transfer   --print_freq 300 --alpha 0.5 --delta_r 10 --seed 1 --load_initial   --lambda_value 1 --regrowth_param 1 --n_cluster 2  --mask_sparse_initialization False