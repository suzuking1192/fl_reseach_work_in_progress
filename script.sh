#!/bin/bash

#SBATCH --job-name=yuto
#SBATCH --partition=math-alderaan
#SBATCH --time=10:00:00            # Max wall-clock time 1 hour
#SBATCH --ntasks=1                # number of cores 

python main.py   --dataset cifar10 --model lenet5 --ks 5 --in_ch 3 --nclass 2 --nsample_pc 20 --noniid --shard  --load_data   --lr 0.01   --momentum 0.5  --pruning_target 70 --dist_thresh 0.0001 --acc_thresh 50  --local_bs 32 --local_ep 10 --pruning_percent 20 --num_users 10 --rounds 100 --frac 1 --algorithm fedavg --print_freq 50 --alpha 0.2 --delta_r 10 --seed 3 --load_initial  --layer_wise_fill_weights True --parameter_to_multiply_avg 0.8 --regrowth_param 0.8 