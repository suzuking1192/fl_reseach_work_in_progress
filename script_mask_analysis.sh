#!/bin/bash

python mask_analysis.py --num_users 10 --algorithm fedspa --dataset cifar10 --percentage 0.2 --top True --seed 3 --layers_from_last 1

