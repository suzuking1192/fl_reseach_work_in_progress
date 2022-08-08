#!/bin/bash

#SBATCH --job-name=yuto
#SBATCH --partition=math-alderaan
#SBATCH --time=90:00:00            # Max wall-clock time 90 hour
#SBATCH --ntasks=1                # number of cores 

python efficiency_analysis.py