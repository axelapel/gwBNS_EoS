#!/bin/sh
#SBATCH -J Job_training_rNVP
#SBATCH -p volta
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 16:00:00
#SBATCH -e ../log_train/%N.%j.%a.err
#SBATCH -o ../log_train/%N.%j.%a.out

python3 training.py