#!/bin/sh
#SBATCH -J Job_training_rNVP
#SBATCH -p kepler
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -t 16:00:00
#SBATCH -A a226
#SBATCH -e ../log_train/%N.%j.%a.err
#SBATCH -o ../log_train/%N.%j.%a.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=axel.lapel@oca.eu

python3 training.py