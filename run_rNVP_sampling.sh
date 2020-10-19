#!/bin/sh
#SBATCH -J Job_rNVP_sampling
#SBATCH -p skylake
#SBATCH -A a226
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -t 2:00:00
#SBATCH -e ../log_mcmc/%N.%j.%a.err
#SBATCH -o ../log_mcmc/%N.%j.%a.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=axel.lapel@oca.eu

python3 rNVP_sampling.py 