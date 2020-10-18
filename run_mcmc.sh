#!/bin/sh
#SBATCH -J Job_mcmc_sampling
#SBATCH -p skylake
#SBATCH -A a226
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 16:00:00
#SBATCH -e ../log_mcmc/%N.%j.%a.err
#SBATCH -o ../log_mcmc/%N.%j.%a.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=axel.lapel@oca.eu

python3 mcmc_sampling.py 