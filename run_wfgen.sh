#!/bin/sh
#SBATCH -J Job_waveform_generation
#SBATCH -p skylake
#SBATCH -A a226
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00
#SBATCH -e /scratch/log_wfgen/%N.%j.%a.err
#SBATCH -o /scratch/log_wfgen/%N.%j.%a.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=axel.lapel@oca.eu

python3 wfgen.py 