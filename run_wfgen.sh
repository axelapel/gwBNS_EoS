#!/bin/sh
#SBATCH -J Job_waveform_generation
#SBATCH -p skylake
#SBATCH -A a226
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-9%5
#SBATCH -t 2:00:00
#SBATCH -e ../log_wfgen/%N.%j.%a.err
#SBATCH -o ../log_wfgen/%N.%j.%a.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=axel.lapel@oca.eu

python3 wfgen.py --index $SLURM_ARRAY_TASK_ID