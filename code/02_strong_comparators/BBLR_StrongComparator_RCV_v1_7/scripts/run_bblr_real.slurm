#!/bin/bash
#SBATCH --job-name=bblr_real
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --time=16:00:00
#SBATCH --output=output%j.log

module load gnu12
module load openmpi4

source ~/.bashrc

micromamba activate mpi_env

python py_real.py --dataset_name 'house_sales' --test_size 0.2 --d_add 0 --subset_size 5000







