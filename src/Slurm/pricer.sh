#!/bin/bash
#SBATCH --job-name=pricer
#SBATCH -p short-40core
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -t 00:05:00
#SBATCH -o HPC_Logs/pricer_%j.out
#SBATCH -e HPC_Logs/pricer_%j.err

module purge
module load slurm
module load mpi4py/latest

cd src

mpirun -n $SLURM_NTASKS python3 Seawulf/pricer.py
