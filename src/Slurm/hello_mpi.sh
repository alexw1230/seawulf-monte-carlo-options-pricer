#!/bin/bash
#SBATCH --job-name=mpi_hello
#SBATCH -p short-40core
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -t 00:05:00
#SBATCH -o mpi_test_%j.out
#SBATCH -e mpi_test_%j.err

module purge
module load mpi4py/latest

mpirun -n $SLURM_NTASKS python3 ../MPIBasicTest/mpi_test.py
