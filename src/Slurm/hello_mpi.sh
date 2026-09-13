#!/bin/bash
#SBATCH --job-name=mpi_hello
#SBATCH -p short-40core
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -t 00:05:00
#SBATCH -o HPC_Logs/mpi_test_%j.out
#SBATCH -e HPC_Logs/mpi_test_%j.err

#Clear out the module bay
module purge

#Load needed modules
module load slurm
module load mpi4py/latest

#Execute
mpirun -n $SLURM_NTASKS python3 src/MPIBasicTest/mpi_test.py
