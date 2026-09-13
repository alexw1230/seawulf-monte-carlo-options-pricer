#!/bin/bash

#Flags
#Job Name
#SBATCH --job-name=mpi_hello

#Partition
#SBATCH -p short-40core

#Number of compute nodes
#SBATCH -N 2

#Number of slurm jobs per node
#SBATCH -n 8

#Wall time limit
#SBATCH -t 00:05:00

#Output file and error file
#SBATCH -o HPC_Logs/mpi_test_%j.out
#SBATCH -e HPC_Logs/mpi_test_%j.err

#Clear out the module bay
module purge

#Load needed modules
module load slurm
module load mpi4py/latest

#Execute
mpirun -n $SLURM_NTASKS python3 src/MPIBasicTest/mpi_test.py
