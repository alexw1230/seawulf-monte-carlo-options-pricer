#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH -p short-40core
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -t 2:00:00
#SBATCH -o HPC_Logs/sweep_%j.out
#SBATCH -e HPC_Logs/sweep_%j.err

module purge
module load slurm
module load mpi4py/latest
cd src

P=$SLURM_NTASKS
RESULTS=../csv/results_P${P}.csv

N_VALUES=(100000 1000000 10000000 100000000 1000000000 10000000000)
 
for N in "${N_VALUES[@]}"; do
    echo "$N"
    mpirun -n $SLURM_NTASKS python3 -m Seawulf.pricer --n "$N" --csv >> "$RESULTS"
done
