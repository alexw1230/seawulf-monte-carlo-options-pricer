#!/bin/bash
#SBATCH --job-name=f32_test
#SBATCH -t 00:15:00
#SBATCH -o HPC_Logs/f32_%j.out
#SBATCH -e HPC_Logs/f32_%j.err

module purge
module load slurm
module load mpi4py/latest

cd src

P=$SLURM_NTASKS
RESULTS=../f32_P${P}.csv
echo "P,N,call,call_se,put,put_se,compute_s,comm_s,total_s" > "$RESULTS"

N=10000000000
REPEATS=5

for i in $(seq 1 $REPEATS); do
    mpirun -n $SLURM_NTASKS python3 -m Seawulf.pricer_f32 --n "$N" --csv >> "$RESULTS"
done