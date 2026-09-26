#!/bin/bash
#SBATCH --job-name=gpu_check
#SBATCH -p a100
#SBATCH --gpus=1
#SBATCH -t 00:10:00
#SBATCH -o HPC_Logs/gpu_check_%j.out
#SBATCH -e HPC_Logs/gpu_check_%j.err

module purge
module load slurm
module load mpi4py/latest
module load cuda120/toolkit/12.0
source /gpfs/scratch/awiegand/venvs/gpu/bin/activate
export CUPY_CACHE_DIR=/gpfs/scratch/awiegand/.cupy/kernel_cache   # compiled GPU programs

echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo

python3 src/gpu/gpu_check.py