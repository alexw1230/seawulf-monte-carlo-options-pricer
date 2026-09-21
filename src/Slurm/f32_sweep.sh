#!/bin/bash
sbatch --partition=short-96core --nodes=1 --ntasks=1   src/Slurm/f32_single.sh
echo "submitted P=1 (float32)"

sbatch --partition=short-96core --nodes=1 --ntasks=96  src/Slurm/f32_single.sh
echo "submitted P=96 (float32)"

sbatch --partition=short-96core --nodes=2 --ntasks=192 src/Slurm/f32_single.sh
echo "submitted P=192 (float32)"
