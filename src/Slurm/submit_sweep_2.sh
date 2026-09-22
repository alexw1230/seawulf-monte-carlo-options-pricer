#!/bin/bash
# Run from the repo root on milan1/milan2/xeonmax.
sbatch --nodes=1 --ntasks=96  src/Slurm/sweep_v2.sh
echo "submitted P=1..96 (1 node)"

sbatch --nodes=2 --ntasks=192 --ntasks-per-node=96 src/Slurm/sweep_v2.sh
echo "submitted P=192 (2 nodes)"
