#!/bin/bash
#SBATCH --job-name=gpu_pricer
#SBATCH -p a100
#SBATCH --gpus=1
#SBATCH -t 00:20:00
#SBATCH -o HPC_Logs/gpu_step2_%j.out
#SBATCH -e HPC_Logs/gpu_step2_%j.err
# Submit from the repo root:  sbatch src/Slurm/gpu_step2.sh

module purge
module load slurm
module load mpi4py/latest
module load cuda120/toolkit/12.0
source /gpfs/scratch/awiegand/venvs/gpu/bin/activate
export CUPY_CACHE_DIR=/gpfs/scratch/awiegand/.cupy/kernel_cache

ROOT=$SLURM_SUBMIT_DIR
mkdir -p "$ROOT/gpu/csv"
cd "$ROOT/src" || exit 1

log () { echo "[$(date +%H:%M:%S)] $*"; }
log "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

HEADER="P,N,call,call_se,put,put_se,compute_s,comm_s,total_s"

# 1. One readable run first: price, SE, throughput, estimated bandwidth.
log "Readable run, N=1e10:"
python3 -m gpu.gpu_pricer --n 10000000000

# 2. N sweep (validation + runtime vs N), default batch 1e8, 5 reps.
OUT="$ROOT/gpu/csv/step2_nsweep_${SLURM_JOB_ID}.csv"
echo "$HEADER" > "$OUT"
for N in 1000000 10000000 100000000 1000000000 10000000000; do
    for i in 1 2 3 4 5; do
        python3 -m gpu.gpu_pricer --n "$N" --csv >> "$OUT"
    done
    log "N=$N done (last: $(tail -1 "$OUT" | cut -d, -f7)s)"
done

# 3. Batch sweep at N=1e10: does the GPU want big batches like we expect?
OUT="$ROOT/gpu/csv/step2_batch_${SLURM_JOB_ID}.csv"
echo "batch,$HEADER" > "$OUT"
for B in 1000000 10000000 100000000 500000000; do
    for i in 1 2 3; do
        LINE=$(python3 -m gpu.gpu_pricer --n 10000000000 --batch "$B" --csv)
        echo "$B,$LINE" >> "$OUT"
    done
    log "batch=$B done (last: $(echo "$LINE" | cut -d, -f7)s)"
done

log "All done."