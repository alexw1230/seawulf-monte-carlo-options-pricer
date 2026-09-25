#!/bin/bash
#SBATCH --job-name=gpu_step3
#SBATCH -p a100
#SBATCH --gpus=1
#SBATCH -t 00:20:00
#SBATCH -o HPC_Logs/gpu_step3_%j.out
#SBATCH -e HPC_Logs/gpu_step3_%j.err

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

# 1. Correctness first: the RNG must pass its known-answer tests, and the
#    GPU kernel must match the CPU reference before any timing means anything.
log "Philox known-answer tests:"
python3 gpu/philox_ref.py || { log "KAT failed; stopping"; exit 1; }
log "GPU kernel vs CPU reference:"
python3 -m gpu.fused_pricer --check || { log "GPU/CPU mismatch; stopping"; exit 1; }

# 2. One readable run.
log "Readable run, N=1e10:"
python3 -m gpu.fused_pricer --n 10000000000

# 3. N sweep, 5 reps, same N values as step 2 for direct comparison.
OUT="$ROOT/gpu/csv/step3_nsweep_${SLURM_JOB_ID}.csv"
echo "P,N,call,call_se,put,put_se,compute_s,comm_s,total_s" > "$OUT"
for N in 1000000 10000000 100000000 1000000000 10000000000 100000000000; do
    for i in 1 2 3 4 5; do
        python3 -m gpu.fused_pricer --n "$N" --csv >> "$OUT"
    done
    log "N=$N done (last: $(tail -1 "$OUT" | cut -d, -f7)s)"
done

log "All done."
