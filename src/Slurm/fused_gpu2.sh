#!/bin/bash
#SBATCH --job-name=gpu_step3
#SBATCH -p a100
#SBATCH --gpus=1
#SBATCH -t 00:20:00
#SBATCH -o HPC_Logs/gpu_step3_%j.out
#SBATCH -e HPC_Logs/gpu_step3_%j.err
# Submit from the repo root:  sbatch src/Slurm/gpu/gpu_step3.sh
#
# Every stage must print evidence that it actually ran; an exit code of 0
# alone is not trusted (an empty .py file also exits 0).

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
die () { log "STOPPING: $*"; exit 1; }

log "Node: $(hostname)"
log "fused_pricer.py: $(wc -l < gpu/fused_pricer.py) lines, md5 $(md5sum gpu/fused_pricer.py | cut -c1-32)"
log "philox_ref.py:   $(wc -l < gpu/philox_ref.py) lines"

# 1. Philox known-answer tests: need exactly 3 OK lines.
log "Philox known-answer tests:"
KAT=$(python3 gpu/philox_ref.py 2>&1); echo "$KAT"
[ "$(grep -c ': OK$' <<< "$KAT")" -eq 3 ] || die "known-answer tests did not all pass"

# 2. GPU kernel vs CPU reference: need exactly 3 OK lines.
log "GPU kernel vs CPU reference:"
CHECK=$(python3 -m gpu.fused_pricer --check 2>&1); echo "$CHECK"
[ "$(grep -c ' OK$' <<< "$CHECK")" -eq 3 ] || die "GPU result does not match CPU reference"

# 3. One readable run: must report a time.
log "Readable run, N=1e10:"
RUN=$(python3 -m gpu.fused_pricer --n 10000000000 2>&1); echo "$RUN"
grep -q '^Time = ' <<< "$RUN" || die "readable run printed no timing"

# 4. N sweep, 5 reps. Every run must produce a CSV row.
OUT="$ROOT/gpu/csv/step3_nsweep_${SLURM_JOB_ID}.csv"
echo "P,N,call,call_se,put,put_se,compute_s,comm_s,total_s" > "$OUT"
for N in 1000000 10000000 100000000 1000000000 10000000000 100000000000; do
    for i in 1 2 3 4 5; do
        LINE=$(python3 -m gpu.fused_pricer --n "$N" --csv)
        [[ "$LINE" == 1,"$N",* ]] || die "N=$N rep=$i produced no CSV row (got: '$LINE')"
        echo "$LINE" >> "$OUT"
    done
    log "N=$N done (last: $(echo "$LINE" | cut -d, -f7)s)"
done

log "All done. Results: $OUT"
