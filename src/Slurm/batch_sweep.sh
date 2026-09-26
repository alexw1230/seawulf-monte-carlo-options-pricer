#!/bin/bash
#SBATCH --job-name=batch_sweep
#SBATCH -p hbm-short-96core
#SBATCH -N 1
#SBATCH --ntasks=96
#SBATCH -t 01:15:00
#SBATCH -o HPC_Logs/batch_sweep_%j.out
#SBATCH -e HPC_Logs/batch_sweep_%j.err

module purge
module load slurm
module load mpi4py/latest
module load numactl/2.0.16

cd src

log () { echo "[$(date +%H:%M:%S)] $*"; }

if ! grep -q "batch=args.batch" Seawulf/pricer.py; then
    log "ERROR: Seawulf/pricer.py does not pass batch=args.batch to run_chunk()."
    log "       Every run would silently use batch=1e6. Fix pricer.py first."
    exit 1
fi

RESULTS=../csv/batch_sweep_${SLURM_JOB_ID}.csv
echo "mem,batch,P,N,call,call_se,put,put_se,compute_s,comm_s,total_s" > "$RESULTS"
log "Node: $(hostname)"

DDR_POLICY="--membind=0-7"
HBM_POLICY="--preferred-many=8-15"

BATCHES=(1000 10000 30000 100000 1000000 10000000)

N_P1=1000000000     
N_P96=10000000000   
REPS_P1=2
REPS_P96=3

run_case () {
    local LABEL=$1 POLICY=$2 NRANKS=$3 N=$4 BATCH=$5 REPS=$6
    for i in $(seq 1 "$REPS"); do
        LINE=$(timeout 900 mpirun -n "$NRANKS" --map-by numa --bind-to core \
                 numactl $POLICY -- \
                 python3 -m Seawulf.pricer --n "$N" --batch "$BATCH" --csv)
        STATUS=$?
        if [ $STATUS -ne 0 ] || [ -z "$LINE" ]; then
            log "FAILED mem=$LABEL batch=$BATCH P=$NRANKS rep=$i (exit $STATUS)"
            continue
        fi
        echo "${LABEL},${BATCH},${LINE}" >> "$RESULTS"
        COMPUTE=$(echo "$LINE" | cut -d, -f7)
        RATE=$(awk -v n="$N" -v p="$NRANKS" -v t="$COMPUTE" \
               'BEGIN { printf "%.1f", n / t / p / 1e6 }')
        log "mem=$LABEL batch=$BATCH P=$NRANKS rep=$i -> ${COMPUTE}s  (${RATE}M trials/s per core)"
    done
}

for B in "${BATCHES[@]}"; do
    run_case ddr "$DDR_POLICY" 96 $N_P96 "$B" $REPS_P96
    run_case hbm "$HBM_POLICY" 96 $N_P96 "$B" $REPS_P96
done

for B in "${BATCHES[@]}"; do
    run_case ddr "$DDR_POLICY" 1 $N_P1 "$B" $REPS_P1
    run_case hbm "$HBM_POLICY" 1 $N_P1 "$B" $REPS_P1
done

log "All done. Results: $RESULTS"
