#!/bin/bash
#SBATCH --job-name=hbm_ddr_v3
#SBATCH -p hbm-short-96core
#SBATCH -N 1
#SBATCH --ntasks=96
#SBATCH -t 02:00:00
#SBATCH -o HPC_Logs/hbm_ddr_v3_%j.out
#SBATCH -e HPC_Logs/hbm_ddr_v3_%j.err

# Follows SeaWulf's documented HBM recipe:
#   module load numactl/2.0.16
#   mpirun --map-by numa numactl --preferred-many=8-15 ./a.out
# Submit from milan1/milan2/xeonmax.

module purge
module load slurm
module load mpi4py/latest
module load numactl/2.0.16

cd src

RESULTS=../csv/hbm_vs_ddr_v3_${SLURM_JOB_ID}.csv
NUMASTAT_LOG=../csv/numastat_${SLURM_JOB_ID}.txt

log () { echo "[$(date +%H:%M:%S)] $*"; }

log "Node: $(hostname)"
log "numactl: $(which numactl) ($(numactl --version 2>&1 | head -1))"
if ! numactl --help 2>&1 | grep -q -- "--preferred-many"; then
    log "ERROR: this numactl does not support --preferred-many."
    log "       Check 'module avail numactl' and fix the module load line."
    exit 1
fi

DDR_POLICY="--membind=0-7"          # strict DDR5 (worked fine in v2)
HBM_POLICY="--preferred-many=8-15"  # SeaWulf's recommended HBM policy

echo "mem,P,N,call,call_se,put,put_se,compute_s,comm_s,total_s" > "$RESULTS"
: > "$NUMASTAT_LOG"

# run_case LABEL "NUMACTL_POLICY" NRANKS N REPEATS SNAPSHOT_DELAY_S
run_case () {
    local LABEL=$1 POLICY=$2 NRANKS=$3 N=$4 REPS=$5 DELAY=$6
    for i in $(seq 1 "$REPS"); do
        log "start mem=$LABEL P=$NRANKS N=$N rep=$i"

        # Background snapshot of where the running ranks' memory lives.
        # Columns 0-7 = DDR5, 8-15 = HBM. This is the proof HBM was used.
        (
            sleep "$DELAY"
            {
                echo "===== mem=$LABEL P=$NRANKS N=$N rep=$i at $(date +%H:%M:%S) ====="
                numastat -p python3 2>&1 | tail -n 8
                echo
            } >> "$NUMASTAT_LOG"
        ) &
        SNAP_PID=$!

        LINE=$(timeout 900 mpirun -n "$NRANKS" --map-by numa --bind-to core \
                 numactl $POLICY -- \
                 python3 -m Seawulf.pricer --n "$N" --csv)
        STATUS=$?
        wait $SNAP_PID 2>/dev/null

        if [ $STATUS -ne 0 ] || [ -z "$LINE" ]; then
            log "FAILED mem=$LABEL P=$NRANKS rep=$i (exit $STATUS; 124 = timed out)"
            return 1
        fi
        echo "${LABEL},${LINE}" >> "$RESULTS"
        log "done  mem=$LABEL P=$NRANKS rep=$i -> $(echo "$LINE" | cut -d, -f7)s compute"
    done
}

# 1. Smoke test (~3s each). Snapshot at 1.5s, mid-run.
SMOKE_N=100000000
run_case ddr "$DDR_POLICY" 1 $SMOKE_N 1 1.5 || exit 1
run_case hbm "$HBM_POLICY" 1 $SMOKE_N 1 1.5 || { log "HBM smoke test failed; see .err"; exit 1; }

# 2. Full node (~10s per run). Snapshot at 5s.
BIG_N=10000000000
run_case ddr "$DDR_POLICY" 96 $BIG_N 5 5
run_case hbm "$HBM_POLICY" 96 $BIG_N 5 5

# 3. Single core, 2 repeats (~275s per run). Snapshot at 60s.
run_case ddr "$DDR_POLICY" 1 $BIG_N 2 60
run_case hbm "$HBM_POLICY" 1 $BIG_N 2 60

log "All done. Results: $RESULTS   Memory placement: $NUMASTAT_LOG"
