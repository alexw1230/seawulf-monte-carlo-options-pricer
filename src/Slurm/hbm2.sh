#!/bin/bash
#SBATCH --job-name=hbm_ddr_v2
#SBATCH -p hbm-short-96core
#SBATCH -N 1
#SBATCH --ntasks=96
#SBATCH -t 02:00:00
#SBATCH -o HPC_Logs/hbm_ddr_v2_%j.out
#SBATCH -e HPC_Logs/hbm_ddr_v2_%j.err

module purge
module load slurm
module load mpi4py/latest

cd src

RESULTS=../csv/hbm_vs_ddr_v2_${SLURM_JOB_ID}.csv
NUMA_LOG=../csv/numa_layout_${SLURM_JOB_ID}.txt

log () { echo "[$(date +%H:%M:%S)] $*"; }

numactl -H > "$NUMA_LOG"
log "Node: $(hostname)"
cat "$NUMA_LOG"

DDR_NODES=$(awk '/^node [0-9]+ cpus:/ {l=$0; sub(/^node [0-9]+ cpus: */,"",l); if (l!="") print $2}' "$NUMA_LOG" | paste -sd, -)
HBM_NODES=$(awk '/^node [0-9]+ cpus:/ {l=$0; sub(/^node [0-9]+ cpus: */,"",l); if (l=="") print $2}' "$NUMA_LOG" | paste -sd, -)
log "DDR (CPU-attached) NUMA nodes: $DDR_NODES"
log "HBM (memory-only) NUMA nodes:  $HBM_NODES"

if [ -z "$HBM_NODES" ] || [ -z "$DDR_NODES" ]; then
    log "ERROR: no memory-only NUMA nodes found. This node is probably not in"
    log "       flat mode (e.g. HBM configured as cache), so --membind to HBM"
    log "       is not possible here. See $NUMA_LOG. Exiting."
    exit 1
fi

echo "mem,P,N,call,call_se,put,put_se,compute_s,comm_s,total_s" > "$RESULTS"

run_case () {
    local LABEL=$1 MEMBIND=$2 NRANKS=$3 N=$4 REPS=$5
    for i in $(seq 1 "$REPS"); do
        log "start mem=$LABEL P=$NRANKS N=$N rep=$i"
        LINE=$(timeout 900 mpirun -n "$NRANKS" --bind-to core \
                 numactl --membind="$MEMBIND" -- \
                 python3 -m Seawulf.pricer --n "$N" --csv)
        STATUS=$?
        if [ $STATUS -ne 0 ] || [ -z "$LINE" ]; then
            log "FAILED mem=$LABEL P=$NRANKS rep=$i (exit $STATUS; 124 = timed out after 15 min)"
            return 1
        fi
        echo "${LABEL},${LINE}" >> "$RESULTS"
        log "done  mem=$LABEL P=$NRANKS rep=$i -> $(echo "$LINE" | cut -d, -f7)s compute"
    done
}

SMOKE_N=100000000   # 1e8: ~3s at P=1 on DDR
run_case ddr "$DDR_NODES" 1 $SMOKE_N 1 || exit 1
run_case hbm "$HBM_NODES" 1 $SMOKE_N 1 || { log "HBM smoke test failed; see .err"; exit 1; }

BIG_N=10000000000
run_case ddr "$DDR_NODES" 96 $BIG_N 5
run_case hbm "$HBM_NODES" 96 $BIG_N 5

run_case ddr "$DDR_NODES" 1 $BIG_N 2
run_case hbm "$HBM_NODES" 1 $BIG_N 2

log "All done. Results: $RESULTS"
