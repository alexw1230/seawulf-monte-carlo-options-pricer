#!/bin/bash
#SBATCH --job-name=spread_test
#SBATCH -p hbm-short-96core
#SBATCH -N 4
#SBATCH --ntasks-per-node=96
#SBATCH -t 00:40:00
#SBATCH -o HPC_Logs/spread_test_%j.out
#SBATCH -e HPC_Logs/spread_test_%j.err

module purge
module load slurm
module load mpi4py/latest
module load numactl/2.0.16

cd src

log () { echo "[$(date +%H:%M:%S)] $*"; }

if ! grep -q "batch=args.batch" Seawulf/pricer.py; then
    log "ERROR: pricer.py does not pass batch=args.batch. Fix it first."; exit 1
fi

DIAG=../HPC_Logs/spread_test_${SLURM_JOB_ID}
mkdir -p "$DIAG"
RESULTS=../csv/spread_test_${SLURM_JOB_ID}.csv
echo "config,mem,nodes,per_node,P,N,call,call_se,put,put_se,compute_s,comm_s,total_s" > "$RESULTS"

N=10000000000
BATCH=30000
REPS=5
DDR="--membind=0-7"
HBM="--preferred-many=8-15"

log "Nodes: $SLURM_JOB_NODELIST"

# run_config NAME MEMLABEL MEMPOLICY NODES PER_NODE PER_SOCKET REPS
run_config () {
    local NAME=$1 MEM=$2 POLICY=$3 NODES=$4 PER_NODE=$5 PER_SOCKET=$6 R=$7
    local P=$((NODES * PER_NODE))
    local MAP="--map-by ppr:${PER_SOCKET}:package --bind-to core"

    # Record placement once per config: want PER_NODE ranks on each of
    # the first NODES hosts, split evenly across both sockets.
    mpirun -n "$P" $MAP --report-bindings hostname \
        > "$DIAG/hosts_${NAME}_${MEM}.txt" 2> "$DIAG/bindings_${NAME}_${MEM}.txt"
    log "$NAME/$MEM placement: $(sort "$DIAG/hosts_${NAME}_${MEM}.txt" | uniq -c | awk '{printf "%s=%s ", $2, $1}')"

    for i in $(seq 1 "$R"); do
        LINE=$(timeout 600 mpirun -n "$P" $MAP \
                 numactl $POLICY -- \
                 python3 -m Seawulf.pricer --n "$N" --batch "$BATCH" --csv)
        if [ $? -ne 0 ] || [ -z "$LINE" ]; then
            log "FAILED $NAME/$MEM rep=$i"; continue
        fi
        echo "${NAME},${MEM},${NODES},${PER_NODE},${LINE}" >> "$RESULTS"
        log "$NAME/$MEM rep=$i -> $(echo "$LINE" | cut -d, -f7)s"
    done
}

# Baseline on these same nodes (one run: ~151 s, run-to-run spread ~0.1%)
run_config p1        ddr "$DDR" 1 1  1  1

# Single node: spread vs packed
run_config 1x48      ddr "$DDR" 1 48 24 $REPS
run_config 1x96      ddr "$DDR" 1 96 48 $REPS

# P=96: 2 nodes half-full vs 1 node full (above)
run_config 2x48      ddr "$DDR" 2 48 24 $REPS

# P=192: the main comparison
run_config 2x96      ddr "$DDR" 2 96 48 $REPS
run_config 4x48      ddr "$DDR" 4 48 24 $REPS
run_config 4x48      hbm "$HBM" 4 48 24 $REPS

# Everything available, for maximum throughput
run_config 4x96      ddr "$DDR" 4 96 48 $REPS

log "Done. Results: $RESULTS   Placement logs: $DIAG/"
