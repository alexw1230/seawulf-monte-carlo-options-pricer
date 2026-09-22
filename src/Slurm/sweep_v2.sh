#!/bin/bash
#SBATCH --job-name=sweep_v2
#SBATCH -p hbm-short-96core
#SBATCH -t 02:30:00
#SBATCH -o HPC_Logs/sweep_v2_%j.out
#SBATCH -e HPC_Logs/sweep_v2_%j.err
# Nodes/tasks are set by submit_sweep_v2.sh:
#   1 node  -> runs every P in P_VALUES (all on the same node)
#   2 nodes -> runs P=192 only
# Submit from milan1/milan2/xeonmax.

module purge
module load slurm
module load mpi4py/latest
module load numactl/2.0.16

cd src

log () { echo "[$(date +%H:%M:%S)] $*"; }

if ! grep -q "batch=args.batch" Seawulf/pricer.py; then
    log "ERROR: pricer.py does not pass batch=args.batch to run_chunk(). Fix it first."
    exit 1
fi

BATCH=30000
MEM_POLICY="--membind=0-7"     # DDR5 only: the ordinary-hardware result
N_VALUES=(100000 1000000 10000000 100000000 1000000000 10000000000)
REPEATS=5

if [ "$SLURM_JOB_NUM_NODES" -eq 1 ]; then
    # 12 = one SNC NUMA region, 48 = one socket, 96 = full node
    P_VALUES=(1 2 4 8 12 16 24 32 48 64 96)
else
    P_VALUES=(192)
fi

log "Nodes: $SLURM_JOB_NODELIST  batch=$BATCH  P values: ${P_VALUES[*]}"

for P in "${P_VALUES[@]}"; do
    RESULTS=../csv/v2_results_P${P}.csv
    echo "P,N,call,call_se,put,put_se,compute_s,comm_s,total_s" > "$RESULTS"
    for N in "${N_VALUES[@]}"; do
        for i in $(seq 1 $REPEATS); do
            LINE=$(timeout 900 mpirun -n "$P" --map-by numa --bind-to core \
                     numactl $MEM_POLICY -- \
                     python3 -m Seawulf.pricer --n "$N" --batch "$BATCH" --csv)
            if [ $? -ne 0 ] || [ -z "$LINE" ]; then
                log "FAILED P=$P N=$N rep=$i"
                continue
            fi
            echo "$LINE" >> "$RESULTS"
        done
        log "P=$P N=$N done (last compute: $(echo "$LINE" | cut -d, -f7)s)"
    done
done

log "All done."
