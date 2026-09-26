#!/bin/bash
#SBATCH --job-name=p192_diag
#SBATCH -p hbm-short-96core
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH -t 00:20:00
#SBATCH -o HPC_Logs/p192_diag_%j.out
#SBATCH -e HPC_Logs/p192_diag_%j.err

module purge
module load slurm
module load mpi4py/latest
module load numactl/2.0.16

cd src

log () { echo "[$(date +%H:%M:%S)] $*"; }
DIAG=../HPC_Logs/p192_diag_${SLURM_JOB_ID}
mkdir -p "$DIAG"

log "Nodes: $SLURM_JOB_NODELIST"
log "mpirun: $(which mpirun)"
mpirun --version 2>&1 | head -1

#Hyperthreading check
log "CPU topology per node:"
srun -N 2 --ntasks-per-node=1 bash -c \
  'echo "--- $(hostname)"; lscpu | grep -E "^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)"'

declare -A MAPS=(
  [numa]="--map-by numa --bind-to core"
  [ppr]="--map-by ppr:96:node --bind-to core"
  [default]="--bind-to core"
)

for NAME in numa ppr default; do
    FLAGS=${MAPS[$NAME]}
    log "=== Placement: $NAME ($FLAGS) ==="
    mpirun -n 192 $FLAGS --report-bindings hostname \
        > "$DIAG/hosts_$NAME.txt" 2> "$DIAG/bindings_$NAME.txt"
    if [ $? -ne 0 ]; then
        log "mpirun failed for $NAME; see $DIAG/bindings_$NAME.txt"
        continue
    fi
    echo "Ranks per host:"
    sort "$DIAG/hosts_$NAME.txt" | uniq -c
    DUPES=$(grep "MCW rank" "$DIAG/bindings_$NAME.txt" \
            | sed -E 's/^\[([^:]+):[^]]*\] MCW rank [0-9]+ bound to /\1 /' \
            | sort | uniq -d | wc -l)
    echo "Core bindings shared by 2+ ranks: $DUPES (want 0)"
done


N=10000000000
BATCH=30000
RESULTS=../csv/p192_diag_${SLURM_JOB_ID}.csv
echo "map,P,N,call,call_se,put,put_se,compute_s,comm_s,total_s" > "$RESULTS"

for NAME in numa ppr default; do
    FLAGS=${MAPS[$NAME]}
    for i in 1 2; do
        LINE=$(timeout 300 mpirun -n 192 $FLAGS \
                 numactl --membind=0-7 -- \
                 python3 -m Seawulf.pricer --n "$N" --batch "$BATCH" --csv)
        if [ $? -ne 0 ] || [ -z "$LINE" ]; then
            log "timing FAILED map=$NAME rep=$i"
            continue
        fi
        echo "${NAME},${LINE}" >> "$RESULTS"
        log "timing map=$NAME rep=$i -> $(echo "$LINE" | cut -d, -f7)s compute"
    done
done

log "Done. Raw placement files in $DIAG/, timings in $RESULTS"
