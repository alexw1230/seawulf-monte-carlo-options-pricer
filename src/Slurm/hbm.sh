#!/bin/bash
#SBATCH --job-name=hbm_ddr_test
#SBATCH -p hbm-short-96core
#SBATCH -N 1
#SBATCH --ntasks=96
#SBATCH -t 01:30:00
#SBATCH -o HPC_Logs/hbm_ddr_%j.out
#SBATCH -e HPC_Logs/hbm_ddr_%j.err

# NOTE: hbm-96core is only reachable from the milan1/milan2/xeonmax login
# nodes, not login1/login2 -- submit this job from one of those. Those
# login nodes may also require a different slurm module
# (slurm/seawulf3/21.08.8 per SBU's KB) instead of the bare "module load
# slurm" below -- if the module load fails, run `module avail slurm` and
# swap in whatever version it lists.

module purge
module load slurm
module load mpi4py/latest

cd src

N=10000000000
REPEATS=5
RESULTS=../csv/hbm_vs_ddr_${SLURM_JOB_ID}.csv
NUMA_LOG=../csv/numa_layout_${SLURM_JOB_ID}.txt

# --- Detect NUMA layout on this node ---------------------------------
# In flat/SNC4 mode, SPR nodes expose 16 NUMA regions: some have cores
# and local DDR5 attached, some are memory-only HBM regions with no
# cores of their own. We classify by whether "cpus:" has anything after
# it. ALWAYS check $NUMA_LOG by hand before trusting results below --
# this classification is a best-effort guess, not a guarantee.
numactl -H > "$NUMA_LOG"

DDR_NODES=$(awk '/^node [0-9]+ cpus:/ {
    line=$0; sub(/^node [0-9]+ cpus: */,"",line);
    if (line != "") print $2
}' "$NUMA_LOG" | paste -sd, -)

HBM_NODES=$(awk '/^node [0-9]+ cpus:/ {
    line=$0; sub(/^node [0-9]+ cpus: */,"",line);
    if (line == "") print $2
}' "$NUMA_LOG" | paste -sd, -)

echo "Detected DDR (CPU-attached) NUMA nodes: $DDR_NODES"
echo "Detected HBM (memory-only) NUMA nodes:  $HBM_NODES"

if [ -z "$HBM_NODES" ] || [ -z "$DDR_NODES" ]; then
    echo "WARNING: auto-detection failed, falling back to documented layout"
    echo "         (0-7 DDR, 8-15 HBM). VERIFY against $NUMA_LOG -- this"
    echo "         fallback may not match the actual node."
    DDR_NODES="0,1,2,3,4,5,6,7"
    HBM_NODES="8,9,10,11,12,13,14,15"
fi

# --- Run both memory conditions at both P=1 (uncontended) and --------
# --- P=96 (full node, matches the project's existing P=96 point) -----
echo "mem,P,N,call,call_se,put,put_se,compute_s,comm_s,total_s" > "$RESULTS"

run_case () {
    local MEM_LABEL=$1
    local MEMBIND=$2
    local NRANKS=$3
    for i in $(seq 1 "$REPEATS"); do
        LINE=$(mpirun -n "$NRANKS" --bind-to core \
                numactl --membind="$MEMBIND" -- \
                python3 -m Seawulf.pricer --n "$N" --csv)
        if [ -z "$LINE" ]; then
            echo "WARNING: empty output for mem=$MEM_LABEL P=$NRANKS rep=$i (check .err log)"
            continue
        fi
        echo "${MEM_LABEL},${LINE}" >> "$RESULTS"
    done
}

run_case ddr "$DDR_NODES" 1
run_case hbm "$HBM_NODES" 1
run_case ddr "$DDR_NODES" 96
run_case hbm "$HBM_NODES" 96

echo "Done. Results: $RESULTS"
echo "NUMA layout for reference: $NUMA_LOG"
