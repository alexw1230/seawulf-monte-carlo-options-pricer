#!/bin/bash
#SBATCH --job-name=freq_test
#SBATCH -p hbm-short-96core
#SBATCH -N 1
#SBATCH --ntasks=96
#SBATCH -t 00:45:00
#SBATCH -o HPC_Logs/freq_test_%j.out
#SBATCH -e HPC_Logs/freq_test_%j.err
# Hypothesis: per-core throughput falls past P=48 because the chip lowers
# its clock as more cores become active (shared power budget).
# Prediction if true: GHz falls ~25% from P=48 to P=96, IPC stays flat.
# Submit from milan1/milan2/xeonmax.

module purge
module load slurm
module load mpi4py/latest
module load numactl/2.0.16

cd src

log () { echo "[$(date +%H:%M:%S)] $*"; }
OUT=../HPC_Logs/freq_test_${SLURM_JOB_ID}
mkdir -p "$OUT"
SUMMARY=../csv/freq_test_${SLURM_JOB_ID}.csv

log "Node: $(hostname)"

# Weak scaling: fixed work per rank (~15 s), so python startup is a small
# fraction of what perf measures and per-rank times compare directly.
PER_RANK=1000000000
BATCH=30000
P_VALUES=(1 24 48 64 80 96)

HAVE_PERF=0
if command -v perf >/dev/null 2>&1 && \
   perf stat -e cycles:u,instructions:u -x, true >/dev/null 2>&1; then
    HAVE_PERF=1
    log "perf available: measuring cycles and instructions per rank"
else
    log "perf unavailable or blocked: using /proc/cpuinfo sampling only"
fi

echo "P,compute_s,Mtrials_per_core,cpuinfo_GHz,perf_GHz,perf_IPC" > "$SUMMARY"

for P in "${P_VALUES[@]}"; do
    N=$((P * PER_RANK))
    rm -f "$OUT"/perf_P${P}_*.txt "$OUT/mhz_P${P}.txt"

    # Sample the P fastest cores' MHz every 2 s while the job runs
    # (busy cores run fast, idle ones drop to low clocks).
    (
        sleep 4   # skip python startup
        while true; do
            grep "cpu MHz" /proc/cpuinfo | awk '{print $4}' | sort -nr | head -n "$P" \
              | awk '{s+=$1} END {printf "%.0f\n", s/NR}' >> "$OUT/mhz_P${P}.txt"
            sleep 2
        done
    ) &
    SAMPLER=$!

    if [ $HAVE_PERF -eq 1 ]; then
        LINE=$(mpirun -n "$P" --map-by numa --bind-to core bash -c \
          'R=${OMPI_COMM_WORLD_RANK:-$PMIX_RANK}
           exec perf stat -x, -e cycles:u,instructions:u,task-clock \
             -o '"$OUT"'/perf_P'"$P"'_r${R}.txt \
             numactl --membind=0-7 -- \
             python3 -m Seawulf.pricer --n '"$N"' --batch '"$BATCH"' --csv')
    else
        LINE=$(mpirun -n "$P" --map-by numa --bind-to core \
                 numactl --membind=0-7 -- \
                 python3 -m Seawulf.pricer --n "$N" --batch "$BATCH" --csv)
    fi
    kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null

    if [ -z "$LINE" ]; then log "FAILED P=$P"; continue; fi
    T=$(echo "$LINE" | cut -d, -f7)
    RATE=$(awk -v n="$PER_RANK" -v t="$T" 'BEGIN{printf "%.1f", n/t/1e6}')
    CPU_GHZ=$(awk '{s+=$1} END {if (NR) printf "%.2f", s/NR/1000; else print "NA"}' "$OUT/mhz_P${P}.txt")

    PERF_GHZ=NA; PERF_IPC=NA
    if [ $HAVE_PERF -eq 1 ]; then
        # perf -x, lines: value,unit,event,...  task-clock is in msec.
        read PERF_GHZ PERF_IPC < <(cat "$OUT"/perf_P${P}_r*.txt | awk -F, '
            $3 ~ /^cycles/       {c+=$1}
            $3 ~ /^instructions/ {i+=$1}
            $3 ~ /^task-clock/   {t+=$1}
            END { if (t>0 && c>0) printf "%.2f %.2f", c/(t*1e6), i/c; else print "NA NA" }')
    fi

    echo "$P,$T,$RATE,$CPU_GHZ,$PERF_GHZ,$PERF_IPC" >> "$SUMMARY"
    log "P=$P  ${T}s  ${RATE}M trials/s/core  cpuinfo=${CPU_GHZ}GHz  perf=${PERF_GHZ}GHz  IPC=${PERF_IPC}"
done

log "Done. Summary: $SUMMARY"
