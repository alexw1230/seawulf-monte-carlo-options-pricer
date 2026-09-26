CONFIGS=(
    "96 2"
    "96 4"
    "192 4"
    "192 8"
)

for cfg in "${CONFIGS[@]}"; do
    read -r P NODES <<< "$cfg"
    PER_NODE=$((P / NODES))

    sbatch --partition=short-96core --nodes="$NODES" --ntasks-per-node="$PER_NODE" \
        src/Slurm/job2.sh
    echo "submitted P=$P across $NODES node(s), $PER_NODE cores/node"
done
