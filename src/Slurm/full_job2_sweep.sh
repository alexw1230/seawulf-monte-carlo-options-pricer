CONFIGS=(
    "96 2"    # 48 cores/node, vs existing P=96 baseline at 96 cores/node (1 node)
    "96 4"    # 24 cores/node
    "192 4"   # 48 cores/node, vs existing P=192 baseline at 96 cores/node (2 nodes)
    "192 8"   # 24 cores/node
)

for cfg in "${CONFIGS[@]}"; do
    read -r P NODES <<< "$cfg"
    PER_NODE=$((P / NODES))

    sbatch --partition=short-96core --nodes="$NODES" --ntasks-per-node="$PER_NODE" \
        src/Slurm/job2.sh
    echo "submitted P=$P across $NODES node(s), $PER_NODE cores/node"
done
