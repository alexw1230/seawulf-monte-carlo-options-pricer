#!/bin/bash
P_VALUES=(1 2 4 8 16 32 40 96 192)

for P in "${P_VALUES[@]}"; do
    if [ "$P" -le 40 ]; then
        PART=short-40core
        NODES=1
    elif [ "$P" -eq 96 ]; then
        PART=short-96core
        NODES=1
    else
        PART=short-96core
        NODES=2
    fi

    sbatch --partition="$PART" --nodes="$NODES" --ntasks="$P" src/Slurm/sweep.sh
    echo "submitted P=$P ($PART, $NODES node(s))"
done
