#!/bin/bash

cd csv

OUT=combined_sweep.csv
echo "P,N,call,call_se,put,put_se,compute_s,comm_s,total_s" > "$OUT"

count=0
for f in results_P*.csv; do
    tail -n +2 "$f" >> "$OUT"
    count=$((count+1))
done

echo "merged $count files into $OUT"
wc -l "$OUT"
