#!/bin/bash
# One ~150 s chunk of the repeats (replica 1): first panel of P0, P1, P2 with games left.
cd "$(dirname "$0")/.."
for P in P0 P1 P2; do
  N=0; for S in A B C D ABCD; do f=results/main_r1/$P/$S.jsonl; [ -f $f ] && N=$((N+$(wc -l < $f))); done
  if [ $N -lt 1470 ]; then
    timeout 172 python3 scripts/20_run_panel.py --panel $P --replica 1 --subgroups A,B,C,D,ABCD --budget 125 --grace 20 \
      --max-spend 12 --games-in-flight 200 --concurrency 200 --check-every 300 2>&1 | grep -v "^\[retry" | grep -E "^P[0-9]|manifest" 
    exit 0
  fi
done
echo "ALL REPEATS DONE"
