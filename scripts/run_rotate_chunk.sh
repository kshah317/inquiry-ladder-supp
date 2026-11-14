#!/bin/bash
# One chunk of the rotating-proposer run: first of P1, P2 with full-team episodes left.
cd "$(dirname "$0")/.."
for P in P1 P2; do
  f=results/rotate/$P/ABCD.jsonl; N=0; [ -f $f ] && N=$(python3 -c "import json,sys;print(len({json.loads(l)['pid'] for l in open('$f')}))")
  if [ $N -lt 294 ]; then
    timeout 172 python3 scripts/20_run_panel.py --panel $P --subgroups ABCD --out rotate --question-rule rotate --budget 120 --grace 40 \
      --max-spend 4 --games-in-flight 110 --concurrency 440 --check-every 300 2>&1 | grep -v "^\[retry" | grep -E "^P[0-9]|manifest|budget"
    exit 0
  fi
done
echo "ALL ROTATE DONE"
