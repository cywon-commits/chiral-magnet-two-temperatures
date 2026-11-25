#!/bin/sh
# base case (K = h = 0) was run first with NEQ, NMEAS = 1000, 4000; the 5 cases below with 700, 3200.  4 seeds each, 4 at a time.  args: D K hx hz L seed tag
i=0
for c in "0.05 0 0 Kp05" "-0.05 0 0 Km05" "0 0 0.05 hz05" "0 0 0.10 hz10" "0 0.10 0 hx10"; do
  set -- $c
  for s in 1 2 3 4; do
    python3 review/mc_Kh.py 0.3 $1 $2 $3 96 $s $4 > review/data/log_Kh_$4_$s.txt 2>&1 &
    i=$((i+1)); [ $((i % 4)) -eq 0 ] && wait
  done
done
wait
