#!/bin/sh
# Restartable job queue for task 1: skips runs whose output is complete (no checkpoint left), resumes the rest.
cd "$(dirname "$0")/.."
mkdir -p logs data/raw
python3 - <<'PY' > logs/pending.txt
import os, numpy as np
for line in open('src/jobs_task1.txt'):
    proto, D, L, s = line.split(); fn = 'data/raw/t1_%s_D%03d_L%s_s%s.npz' % (proto, round(float(D) * 100), L, s)
    if os.path.exists(fn) and not os.path.exists(fn.replace('.npz', '.ckpt.npz')):
        n = len(np.load(fn)['T'])
        if n == (42 if proto == 'anneal' else 21): continue
    print(line.strip())
PY
echo "pending: $(wc -l < logs/pending.txt)"
cat logs/pending.txt | xargs -P 4 -L 1 sh -c 'python3 src/run_task1.py $0 $1 $2 $3 >> logs/t1_$0_$1_$2_$3.log 2>&1'
