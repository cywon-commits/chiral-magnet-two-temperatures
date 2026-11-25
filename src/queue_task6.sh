#!/bin/sh
# Restartable queue for task 6b (per-job arguments in src/jobs_task6.txt; completed runs have <run_dir>/DONE).
cd "$(dirname "$0")/.."; mkdir -p logs data/raw
python3 - <<'PY' > logs/pending6.txt
import os
for line in open('src/jobs_task6.txt'):
    w = line.split(); tag, D, L, s = w[:4]
    if os.path.exists('data/raw/t2_%s_D%03d_L%s_s%s/DONE' % (tag, round(float(D) * 100), L, s)): continue
    print(line.strip())
PY
echo "pending: $(wc -l < logs/pending6.txt)"
cat logs/pending6.txt | xargs -P 4 -L 1 sh -c 'python3 src/run_task2.py "$@" >> logs/t6_$1_$2_$3.log 2>&1' _
