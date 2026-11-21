#!/bin/sh
# Restartable queue for task 2: usage src/queue_task2.sh <jobfile> [extra args for run_task2.py]
cd "$(dirname "$0")/.."; JOBS=$1; shift; EXTRA="$*"
mkdir -p logs data/raw
python3 - "$JOBS" <<'PY' > logs/pending2.txt
import os, sys
for line in open(sys.argv[1]):
    tag, D, L, s = line.split(); fn = 'data/raw/t2_%s_D%03d_L%s_s%s.npz' % (tag, round(float(D) * 100), L, s)
    if os.path.exists(fn) and not os.path.exists(fn.replace('.npz', '.ckpt.npz')): continue
    print(line.strip())
PY
echo "pending: $(wc -l < logs/pending2.txt)"
cat logs/pending2.txt | xargs -P 4 -L 1 sh -c "python3 src/run_task2.py \$0 \$1 \$2 \$3 $EXTRA >> logs/t2_\$0_\$1_\$2_\$3.log 2>&1"
