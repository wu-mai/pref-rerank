#!/bin/bash
set -e
cd /home/tylg-jsjxy-3/guochaoyi/服务器3/wise111/wise2026_personalized_evidence_rag
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MAX_SESSIONS=100000
LR=0.05
EPOCHS=50

VARIANTS="full no_user_bias linear_only random_schema classic"

echo "=== Train 5 variants ==="
date
for V in $VARIANTS; do
    echo "--- training $V ---"
    python3 scripts/train.py \
        --variant $V --seed 42 \
        --max-sessions $MAX_SESSIONS \
        --lr $LR --epochs $EPOCHS \
        --output results/r100k_${V}.json 2>&1 | tail -5
done
echo ""

echo "=== Evaluate 5 variants ==="
date
for V in $VARIANTS; do
    echo "--- evaluating $V ---"
    python3 scripts/evaluate.py \
        --model results/r100k_${V}.json \
        --variant $V --max-sessions $MAX_SESSIONS \
        --output results/r100k_eval_${V}.json 2>&1 | tail -3
done
echo ""

echo "=== Final Results ==="
date
for V in $VARIANTS; do
    python3 -c "
import json
d = json.load(open('results/r100k_eval_${V}.json'))
print(f'${V:18s} n={d[\"n_eval_sessions\"]:3d} baseline={d[\"baseline\"][\"ndcg10\"]:.4f} personalized={d[\"personalized\"][\"ndcg10\"]:.4f} delta={d[\"delta_ndcg10\"]:+.4f}')
"
done
