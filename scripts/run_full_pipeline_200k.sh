#!/bin/bash
# Full pipeline: prefetch → train 4 variants → eval 4 variants
# Uses existing schema cache (most histories already cached from earlier runs)
set -e
cd /home/tylg-jsjxy-3/guochaoyi/服务器3/wise111/wise2026_personalized_evidence_rag

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MAX_SESSIONS=200000
LR=0.05
EPOCHS=50

echo "=== Step 1: Prefetch features on $MAX_SESSIONS sessions ==="
date
python3 scripts/prefetch.py --max-sessions $MAX_SESSIONS 2>&1 | tail -5
echo ""

VARIANTS="full no_user_bias linear_only random_schema classic"

echo "=== Step 2: Train 5 variants ==="
date
for V in $VARIANTS; do
    echo "--- training $V ---"
    python3 scripts/train.py \
        --variant $V --seed 42 \
        --max-sessions $MAX_SESSIONS \
        --lr $LR --epochs $EPOCHS \
        --output results/r200k_${V}.json 2>&1 | tail -3
done
echo ""

echo "=== Step 3: Evaluate 5 variants ==="
date
for V in $VARIANTS; do
    echo "--- evaluating $V ---"
    python3 scripts/evaluate.py \
        --model results/r200k_${V}.json \
        --variant $V --max-sessions $MAX_SESSIONS \
        --output results/r200k_eval_${V}.json 2>&1 | tail -3
done
echo ""

echo "=== Final Results Summary ==="
date
for V in $VARIANTS; do
    python3 -c "
import json
d = json.load(open('results/r200k_eval_${V}.json'))
print(f'${V:18s} baseline={d[\"baseline\"][\"ndcg10\"]:.4f} personalized={d[\"personalized\"][\"ndcg10\"]:.4f} delta={d[\"delta_ndcg10\"]:+.4f}')
"
done
