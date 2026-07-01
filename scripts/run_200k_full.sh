#!/bin/bash
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
        --variant $V --seed 42 --extractor phi3 \
        --max-sessions $MAX_SESSIONS \
        --lr $LR --epochs $EPOCHS \
        --output results/r200k_${V}.json 2>&1 | tail -5
done
echo ""

echo "=== Step 3: Evaluate 5 variants ==="
date
for V in $VARIANTS; do
    echo "--- evaluating $V ---"
    python3 scripts/evaluate.py \
        --model results/r200k_${V}.json \
        --variant $V --extractor phi3 --max-sessions $MAX_SESSIONS \
        --output results/r200k_eval_${V}.json 2>&1 | tail -3
done
echo ""

echo "=== Final Results ==="
date
for V in $VARIANTS; do
    python3 -c "
import json
d = json.load(open('results/r200k_eval_${V}.json'))
ci = d['paired_bootstrap_ci']
print('${V}|n=' + str(d['n_eval_sessions']) + '|baseline=' + str(round(d['baseline']['ndcg10'],4)) + '|pers=' + str(round(d['personalized']['ndcg10'],4)) + '|delta=' + str(round(d['delta_ndcg10'],4)) + '|CI=[' + str(round(ci['ci_low'],4)) + ',' + str(round(ci['ci_high'],4)) + ']|sig=' + str(ci['significant']))
"
done
