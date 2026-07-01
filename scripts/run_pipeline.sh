#!/bin/bash
# Auto pipeline: wait for prewarm, then train, then eval.
# Pass PID=0 to skip wait (use existing cache immediately).
set -e
cd /home/tylg-jsjxy-3/guochaoyi/服务器3/wise111/wise2026_personalized_evidence_rag

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PREWARM_PID="${1:-1611223}"
MAX_SESSIONS="${2:-200000}"

echo "=== Phase 0: Waiting for prewarm (PID $PREWARM_PID) ==="
if [ "$PREWARM_PID" != "0" ] && [ -d "/proc/$PREWARM_PID" ]; then
    while [ -d "/proc/$PREWARM_PID" ]; do
        sleep 30
        N=$(ls cache/schema/ 2>/dev/null | wc -l)
        echo "  [$(date +%H:%M:%S)] prewarm running, $N schemas cached"
    done
    echo "  prewarm done. $(ls cache/schema/ | wc -l) schemas cached."
else
    echo "  prewarm skipped (PID=$PREWARM_PID). Using existing $(ls cache/schema/ | wc -l) cached schemas."
fi

echo ""
echo "=== Phase 1: Train R005 (full variant, max_sessions=$MAX_SESSIONS) ==="
python3 scripts/train.py \
    --variant full --seed 42 \
    --max-sessions "$MAX_SESSIONS" \
    --output results/r005_full_seed42.json 2>&1 | tail -40
echo ""

echo "=== Phase 2: Evaluate R005 ==="
python3 scripts/evaluate.py \
    --model results/r005_full_seed42.json \
    --variant full \
    --max-sessions "$MAX_SESSIONS" \
    --output results/r005_eval.json 2>&1 | tail -20
echo ""

echo "=== Final Result ==="
cat results/r005_eval.json
