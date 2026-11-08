#!/usr/bin/env bash
# Run all four personalization baselines (B1-B4) end-to-end.
#
# After this completes, fill the numeric entries in
# paper_repo/tables/table5_baselines.tex with the JSON outputs produced.
#
# Usage:
#   bash scripts/baselines/run_all_baselines.sh \
#       --session-cache cache/aol_sessions.json \
#       --feature-cache cache/features_queryflow

set -euo pipefail
SESSION_CACHE="cache/aol_sessions.json"
FEATURE_CACHE="cache/features_queryflow"
N_NEG=9
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-cache) SESSION_CACHE="$2"; shift 2;;
    --feature-cache) FEATURE_CACHE="$2"; shift 2;;
    --n-negatives)   N_NEG="$2"; shift 2;;
    --seed)          SEED="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

OUT_DIR="results/baselines"
mkdir -p "$OUT_DIR"

echo "==> B1: popularity"
python scripts/baselines/b1_popularity.py \
  --session-cache "$SESSION_CACHE" --feature-cache "$FEATURE_CACHE" \
  --n-negatives "$N_NEG" --seed "$SEED" \
  | tee "$OUT_DIR/b1_popularity.json"

echo "==> B2: hist-lex"
python scripts/baselines/b2_hist_lex.py \
  --session-cache "$SESSION_CACHE" --feature-cache "$FEATURE_CACHE" \
  --n-negatives "$N_NEG" --seed "$SEED" \
  | tee "$OUT_DIR/b2_hist_lex.json"

echo "==> B3: pclick"
python scripts/baselines/b3_pclick.py \
  --session-cache "$SESSION_CACHE" --feature-cache "$FEATURE_CACHE" \
  --n-negatives "$N_NEG" --seed "$SEED" \
  | tee "$OUT_DIR/b3_pclick.json"

echo "==> B4: hrm"
python scripts/baselines/b4_hrm.py \
  --session-cache "$SESSION_CACHE" --feature-cache "$FEATURE_CACHE" \
  --n-negatives "$N_NEG" --seed "$SEED" \
  | tee "$OUT_DIR/b4_hrm.json"

echo "All four baselines complete. JSON results in $OUT_DIR/"