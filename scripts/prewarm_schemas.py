"""
Pre-warm schema cache: pre-extract all unique user histories from train+dev pairs.

Run this BEFORE train.py to avoid interleaving LLM calls with training.
After this, train.py's featurize step is fast (all schemas cached).

Usage:
    python scripts/prewarm_schemas.py --max-sessions 200000
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aol_session import (  # noqa: E402
    load_aol_sessions_cached_or_fresh,
    build_training_pairs,
    split_sessions,
)
from src.extract.schema_extractor import SchemaExtractor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prewarm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sessions", type=int, default=200000)
    ap.add_argument("--history-k", type=int, default=3)
    ap.add_argument("--topk-debias", type=int, default=3)
    ap.add_argument("--schema-cache", default="cache/schema")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logger.info("Loading sessions...")
    sessions = load_aol_sessions_cached_or_fresh(
        min_session_len=args.history_k + 1,
        max_sessions=args.max_sessions,
        seed=args.seed,
    )
    train_s, dev_s, test_s = split_sessions(sessions, seed=args.seed)
    logger.info("Split: train=%d dev=%d test=%d", len(train_s), len(dev_s), len(test_s))

    train_pairs = build_training_pairs(train_s, history_k=args.history_k, topk_debias=args.topk_debias, seed=args.seed)
    dev_pairs = build_training_pairs(dev_s, history_k=args.history_k, topk_debias=args.topk_debias, seed=args.seed)
    logger.info("Pairs: train=%d dev=%d", len(train_pairs), len(dev_pairs))

    all_pairs = train_pairs + dev_pairs
    histories = [p.user_history for p in all_pairs]

    # Dedupe (many pairs share history)
    seen = set()
    unique_histories = []
    for h in histories:
        import json
        key = json.dumps(h, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_histories.append(h)
    logger.info("Unique histories: %d (from %d pairs)", len(unique_histories), len(all_pairs))

    extractor = SchemaExtractor(cache_dir=args.schema_cache)
    logger.info("Pre-warming schemas (10 workers, may take 5-15 min for 500+ unique histories)...")
    schemas = extractor.extract_batch(unique_histories, max_workers=10, desc="prewarm")
    n_ok = sum(1 for s in schemas if s is not None)
    logger.info("Done. %d/%d schemas successfully extracted", n_ok, len(schemas))


if __name__ == "__main__":
    main()
