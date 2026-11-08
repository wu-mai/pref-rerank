"""
Precompute (s_base, h) features for all (query, doc) pairs.

Backbone options:
  --backbone queryflow : BM25 + BGE embeddings over click-query pseudo-docs (no doc text needed)
  --backbone monot5    : monoT5-220M cross-encoder (requires AOL docs download)

Default: queryflow (works offline with cached BGE model).
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
from src.model.queryflow_backbone import (  # noqa: E402
    QueryFlowBackbone,
    FeatureCache,
    precompute_features,
    build_doc_click_queries,
    EMBED_DIM,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prefetch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="queryflow", choices=["queryflow", "monot5"])
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--history-k", type=int, default=3)
    ap.add_argument("--topk-debias", type=int, default=3)
    ap.add_argument("--feature-cache", default="cache/features_queryflow")
    ap.add_argument("--doc-click-cache", default="cache/data/doc_click_queries.pkl")
    ap.add_argument("--encoder-name", default="cache/bge_local",
                    help="Local sentence encoder (path or HF name)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.backbone != "queryflow":
        raise SystemExit("monot5 backbone not available (AOL docs not downloaded). Use --backbone queryflow.")

    logger.info("Loading sessions (from cache, fast)...")
    sessions = load_aol_sessions_cached_or_fresh(
        min_session_len=args.history_k + 1,
        max_sessions=args.max_sessions,
        seed=args.seed,
    )
    train_s, dev_s, test_s = split_sessions(sessions, seed=args.seed)
    pairs = (
        build_training_pairs(train_s, history_k=args.history_k, topk_debias=args.topk_debias, seed=args.seed)
        + build_training_pairs(dev_s, history_k=args.history_k, topk_debias=args.topk_debias, seed=args.seed)
        + build_training_pairs(test_s, history_k=args.history_k, topk_debias=args.topk_debias, seed=args.seed)
    )

    # Collect unique (query, doc) pairs
    seen = set()
    pair_iter = []
    for p in pairs:
        for q, d in [(p.eval_query, p.pos_url), (p.eval_query, p.neg_url)]:
            if (q, d) not in seen:
                seen.add((q, d))
                pair_iter.append((q, d))
    logger.info("Unique (query, doc) pairs to featurize: %d (from %d training pairs)", len(pair_iter), len(pairs))

    # Load doc -> click queries map from prepare_data.py cache
    doc_click_cache = Path(args.doc_click_cache)
    if not doc_click_cache.exists():
        # Fall back to prepare_data.py's default location
        doc_click_cache = Path("cache/data/doc_click_queries.pkl")
    if doc_click_cache.exists():
        import pickle
        logger.info("Loading cached doc_click_queries from %s", doc_click_cache)
        doc_click_queries = pickle.loads(doc_click_cache.read_bytes())
    else:
        raise SystemExit(
            "doc_click_queries cache not found. Run `python scripts/prepare_data.py` first."
        )

    backbone = QueryFlowBackbone(doc_click_queries, encoder_name=args.encoder_name)
    cache = FeatureCache(args.feature_cache)
    logger.info("Cache has %d pre-existing entries", len(cache))

    precompute_features(
        backbone=backbone,
        pairs=iter(pair_iter),
        cache=cache,
        batch_size=args.batch_size,
    )
    logger.info("Done. Total cached: %d", len(cache))


if __name__ == "__main__":
    main()
