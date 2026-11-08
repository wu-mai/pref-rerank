"""Shared harness for personalization baselines.

Mirrors the evaluation protocol of scripts/evaluate.py EXACTLY:
  - same session split (user-disjoint, seed=42)
  - same candidate construction (build_eval_candidates: 1 positive +
    n_negatives negatives drawn from the same query's cross-session click pool)
  - same metrics (session-level nDCG@10, paired bootstrap 95% CI)

Only the scoring function differs per baseline.
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.aol_session import (  # noqa: E402
    load_aol_sessions_cached_or_fresh,
    build_training_pairs,
    split_sessions,
)
from src.eval.metrics import paired_bootstrap_ci, ndcg_at_k  # noqa: E402
from src.model.queryflow_backbone import FeatureCache, QueryFlowBackbone  # noqa: E402

logger = logging.getLogger("baselines_common")


def load_protocol(
    session_cache: str = "cache/data/sessions.pkl",
    feature_cache: str = "cache/features_queryflow",
    doc_click_cache: str = "cache/data/doc_click_queries.pkl",
    n_negatives: int = 9,
    seed: int = 42,
    with_backbone: bool = False,
    encoder_name: str = "/root/gcy/Rerank/cache/bge_local",
    full_neg_pool: bool = True,
):
    """Load sessions, build pairs, and build eval candidates exactly like
    scripts/evaluate.py does (full_neg_pool=True: neg pool built from ALL
    clicks in the test split, matching the paper's evaluation protocol).

    Returns (cands, cache, train_pairs, dev_pairs, [backbone])."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate import build_eval_candidates  # noqa: E402
    from src.data.aol_session import TrainingPair  # noqa: E402
    from collections import defaultdict  # noqa: E402

    cache = FeatureCache(feature_cache)
    sessions = load_aol_sessions_cached_or_fresh(cache_path=session_cache)
    train_s, dev_s, test_s = split_sessions(sessions, seed=seed)
    train_pairs = build_training_pairs(train_s, seed=seed)
    dev_pairs = build_training_pairs(dev_s, seed=seed)
    test_pairs = build_training_pairs(test_s, seed=seed)
    logger.info("Pairs: train=%d dev=%d test=%d", len(train_pairs), len(dev_pairs), len(test_pairs))

    if full_neg_pool:
        # EXACTLY the neg-pool construction of scripts/evaluate.py:
        # all clicks from the test split, not just test_pairs.
        test_query_to_urls: dict[str, list[str]] = defaultdict(list)
        for s in test_s:
            for ev in s.events:
                if ev.query and ev.query.strip():
                    for it in ev.clicked_items:
                        test_query_to_urls[ev.query].append(it.doc_id)
        neg_pool_pairs = []
        for q, urls in test_query_to_urls.items():
            if len(urls) >= 2:
                for i, u in enumerate(urls):
                    neg_pool_pairs.append(TrainingPair(
                        session_id="", user_history=[], eval_query=q,
                        pos_url=u, neg_url=urls[(i + 1) % len(urls)],
                        pos_rank=None, neg_rank=None,
                    ))
        logger.info("Built full neg pool: %d entries (%d unique queries)",
                    len(neg_pool_pairs), len(test_query_to_urls))
    else:
        neg_pool_pairs = test_pairs

    cands = build_eval_candidates(test_pairs, neg_pool_pairs, n_negatives=n_negatives, seed=seed)
    logger.info("Eval candidates: %d", len(cands))

    if not with_backbone:
        return cands, cache, train_pairs, dev_pairs

    doc_click = Path(doc_click_cache)
    if not doc_click.exists():
        raise SystemExit(f"doc_click_queries cache not found: {doc_click}")
    doc_click_queries = pickle.loads(doc_click.read_bytes())
    backbone = QueryFlowBackbone(doc_click_queries, encoder_name=encoder_name)
    return cands, cache, train_pairs, dev_pairs, backbone


def ndcg_from_scores(scores: list[float]) -> float:
    """nDCG@10 with the positive at index 0 of the candidate list."""
    order = np.argsort(-np.array(scores), kind="stable")
    ranked_rel = [1.0 if i == 0 else 0.0 for i in order]
    return ndcg_at_k(ranked_rel, 10)


def report(system: str, ndcg_base: list[float], ndcg_pers: list[float], seed: int, extra: dict | None = None):
    import json

    mean, lo, hi = paired_bootstrap_ci(ndcg_pers, ndcg_base, n_bootstrap=10000, seed=seed)
    result = {
        "system": system,
        "n_sessions": len(ndcg_base),
        "baseline_ndcg10": float(np.mean(ndcg_base)),
        "personalized_ndcg10": float(np.mean(ndcg_pers)),
        "delta": float(mean),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }
    if extra:
        result.update(extra)
    print(json.dumps(result, indent=2))