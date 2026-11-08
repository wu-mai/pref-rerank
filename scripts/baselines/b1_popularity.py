"""B1: Popularity-based re-ranking baseline.

Source: Craswell & Szummer, SIGIR 2007 (CCF A) - "Random Walks on the Click Graph".

For each (query, candidate) pair, compute the global click count across
the training split. Re-rank candidates by:

    s'(q, d) = s_base(q, d) + alpha * clicks(d, q)

where alpha is tuned on the dev set. Zero-training personalization
baseline: the only learned parameter is the scalar alpha.

Uses the SAME evaluation protocol as scripts/evaluate.py (see common.py).
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.baselines.common import load_protocol, ndcg_from_scores, report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("b1_popularity")


def compute_click_counts(train_pairs) -> dict[tuple[str, str], int]:
    """Map (query, url) -> click count across train."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for p in train_pairs:
        counts[(p.eval_query, p.pos_url)] += 1
    return dict(counts)


def evaluate(cands, cache, click_counts, alpha: float) -> list[float]:
    """Per-session nDCG@10 under s_base + alpha * clicks."""
    ndcgs = []
    for c in cands:
        urls = [c["pos_url"]] + c["neg_urls"]
        scores = []
        ok = True
        for url in urls:
            pair = cache.get(c["eval_query"], url)
            if pair is None:
                ok = False
                break
            s_base = pair[0]
            clicks = click_counts.get((c["eval_query"], url), 0)
            scores.append(s_base + alpha * float(clicks))
        if ok:
            ndcgs.append(ndcg_from_scores(scores))
    return ndcgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-cache", default="cache/data/sessions.pkl")
    ap.add_argument("--feature-cache", default="cache/features_queryflow")
    ap.add_argument("--n-negatives", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha-grid", nargs="+", type=float,
                    default=[0.0, 0.01, 0.1, 0.5, 1.0, 5.0])
    args = ap.parse_args()

    cands, cache, train_pairs, dev_pairs = load_protocol(
        session_cache=args.session_cache,
        feature_cache=args.feature_cache,
        n_negatives=args.n_negatives,
        seed=args.seed,
    )
    click_counts = compute_click_counts(train_pairs)
    logger.info("Click-count vocabulary: %d (query, url) pairs", len(click_counts))

    # Tune alpha on dev candidates (same construction, dev split)
    from evaluate import build_eval_candidates  # noqa: E402
    dev_cands = build_eval_candidates(dev_pairs, dev_pairs, n_negatives=args.n_negatives, seed=args.seed)
    best_alpha, best_ndcg = 0.0, -1.0
    for alpha in args.alpha_grid:
        nd = evaluate(dev_cands, cache, click_counts, alpha)
        m = float(np.mean(nd)) if nd else 0.0
        logger.info("  alpha=%.3f  dev nDCG@10=%.4f (n=%d)", alpha, m, len(nd))
        if m > best_ndcg:
            best_alpha, best_ndcg = alpha, m
    logger.info("Best alpha on dev: %.3f", best_alpha)

    ndcg_base = evaluate(cands, cache, click_counts, alpha=0.0)
    ndcg_pers = evaluate(cands, cache, click_counts, alpha=best_alpha)
    # Keep only sessions present in both (alpha=0 base is a subset union)
    n = min(len(ndcg_base), len(ndcg_pers))
    report("popularity", ndcg_base[:n], ndcg_pers[:n], args.seed,
           extra={"alpha": best_alpha})


if __name__ == "__main__":
    main()