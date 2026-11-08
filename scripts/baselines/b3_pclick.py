"""B3: Position-bias-aware re-ranker (P-Click style).

Source: Joachims, Swaminathan, Schnabel - WSDM 2017 (CCF A) -
"Unbiased Learning-to-Rank with Biased Feedback".

We implement a simple, faithful version of unbiased LTR:
    s'(q, d) = s_base(q, d) + beta(q, d)

where beta(q, d) is a learned scalar per (query, URL) pair from the
training click log. Training optimizes a pairwise loss with inverse
propensity weighting so that high-ranked clicks are down-weighted
relative to lower-ranked ones.

This isolates the contribution of a learned query-URL bias from the
bilinear user-document interaction in Pref-Rerank.

Uses the SAME evaluation protocol as scripts/evaluate.py.
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
logger = logging.getLogger("b3_pclick")


def fit_pairwise_ips(
    train_pairs,
    cache,
    n_epochs: int = 5,
    lr: float = 1e-2,
    l2: float = 1e-4,
    click_decay: float = 0.85,
    seed: int = 42,
) -> dict[tuple[str, str], float]:
    """Fit a learned scalar beta(q, d) via pairwise IPS loss."""
    rng = np.random.RandomState(seed)
    beta: dict[tuple[str, str], float] = defaultdict(float)

    # Filter to pairs that have cached features (avoid KeyError)
    train_pairs = [
        p for p in train_pairs
        if cache.get(p.eval_query, p.pos_url) is not None
    ]
    by_query: dict[str, list] = defaultdict(list)
    for p in train_pairs:
        by_query[p.eval_query].append(p)

    for epoch in range(n_epochs):
        total_loss, n_pairs = 0.0, 0
        for q, pairs in by_query.items():
            if len(pairs) < 2:
                continue
            for p_pos in pairs:
                neg_pool = [pp for pp in pairs if pp.pos_url != p_pos.pos_url]
                if not neg_pool:
                    continue
                p_neg = neg_pool[rng.randint(len(neg_pool))]
                s_pos_base = cache.get(p_pos.eval_query, p_pos.pos_url)[0]
                s_neg_base = cache.get(p_neg.eval_query, p_neg.pos_url)[0]
                w_pos = click_decay ** max(1, p_pos.pos_rank or 1)
                w_neg = click_decay ** max(1, p_neg.pos_rank or 1)

                diff = (s_pos_base + beta[(q, p_pos.pos_url)]) - \
                       (s_neg_base + beta[(q, p_neg.pos_url)])
                sig = 1.0 / (1.0 + np.exp(-diff))
                grad = -w_pos * (sig - 1.0)
                beta[(q, p_pos.pos_url)] -= lr * (grad + l2 * beta[(q, p_pos.pos_url)])
                beta[(q, p_neg.pos_url)] -= lr * (-grad + l2 * beta[(q, p_neg.pos_url)])
                total_loss += -w_pos * np.log(max(sig, 1e-7))
                n_pairs += 1
        logger.info("Epoch %d  loss=%.4f  pairs=%d  beta_vocab=%d",
                    epoch + 1, total_loss / max(1, n_pairs), n_pairs, len(beta))
    return dict(beta)


def evaluate(cands, cache, beta) -> tuple[list[float], list[float]]:
    base, pers = [], []
    for c in cands:
        urls = [c["pos_url"]] + c["neg_urls"]
        s_base, s_pclick = [], []
        ok = True
        for url in urls:
            sb = cache.get(c["eval_query"], url)
            if sb is None:
                ok = False
                break
            s_base.append(sb[0])
            s_pclick.append(sb[0] + beta.get((c["eval_query"], url), 0.0))
        if ok:
            base.append(ndcg_from_scores(s_base))
            pers.append(ndcg_from_scores(s_pclick))
    return base, pers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-cache", default="cache/data/sessions.pkl")
    ap.add_argument("--feature-cache", default="cache/features_queryflow")
    ap.add_argument("--n-negatives", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--l2", type=float, default=1e-4)
    args = ap.parse_args()

    cands, cache, train_pairs, _ = load_protocol(
        session_cache=args.session_cache,
        feature_cache=args.feature_cache,
        n_negatives=args.n_negatives,
        seed=args.seed,
    )
    beta = fit_pairwise_ips(
        train_pairs, cache,
        n_epochs=args.n_epochs, lr=args.lr, l2=args.l2, seed=args.seed,
    )
    ndcg_base, ndcg_pers = evaluate(cands, cache, beta)
    report("pclick", ndcg_base, ndcg_pers, args.seed,
           extra={"n_beta_params": len(beta)})


if __name__ == "__main__":
    main()