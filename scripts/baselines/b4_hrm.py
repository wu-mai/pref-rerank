"""B4: History-aware ranking model (HRM).

Source: Bennett, White, Chu, Dumais et al. - SIGIR 2012 (CCF A) -
"Modeling the impact of short- and long-term behavior on search personalization".

HRM is the classic history-aware query-rewriting baseline. The current
query q is re-scored against the candidate document d under a learned
log-linear model:

    s'(q, d, H) = s_base(q, d) + sum_{q' in H} beta_{q'} * BM25(q', d)

where the per-history-query weights beta_{q'} are learned jointly from
click logs. We use the QueryFlowBackbone to compute BM25(q', d) for each
history query q' (no training-time featurization needed beyond the
existing doc_click_queries cache).

Uses the SAME evaluation protocol as scripts/evaluate.py.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.baselines.common import load_protocol, ndcg_from_scores, report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("b4_hrm")


def fit_hrm(
    train_pairs,
    bm25,
    n_epochs: int = 5,
    lr: float = 1e-2,
    l2: float = 1e-4,
    seed: int = 42,
) -> dict[str, float]:
    """Fit per-history-query weight beta_{q'} via pairwise ranking loss.

    Score:
        s(p, d) = s_base(p.eval_query, d) + sum_{q' in p.user_history} beta_{q'} * BM25(q', d)

    where s_base is approximated by the BM25 score for the eval query (since
    on this small training set the more nuanced cache features aren't needed
    for fitting HRM's own beta terms).
    """
    rng = np.random.RandomState(seed)
    beta: dict[str, float] = defaultdict(float)

    train_pairs = [
        p for p in train_pairs
        if p.pos_url and any(hq for hq, _ in p.user_history)
    ]
    for epoch in range(n_epochs):
        total_loss, n_pairs = 0.0, 0
        shuffled = list(train_pairs)
        rng.shuffle(shuffled)
        for p_pos in shuffled:
            same_q = [q for q in train_pairs
                      if q.eval_query == p_pos.eval_query
                      and q.session_id != p_pos.session_id
                      and any(hq for hq, _ in q.user_history)]
            if not same_q:
                continue
            p_neg = same_q[rng.randint(len(same_q))]

            sb_pos = bm25.score(p_pos.eval_query, p_pos.pos_url)
            sb_neg = bm25.score(p_neg.eval_query, p_neg.pos_url)

            hist_pos = [hq for hq, _ in p_pos.user_history if hq]
            hist_neg = [hq for hq, _ in p_neg.user_history if hq]
            history_bonus_pos = sum(beta[hq] * bm25.score(hq, p_pos.pos_url) for hq in hist_pos)
            history_bonus_neg = sum(beta[hq] * bm25.score(hq, p_neg.pos_url) for hq in hist_neg)

            diff = (sb_pos + history_bonus_pos) - (sb_neg + history_bonus_neg)
            sig = 1.0 / (1.0 + np.exp(-diff))
            grad = -(sig - 1.0)
            for hq in hist_pos:
                grad_h = grad * bm25.score(hq, p_pos.pos_url)
                beta[hq] -= lr * (grad_h + l2 * beta[hq])
            for hq in hist_neg:
                grad_h = -grad * bm25.score(hq, p_neg.pos_url)
                beta[hq] -= lr * (grad_h + l2 * beta[hq])
            total_loss += -np.log(max(sig, 1e-7))
            n_pairs += 1
        logger.info("Epoch %d  loss=%.4f  pairs=%d  beta_vocab=%d",
                    epoch + 1, total_loss / max(1, n_pairs), n_pairs, len(beta))
    return dict(beta)


def evaluate(cands, cache, beta, bm25) -> tuple[list[float], list[float]]:
    base, pers = [], []
    for c in cands:
        urls = [c["pos_url"]] + c["neg_urls"]
        s_base, s_hrm = [], []
        ok = True
        hist_queries = [hq for hq, _ in c["user_history"] if hq]
        for url in urls:
            sb = cache.get(c["eval_query"], url)
            if sb is None:
                ok = False
                break
            s_base.append(sb[0])
            history_bonus = sum(beta.get(hq, 0.0) * bm25.score(hq, url) for hq in hist_queries)
            s_hrm.append(sb[0] + history_bonus)
        if ok:
            base.append(ndcg_from_scores(s_base))
            pers.append(ndcg_from_scores(s_hrm))
    return base, pers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-cache", default="cache/data/sessions.pkl")
    ap.add_argument("--feature-cache", default="cache/features_queryflow")
    ap.add_argument("--doc-click-cache", default="cache/data/doc_click_queries.pkl")
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
    from src.model.queryflow_backbone import BM25  # noqa: E402
    doc_click = pickle.loads(Path(args.doc_click_cache).read_bytes())
    bm25 = BM25(doc_click)
    beta = fit_hrm(train_pairs, bm25, n_epochs=args.n_epochs, lr=args.lr, l2=args.l2, seed=args.seed)
    ndcg_base, ndcg_pers = evaluate(cands, cache, beta, bm25)
    report("hrm", ndcg_base, ndcg_pers, args.seed,
           extra={"n_beta_params": len(beta)})


if __name__ == "__main__":
    main()