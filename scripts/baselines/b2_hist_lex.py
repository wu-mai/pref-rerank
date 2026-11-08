"""B2: Query-history lexicalization baseline.

Source: Craswell et al., WSDM 2008 (CCF A). The query-history
lexicalization idea is the standard zero-training session-search baseline.

For each evaluation query q and session history H = {q_1, ..., q_K},
build an augmented query Q = q + " " + " ".join(H), then re-score with
the same BM25 backbone. No training is required; the only signal is that
the current candidate's past click history shares terms with the user's
recent query history.

BM25 scores for the augmented query are computed directly via the backbone
(not via the feature cache, since lex_q is a fresh concatenation that has
not been prefetched).

Uses the SAME evaluation protocol as scripts/evaluate.py.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.baselines.common import load_protocol, ndcg_from_scores, report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("b2_hist_lex")


def build_lex_query(eval_query: str, history: list[tuple[str, str]]) -> str:
    past_queries = [hq for hq, _ in history if hq]
    return eval_query + " " + " ".join(past_queries)


def evaluate(cands, cache, bm25) -> tuple[list[float], list[float]]:
    """Per-session nDCG@10 under (s_base from cache, BM25 over lex_q via BM25)."""
    base, pers = [], []
    for c in cands:
        urls = [c["pos_url"]] + c["neg_urls"]
        s_base, s_lex = [], []
        ok = True
        lex_q = build_lex_query(c["eval_query"], c["user_history"])
        for url in urls:
            sb = cache.get(c["eval_query"], url)
            if sb is None:
                ok = False
                break
            s_base.append(sb[0])
            s_lex.append(bm25.score(lex_q, url))
        if ok:
            base.append(ndcg_from_scores(s_base))
            pers.append(ndcg_from_scores(s_lex))
    return base, pers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-cache", default="cache/data/sessions.pkl")
    ap.add_argument("--feature-cache", default="cache/features_queryflow")
    ap.add_argument("--doc-click-cache", default="cache/data/doc_click_queries.pkl")
    ap.add_argument("--n-negatives", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cands, cache, _, _ = load_protocol(
        session_cache=args.session_cache,
        feature_cache=args.feature_cache,
        n_negatives=args.n_negatives,
        seed=args.seed,
    )
    # Build BM25 directly (no sentence encoder — we only need BM25 scoring)
    from src.model.queryflow_backbone import BM25  # noqa: E402
    import pickle
    doc_click = pickle.loads(Path(args.doc_click_cache).read_bytes())
    bm25 = BM25(doc_click)
    ndcg_base, ndcg_pers = evaluate(cands, cache, bm25)
    report("hist-lex", ndcg_base, ndcg_pers, args.seed)


if __name__ == "__main__":
    main()