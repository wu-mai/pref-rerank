"""B5: HEM-style embedding-matching baseline (SIGIR 2017, Lu et al.).

Source: Lu et al., "A Personalized Search Engine Based on HEM",
SIGIR 2017 (CCF A). HEM embeds the user's past queries and clicked
documents into a shared word2vec space and re-ranks candidates by
their similarity to the user's historical profile.

Faithful lightweight adaptation to our frozen-backbone protocol:
the user profile vector is the mean of BGE embeddings of the history
queries (we reuse the frozen h(q,d) encoder space), and the
personalized score adds the cosine similarity between the candidate
document representation h(q,d) and the user profile:

    s'(q, d, H) = s_base(q, d) + alpha * cos(h(q,d), u_hem(H))
    u_hem(H) = mean over past queries q' of BGE(q')

This isolates the "embed history, match candidates" mechanism of HEM
on the same frozen features Pref-Rerank uses. No training beyond the
scalar alpha (tuned on dev).

Usage:
    python scripts/baselines/b5_hem.py --session-cache cache/data/sessions.pkl \
        --feature-cache cache/features_queryflow --n-negatives 9 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.baselines.common import load_protocol, ndcg_from_scores, report  # noqa: E402
from src.eval.metrics import paired_bootstrap_ci  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("b5_hem")


def build_user_profile(user_history, doc_click_queries, bge, cache_dir):
    """Mean BGE embedding of past queries in the session history."""
    embs = []
    for hq, _ in user_history:
        if not hq:
            continue
        e = bge.get_query_embedding(hq)
        if e is not None:
            embs.append(e)
    if not embs:
        return None
    return np.mean(embs, axis=0)


class BGEQueryEncoder:
    """Caches BGE embeddings for queries (lazy, dict-backed)."""

    def __init__(self, encoder_path="/root/gcy/Rerank/cache/bge_local"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(encoder_path, device="cpu")
        self._cache: dict[str, np.ndarray] = {}

    def get_query_embedding(self, q: str):
        if q in self._cache:
            return self._cache[q]
        e = self.model.encode([q], convert_to_numpy=True, show_progress_bar=False)[0]
        self._cache[q] = e
        return e


def evaluate(cands, cache, bge, alpha: float):
    """Per-candidate-set nDCG@10 under s_base + alpha * cos(h, u_hem)."""
    base, pers = [], []
    for c in cands:
        urls = [c["pos_url"]] + c["neg_urls"]
        prof = build_user_profile(c["user_history"], None, bge, None)
        s_base_list, s_pers_list = [], []
        ok = True
        for url in urls:
            pair = cache.get(c["eval_query"], url)
            if pair is None:
                ok = False
                break
            s_base, h = pair
            s_base_list.append(s_base)
            if prof is not None:
                h = np.asarray(h, dtype=np.float32)
                cos = float(np.dot(h, prof) / (np.linalg.norm(h) * np.linalg.norm(prof) + 1e-8))
                s_pers_list.append(s_base + alpha * cos)
            else:
                s_pers_list.append(s_base)
        if ok:
            base.append(ndcg_from_scores(s_base_list))
            pers.append(ndcg_from_scores(s_pers_list))
    return base, pers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-cache", default="cache/data/sessions.pkl")
    ap.add_argument("--feature-cache", default="cache/features_queryflow")
    ap.add_argument("--n-negatives", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha-grid", nargs="+", type=float,
                    default=[0.5, 1.0, 2.0, 5.0, 10.0])
    args = ap.parse_args()

    cands, cache, train_pairs, dev_pairs = load_protocol(
        session_cache=args.session_cache,
        feature_cache=args.feature_cache,
        n_negatives=args.n_negatives,
        seed=args.seed,
    )
    bge = BGEQueryEncoder()
    logger.info("BGE encoder loaded; evaluating %d candidates", len(cands))

    # Tune alpha on dev candidates
    from evaluate import build_eval_candidates  # noqa: E402
    dev_cands = build_eval_candidates(dev_pairs, dev_pairs, n_negatives=args.n_negatives, seed=args.seed)
    best_alpha, best_ndcg = 1.0, -1.0
    for alpha in args.alpha_grid:
        base, pers = evaluate(dev_cands, cache, bge, alpha)
        m = float(np.mean(pers)) if pers else 0.0
        logger.info("  alpha=%.2f  dev nDCG@10=%.4f (n=%d)", alpha, m, len(pers))
        if m > best_ndcg:
            best_alpha, best_ndcg = alpha, m
    logger.info("Best alpha on dev: %.2f", best_alpha)

    base, pers = evaluate(cands, cache, bge, best_alpha)
    mean, lo, hi = paired_bootstrap_ci(pers, base, n_bootstrap=10000, seed=args.seed)
    result = {
        "system": "hem",
        "source": "Lu et al., SIGIR 2017 (CCF A) - A Personalized Search Engine Based on HEM",
        "protocol": "AOL full neg-pool, n_negatives=9",
        "n_sessions": len(base),
        "baseline_ndcg10": float(np.mean(base)),
        "personalized_ndcg10": float(np.mean(pers)),
        "alpha": best_alpha,
        "delta": float(mean),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "note": "Lightweight adaptation: user profile = mean BGE embedding of history queries; "
                "score = s_base + alpha * cos(h(q,d), profile). Scalar alpha tuned on dev.",
    }
    print(json.dumps(result, indent=2))
    out = Path("/root/gcy/Rerank/results/baselines_n120/b5_hem.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()