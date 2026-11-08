"""Shared harness for MS-MARCO TREC DL 2019 evaluation.

Mirrors scripts/baselines/common.py but for MS-MARCO:
  - load TREC DL 2019 judged qrels (43 queries with binary relevance)
  - build candidates (1 positive + 9 negatives per (query, positive) pair)
  - load monoT5 precomputed features (s_base, h) for every (q, d)
  - return per-candidate score lists compatible with paired bootstrap CI

Only the scoring function differs per baseline.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.metrics import paired_bootstrap_ci, ndcg_at_k  # noqa: E402

logger = logging.getLogger("msmarco_common")


def load_candidates(candidate_cache: str = "cache/msmarco/candidates.jsonl"):
    """Load the candidate set built by msmarco_prefetch.py."""
    p = Path(candidate_cache)
    if not p.exists():
        raise SystemExit(
            f"candidate cache {p} missing. Run scripts/msmarco_prefetch.py first."
        )
    return [json.loads(l) for l in p.read_text().splitlines()]


def ndcg_from_scores(scores: list[float]) -> float:
    """nDCG@10 with the positive at index 0 of the candidate list."""
    order = np.argsort(-np.array(scores), kind="stable")
    ranked_rel = [1.0 if i == 0 else 0.0 for i in order]
    return ndcg_at_k(ranked_rel, 10)


def evaluate(cands, score_fn) -> list[float]:
    """Per-candidate nDCG@10 under score_fn(qid, qtext, doc_id) -> float."""
    ndcgs = []
    for c in cands:
        urls = [c["pos_doc_id"]] + c["neg_doc_ids"]
        scores = []
        ok = True
        for d in urls:
            s = score_fn(c["query_text"], d)
            if s is None:
                ok = False
                break
            scores.append(s)
        if ok:
            ndcgs.append(ndcg_from_scores(scores))
    return ndcgs


def score_with_cache(cache, query_text: str, doc_id: str) -> float | None:
    """Look up cached (s_base, h) for a (query_text, doc_id) pair.

    The msmarco_prefetch writes cache entries keyed by sha256(query||doc_id),
    bypassing the default pair_hash in monot5_backbone.
    """
    import hashlib
    key = hashlib.sha256(f"{query_text}\t{doc_id}".encode()).hexdigest()[:16]
    feat = cache._mem.get(key)
    if feat is None:
        return None
    return feat[0]


def report(system: str, ndcg_base: list[float], ndcg_pers: list[float], seed: int = 42, extra: dict | None = None):
    mean, lo, hi = paired_bootstrap_ci(ndcg_pers, ndcg_base, n_bootstrap=10000, seed=seed)
    result = {
        "system": system,
        "n_candidates": len(ndcg_base),
        "baseline_ndcg10": float(np.mean(ndcg_base)),
        "personalized_ndcg10": float(np.mean(ndcg_pers)),
        "delta": float(mean),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }
    if extra:
        result.update(extra)
    print(json.dumps(result, indent=2))
    return result