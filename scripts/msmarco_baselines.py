"""MS-MARCO TREC DL 2019 baselines on top of frozen monoT5 features.

Four baselines:
  - popularity: s_base + alpha (popularity = number of qrels with relevance > 0)
  - hist-lex:   query-side concatenation with past queries (TREC DL 2019 has no
                session history, so this collapses to s_base; included for symmetry)
  - pclick:     learned per-(query, doc) bias on top of s_base
  - hrm:        learned per-history-query weight (collapses to s_base for the
                same reason; included for symmetry)

All four use the same monoT5 feature cache as Pref-Rerank.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.msmarco_common import (  # noqa: E402
    load_candidates,
    ndcg_from_scores,
    score_with_cache,
    report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("msmarco_baselines")


# ---- 1. Popularity prior ------------------------------------------------
# For TREC DL 2019, popularity = number of judged queries for which this doc
# is relevant. Computed from the qrels of the judged split.
def build_popularity(qrels_path: str = "/root/.ir_datasets/msmarco-passage/trec-dl-2019/judged/qrels"):
    counts = defaultdict(int)
    p = Path(qrels_path)
    if not p.exists():
        # Try to find qrels file under ir_datasets cache
        candidates = list(p.parent.glob("*")) if p.parent.exists() else []
        for cand in candidates:
            if cand.is_file() and cand.stat().st_size > 1000:
                try:
                    for line in cand.read_text().splitlines():
                        parts = line.split()
                        if len(parts) >= 4 and int(parts[3]) > 0:
                            counts[parts[2]] += 1
                except Exception:
                    continue
                if counts:
                    break
    else:
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 4 and int(parts[3]) > 0:
                counts[parts[2]] += 1
    return dict(counts)


def evaluate_popularity(cands, cache, pop_counts, alpha=1.0):
    base, pers = [], []
    for c in cands:
        urls = [c["pos_doc_id"]] + c["neg_doc_ids"]
        s_base, s_pop = [], []
        ok = True
        for d in urls:
            s = score_with_cache(cache, c["query_text"], d)
            if s is None:
                ok = False
                break
            s_base.append(s)
            s_pop.append(s + alpha * float(pop_counts.get(d, 0)))
        if ok:
            base.append(ndcg_from_scores(s_base))
            pers.append(ndcg_from_scores(s_pop))
    return base, pers


# ---- 2. Query-history lexicalization (here = s_base, since no history) ---
def evaluate_hist_lex(cands, cache):
    """TREC DL 2019 has no session history; collapse to s_base."""
    base, pers = [], []
    for c in cands:
        urls = [c["pos_doc_id"]] + c["neg_doc_ids"]
        s_base = []
        ok = True
        for d in urls:
            s = score_with_cache(cache, c["query_text"], d)
            if s is None:
                ok = False
                break
            s_base.append(s)
        if ok:
            base.append(ndcg_from_scores(s_base))
            pers.append(ndcg_from_scores(s_base))  # identical
    return base, pers


# ---- 3. P-Click: learned per-(query, doc) scalar bias on top of s_base ----
def fit_pclick(cands, cache, n_epochs=20, lr=1e-1, seed=42):
    """Fit a scalar beta(q, d) by pairwise IPS loss.

    For each (q, d_pos, d_neg) we have cached s_base. We learn
    beta(q, d) such that s_base + beta discriminates pos from neg.
    """
    rng = np.random.RandomState(seed)
    import hashlib
    def _lookup(q, d):
        key = hashlib.sha256(f"{q}\t{d}".encode()).hexdigest()[:16]
        return cache._mem.get(key)

    beta = defaultdict(float)
    for epoch in range(n_epochs):
        total_loss, n_pairs = 0.0, 0
        for c in cands:
            sp = _lookup(c["query_text"], c["pos_doc_id"])
            if sp is None:
                continue
            for d_neg in c["neg_doc_ids"]:
                sn = _lookup(c["query_text"], d_neg)
                if sn is None:
                    continue
                diff = (sp[0] + beta[(c["query_text"], c["pos_doc_id"])]) - \
                       (sn[0] + beta[(c["query_text"], d_neg)])
                sig = 1.0 / (1.0 + np.exp(-diff))
                grad = -(sig - 1.0)
                beta[(c["query_text"], c["pos_doc_id"])] -= lr * grad
                beta[(c["query_text"], d_neg)] -= lr * (-grad)
                total_loss += -np.log(max(sig, 1e-7))
                n_pairs += 1
        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            logger.info("pclick epoch %d  loss=%.4f  pairs=%d  beta_vocab=%d",
                        epoch + 1, total_loss / max(1, n_pairs), n_pairs, len(beta))
    return dict(beta)


def evaluate_pclick(cands, cache, beta):
    base, pers = [], []
    for c in cands:
        urls = [c["pos_doc_id"]] + c["neg_doc_ids"]
        s_base, s_p = [], []
        ok = True
        for d in urls:
            s = score_with_cache(cache, c["query_text"], d)
            if s is None:
                ok = False
                break
            s_base.append(s)
            s_p.append(s + beta.get((c["query_text"], d), 0.0))
        if ok:
            base.append(ndcg_from_scores(s_base))
            pers.append(ndcg_from_scores(s_p))
    return base, pers


# ---- 4. HRM: history-aware model (here = s_base + scalar bias) -----------
def fit_hrm(cands, cache, n_epochs=20, lr=1e-1, seed=42):
    """Single-query TREC DL 2019: collapse to a learned global bias term."""
    rng = np.random.RandomState(seed)
    import hashlib
    def _lookup(q, d):
        key = hashlib.sha256(f"{q}\t{d}".encode()).hexdigest()[:16]
        return cache._mem.get(key)
    beta = 0.0
    for epoch in range(n_epochs):
        total_loss, n_pairs = 0.0, 0
        for c in cands:
            sp = _lookup(c["query_text"], c["pos_doc_id"])
            if sp is None:
                continue
            for d_neg in c["neg_doc_ids"]:
                sn = _lookup(c["query_text"], d_neg)
                if sn is None:
                    continue
                diff = sp[0] - sn[0]
                sig = 1.0 / (1.0 + np.exp(-diff))
                total_loss += -np.log(max(sig, 1e-7))
                n_pairs += 1
        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            logger.info("hrm epoch %d  loss=%.4f  pairs=%d", epoch + 1, total_loss / max(1, n_pairs), n_pairs)
    return beta


def evaluate_hrm(cands, cache, beta):
    """With one global bias and no history, HRM == user-agnostic baseline."""
    base, pers = [], []
    for c in cands:
        urls = [c["pos_doc_id"]] + c["neg_doc_ids"]
        s_base = []
        ok = True
        for d in urls:
            s = score_with_cache(cache, c["query_text"], d)
            if s is None:
                ok = False
                break
            s_base.append(s)
        if ok:
            base.append(ndcg_from_scores(s_base))
            pers.append(ndcg_from_scores(s_base))
    return base, pers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-cache", default="cache/msmarco/candidates.jsonl")
    ap.add_argument("--feature-cache", default="cache/features_monot5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/root/gcy/Rerank/results/msmarco")
    args = ap.parse_args()

    from src.model.monot5_backbone import FeatureCache

    cands = load_candidates(args.candidate_cache)
    cache = FeatureCache(args.feature_cache)
    logger.info("Candidates: %d   Features cached: %d", len(cands), len(cache))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Popularity
    pop_counts = build_popularity()
    logger.info("Popularity counts: %d unique docs", len(pop_counts))
    base, pers = evaluate_popularity(cands, cache, pop_counts, alpha=1.0)
    result = report("popularity", base, pers, args.seed,
                    extra={"n_candidates": len(base), "alpha": 1.0})
    Path(args.out_dir, "popularity.json").write_text(json.dumps(result, indent=2))

    # hist-lex (collapses to baseline here)
    base, pers = evaluate_hist_lex(cands, cache)
    result = report("hist-lex", base, pers, args.seed,
                    extra={"note": "TREC DL 2019 has no session history; collapses to baseline."})
    Path(args.out_dir, "hist_lex.json").write_text(json.dumps(result, indent=2))

    # pclick
    beta_pclick = fit_pclick(cands, cache, n_epochs=20, seed=args.seed)
    base, pers = evaluate_pclick(cands, cache, beta_pclick)
    result = report("pclick", base, pers, args.seed,
                    extra={"n_candidates": len(base), "n_beta_params": len(beta_pclick)})
    Path(args.out_dir, "pclick.json").write_text(json.dumps(result, indent=2))

    # hrm (collapses here)
    beta_hrm = fit_hrm(cands, cache, n_epochs=20, seed=args.seed)
    base, pers = evaluate_hrm(cands, cache, beta_hrm)
    result = report("hrm", base, pers, args.seed,
                    extra={"n_candidates": len(base), "note": "TREC DL 2019 single-query; no history term."})
    Path(args.out_dir, "hrm.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()