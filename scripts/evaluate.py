"""
Evaluate a trained Pref-Rerank model on test sessions.

Computes session-level nDCG@10, MRR, Recall@10, Recall@20 for:
  - user-agnostic baseline (s_base only)
  - personalized model (s = s_base + w^T u + h^T A u)

Then paired bootstrap 95% CI on the nDCG@10 difference (R021).
Stratified eval: high-rank (≤3) vs low-rank (>3) clicks (R006).

NOTE on ground truth (CRITICAL):
  AOL test sessions provide a single clicked doc per eval query. We treat the
  clicked doc as relevance=1 and all other candidate docs as relevance=0.
  This IS the dataset's actual ground truth (click signal), NOT another model's
  output. This satisfies the experiment-bridge ground-truth rule.

  For a proper ranking metric, we need MULTIPLE candidates per eval query. We
  build the candidate set from: the clicked doc + 9 negative docs drawn from
  the same query's cross-session click pool (same approach as training pairs).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aol_session import (  # noqa: E402
    Session,
    load_aol_sessions_cached_or_fresh,
    build_training_pairs,
    split_sessions,
    TrainingPair,
)
from src.eval.metrics import (  # noqa: E402
    SessionEval,
    aggregate_session_evals,
    paired_bootstrap_ci,
    stratified_eval,
    ndcg_at_k,
    mrr,
    recall_at_k,
)
from src.extract.schema_extractor import SchemaExtractor, random_schema  # noqa: E402
from src.model.queryflow_backbone import FeatureCache, load_features  # noqa: E402
from src.model.score_intervention import (  # noqa: E402
    ScoreIntervention,
    ScoreInterventionConfig,
    ClassicFeatures,
    classic_to_vector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval")


def build_eval_candidates(
    test_pairs: list[TrainingPair],
    neg_pool_pairs: list[TrainingPair],
    n_negatives: int = 9,
    seed: int = 42,
) -> list[dict]:
    """For each test pair, build candidate set: 1 positive + n_negatives negatives.

    neg_pool_pairs: pairs to mine negatives from. Should be the SAME split as
    test_pairs (user-disjoint from train/dev) to avoid leakage.

    NOTE (from GPT-5.4 review): AOL has no full-SERP impressions, so negatives
    are "cross-session clicked distractors" — URLs clicked for the same query by
    OTHER users in the same split. This is a PROXY for true non-relevance.
    The paper must report this as "click-based reranking under sampled candidate
    sets", not as general relevance evaluation.
    """
    from collections import defaultdict

    rng = random.Random(seed)
    query_to_neg_urls: dict[str, list[str]] = defaultdict(list)
    for p in neg_pool_pairs:
        # Skip pairs with empty query (AOL has some empty-query logs)
        if not p.eval_query or not p.eval_query.strip():
            continue
        query_to_neg_urls[p.eval_query].append(p.neg_url)
        query_to_neg_urls[p.eval_query].append(p.pos_url)

    out = []
    for p in test_pairs:
        # Skip pairs with empty query
        if not p.eval_query or not p.eval_query.strip():
            continue
        # Exclude history URLs from negatives (same rule as training)
        history_urls = {u for (_, u) in p.user_history}
        neg_pool = [
            u
            for u in query_to_neg_urls.get(p.eval_query, [])
            if u != p.pos_url and u not in history_urls
        ]
        if len(neg_pool) < n_negatives:
            continue
        negs = rng.sample(neg_pool, n_negatives)
        out.append({
            "session_id": p.session_id,
            "eval_query": p.eval_query,
            "user_history": p.user_history,
            "pos_url": p.pos_url,
            "neg_urls": negs,
            "pos_rank": p.pos_rank,
        })
    return out


def score_candidates_baseline(cand: dict, cache: FeatureCache) -> list[float] | None:
    """User-agnostic baseline: score candidates by s_base only."""
    urls = [cand["pos_url"]] + cand["neg_urls"]
    scores = []
    for url in urls:
        try:
            s_base, _ = load_features(cand["eval_query"], url, cache)
        except KeyError:
            return None
        scores.append(float(s_base))
    return scores


def score_candidates_personalized(
    cand: dict,
    cache: FeatureCache,
    model: ScoreIntervention,
    extractor: SchemaExtractor,
    variant: str,
    seed: int,
    u_dim: int = 16,
) -> list[float] | None:
    """Personalized: score by s = s_base + w^T u + h^T A u."""
    # Get user vector
    if variant in ("full", "no_user_bias", "linear_only"):
        schema = extractor.extract(cand["user_history"])
        if schema is None:
            return None
        u = torch.tensor([schema.to_vector()], dtype=torch.float32)
    elif variant == "random_schema":
        u = torch.tensor([random_schema(abs(hash(cand["session_id"])) + seed).to_vector()], dtype=torch.float32)
    elif variant in ("classic", "concat_linear", "film"):
        # Build classic features
        history = cand["user_history"]
        eval_q = cand["eval_query"]
        eval_tokens = set(eval_q.lower().split())
        overlaps = []
        for hq, _ in history:
            ht = set(hq.lower().split())
            if eval_tokens or ht:
                overlaps.append(len(eval_tokens & ht) / len(eval_tokens | ht))
        feat = ClassicFeatures(
            bm25_click_overlap=float(np.mean(overlaps)) if overlaps else 0.0,
            avg_dwell_proxy=float(len(history)),
            avg_rank=3.0,
            history_len=float(len(history)),
            query_len=float(len(eval_tokens)),
            has_reformulation=1.0 if any(overlaps) else 0.0,
        )
        u = classic_to_vector(feat, u_dim=u_dim).unsqueeze(0)
    else:
        raise ValueError(variant)

    urls = [cand["pos_url"]] + cand["neg_urls"]
    s_bases, hs = [], []
    for url in urls:
        try:
            s_base, h = load_features(cand["eval_query"], url, cache)
        except KeyError:
            return None
        s_bases.append(float(s_base))
        hs.append(h)
    s_base_t = torch.tensor(s_bases, dtype=torch.float32)
    h_t = torch.tensor(hs, dtype=torch.float32)
    u_t = u.expand(len(urls), -1)
    with torch.no_grad():
        scores = model(s_base_t, h_t, u_t).tolist()
    return scores


def eval_one(cand: dict, scores: list[float]) -> SessionEval:
    """Compute session-level metrics from scores. Relevance = [1, 0, 0, ...]."""
    ranked = sorted(zip(scores, [1] + [0] * (len(scores) - 1)), reverse=True)
    ranked_rel = [r for _, r in ranked]
    return SessionEval(
        session_id=cand["session_id"],
        ndcg10=ndcg_at_k(ranked_rel, k=10),
        mrr_val=mrr(ranked_rel),
        recall10=recall_at_k(ranked_rel, total_relevant=1, k=10),
        recall20=recall_at_k(ranked_rel, total_relevant=1, k=20),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Trained model JSON from train.py")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--feature-cache", default="cache/features_queryflow")
    ap.add_argument("--schema-cache", default="cache/schema")
    ap.add_argument("--extractor", default="codex", choices=["codex", "phi3"])
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--history-k", type=int, default=3)
    ap.add_argument("--topk-debias", type=int, default=3)
    ap.add_argument("--n-negatives", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--u-dim", type=int, default=16, help="User vector dim (must match training)")
    args = ap.parse_args()

    # Load model
    model_state = json.loads(Path(args.model).read_text())
    cfg = ScoreInterventionConfig(variant=args.variant, u_dim=args.u_dim)
    model = ScoreIntervention(cfg)
    sd = {k: torch.tensor(v) for k, v in model_state["state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()

    # Load data — build test pairs ONLY from test split (no cross-split leakage)
    sessions = load_aol_sessions_cached_or_fresh(min_session_len=args.history_k + 1, max_sessions=args.max_sessions, seed=args.seed)
    train_s, dev_s, test_s = split_sessions(sessions, seed=args.seed)
    test_pairs = build_training_pairs(test_s, history_k=args.history_k, topk_debias=args.topk_debias, seed=args.seed)
    # Build a comprehensive neg pool from ALL clicks in test_s (not just test_pairs)
    # This gives more negatives per query, preventing eval collapse
    from collections import defaultdict
    test_query_to_urls: dict[str, list[str]] = defaultdict(list)
    for s in test_s:
        for ev in s.events:
            if ev.query and ev.query.strip():
                for it in ev.clicked_items:
                    test_query_to_urls[ev.query].append(it.doc_id)
    # Convert to TrainingPair-like objects (just need eval_query + pos_url/neg_url)
    from src.data.aol_session import TrainingPair
    neg_pool_pairs = []
    for q, urls in test_query_to_urls.items():
        if len(urls) >= 2:
            for i, u in enumerate(urls):
                neg_pool_pairs.append(TrainingPair(
                    session_id="", user_history=[], eval_query=q,
                    pos_url=u, neg_url=urls[(i+1) % len(urls)],
                    pos_rank=None, neg_rank=None,
                ))
    logger.info("Built neg pool: %d entries from test_s (%d unique queries)", len(neg_pool_pairs), len(test_query_to_urls))

    candidates = build_eval_candidates(test_pairs, neg_pool_pairs, n_negatives=args.n_negatives, seed=args.seed)
    logger.info("Eval candidates: %d (from %d test pairs)", len(candidates), len(test_pairs))

    cache = FeatureCache(args.feature_cache)
    if args.extractor == "phi3":
        from src.extract.phi3_extractor import Phi3SchemaExtractor
        extractor = Phi3SchemaExtractor(cache_dir=args.schema_cache)
        logger.info("Using Phi-3-mini local extractor")
    else:
        extractor = SchemaExtractor(cache_dir=args.schema_cache)

    # Score
    base_evals, pers_evals = [], []
    for i, cand in enumerate(candidates):
        bs = score_candidates_baseline(cand, cache)
        ps = score_candidates_personalized(cand, cache, model, extractor, args.variant, args.seed, u_dim=args.u_dim)
        if bs is None or ps is None:
            continue
        base_evals.append(eval_one(cand, bs))
        pers_evals.append(eval_one(cand, ps))
        if (i + 1) % 100 == 0:
            logger.info("Scored %d/%d", i + 1, len(candidates))

    # Aggregate
    base_agg = aggregate_session_evals(base_evals)
    pers_agg = aggregate_session_evals(pers_evals)

    # Paired bootstrap CI on nDCG@10
    base_ndcg = [e.ndcg10 for e in base_evals]
    pers_ndcg = [e.ndcg10 for e in pers_evals]
    mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(pers_ndcg, base_ndcg, seed=args.seed)

    # Stratified by pos_rank
    high_rank_sids = {e.session_id for e in pers_evals if any(c["session_id"] == e.session_id and (c["pos_rank"] or 99) <= 3 for c in candidates)}
    base_high, base_low = stratified_eval(base_evals, high_rank_sids)
    pers_high, pers_low = stratified_eval(pers_evals, high_rank_sids)

    out = {
        "variant": args.variant,
        "model_file": args.model,
        "n_eval_sessions": len(base_evals),
        "baseline": base_agg,
        "personalized": pers_agg,
        "delta_ndcg10": pers_agg["ndcg10"] - base_agg["ndcg10"],
        "paired_bootstrap_ci": {
            "mean_diff": mean_diff,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "significant": ci_lo > 0 or ci_hi < 0,
        },
        "stratified": {
            "high_rank_le3": {
                "n": base_high["n_sessions"],
                "baseline_ndcg10": base_high["ndcg10"],
                "personalized_ndcg10": pers_high["ndcg10"],
                "delta": pers_high["ndcg10"] - base_high["ndcg10"],
            },
            "low_rank_gt3": {
                "n": base_low["n_sessions"],
                "baseline_ndcg10": base_low["ndcg10"],
                "personalized_ndcg10": pers_low["ndcg10"],
                "delta": pers_low["ndcg10"] - base_low["ndcg10"],
            },
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    logger.info("Results saved to %s", args.output)
    logger.info("Baseline nDCG@10=%.4f, Personalized nDCG@10=%.4f, delta=%+.4f (95%% CI [%.4f, %.4f])",
                base_agg["ndcg10"], pers_agg["ndcg10"], out["delta_ndcg10"], ci_lo, ci_hi)


if __name__ == "__main__":
    main()
