"""B5: Pref-Rerank classic under the SAME baseline harness protocol.

Loads the trained classic model (results/r200k_classic_seed42.json),
scores candidates with s = s_base + w^T u + h^T A u, and reports
per-session nDCG@10 vs the user-agnostic baseline under the exact
same candidate construction as b1-b4 (common.load_protocol).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.baselines.common import load_protocol, ndcg_from_scores, report  # noqa: E402
from src.model.score_intervention import (  # noqa: E402
    ScoreIntervention, ScoreInterventionConfig,
    ClassicFeatures, classic_to_vector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("b5_pref_rerank")


def classic_features_from_history(eval_query: str, history: list[tuple[str, str]]) -> ClassicFeatures:
    """Same feature computation as scripts/train.py _classic_features_from_history."""
    eval_tokens = set(eval_query.lower().split())
    overlaps = []
    for hq, _ in history:
        ht = set(hq.lower().split())
        if eval_tokens or ht:
            overlaps.append(len(eval_tokens & ht) / len(eval_tokens | ht))
    return ClassicFeatures(
        bm25_click_overlap=float(np.mean(overlaps)) if overlaps else 0.0,
        avg_dwell_proxy=float(len(history)),
        avg_rank=3.0,
        history_len=float(len(history)),
        query_len=float(len(eval_tokens)),
        has_reformulation=1.0 if any(overlaps) else 0.0,
    )


def evaluate(cands, cache, model, u_dim: int) -> tuple[list[float], list[float]]:
    base, pers = [], []
    for c in cands:
        urls = [c["pos_url"]] + c["neg_urls"]
        s_base, s_pers = [], []
        ok = True
        feat = classic_features_from_history(c["eval_query"], c["user_history"])
        u = classic_to_vector(feat, u_dim=u_dim)
        for url in urls:
            pair = cache.get(c["eval_query"], url)
            if pair is None:
                ok = False
                break
            s_b, h = pair
            s_base.append(float(s_b))
            s_pers.append(float(model(torch.tensor([s_b]), torch.tensor([h]), u.unsqueeze(0))[0]))
        if ok:
            base.append(ndcg_from_scores(s_base))
            pers.append(ndcg_from_scores(s_pers))
    return base, pers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-cache", default="cache/data/sessions.pkl")
    ap.add_argument("--feature-cache", default="cache/features_queryflow")
    ap.add_argument("--model", default="results/r200k_classic_seed42.json")
    ap.add_argument("--n-negatives", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--u-dim", type=int, default=16)
    args = ap.parse_args()

    cands, cache, _, _ = load_protocol(
        session_cache=args.session_cache,
        feature_cache=args.feature_cache,
        n_negatives=args.n_negatives,
        seed=args.seed,
    )

    spec = json.loads(Path(args.model).read_text())
    cfg = ScoreInterventionConfig(variant="classic", u_dim=args.u_dim)
    model = ScoreIntervention(cfg)
    model.load_state_dict({k: torch.tensor(v) for k, v in spec["state_dict"].items()})
    model.eval()
    logger.info("Loaded model %s (params=%d)", args.model, model.trainable_param_count())

    ndcg_base, ndcg_pers = evaluate(cands, cache, model, args.u_dim)
    report("pref-rerank-classic", ndcg_base, ndcg_pers, args.seed,
           extra={"model": args.model})


if __name__ == "__main__":
    main()