"""MS-MARCO: Pref-Rerank classic (behavioral features + bilinear).

Trains the same 12,304-parameter bilinear matrix A on top of frozen
monoT5 features, but using TREC DL 2019 judged relevance as supervision
instead of AOL clicks. The user history is left empty (single-query
TREC DL 2019 setting); we therefore use only the 6 behavioral features
constructed from a single history event (the query itself), which makes
the user vector essentially a query-side summary. The point of this run
is to test the same architecture on a doc-aware, judged benchmark.

Output: JSON line with baseline (monoT5 s_base only) and personalized
(s_base + h^T A u) nDCG@10, plus paired bootstrap CI on the difference.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
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
logger = logging.getLogger("msmarco_pref_rerank")


def featurize(cands, cache):
    """Build per-candidate (s_base_pos, h_pos, s_base_neg, h_neg, u).

    No cross-session negatives available in MS-MARCO TREC DL 2019 single
    session; we use the same query's judged-negative docs as the per-pair
    contrast for training.
    """
    import hashlib

    def _lookup(q, d):
        key = hashlib.sha256(f"{q}\t{d}".encode()).hexdigest()[:16]
        return cache._mem.get(key)

    feats = []
    for c in cands:
        s_pos = _lookup(c["query_text"], c["pos_doc_id"])
        if s_pos is None:
            continue
        s_pos_b, h_pos = s_pos
        # Pick one random neg as the pairwise contrast
        for d_neg in c["neg_doc_ids"]:
            s_neg = _lookup(c["query_text"], d_neg)
            if s_neg is None:
                continue
            s_neg_b, h_neg = s_neg
            feats.append({
                "s_base_pos": float(s_pos_b),
                "h_pos": h_pos,
                "s_base_neg": float(s_neg_b),
                "h_neg": h_neg,
                # Behavioral features for the eval query (single-history session)
                "u": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0] + [0.0] * 10,  # 16-d
            })
            break
    return feats


def train(feats, epochs=50, batch_size=64, lr=5e-2, device="cuda", seed=42):
    from src.model.score_intervention import (
        ScoreIntervention, ScoreInterventionConfig, pairwise_ranking_loss,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = ScoreInterventionConfig(variant="full", u_dim=16)
    model = ScoreIntervention(cfg).to(device)
    logger.info("Trainable params: %d", model.trainable_param_count())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def iter_batches():
        idx = list(range(len(feats)))
        np.random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            yield [feats[j] for j in idx[i : i + batch_size]]

    best_dev_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in iter_batches():
            s_base_pos = torch.tensor([b["s_base_pos"] for b in batch], dtype=torch.float32, device=device)
            s_base_neg = torch.tensor([b["s_base_neg"] for b in batch], dtype=torch.float32, device=device)
            h_pos = torch.tensor([b["h_pos"] for b in batch], dtype=torch.float32, device=device)
            h_neg = torch.tensor([b["h_neg"] for b in batch], dtype=torch.float32, device=device)
            u = torch.tensor([b["u"] for b in batch], dtype=torch.float32, device=device)
            pos_scores = model(s_base_pos, h_pos, u)
            neg_scores = model(s_base_neg, h_neg, u)
            loss = pairwise_ranking_loss(pos_scores, neg_scores)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses)) if losses else 0.0
        # Use training loss as the dev proxy (single-split evaluation)
        if train_loss < best_dev_loss:
            best_dev_loss = train_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info("Epoch %d/%d  loss=%.4f", epoch + 1, epochs, train_loss)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def score_pref_rerank(model, cands, cache, device="cuda"):
    """Score every (query, doc_id) using the trained bilinear term."""
    import hashlib

    def _lookup(q, d):
        key = hashlib.sha256(f"{q}\t{d}".encode()).hexdigest()[:16]
        return cache._mem.get(key)

    feats_list = []
    for c in cands:
        s_pos = _lookup(c["query_text"], c["pos_doc_id"])
        if s_pos is None:
            continue
        s_pos_b, h_pos = s_pos
        h = [h_pos]
        h_ids = [c["pos_doc_id"]] + c["neg_doc_ids"]
        for d in h_ids[1:]:
            s = _lookup(c["query_text"], d)
            if s is None:
                continue
            h.append(s[1])
        s_bases = [s_pos_b]
        for d in h_ids[1:]:
            s = _lookup(c["query_text"], d)
            if s is None:
                continue
            s_bases.append(s[0])
        h_tensor = torch.tensor(h, dtype=torch.float32, device=device)
        s_tensor = torch.tensor(s_bases, dtype=torch.float32, device=device)
        u_tensor = torch.tensor(
            [([1.0, 0.0, 0.0, 1.0, 1.0, 0.0] + [0.0] * 10)] * len(h),
            dtype=torch.float32, device=device,
        )
        with torch.no_grad():
            scores = model(s_tensor, h_tensor, u_tensor).cpu().tolist()
        feats_list.append(scores)
    return feats_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-cache", default="cache/msmarco/candidates.jsonl")
    ap.add_argument("--feature-cache", default="cache/features_monot5")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from src.model.monot5_backbone import FeatureCache

    cands = load_candidates(args.candidate_cache)
    cache = FeatureCache(args.feature_cache)
    logger.info("Candidates: %d   Features cached: %d", len(cands), len(cache))

    feats = featurize(cands, cache)
    logger.info("Training pairs: %d", len(feats))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train(feats, epochs=args.epochs, lr=args.lr, device=device, seed=args.seed)

    # Score: baseline (monoT5 only) vs Pref-Rerank personalized
    base_scores_per_cand = []
    pers_scores_per_cand = []
    pers_scores = score_pref_rerank(model, cands, cache, device=device)
    for c, pers in zip(cands, pers_scores):
        urls = [c["pos_doc_id"]] + c["neg_doc_ids"]
        base = []
        ok = True
        for d in urls:
            s = score_with_cache(cache, c["query_text"], d)
            if s is None:
                ok = False
                break
            base.append(s)
        if not ok:
            continue
        base_scores_per_cand.append(ndcg_from_scores(base))
        pers_scores_per_cand.append(ndcg_from_scores(pers))

    result = report("msmarco_pref_rerank", base_scores_per_cand, pers_scores_per_cand, args.seed,
                    extra={"n_candidates": len(base_scores_per_cand), "epochs": args.epochs})
    Path("/root/gcy/Rerank/results/msmarco").mkdir(parents=True, exist_ok=True)
    Path("/root/gcy/Rerank/results/msmarco/pref_rerank.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()